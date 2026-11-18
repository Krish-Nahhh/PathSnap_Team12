"""
decision_tree.py – Decision Tree candidate classifier for PathSnap map-matching.

Responsibility: Krithika S
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
Provides a Decision Tree binary classifier that scores each road-arc candidate
with P(candidate == true arc).  Follows the same two-function interface as
every other model in PathSnap so the shared evaluator and Viterbi decoder
can consume it without changes.

INTERFACE CONTRACT  (see MODEL_OWNERSHIP.md)
--------------------------------------------
  train_decision_tree(training_rows, labels) -> fitted sklearn Pipeline
  score_candidates(model, candidates)        -> list[float]  (same order)

DESIGN NOTES
------------
* Decision trees are inherently interpretable — the learned rules can be
  exported as text or rendered as a graphviz plot (see `export_tree_rules`).
* class_weight='balanced' compensates for the ~1:9 positive/negative
  imbalance produced when top_k=10 candidates are generated per fix.
* ccp_alpha (cost-complexity pruning) is the primary regulariser;
  it removes subtrees whose improvement in impurity falls below the threshold.
  Tune it with cross-validation instead of max_depth alone.
* The same 6-feature vector used by RandomForest is reused here for fair
  direct comparison — only the model type changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .candidates import Candidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering  (matches random_forest.py for direct comparison)
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "distance_m",          # perpendicular distance from GPS fix to arc (m)
    "heading_error_deg",   # angular diff between GPS bearing and arc direction (°)
    "heading_available",   # 1 if heading was computable, 0 otherwise
    "candidate_rank",      # 1-based rank by distance (smaller = closer)
    "inv_rank",            # 1 / candidate_rank (nonlinear top-candidate emphasis)
    "dist_x_rank",         # distance_m × rank  (joint distance-rank penalty)
)


def candidate_features(candidate: "Candidate") -> tuple[float, ...]:
    """
    Map one Candidate namedtuple to a fixed-length numeric feature vector.

    Missing heading errors are imputed with 0.0 (neutral) and signalled by
    the ``heading_available`` flag so the tree can branch on sensor availability.

    Parameters
    ----------
    candidate : Candidate
        One row from candidates.generate_candidates().

    Returns
    -------
    tuple of float
        Feature values in the order defined by FEATURE_NAMES.
    """
    error = candidate.heading_error_deg
    heading_available = int(error is not None)
    imputed_error = 0.0 if error is None else error   # neutral imputation

    rank = candidate.candidate_rank
    inv_rank = 1.0 / rank                             # up-weights rank-1
    dist_x_rank = candidate.distance_m * rank         # joint penalty

    return (
        candidate.distance_m,
        imputed_error,
        float(heading_available),
        float(rank),
        inv_rank,
        dist_x_rank,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_decision_tree(
    training_rows: Sequence[tuple[str, "Candidate"]],
    labels: dict[tuple[str, int], int],
    *,
    max_depth: int | None = 15,
    min_samples_leaf: int = 8,
    ccp_alpha: float = 1e-4,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> object:
    """
    Fit a Decision Tree binary classifier on candidate rows from training trips.

    The classifier learns P(candidate == true arc) for every candidate row.
    Because one fix produces top_k candidates with exactly 1 positive label,
    the dataset is ~1:9 positive/negative — corrected by class_weight='balanced'.

    Parameters
    ----------
    training_rows : sequence of (trip_id, Candidate)
        All candidates from the training partition (or the training fold).
        Never mix validation observations in here.
    labels : dict mapping (trip_id, observation_index) -> true_arc_id
        Ground-truth arc IDs for every labeled observation.
        Unlabeled observations are silently skipped.
    max_depth : int or None, default 15
        Maximum tree depth.  Deeper trees memorise training data; control
        overfitting primarily via ccp_alpha and min_samples_leaf.
        Set to None for fully grown trees (pruned only by ccp_alpha).
    min_samples_leaf : int, default 8
        A leaf must have at least this many training samples.
        Prevents the tree from learning single-sample rules.
    ccp_alpha : float, default 1e-4
        Complexity parameter for Minimal Cost-Complexity Pruning (post-pruning).
        0.0 = no pruning; larger values produce smaller trees.
        Tune with `sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path`.
    class_weight : str, default 'balanced'
        Passes directly to DecisionTreeClassifier; 'balanced' up-weights the
        rare positive class by n_negative / n_positive ≈ top_k − 1.
    random_state : int, default 42
        Seed for reproducibility (affects feature tie-breaking in splits).

    Returns
    -------
    fitted sklearn Pipeline  (StandardScaler -> DecisionTreeClassifier)
        Call score_candidates(model, candidates) to get probabilities.
        StandardScaler is kept for interface consistency with LR/NB; trees
        are scale-invariant but the shared evaluator treats all models the same.

    Raises
    ------
    ValueError
        When no labeled observations are found, or only one class is present
        in the assembled training targets.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    # ── 1. Group candidates by (trip_id, observation_index) ─────────────────
    groups: dict[tuple[str, int], list["Candidate"]] = {}
    for trip_id, candidate in training_rows:
        key = (trip_id, candidate.observation_index)
        if key in labels:                              # skip unlabeled observations
            groups.setdefault(key, []).append(candidate)

    if not groups:
        raise ValueError(
            "No labeled observations found in training_rows. "
            "Verify that trip IDs in training_rows match keys in labels."
        )

    # ── 2. Build feature matrix and binary target vector ─────────────────────
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    skipped_miss = 0

    for key, candidates in groups.items():
        true_arc = labels[key]
        # Skip observations where the true arc was never generated as a candidate
        # (candidate miss) — there is no positive training example for them.
        if not any(c.arc_id == true_arc for c in candidates):
            skipped_miss += 1
            continue
        for c in candidates:
            features.append(candidate_features(c))
            targets.append(int(c.arc_id == true_arc))   # 1=true arc, 0=false

    if skipped_miss:
        logger.warning(
            "Skipped %d candidate-miss observations (%.1f%% of labeled set).",
            skipped_miss,
            100.0 * skipped_miss / len(groups),
        )

    unique_classes = set(targets)
    if unique_classes != {0, 1}:
        raise ValueError(
            f"Training requires both positive and negative candidate examples. "
            f"Found only classes: {unique_classes}. "
            "Ensure training data covers multiple trips with varied candidates."
        )

    logger.info(
        "Training DecisionTree on %d candidate rows (%d observations, "
        "%d positive, %d negative).",
        len(features),
        len(groups) - skipped_miss,
        targets.count(1),
        targets.count(0),
    )

    # ── 3. Build and fit the sklearn Pipeline ────────────────────────────────
    pipeline = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            class_weight=class_weight,
            random_state=random_state,
        ),
    )
    pipeline.fit(features, targets)

    # ── 4. Log tree statistics for interpretability ──────────────────────────
    dt: DecisionTreeClassifier = pipeline.named_steps["decisiontreeclassifier"]
    logger.info(
        "DecisionTree fitted: depth=%d, leaves=%d, nodes=%d, "
        "ccp_alpha=%.2e",
        dt.get_depth(),
        dt.get_n_leaves(),
        dt.tree_.node_count,
        ccp_alpha,
    )

    return pipeline


# ---------------------------------------------------------------------------
# Scoring / Inference
# ---------------------------------------------------------------------------

def score_candidates(model: object, candidates: Sequence["Candidate"]) -> list[float]:
    """
    Return P(true arc) for each candidate in input order.

    The probabilities come from the leaf-node class fraction of the fitted
    decision tree, calibrated by class_weight during training.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The pipeline returned by train_decision_tree().
    candidates : sequence of Candidate
        All candidates for a **single** GPS observation.
        Pass an empty list when the observation has no candidates.

    Returns
    -------
    list of float
        P(true arc) for each candidate, in input order.
        Returns [] when candidates is empty.
    """
    if not candidates:
        return []

    features = [candidate_features(c) for c in candidates]
    # predict_proba[:, 1] = probability of the positive class (true arc)
    return model.predict_proba(features)[:, 1].tolist()


# ---------------------------------------------------------------------------
# Interpretability helper
# ---------------------------------------------------------------------------

def export_tree_rules(model: object, *, max_depth: int = 4) -> str:
    """
    Export the decision tree's learned rules as human-readable text.

    Useful for understanding what features the tree relies on most and
    for documenting the model's decision logic in the technical report.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The pipeline returned by train_decision_tree().
    max_depth : int, default 4
        Depth limit for the exported text (keeps output manageable).

    Returns
    -------
    str
        Multiline string with the tree's decision rules.
    """
    from sklearn.tree import export_text

    dt = model.named_steps["decisiontreeclassifier"]
    return export_text(dt, feature_names=list(FEATURE_NAMES), max_depth=max_depth)
