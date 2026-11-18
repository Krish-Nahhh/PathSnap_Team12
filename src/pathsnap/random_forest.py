"""
random_forest.py – Random Forest candidate classifier for PathSnap map-matching.

Responsibility: Krishna H Zalavadiya
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
Given a stream of (trip_id, Candidate) rows from candidates.py and a label
mapping {(trip_id, observation_index): true_arc_id}, this module:

  1. Extracts a rich feature vector for every candidate (distance, heading
     error, rank, rank-normalised distance, missing-heading flag).
  2. Trains a scikit-learn RandomForestClassifier with class-weight balancing
     (true-arc rows are rare – typically 1 out of top_k candidates).
  3. Exposes `score_candidates(model, candidates)` that returns P(true arc)
     for each candidate, in original list order, so the shared evaluator and
     Viterbi decoder can consume the output directly.

INTERFACE CONTRACT (see MODEL_OWNERSHIP.md)
-------------------------------------------
  train_random_forest(training_rows, labels) -> fitted sklearn Pipeline
  score_candidates(model, candidates)        -> list[float]  (same order)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .candidates import Candidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "distance_m",          # perpendicular distance from GPS fix to arc (m)
    "heading_error_deg",   # angular difference between GPS bearing and arc bearing (deg)
    "heading_available",   # 1 if heading was computable, 0 otherwise (indicator)
    "candidate_rank",      # 1-based rank by distance (smaller = closer)
    "inv_rank",            # 1 / candidate_rank  (nonlinear emphasis on top candidates)
    "dist_x_rank",         # distance_m * candidate_rank  (joint penalty feature)
)


def candidate_features(candidate: "Candidate") -> tuple[float, ...]:
    """
    Map a single Candidate namedtuple to a fixed-length numeric feature vector.

    Missing heading errors are imputed with 0.0 (neutral) and flagged by the
    ``heading_available`` indicator so the model learns separate behaviour for
    observations where GPS heading was unavailable (first fix, long gaps, etc.).

    Parameters
    ----------
    candidate : Candidate
        A single row from candidates.generate_candidates().

    Returns
    -------
    tuple of floats
        Feature values in the order defined by FEATURE_NAMES.
    """
    error = candidate.heading_error_deg
    heading_available = int(error is not None)
    imputed_error = 0.0 if error is None else error        # 0 is neutral imputation

    rank = candidate.candidate_rank
    inv_rank = 1.0 / rank                                  # higher for top-1 candidate
    dist_x_rank = candidate.distance_m * rank              # joint distance+rank penalty

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

def train_random_forest(
    training_rows: Sequence[tuple[str, "Candidate"]],
    labels: dict[tuple[str, int], int],
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 4,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> object:
    """
    Fit a Random Forest binary classifier on candidate rows from training trips.

    The classifier learns P(candidate == true arc) for each row.
    Because only 1 candidate per observation is the true arc (top_k is
    typically 10), the dataset is naturally imbalanced; we correct this via
    class_weight='balanced'.

    Parameters
    ----------
    training_rows : sequence of (trip_id, Candidate)
        All (trip_id, Candidate) pairs belonging to the **training** partition
        for this fold.  The caller must supply only training data — never
        mix validation observations here.
    labels : dict mapping (trip_id, observation_index) -> true_arc_id
        Ground-truth arc IDs for every labeled observation in the training set.
        Unlabeled observations are silently skipped.
    n_estimators : int, default 300
        Number of trees.  300 gives a solid bias/variance trade-off; tune
        upward if the CV score is still improving past this point.
    max_depth : int or None, default None
        Maximum tree depth (None = fully grown).  Fully grown trees with
        balanced class_weight and min_samples_leaf give low bias; the ensemble
        controls variance.
    min_samples_leaf : int, default 4
        Minimum samples per leaf — acts as a light regulariser to prevent
        overfitting on rare arc-IDs.
    max_features : str, default 'sqrt'
        Number of features considered at each split (scikit-learn default for
        classification). 'sqrt' reduces correlation between trees.
    random_state : int, default 42
        Seed for full reproducibility of tree growth and bootstrap samples.

    Returns
    -------
    fitted sklearn Pipeline  (StandardScaler -> RandomForestClassifier)
        Call score_candidates(model, candidates) to get probabilities.
        StandardScaler is kept for interface consistency with LR/NB so the
        shared evaluator can treat all models identically; RF itself is
        invariant to monotonic scaling.

    Raises
    ------
    ValueError
        If the assembled training set contains only one class (all true or
        all false candidates), which would make supervised training impossible.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # ── 1. Group candidates by (trip_id, observation_index) ─────────────────
    groups: dict[tuple[str, int], list["Candidate"]] = {}
    for trip_id, candidate in training_rows:
        key = (trip_id, candidate.observation_index)
        if key in labels:                              # skip unlabeled observations
            groups.setdefault(key, []).append(candidate)

    if not groups:
        raise ValueError(
            "No labeled observations found in training_rows. "
            "Check that trip IDs match between training_rows and labels."
        )

    # ── 2. Build feature matrix and binary target vector ─────────────────────
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    skipped_miss = 0

    for key, candidates in groups.items():
        true_arc = labels[key]
        # Skip observations where the true arc was never generated as a candidate
        # (candidate miss) — including them would add rows with no positive label.
        if not any(c.arc_id == true_arc for c in candidates):
            skipped_miss += 1
            continue
        for c in candidates:
            features.append(candidate_features(c))
            targets.append(int(c.arc_id == true_arc))   # 1 = true arc, 0 = false

    if skipped_miss:
        logger.warning(
            "Skipped %d observations where the true arc was absent from "
            "the candidate list (miss rate: %.1f%%).",
            skipped_miss,
            100.0 * skipped_miss / len(groups),
        )

    unique_classes = set(targets)
    if unique_classes != {0, 1}:
        raise ValueError(
            f"Training requires both positive (true arc) and negative candidates. "
            f"Found only classes: {unique_classes}. "
            "Ensure training_rows spans multiple trips with varied candidates."
        )

    logger.info(
        "Training RandomForest on %d candidate rows (%d labeled observations, "
        "%d positive, %d negative).",
        len(features),
        len(groups) - skipped_miss,
        targets.count(1),
        targets.count(0),
    )

    # ── 3. Build and fit the sklearn Pipeline ────────────────────────────────
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",    # corrects for ~1-positive-in-top_k imbalance
            n_jobs=-1,                  # use all available CPU cores for speed
            random_state=random_state,
        ),
    )
    pipeline.fit(features, targets)

    # ── 4. Log feature importances for interpretability ──────────────────────
    rf: RandomForestClassifier = pipeline.named_steps["randomforestclassifier"]
    importance_pairs = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1]
    )
    logger.info(
        "Feature importances (descending): %s",
        ", ".join(f"{name}={imp:.3f}" for name, imp in importance_pairs),
    )

    return pipeline


# ---------------------------------------------------------------------------
# Scoring / Inference
# ---------------------------------------------------------------------------

def score_candidates(model: object, candidates: Sequence["Candidate"]) -> list[float]:
    """
    Return the probability that each candidate is the true arc.

    Probabilities are in the same order as the input list.  Pass the full
    candidate list for a single GPS fix; the evaluator selects the argmax,
    and the Viterbi decoder uses these as emission probabilities.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The pipeline returned by train_random_forest().
    candidates : sequence of Candidate
        All candidates for a **single** GPS observation (one fix's candidate
        list).  Pass an empty list if the observation has no candidates.

    Returns
    -------
    list of float
        P(true arc) for each candidate, in input order.
        Returns [] when candidates is empty.
    """
    if not candidates:
        return []

    features = [candidate_features(c) for c in candidates]
    # predict_proba returns [P(class=0), P(class=1)]; take column 1 (positive)
    return model.predict_proba(features)[:, 1].tolist()

