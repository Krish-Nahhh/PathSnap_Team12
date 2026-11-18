"""
hist_gradient_boosting.py – HistGradientBoosting candidate classifier for PathSnap.

Responsibility: Krithika S
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
Provides a Histogram-based Gradient Boosting (HGB) binary classifier — the
scikit-learn equivalent of LightGBM — that scores each road-arc candidate with
P(candidate == true arc).  Follows the same two-function interface as every
other PathSnap model.

INTERFACE CONTRACT  (see MODEL_OWNERSHIP.md)
--------------------------------------------
  train_hist_gradient_boosting(training_rows, labels) -> fitted sklearn Pipeline
  score_candidates(model, candidates)                 -> list[float] (same order)

DESIGN NOTES
------------
* HGB natively handles missing values, so we can pass raw NaN for missing
  heading errors — no imputation step needed.  This is tracked via a separate
  `heading_available` indicator column so the model can distinguish 0° error
  from "no heading".
* class_weight='balanced' compensates for the ~1:9 positive/negative imbalance.
* learning_rate + max_iter are the primary trade-off knobs; smaller lr needs
  more iterations (and is slower but often more accurate).
* early_stopping='auto' uses 10% of training data as an internal validation
  set and stops early if validation loss doesn't improve, preventing overfitting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .candidates import Candidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering  (matches decision_tree.py / random_forest.py)
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
    the `heading_available` flag.  HGB can natively handle NaN, but we use
    explicit imputation + flag for consistency with other models.

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
    imputed_error = 0.0 if error is None else error

    rank = candidate.candidate_rank
    inv_rank = 1.0 / rank
    dist_x_rank = candidate.distance_m * rank

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

def train_hist_gradient_boosting(
    training_rows: Sequence[tuple[str, "Candidate"]],
    labels: dict[tuple[str, int], int],
    *,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1e-2,
    early_stopping: str = "auto",
    random_state: int = 42,
) -> object:
    """
    Fit a HistGradientBoosting binary classifier on candidate rows.

    Parameters
    ----------
    training_rows : sequence of (trip_id, Candidate)
        All candidates from the training partition (or the training fold).
    labels : dict mapping (trip_id, observation_index) -> true_arc_id
        Ground-truth arc IDs for every labeled observation.
    max_iter : int, default 200
        Maximum number of boosting iterations (trees).  With early_stopping,
        the actual number of trees may be lower.
    learning_rate : float, default 0.05
        Step size shrinkage.  Smaller values need more iterations but often
        generalise better.  Tune with CV.
    max_leaf_nodes : int, default 31
        Maximum number of leaves per tree.  Controls model capacity.
    min_samples_leaf : int, default 20
        Minimum samples per leaf.  Acts as a regulariser.
    l2_regularization : float, default 0.01
        L2 penalty on leaf values.  Reduces overfitting on noisy road data.
    early_stopping : str, default 'auto'
        'auto' uses 10% of training data as an internal val set.
        Set to 'off' when training with very small datasets.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    fitted sklearn Pipeline  (StandardScaler -> HistGradientBoostingClassifier)

    Raises
    ------
    ValueError
        When no labeled observations are found or only one class is present.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # ── 1. Group candidates by (trip_id, observation_index) ─────────────────
    groups: dict[tuple[str, int], list["Candidate"]] = {}
    for trip_id, candidate in training_rows:
        key = (trip_id, candidate.observation_index)
        if key in labels:
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
        if not any(c.arc_id == true_arc for c in candidates):
            skipped_miss += 1
            continue
        for c in candidates:
            features.append(candidate_features(c))
            targets.append(int(c.arc_id == true_arc))

    if skipped_miss:
        logger.warning(
            "Skipped %d candidate-miss observations (%.1f%%).",
            skipped_miss,
            100.0 * skipped_miss / len(groups),
        )

    if set(targets) != {0, 1}:
        raise ValueError(
            f"Training requires both positive and negative examples. "
            f"Found only classes: {set(targets)}."
        )

    logger.info(
        "Training HistGradientBoosting on %d candidate rows (%d observations, "
        "%d positive, %d negative).",
        len(features),
        len(groups) - skipped_miss,
        targets.count(1),
        targets.count(0),
    )

    # ── 3. Build and fit the sklearn Pipeline ────────────────────────────────
    pipeline = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            class_weight="balanced",
            early_stopping=early_stopping,
            random_state=random_state,
        ),
    )
    pipeline.fit(features, targets)

    hgb = pipeline.named_steps["histgradientboostingclassifier"]
    actual_iters = hgb.n_iter_
    logger.info("HistGradientBoosting fitted: %d trees used (of %d max).", actual_iters, max_iter)

    return pipeline


# ---------------------------------------------------------------------------
# Scoring / Inference
# ---------------------------------------------------------------------------

def score_candidates(model: object, candidates: Sequence["Candidate"]) -> list[float]:
    """
    Return P(true arc) for each candidate in input order.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The pipeline returned by train_hist_gradient_boosting().
    candidates : sequence of Candidate
        All candidates for a **single** GPS observation.

    Returns
    -------
    list of float
        P(true arc) for each candidate, in input order. Returns [] if empty.
    """
    if not candidates:
        return []
    features = [candidate_features(c) for c in candidates]
    return model.predict_proba(features)[:, 1].tolist()
