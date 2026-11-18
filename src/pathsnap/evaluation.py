"""
evaluation.py – Shared model evaluator for PathSnap map-matching.

Responsibility: Krishna H Zalavadiya (shared module)
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
Provides a single, model-agnostic evaluation harness used by every classifier
in PathSnap.  Any model that exposes the two-function interface:

    train_fn(training_rows, labels)  -> fitted model
    score_fn(model, candidates)      -> list[float]  (same order)

can be dropped into evaluate_model() or cross_validate() without changes.

METRICS COMPUTED
----------------
  Per-observation (pointwise):
    • Top-1 Accuracy        – fraction of observations where the highest-scored
                              candidate is the true arc.
    • Macro F1-Score        – F1 averaged equally over both classes (true/false
                              arc), penalises the model for ignoring the rare
                              positive class.
    • Log Loss              – cross-entropy of the probability scores, measuring
                              calibration quality.

  Per-sequence (route quality):
    • Route Continuity      – fraction of **consecutive fix pairs** where the
                              predicted arcs are topologically connected in the
                              road network (share a node), giving a measure of
                              how smooth and physically plausible the predicted
                              route is.

  Cross-validation (generalisation):
    • CV Macro F1-Score     – mean ± std of Macro F1 over trip-stratified folds.

  Efficiency:
    • Inference Time (ms)   – wall-clock time to score all candidates in the
                              test set, in milliseconds.
    • Per-observation time  – Inference Time / number of scored observations.

  Coverage:
    • Candidate Miss Rate   – fraction of test observations that have zero
                              candidates (the true arc was outside the search
                              radius); these cannot be predicted correctly.

USAGE
-----
    from pathsnap.evaluation import cross_validate, evaluate_model, print_report
    from pathsnap.random_forest import train_random_forest, score_candidates

    results = cross_validate(
        all_rows, labels, arcs_adjacency,
        train_fn=train_random_forest,
        score_fn=score_candidates,
        n_splits=5,
    )
    print_report(results, model_name="RandomForest")

    # Or single hold-out evaluation:
    report = evaluate_model(
        train_rows, test_rows, test_labels, arcs_adjacency,
        train_fn=train_random_forest,
        score_fn=score_candidates,
    )
    print_report(report, model_name="RandomForest")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from itertools import islice
from typing import Callable, Sequence, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases and result containers
# ---------------------------------------------------------------------------

# A label mapping: (trip_id, observation_index) -> true_arc_id
Labels = dict[tuple[str, int], int]

# Arc adjacency set: set of (arc_id_a, arc_id_b) for topologically connected arcs.
# Two arcs are connected if the end-node of arc_a == start-node of arc_b.
# Build this once from the .arcs files; pass it into every evaluation call.
ArcAdjacency = set[tuple[int, int]]


class EvalReport(TypedDict):
    """Structured container for all evaluation metrics."""

    # ── Pointwise metrics ───────────────────────────────────────────────────
    top1_accuracy: float          # fraction correctly matched at top-1
    macro_f1: float               # macro-averaged F1 (binary: true/false arc)
    log_loss: float               # cross-entropy loss on probability outputs

    # ── Route quality ───────────────────────────────────────────────────────
    route_continuity: float       # fraction of consecutive arc pairs that connect

    # ── Coverage ────────────────────────────────────────────────────────────
    candidate_miss_rate: float    # fraction of test obs with zero candidates

    # ── Efficiency ──────────────────────────────────────────────────────────
    inference_time_ms: float      # total inference wall-clock time (ms)
    inference_time_per_obs_ms: float  # ms per labeled observation

    # ── Counts ──────────────────────────────────────────────────────────────
    n_observations: int           # total labeled observations in test set
    n_scored: int                 # observations that had ≥1 candidate


class CVReport(TypedDict):
    """Cross-validation summary across all folds."""

    cv_macro_f1_mean: float       # mean CV Macro F1 across folds
    cv_macro_f1_std: float        # std deviation of CV Macro F1
    cv_top1_accuracy_mean: float
    cv_top1_accuracy_std: float
    cv_log_loss_mean: float
    cv_log_loss_std: float
    cv_route_continuity_mean: float
    cv_route_continuity_std: float
    cv_inference_time_ms_mean: float
    n_folds: int
    fold_reports: list[EvalReport]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_adjacency_from_arcs(arcs: Sequence[tuple[int, int]]) -> ArcAdjacency:
    """
    Build the arc-to-arc adjacency set from raw (start_node, end_node) arc data.

    Two arcs A and B are topologically connected (A -> B) when
    end_node(A) == start_node(B).  This set is used by route_continuity().

    Parameters
    ----------
    arcs : sequence of (start_node_id, end_node_id)
        Zero-indexed arc endpoints, in arc-ID order (row 0 = arc 0, etc.).

    Returns
    -------
    set of (arc_id_a, arc_id_b)
        Directed adjacency pairs.
    """
    # Index start-node -> list of arc_ids that begin there
    start_index: dict[int, list[int]] = defaultdict(list)
    for arc_id, (start, _end) in enumerate(arcs):
        start_index[start].append(arc_id)

    adjacency: ArcAdjacency = set()
    for arc_a, (_start, end) in enumerate(arcs):
        # arc_a connects to every arc_b that starts where arc_a ends
        for arc_b in start_index.get(end, ()):
            adjacency.add((arc_a, arc_b))
    return adjacency


def _group_candidates_by_obs(
    rows: Sequence[tuple[str, object]],
) -> dict[tuple[str, int], list]:
    """
    Collect all candidates for each (trip_id, observation_index) key.

    Parameters
    ----------
    rows : sequence of (trip_id, Candidate)

    Returns
    -------
    dict mapping (trip_id, obs_index) -> list of Candidate
    """
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for trip_id, candidate in rows:
        key = (trip_id, candidate.observation_index)
        groups[key].append(candidate)
    return dict(groups)


def _select_top1(
    groups: dict[tuple[str, int], list],
    score_fn: Callable,
    model: object,
) -> tuple[dict[tuple[str, int], int | None], dict[tuple[str, int], list[float]], float]:
    """
    Run inference on all candidate groups and return top-1 arc predictions.

    Parameters
    ----------
    groups : dict of (trip_id, obs_index) -> list of Candidate
    score_fn : callable matching the shared interface
    model : fitted model

    Returns
    -------
    predictions : dict (trip_id, obs_index) -> arc_id or None
        None when the candidate list is empty (candidate miss).
    all_probs : dict (trip_id, obs_index) -> list[float]
        Probability vector for each observation.
    inference_time_ms : float
        Total wall-clock time for the scoring phase, in milliseconds.
    """
    predictions: dict[tuple[str, int], int | None] = {}
    all_probs: dict[tuple[str, int], list[float]] = {}

    t0 = time.perf_counter()

    for key, candidates in groups.items():
        probs = score_fn(model, candidates)   # list[float], same order as candidates
        all_probs[key] = probs
        if probs:
            # Select the candidate with the highest probability as top-1 prediction
            best_idx = int(np.argmax(probs))
            predictions[key] = candidates[best_idx].arc_id
        else:
            predictions[key] = None           # no candidates — cannot predict

    inference_time_ms = (time.perf_counter() - t0) * 1_000.0
    return predictions, all_probs, inference_time_ms


# ---------------------------------------------------------------------------
# Core metric functions (public, so they can be unit-tested independently)
# ---------------------------------------------------------------------------

def top1_accuracy(
    predictions: dict[tuple[str, int], int | None],
    labels: Labels,
) -> float:
    """
    Fraction of labeled observations where the top-1 predicted arc is correct.

    Observations with no candidates (prediction = None) count as incorrect.

    Parameters
    ----------
    predictions : dict (trip_id, obs_index) -> arc_id or None
    labels : dict (trip_id, obs_index) -> true_arc_id

    Returns
    -------
    float in [0, 1]
    """
    if not labels:
        return float("nan")

    correct = sum(
        1 for key, true_arc in labels.items()
        if predictions.get(key) == true_arc
    )
    return correct / len(labels)


def macro_f1_score(
    predictions: dict[tuple[str, int], int | None],
    labels: Labels,
    groups: dict[tuple[str, int], list],
) -> float:
    """
    Macro-averaged F1 score treating each candidate row as binary
    (true arc = positive, all others = negative).

    'Macro' means F1 is computed per class and then averaged, so the rare
    positive class (true arc) is weighted equally with the abundant negative
    class — the right choice for the imbalanced candidate ranking problem.

    Parameters
    ----------
    predictions : dict (trip_id, obs_index) -> predicted arc_id or None
    labels : dict (trip_id, obs_index) -> true_arc_id
    groups : dict (trip_id, obs_index) -> list of Candidate
        Used to enumerate all candidate rows (not just the top-1 prediction).

    Returns
    -------
    float in [0, 1]
    """
    from sklearn.metrics import f1_score

    y_true: list[int] = []
    y_pred: list[int] = []

    for key, true_arc in labels.items():
        candidates = groups.get(key, [])
        if not candidates:
            continue  # no candidates — skip this observation for per-row F1
        predicted_arc = predictions.get(key)  # the arc chosen by argmax of probs
        for c in candidates:
            y_true.append(int(c.arc_id == true_arc))
            # Predict 1 only for the arc that was selected as the top-1 prediction
            y_pred.append(int(c.arc_id == predicted_arc) if predicted_arc is not None else 0)

    if not y_true:
        return float("nan")

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def log_loss_score(
    all_probs: dict[tuple[str, int], list[float]],
    labels: Labels,
    groups: dict[tuple[str, int], list],
    eps: float = 1e-15,
) -> float:
    """
    Cross-entropy (log loss) of the model's probability outputs.

    Measures calibration: a perfect model assigns probability 1.0 to the true
    arc and 0.0 to all others.  Lower is better.

    Parameters
    ----------
    all_probs : dict (trip_id, obs_index) -> list[float]  (from _select_top1)
    labels : dict (trip_id, obs_index) -> true_arc_id
    groups : dict (trip_id, obs_index) -> list of Candidate
    eps : float, default 1e-15
        Clip probabilities to [eps, 1-eps] to avoid log(0).

    Returns
    -------
    float >= 0   (lower is better)
    """
    y_true_flat: list[int] = []
    y_prob_flat: list[float] = []

    for key, true_arc in labels.items():
        candidates = groups.get(key, [])
        probs = all_probs.get(key, [])
        if not candidates or not probs:
            continue
        for c, p in zip(candidates, probs):
            label = int(c.arc_id == true_arc)
            prob = float(np.clip(p, eps, 1 - eps))
            y_true_flat.append(label)
            y_prob_flat.append(prob)

    if not y_true_flat:
        return float("nan")

    from sklearn.metrics import log_loss
    # Provide both columns: [P(negative), P(positive)]
    probs_2col = [[1.0 - p, p] for p in y_prob_flat]
    return float(log_loss(y_true_flat, probs_2col, labels=[0, 1]))


def route_continuity(
    predictions: dict[tuple[str, int], int | None],
    arcs_adjacency: ArcAdjacency,
) -> float:
    """
    Fraction of consecutive fix pairs whose predicted arcs are topologically
    connected in the road network.

    A route is 'continuous' at a transition (obs_i -> obs_{i+1}) when:
      • Both predictions are non-None (both observations have candidates), AND
      • (arc_i, arc_{i+1}) is in the adjacency set, i.e. they share a node.

    This metric penalises 'teleporting' routes that jump between disconnected
    road segments, complementing pointwise accuracy with a sequence-level view.

    Parameters
    ----------
    predictions : dict (trip_id, obs_index) -> arc_id or None
        Must include consecutive observations from the same trip; sorted by
        observation_index within each trip.
    arcs_adjacency : set of (arc_id_a, arc_id_b)
        Directed adjacency for the road network.

    Returns
    -------
    float in [0, 1]   (1.0 = perfectly continuous route)
    NaN if fewer than two observations exist.
    """
    # Group predictions by trip and sort by observation index
    by_trip: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    for (trip_id, obs_idx), arc_id in predictions.items():
        by_trip[trip_id].append((obs_idx, arc_id))

    total_transitions = 0
    continuous_transitions = 0

    for trip_id, obs_list in by_trip.items():
        # Sort by observation index to get temporal order
        obs_list.sort(key=lambda x: x[0])
        for (_, arc_a), (_, arc_b) in zip(obs_list, islice(obs_list, 1, None)):
            if arc_a is None or arc_b is None:
                # Cannot evaluate continuity when one side has no prediction
                continue
            total_transitions += 1
            if (arc_a, arc_b) in arcs_adjacency or arc_a == arc_b:
                # Same arc (vehicle stayed) is also continuous
                continuous_transitions += 1

    if total_transitions == 0:
        return float("nan")

    return continuous_transitions / total_transitions


def candidate_miss_rate(
    groups: dict[tuple[str, int], list],
    labels: Labels,
) -> float:
    """
    Fraction of labeled observations that have zero candidates.

    A candidate miss means the true arc was not within the search radius at
    that fix.  It is a hard upper bound on achievable top-1 accuracy: any
    observation with no candidates cannot be predicted correctly.

    Parameters
    ----------
    groups : dict (trip_id, obs_index) -> list of Candidate
    labels : dict (trip_id, obs_index) -> true_arc_id

    Returns
    -------
    float in [0, 1]
    """
    if not labels:
        return float("nan")

    n_miss = sum(1 for key in labels if not groups.get(key))
    return n_miss / len(labels)


# ---------------------------------------------------------------------------
# High-level evaluation functions
# ---------------------------------------------------------------------------

def evaluate_model(
    train_rows: Sequence[tuple[str, object]],
    test_rows: Sequence[tuple[str, object]],
    test_labels: Labels,
    arcs_adjacency: ArcAdjacency,
    *,
    train_fn: Callable,
    score_fn: Callable,
    train_labels: Labels | None = None,
) -> EvalReport:
    """
    Train on train_rows and evaluate on test_rows with test_labels.

    This is the single-split (hold-out) evaluation function.  For
    generalisation estimates use cross_validate() instead.

    Parameters
    ----------
    train_rows : sequence of (trip_id, Candidate)
        Candidates from training trips.
    test_rows : sequence of (trip_id, Candidate)
        Candidates from held-out test trips.
    test_labels : Labels
        Ground-truth arc IDs for every labeled observation in the test set.
    arcs_adjacency : ArcAdjacency
        Built from the full road network; used for route_continuity.
    train_fn : callable
        Model training function matching the PathSnap interface.
        Signature: train_fn(training_rows, labels) -> model
    score_fn : callable
        Candidate scoring function matching the PathSnap interface.
        Signature: score_fn(model, candidates) -> list[float]
    train_labels : Labels or None
        Labels for training rows.  If None, test_labels is used (only valid
        when train and test sets are from different observations, e.g. in
        full data experiments).  Pass explicitly for safety in production.

    Returns
    -------
    EvalReport
        Dictionary of all computed metrics.
    """
    effective_train_labels = train_labels if train_labels is not None else test_labels

    logger.info("Fitting model on %d training rows …", len(train_rows))
    model = train_fn(train_rows, effective_train_labels)

    # Group test candidates by observation
    groups = _group_candidates_by_obs(test_rows)

    # Run inference and time it
    logger.info("Scoring %d test observations …", len(groups))
    predictions, all_probs, infer_ms = _select_top1(groups, score_fn, model)

    n_obs = len(test_labels)
    n_scored = sum(1 for key in test_labels if groups.get(key))

    # Compute all metrics
    acc = top1_accuracy(predictions, test_labels)
    f1 = macro_f1_score(predictions, test_labels, groups)
    ll = log_loss_score(all_probs, test_labels, groups)
    continuity = route_continuity(predictions, arcs_adjacency)
    miss_rate = candidate_miss_rate(groups, test_labels)

    logger.info(
        "Results → Top-1 Acc: %.4f | Macro F1: %.4f | Log Loss: %.4f | "
        "Route Continuity: %.4f | Miss Rate: %.4f | Inference: %.1f ms",
        acc, f1, ll, continuity, miss_rate, infer_ms,
    )

    return EvalReport(
        top1_accuracy=acc,
        macro_f1=f1,
        log_loss=ll,
        route_continuity=continuity,
        candidate_miss_rate=miss_rate,
        inference_time_ms=infer_ms,
        inference_time_per_obs_ms=infer_ms / n_scored if n_scored else float("nan"),
        n_observations=n_obs,
        n_scored=n_scored,
    )


def cross_validate(
    all_rows: Sequence[tuple[str, object]],
    all_labels: Labels,
    arcs_adjacency: ArcAdjacency,
    *,
    train_fn: Callable,
    score_fn: Callable,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVReport:
    """
    Trip-stratified k-fold cross-validation.

    Folds are built by grouping all observations belonging to the same trip
    together (trip-level stratification).  This prevents data leakage: a GPS
    fix from trip T should not appear in both train and test of the same fold,
    because consecutive fixes share road context.

    Parameters
    ----------
    all_rows : sequence of (trip_id, Candidate)
        All candidates from all trips (training partition only — never include
        evaluation/test data here).
    all_labels : Labels
        Ground-truth labels for every labeled observation.
    arcs_adjacency : ArcAdjacency
        Road network adjacency for route continuity metric.
    train_fn : callable
        Training function for the model under evaluation.
    score_fn : callable
        Scoring function for the model under evaluation.
    n_splits : int, default 5
        Number of CV folds.  With 80 training trips, 5-fold gives 64/16 splits.
    random_state : int, default 42
        Seed for fold shuffling (applied to the trip-level split, not the
        individual observation order).

    Returns
    -------
    CVReport
        Aggregated metrics (mean ± std) across all folds, plus individual
        fold EvalReports for detailed inspection.
    """
    import math

    # ── 1. Identify all unique trips in the data ─────────────────────────────
    all_trip_ids = sorted({trip_id for trip_id, _ in all_rows})
    n_trips = len(all_trip_ids)
    logger.info(
        "Cross-validating with %d folds over %d trips …", n_splits, n_trips
    )

    if n_trips < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from only {n_trips} trips. "
            "Reduce n_splits or add more training trips."
        )

    # Shuffle trips (not individual rows) to randomise fold assignment
    rng = np.random.default_rng(random_state)
    shuffled_trips = rng.permutation(all_trip_ids).tolist()

    # Build trip -> rows index for fast fold construction
    rows_by_trip: dict[str, list] = defaultdict(list)
    for item in all_rows:
        rows_by_trip[item[0]].append(item)

    labels_by_trip: dict[str, Labels] = defaultdict(dict)
    for (trip_id, obs_idx), arc_id in all_labels.items():
        labels_by_trip[trip_id][(trip_id, obs_idx)] = arc_id

    # ── 2. Split trips into n_splits folds ───────────────────────────────────
    fold_size = math.ceil(n_trips / n_splits)
    trip_folds: list[list[str]] = [
        shuffled_trips[i * fold_size: (i + 1) * fold_size]
        for i in range(n_splits)
    ]

    # ── 3. Run each fold ─────────────────────────────────────────────────────
    fold_reports: list[EvalReport] = []

    for fold_idx, val_trips in enumerate(trip_folds):
        val_set = set(val_trips)
        train_trips = [t for t in all_trip_ids if t not in val_set]

        # Build rows and labels for this fold
        train_rows_fold = [row for t in train_trips for row in rows_by_trip[t]]
        test_rows_fold = [row for t in val_trips for row in rows_by_trip[t]]

        train_labels_fold: Labels = {}
        test_labels_fold: Labels = {}
        for t in train_trips:
            train_labels_fold.update(labels_by_trip[t])
        for t in val_trips:
            test_labels_fold.update(labels_by_trip[t])

        logger.info(
            "Fold %d/%d — train: %d trips / %d rows | val: %d trips / %d rows",
            fold_idx + 1, n_splits,
            len(train_trips), len(train_rows_fold),
            len(val_trips), len(test_rows_fold),
        )

        report = evaluate_model(
            train_rows_fold,
            test_rows_fold,
            test_labels_fold,
            arcs_adjacency,
            train_fn=train_fn,
            score_fn=score_fn,
            train_labels=train_labels_fold,
        )
        fold_reports.append(report)

    # ── 4. Aggregate across folds ─────────────────────────────────────────────
    def _mean(key: str) -> float:
        vals = [r[key] for r in fold_reports if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key: str) -> float:
        vals = [r[key] for r in fold_reports if not np.isnan(r[key])]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return CVReport(
        cv_macro_f1_mean=_mean("macro_f1"),
        cv_macro_f1_std=_std("macro_f1"),
        cv_top1_accuracy_mean=_mean("top1_accuracy"),
        cv_top1_accuracy_std=_std("top1_accuracy"),
        cv_log_loss_mean=_mean("log_loss"),
        cv_log_loss_std=_std("log_loss"),
        cv_route_continuity_mean=_mean("route_continuity"),
        cv_route_continuity_std=_std("route_continuity"),
        cv_inference_time_ms_mean=_mean("inference_time_ms"),
        n_folds=n_splits,
        fold_reports=fold_reports,
    )


# ---------------------------------------------------------------------------
# Reporting / pretty printing
# ---------------------------------------------------------------------------

def print_report(report: EvalReport | CVReport, *, model_name: str = "Model") -> None:
    """
    Print a human-readable evaluation summary to stdout.

    Accepts either a single-split EvalReport or a cross-validation CVReport.
    """
    sep = "─" * 60

    if "cv_macro_f1_mean" in report:
        # ── CVReport ─────────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  Cross-Validation Results — {model_name}")
        print(sep)
        print(f"  Folds                      : {report['n_folds']}")
        print(f"  CV Macro F1-Score          : {report['cv_macro_f1_mean']:.4f} "
              f"± {report['cv_macro_f1_std']:.4f}")
        print(f"  CV Top-1 Accuracy          : {report['cv_top1_accuracy_mean']:.4f} "
              f"± {report['cv_top1_accuracy_std']:.4f}")
        print(f"  CV Log Loss                : {report['cv_log_loss_mean']:.4f} "
              f"± {report['cv_log_loss_std']:.4f}")
        print(f"  CV Route Continuity        : {report['cv_route_continuity_mean']:.4f} "
              f"± {report['cv_route_continuity_std']:.4f}")
        print(f"  Avg Inference Time         : {report['cv_inference_time_ms_mean']:.1f} ms")
        print(sep)
        print("  Per-fold detail:")
        for i, fr in enumerate(report["fold_reports"], 1):
            print(
                f"    Fold {i}: Top1={fr['top1_accuracy']:.4f}  "
                f"F1={fr['macro_f1']:.4f}  "
                f"LL={fr['log_loss']:.4f}  "
                f"RC={fr['route_continuity']:.4f}  "
                f"T={fr['inference_time_ms']:.1f}ms"
            )
        print(sep)
    else:
        # ── EvalReport ───────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  Evaluation Report — {model_name}")
        print(sep)
        print(f"  Observations (total)       : {report['n_observations']:,}")
        print(f"  Observations (scored)      : {report['n_scored']:,}")
        print(f"  Candidate Miss Rate        : {report['candidate_miss_rate']:.4f} "
              f"({report['candidate_miss_rate'] * 100:.2f}%)")
        print(sep)
        print(f"  Top-1 Accuracy             : {report['top1_accuracy']:.4f}")
        print(f"  Macro F1-Score             : {report['macro_f1']:.4f}")
        print(f"  Log Loss                   : {report['log_loss']:.4f}")
        print(f"  Route Continuity           : {report['route_continuity']:.4f}")
        print(sep)
        print(f"  Inference Time (total)     : {report['inference_time_ms']:.2f} ms")
        print(f"  Inference Time (per obs)   : {report['inference_time_per_obs_ms']:.4f} ms")
        print(sep)


def compare_models(
    reports: dict[str, EvalReport | CVReport],
) -> None:
    """
    Print a compact side-by-side comparison table for multiple models.

    Parameters
    ----------
    reports : dict of model_name -> EvalReport or CVReport
        Pass EvalReports for a consistent comparison (all same type).
    """
    print("\n" + "─" * 90)
    print(f"{'Model':<28} {'Top-1 Acc':>10} {'Macro F1':>10} {'Log Loss':>10} "
          f"{'Continuity':>12} {'Infer(ms)':>10}")
    print("─" * 90)

    for name, report in reports.items():
        if "cv_macro_f1_mean" in report:
            # CVReport — use mean values
            acc = report["cv_top1_accuracy_mean"]
            f1 = report["cv_macro_f1_mean"]
            ll = report["cv_log_loss_mean"]
            rc = report["cv_route_continuity_mean"]
            infer = report["cv_inference_time_ms_mean"]
            suffix = " (CV)"
        else:
            acc = report["top1_accuracy"]
            f1 = report["macro_f1"]
            ll = report["log_loss"]
            rc = report["route_continuity"]
            infer = report["inference_time_ms"]
            suffix = ""

        print(
            f"{name + suffix:<28} {acc:>10.4f} {f1:>10.4f} {ll:>10.4f} "
            f"{rc:>12.4f} {infer:>10.1f}"
        )

    print("─" * 90)


# ---------------------------------------------------------------------------
# Convenience: build adjacency from raw arc file data
# ---------------------------------------------------------------------------

def build_adjacency(arcs: Sequence[tuple[int, int]]) -> ArcAdjacency:
    """
    Public wrapper around _build_adjacency_from_arcs.

    Use this once per road network (i.e. per trip, since each trip has its own
    .arcs file) and cache the result before calling evaluate_model().

    Parameters
    ----------
    arcs : sequence of (start_node, end_node)
        Raw rows from the .arcs file (zero-indexed node IDs).

    Returns
    -------
    ArcAdjacency
        Set of (arc_id_a, arc_id_b) for topologically adjacent arc pairs.
    """
    return _build_adjacency_from_arcs(arcs)

