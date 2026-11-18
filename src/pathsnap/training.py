"""
training.py – Shared model training runner for PathSnap map-matching.

Responsibility: Krishna H Zalavadiya  (shared module)
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
This is the SINGLE entry-point for running all PathSnap model experiments.
It is NOT a model file — model implementations live in their own modules:

    logistic_regression.py    → train_logistic_regression / score_candidates
    random_forest.py          → train_random_forest       / score_candidates
    decision_tree.py          → train_decision_tree       / score_candidates
    gaussian_naive_bayes.py   → train_gaussian_naive_bayes/ score_candidates
    hist_gradient_boosting.py → train_hist_gradient_boosting/score_candidates

This runner:
  1.  Loads all training trip data from the Student folder.
  2.  Builds aligned labels via label_alignment.export_aligned_labels().
  3.  Runs trip-stratified 5-fold cross-validation on every model via
      evaluation.cross_validate().
  4.  Calls evaluation.compare_models() to print the side-by-side metric table.
  5.  Re-fits each model on the full training set and saves the fitted
      pipeline to artifacts/.

USAGE
-----
  # Cross-validate all 5 models (default)
  python -m src.pathsnap.training

  # Train + save final models on full training data
  python -m src.pathsnap.training --export

  # Cross-validate a single model only
  python -m src.pathsnap.training --model random_forest

  # Student-test evaluation after training
  python -m src.pathsnap.training --export --eval-split student_test
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project root (two levels up from this file) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDENT_FOLDER = PROJECT_ROOT / "Student folder" / "Student folder"
TRAIN_DIR      = STUDENT_FOLDER / "train"
TEST_DIR       = STUDENT_FOLDER / "student_test"
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
RESULTS_DIR    = PROJECT_ROOT / "results"

ARTIFACTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _read_trip_file(trip_path: Path, suffix: str, converters: tuple):
    """Read one .track / .nodes / .arcs / .route file into a list of tuples."""
    file_path = trip_path / (trip_path.name + suffix)
    with file_path.open(encoding="utf-8") as f:
        rows = []
        for line in f:
            parts = line.split()
            if parts:
                rows.append(tuple(conv(p) for conv, p in zip(converters, parts)))
    return rows


def load_all_trips(split_dir: Path, split: str) -> tuple:
    """
    Load all trips from one split directory.

    Returns
    -------
    all_rows : list of (trip_id, Candidate)
        Every candidate from every observation of every trip.
    all_labels : dict  (trip_id, observation_index) -> true_arc_id
        Aligned labels built from .route files.
    arcs_adjacency : set  (arc_id_a, arc_id_b)
        Combined adjacency across all trips (used for route continuity metric).
    """
    from .candidates import generate_candidates
    from .evaluation import build_adjacency
    from .label_alignment import align_trip_labels

    all_rows: list = []
    all_labels: dict = {}
    arcs_adjacency: set = set()

    trips = sorted(split_dir.iterdir())
    logger.info("Loading %d trips from '%s' …", len(trips), split_dir.name)

    for trip_path in trips:
        if not trip_path.is_dir():
            continue
        tid = trip_path.name

        try:
            track = _read_trip_file(trip_path, ".track", (float, float, float))
            nodes = _read_trip_file(trip_path, ".nodes", (float, float))
            arcs  = _read_trip_file(trip_path, ".arcs",  (int, int))
            route = _read_trip_file(trip_path, ".route", (int,))
        except (OSError, ValueError) as exc:
            logger.warning("Trip %s: skipping due to read error — %s", tid, exc)
            continue

        route_ids = [r[0] for r in route]

        # ── Generate candidates for every GPS observation ─────────────────
        for _obs_idx, candidates in generate_candidates(track, nodes, arcs):
            for c in candidates:
                all_rows.append((tid, c))

        # ── Align route arcs to GPS observations (DP label assignment) ────
        alignment = align_trip_labels(track, nodes, arcs, route_ids)
        for obs_idx, true_arc in alignment.items():
            all_labels[(tid, obs_idx)] = true_arc

        # ── Accumulate road-network adjacency ─────────────────────────────
        arcs_adjacency |= build_adjacency(arcs)

    logger.info(
        "Loaded %d candidate rows, %d labeled observations from %d trips.",
        len(all_rows),
        len(all_labels),
        len(trips),
    )
    return all_rows, all_labels, arcs_adjacency


# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

def _get_model_registry() -> dict:
    """
    Return the registry of all PathSnap model (train_fn, score_fn) pairs.
    Imports are deferred so individual modules can be tested without loading all.
    """
    from .decision_tree import (
        train_decision_tree,
        score_candidates as dt_score,
    )
    from .gaussian_naive_bayes import (
        train_gaussian_naive_bayes,
        score_candidates as gnb_score,
    )
    from .hist_gradient_boosting import (
        train_hist_gradient_boosting,
        score_candidates as hgb_score,
    )
    from .logistic_regression import (
        train_logistic_regression,
        score_candidates as lr_score,
    )
    from .random_forest import (
        train_random_forest,
        score_candidates as rf_score,
    )

    return {
        "logistic_regression":    (train_logistic_regression,    lr_score),
        "decision_tree":          (train_decision_tree,           dt_score),
        "random_forest":          (train_random_forest,           rf_score),
        "gaussian_naive_bayes":   (train_gaussian_naive_bayes,    gnb_score),
        "hist_gradient_boosting": (train_hist_gradient_boosting,  hgb_score),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_validation(
    model_names: list[str],
    all_rows: list,
    all_labels: dict,
    arcs_adjacency: set,
    n_splits: int = 5,
) -> dict:
    """
    Run trip-stratified k-fold CV on the requested models and print comparison.

    Parameters
    ----------
    model_names : list of str
        Keys from the model registry to evaluate.
    all_rows : list of (trip_id, Candidate)
    all_labels : dict (trip_id, obs_index) -> true_arc_id
    arcs_adjacency : set (arc_id_a, arc_id_b)
    n_splits : int, default 5

    Returns
    -------
    dict  model_name -> CVReport
    """
    from .evaluation import compare_models, cross_validate, print_report

    registry = _get_model_registry()
    cv_reports: dict = {}

    for name in model_names:
        if name not in registry:
            logger.error("Unknown model '%s'. Skipping.", name)
            continue

        train_fn, score_fn = registry[name]
        logger.info("=" * 60)
        logger.info("Cross-validating  %s  (%d folds) …", name, n_splits)
        logger.info("=" * 60)

        try:
            report = cross_validate(
                all_rows,
                all_labels,
                arcs_adjacency,
                train_fn=train_fn,
                score_fn=score_fn,
                n_splits=n_splits,
            )
            cv_reports[name] = report
            print_report(report, model_name=name)
        except Exception as exc:
            logger.exception("Model '%s' failed during cross-validation: %s", name, exc)

    if cv_reports:
        print("\n" + "═" * 70)
        print("  FINAL MODEL COMPARISON (CV means)")
        print("═" * 70)
        compare_models(cv_reports)

    return cv_reports


# ═══════════════════════════════════════════════════════════════════════════
# FULL TRAINING + EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def run_export(
    model_names: list[str],
    all_rows: list,
    all_labels: dict,
) -> None:
    """
    Refit each model on the FULL training dataset and save to artifacts/.

    Saved as  artifacts/<model_name>.joblib
    """
    import joblib

    registry = _get_model_registry()

    for name in model_names:
        if name not in registry:
            continue
        train_fn, _ = registry[name]
        logger.info("Fitting %s on full training set …", name)
        try:
            model = train_fn(all_rows, all_labels)
            out_path = ARTIFACTS_DIR / f"{name}.joblib"
            joblib.dump(model, out_path)
            logger.info("Saved → %s", out_path)
        except Exception as exc:
            logger.exception("Export failed for '%s': %s", name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# STUDENT-TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_eval_split(
    model_names: list[str],
    train_rows: list,
    train_labels: dict,
    test_rows: list,
    test_labels: dict,
    arcs_adjacency: set,
) -> None:
    """
    Train on training data, evaluate on student-test data, print full report.
    """
    from .evaluation import compare_models, evaluate_model, print_report

    registry = _get_model_registry()
    hold_out_reports: dict = {}

    for name in model_names:
        if name not in registry:
            continue
        train_fn, score_fn = registry[name]
        logger.info("Hold-out evaluation: %s …", name)
        try:
            report = evaluate_model(
                train_rows, test_rows, test_labels, arcs_adjacency,
                train_fn=train_fn,
                score_fn=score_fn,
                train_labels=train_labels,
            )
            hold_out_reports[name] = report
            print_report(report, model_name=f"{name} [student_test]")
        except Exception as exc:
            logger.exception("Hold-out eval failed for '%s': %s", name, exc)

    if hold_out_reports:
        print("\n" + "═" * 70)
        print("  STUDENT-TEST HOLD-OUT COMPARISON")
        print("═" * 70)
        compare_models(hold_out_reports)

    # Save results to CSV
    _save_results_csv(hold_out_reports, RESULTS_DIR / "holdout_results.csv")


def _save_results_csv(reports: dict, output_path: Path) -> None:
    """Write a summary CSV of evaluation results for the technical report."""
    if not reports:
        return
    rows = []
    for name, r in reports.items():
        row = {"model": name}
        row.update({k: v for k, v in r.items() if not isinstance(v, list)})
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results CSV saved → %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gaussian_naive_bayes",
    "hist_gradient_boosting",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PathSnap training runner — cross-validate and/or export all models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples
        --------
          # Cross-validate all 5 models with 5 folds (default)
          python -m src.pathsnap.training

          # Cross-validate a single model
          python -m src.pathsnap.training --model random_forest

          # Train + save artifacts
          python -m src.pathsnap.training --export

          # Train + export + evaluate on student-test
          python -m src.pathsnap.training --export --eval-split student_test

          # Tune number of CV folds
          python -m src.pathsnap.training --folds 3
        """),
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        default=None,
        help="Run one specific model only. Omit to run all five.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of trip-stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="After CV, refit on full training set and save to artifacts/.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["student_test"],
        default=None,
        metavar="SPLIT",
        help="After training, evaluate on the given split (student_test).",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation (useful when only --export is needed).",
    )
    args = parser.parse_args(argv)

    models_to_run = [args.model] if args.model else ALL_MODELS

    # ── Load training data ───────────────────────────────────────────────────
    logger.info("Loading training data …")
    train_rows, train_labels, arcs_adj = load_all_trips(TRAIN_DIR, split="train")

    if not train_rows:
        logger.error("No training data loaded. Check the Student folder path.")
        sys.exit(1)

    # ── Cross-validation ─────────────────────────────────────────────────────
    if not args.skip_cv:
        run_cross_validation(
            models_to_run, train_rows, train_labels, arcs_adj, n_splits=args.folds
        )

    # ── Export final models ──────────────────────────────────────────────────
    if args.export:
        run_export(models_to_run, train_rows, train_labels)

    # ── Student-test hold-out evaluation ─────────────────────────────────────
    if args.eval_split == "student_test":
        logger.info("Loading student-test data …")
        test_rows, test_labels, test_adj = load_all_trips(TEST_DIR, split="student_test")
        combined_adj = arcs_adj | test_adj
        run_eval_split(
            models_to_run,
            train_rows, train_labels,
            test_rows,  test_labels,
            combined_adj,
        )

    logger.info("Done.")


# ── textwrap import needed for CLI epilog ────────────────────────────────────
import textwrap   # noqa: E402  (placed here to keep top of file clean)

if __name__ == "__main__":
    main()
