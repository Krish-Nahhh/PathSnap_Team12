"""
tune_rf_bayes.py – Bayesian Optimization for Random Forest in PathSnap.

Uses Optuna (Tree-structured Parzen Estimator / TPE) to efficiently search the
hyperparameter space and maximise trip-stratified cross-validation Macro F1.

Fast vectorised batch scoring is used so each trial completes in seconds on
local CPU.

USAGE
-----
  # Run Bayesian optimization with default settings (15 trials, 3 folds)
  python -m src.pathsnap.tune_rf_bayes

  # Custom budget and 5 folds
  python -m src.pathsnap.tune_rf_bayes --n-trials 25 --folds 5

  # Train and save the best tuned model to artifacts/
  python -m src.pathsnap.tune_rf_bayes --export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import optuna
from sklearn.metrics import f1_score

from .random_forest import candidate_features, train_random_forest
from .training import load_all_trips

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce verbose Optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDENT_FOLDER = PROJECT_ROOT / "Student folder" / "Student folder"
TRAIN_DIR = STUDENT_FOLDER / "train"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "tuning_results"

ARTIFACTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# FAST VECTORISED FOLD EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rf_fold_fast(
    model: object,
    test_rows: Sequence[tuple[str, object]],
    test_labels: dict[tuple[str, int], int],
) -> float:
    """
    Evaluate a fitted model on test_rows using a single batch predict_proba call.
    Runs in ~1-2 seconds instead of minutes.
    """
    if not test_rows or not test_labels:
        return 0.0

    # 1. Vectorised feature extraction for all test candidates in one pass
    features = [candidate_features(c) for _, c in test_rows]
    probs = model.predict_proba(features)[:, 1]

    # 2. Group probability scores and candidates by observation
    obs_groups: dict[tuple[str, int], list[tuple[float, int]]] = defaultdict(list)
    for (trip_id, c), p in zip(test_rows, probs):
        obs_groups[(trip_id, c.observation_index)].append((float(p), int(c.arc_id)))

    # 3. Compute Top-1 selection and binary true/false targets for Macro F1
    y_true: list[int] = []
    y_pred: list[int] = []

    for key, true_arc in test_labels.items():
        cands = obs_groups.get(key)
        if not cands:
            continue
        # Select candidate with highest probability
        best_arc = max(cands, key=lambda x: x[0])[1]
        for _, arc_id in cands:
            y_true.append(int(arc_id == true_arc))
            y_pred.append(int(arc_id == best_arc))

    if not y_true:
        return 0.0

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN OPTIMIZATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_bayesian_optimization(
    all_rows: list,
    all_labels: dict,
    *,
    n_trials: int = 15,
    n_splits: int = 3,
    random_state: int = 42,
) -> tuple[dict, float, optuna.Study]:
    """
    Execute Bayesian Optimization over Random Forest hyperparameters using Optuna TPE.
    """
    # ── 1. Create trip-stratified folds ──────────────────────────────────────
    unique_trips = sorted({trip_id for trip_id, _ in all_rows})
    rng = np.random.default_rng(random_state)
    shuffled_trips = rng.permutation(unique_trips).tolist()

    rows_by_trip: dict[str, list] = defaultdict(list)
    for row in all_rows:
        rows_by_trip[row[0]].append(row)

    labels_by_trip: dict[str, dict] = defaultdict(dict)
    for (trip_id, obs_idx), arc_id in all_labels.items():
        labels_by_trip[trip_id][(trip_id, obs_idx)] = arc_id

    fold_size = int(np.ceil(len(unique_trips) / n_splits))
    trip_folds = [
        shuffled_trips[i * fold_size : (i + 1) * fold_size]
        for i in range(n_splits)
    ]

    # Pre-build fold datasets in memory
    folds_data = []
    for val_trips in trip_folds:
        val_set = set(val_trips)
        train_trips = [t for t in unique_trips if t not in val_set]

        train_rows = [r for t in train_trips for r in rows_by_trip[t]]
        val_rows = [r for t in val_trips for r in rows_by_trip[t]]

        train_labels = {k: v for t in train_trips for k, v in labels_by_trip[t].items()}
        val_labels = {k: v for t in val_trips for k, v in labels_by_trip[t].items()}

        folds_data.append((train_rows, train_labels, val_rows, val_labels))

    logger.info(
        "Prepared %d trip-stratified folds (%d total training trips).",
        n_splits,
        len(unique_trips),
    )

    # ── 2. Define Optuna Objective Function ──────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 80, 250, step=20)
        max_depth_choice = trial.suggest_categorical("max_depth", [10, 15, 20, 25, None])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 12, step=2)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])

        fold_f1s = []
        t0 = time.perf_counter()

        for fold_idx, (tr_rows, tr_lbls, val_rows, val_lbls) in enumerate(folds_data):
            model = train_random_forest(
                tr_rows,
                tr_lbls,
                n_estimators=n_estimators,
                max_depth=max_depth_choice,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state + trial.number,
            )
            f1 = evaluate_rf_fold_fast(model, val_rows, val_lbls)
            fold_f1s.append(f1)

            # Report intermediate value for pruning unpromising trials
            trial.report(float(np.mean(fold_f1s)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_f1 = float(np.mean(fold_f1s))
        elapsed = time.perf_counter() - t0
        logger.info(
            "Trial %2d: Macro F1 = %.4f (took %.1fs) | n_est=%d, depth=%s, leaf=%d, feat=%s",
            trial.number + 1,
            mean_f1,
            elapsed,
            n_estimators,
            str(max_depth_choice),
            min_samples_leaf,
            str(max_features),
        )
        return mean_f1

    # ── 3. Run Optimization with TPE (Bayesian) Sampler ──────────────────────
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info("=" * 65)
    logger.info("Starting Bayesian Optimization (TPE Sampler) — %d trials …", n_trials)
    logger.info("=" * 65)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    logger.info("=" * 65)
    logger.info("BAYESIAN OPTIMIZATION COMPLETE")
    logger.info("Best CV Macro F1 : %.4f", best_score)
    logger.info("Best Parameters  : %s", best_params)
    logger.info("=" * 65)

    return best_params, best_score, study


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization (Optuna TPE) for PathSnap Random Forest.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        metavar="N",
        help="Number of Bayesian optimization trials (default: 15).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        metavar="K",
        help="Number of trip-stratified CV folds (default: 3 for local speed).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="After finding best params, retrain on full training data and save to artifacts/random_forest.joblib",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args(argv)

    logger.info("Loading training data …")
    all_rows, all_labels, _ = load_all_trips(TRAIN_DIR, split="train")
    if not all_rows:
        logger.error("No training data found in %s", TRAIN_DIR)
        sys.exit(1)

    best_params, best_score, _ = run_bayesian_optimization(
        all_rows,
        all_labels,
        n_trials=args.n_trials,
        n_splits=args.folds,
        random_state=args.seed,
    )

    # Save results to JSON
    out_json = RESULTS_DIR / "random_forest_bayes.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"best_score": best_score, "best_params": best_params},
            f,
            indent=2,
            default=str,
        )
    logger.info("Saved best Bayesian parameters → %s", out_json)

    if args.export:
        import joblib

        logger.info("Retraining Random Forest with best parameters on full training data …")
        final_model = train_random_forest(all_rows, all_labels, **best_params)
        export_path = ARTIFACTS_DIR / "random_forest.joblib"
        joblib.dump(final_model, export_path)
        logger.info("Saved final tuned model → %s", export_path)


if __name__ == "__main__":
    main()
