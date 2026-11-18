"""
tuning.py – Hyperparameter tuning for all PathSnap map-matching models.

Responsibility: Shared module
─────────────────────────────────────────────────────────────────────────────
PURPOSE
-------
Provides trip-stratified RandomizedSearchCV / GridSearchCV wrappers for every
PathSnap model.  Because each model exposes the standard two-function interface
(train_fn / score_fn), a single generic tuner can handle all five models.

The tuner wraps each model in a custom sklearn-compatible Estimator so that
sklearn's cross-validation machinery can operate directly on the raw
(trip_id, Candidate) rows without reshaping.

USAGE
-----
  # Tune a single model (default: random_forest)
  python -m src.pathsnap.tuning

  # Tune a specific model
  python -m src.pathsnap.tuning --model decision_tree

  # Tune all models
  python -m src.pathsnap.tuning --all

  # Control search budget
  python -m src.pathsnap.tuning --model hist_gradient_boosting --n-iter 30 --folds 3

  # Save best params to JSON
  python -m src.pathsnap.tuning --model random_forest --export

DESIGN NOTES
------------
* Trip-stratified CV is used (GroupKFold on trip_id) so no GPS fix from
  trip T leaks into both train and test within the same fold.
* Macro F1 is the optimisation target — it penalises the model for ignoring
  the rare positive class (true arc), which is the right choice for the
  highly imbalanced candidate ranking problem.
* Randomized search (n_iter < full grid) is preferred for large param spaces;
  use --exhaustive to switch to GridSearchCV.
* All best params are logged and optionally saved to tuning_results/<model>.json
  so they can be fed back into the train_* functions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
STUDENT_FOLDER = PROJECT_ROOT / "Student folder" / "Student folder"
TRAIN_DIR      = STUDENT_FOLDER / "train"
TUNING_DIR     = PROJECT_ROOT / "tuning_results"
TUNING_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# sklearn-compatible wrapper
# ═══════════════════════════════════════════════════════════════════════════

class _PathSnapEstimator:
    """
    Thin sklearn-compatible wrapper that adapts a PathSnap train_fn/score_fn
    pair to the fit/predict_proba interface expected by sklearn's CV utilities.

    The raw data it operates on is a list of (trip_id, Candidate) rows; labels
    are passed separately via the `all_labels` attribute set at construction
    time.
    """

    def __init__(self, train_fn, score_fn, all_labels: dict, **model_params):
        self.train_fn    = train_fn
        self.score_fn    = score_fn
        self.all_labels  = all_labels
        self.model_params = model_params
        self._model      = None

    def get_params(self, deep: bool = True) -> dict:
        return dict(self.model_params)

    def set_params(self, **params) -> "_PathSnapEstimator":
        self.model_params.update(params)
        return self

    def fit(self, rows: list, _ignored_y=None) -> "_PathSnapEstimator":
        """Train on a list of (trip_id, Candidate) rows."""
        self._model = self.train_fn(rows, self.all_labels, **self.model_params)
        return self

    def predict_proba(self, rows: list) -> np.ndarray:
        """
        Score every candidate row; return a 2-column probability matrix
        [P(false arc), P(true arc)] for compatibility with sklearn scorers.
        """
        from collections import defaultdict

        # Group rows by observation to call score_fn once per observation
        groups: dict[tuple, list] = defaultdict(list)
        for trip_id, candidate in rows:
            groups[(trip_id, candidate.observation_index)].append(candidate)

        # Build a flat list of P(true arc) aligned with the input row order
        prob_cache: dict[tuple, list[float]] = {}
        # Pre-build position map per group
        group_pos: dict[tuple, dict] = {}
        for trip_id, candidate in rows:
            key = (trip_id, candidate.observation_index)
            if key not in group_pos:
                group_pos[key] = {}
            group_pos[key][id(candidate)] = len(group_pos[key])

        row_probs: list[float] = []
        for trip_id, candidate in rows:
            key = (trip_id, candidate.observation_index)
            if key not in prob_cache:
                prob_cache[key] = self.score_fn(self._model, groups[key])
            idx = group_pos[key][id(candidate)]
            probs = prob_cache[key]
            p = probs[idx] if probs and idx < len(probs) else 0.0
            row_probs.append(p)

        prob_arr = np.array(row_probs, dtype=float).reshape(-1, 1)
        return np.hstack([1.0 - prob_arr, prob_arr])

    def score(self, rows: list, _ignored_y=None) -> float:
        """
        Return macro F1 on `rows` using the fitted model.
        Called internally by sklearn CV to rank parameter configurations.
        """
        from sklearn.metrics import f1_score
        from collections import defaultdict

        proba = self.predict_proba(rows)

        # Reconstruct per-observation top-1 predictions for F1 scoring
        groups: dict[tuple, list] = defaultdict(list)
        obs_order: list[tuple] = []  # preserves insertion order
        seen: set = set()
        for trip_id, candidate in rows:
            key = (trip_id, candidate.observation_index)
            groups[key].append(candidate)
            if key not in seen:
                obs_order.append(key)
                seen.add(key)

        y_pred_all: list[int] = []
        y_true_all: list[int] = []

        prob_idx = 0
        for key in obs_order:
            candidates = groups[key]
            n = len(candidates)
            probs_obs = proba[prob_idx: prob_idx + n, 1]
            best = int(np.argmax(probs_obs)) if n > 0 else -1
            true_arc = self.all_labels.get(key)
            for i, c in enumerate(candidates):
                y_true_all.append(int(c.arc_id == true_arc) if true_arc is not None else 0)
                y_pred_all.append(1 if i == best else 0)
            prob_idx += n

        if not y_true_all:
            return 0.0
        return float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════
# Parameter grids for each model
# ═══════════════════════════════════════════════════════════════════════════

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    # LogisticRegression: train_logistic_regression doesn't accept **kwargs yet.
    # Extend it with a C / solver parameter if you want to tune it.
    "logistic_regression": {},

    "decision_tree": {
        "max_depth":        [8, 12, 15, 20, None],
        "min_samples_leaf": [4, 8, 16, 32],
        "ccp_alpha":        [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
    },

    "random_forest": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [10, 20, None],
        "min_samples_leaf": [2, 4, 8],
        "max_features":     ["sqrt", "log2", 0.5],
    },

    "gaussian_naive_bayes": {
        "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
    },

    "hist_gradient_boosting": {
        "max_iter":          [100, 200, 400],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "max_leaf_nodes":    [15, 31, 63],
        "min_samples_leaf":  [10, 20, 40],
        "l2_regularization": [1e-3, 1e-2, 0.1],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_training_data() -> tuple:
    """Load all training trips; return (all_rows, all_labels, arcs_adjacency)."""
    from .training import load_all_trips
    return load_all_trips(TRAIN_DIR, split="train")


# ═══════════════════════════════════════════════════════════════════════════
# Registry helper
# ═══════════════════════════════════════════════════════════════════════════

def _get_registry() -> dict:
    from .decision_tree          import train_decision_tree,          score_candidates as dt_score
    from .gaussian_naive_bayes   import train_gaussian_naive_bayes,   score_candidates as gnb_score
    from .hist_gradient_boosting import train_hist_gradient_boosting,  score_candidates as hgb_score
    from .logistic_regression    import train_logistic_regression,    score_candidates as lr_score
    from .random_forest          import train_random_forest,          score_candidates as rf_score

    return {
        "logistic_regression":    (train_logistic_regression,    lr_score),
        "decision_tree":          (train_decision_tree,           dt_score),
        "random_forest":          (train_random_forest,           rf_score),
        "gaussian_naive_bayes":   (train_gaussian_naive_bayes,    gnb_score),
        "hist_gradient_boosting": (train_hist_gradient_boosting,  hgb_score),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core tuning routine
# ═══════════════════════════════════════════════════════════════════════════

def tune_model(
    model_name: str,
    all_rows: list,
    all_labels: dict,
    *,
    n_iter: int = 20,
    n_splits: int = 5,
    exhaustive: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run hyperparameter search for one PathSnap model.

    Parameters
    ----------
    model_name   : one of the five model keys
    all_rows     : list of (trip_id, Candidate) from load_all_trips
    all_labels   : dict (trip_id, obs_idx) -> true_arc_id
    n_iter       : number of random configurations to sample (RandomizedSearch)
    n_splits     : number of CV folds (trip-stratified)
    exhaustive   : if True use GridSearchCV instead of RandomizedSearchCV
    random_state : random seed

    Returns
    -------
    dict with keys: best_params, best_score, cv_results
    """
    from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV

    registry = _get_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(registry)}")

    train_fn, score_fn = registry[model_name]
    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        logger.warning(
            "No tunable parameters defined for '%s'. "
            "Add entries to PARAM_GRIDS in tuning.py to enable search.",
            model_name,
        )
        return {"best_params": {}, "best_score": float("nan"), "cv_results": {}}

    # ── Build group array for GroupKFold (one integer per row = trip index) ───
    trip_ids     = [row[0] for row in all_rows]
    unique_trips = sorted(set(trip_ids))
    trip_to_int  = {t: i for i, t in enumerate(unique_trips)}
    groups       = np.array([trip_to_int[t] for t in trip_ids])

    # Dummy y array — labels come from all_labels inside the estimator
    y_dummy = np.zeros(len(all_rows), dtype=int)

    # Build estimator with the first value of each param as default
    default_params = {k: v[0] for k, v in param_grid.items()}
    estimator = _PathSnapEstimator(
        train_fn   = train_fn,
        score_fn   = score_fn,
        all_labels = all_labels,
        **default_params,
    )

    cv = GroupKFold(n_splits=n_splits)

    logger.info(
        "Starting %s for '%s' — %d folds, %s iterations, %d training rows.",
        "GridSearch" if exhaustive else "RandomizedSearch",
        model_name,
        n_splits,
        "all" if exhaustive else n_iter,
        len(all_rows),
    )

    if exhaustive:
        searcher = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            refit=False,
            n_jobs=1,
            verbose=2,
            error_score="raise",
        )
    else:
        searcher = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            refit=False,
            n_jobs=1,
            verbose=2,
            random_state=random_state,
            error_score="raise",
        )

    searcher.fit(all_rows, y_dummy, groups=groups)

    best_params = searcher.best_params_
    best_score  = float(searcher.best_score_)

    logger.info("=" * 60)
    logger.info("Best params for '%s': %s", model_name, best_params)
    logger.info("Best CV Macro F1  : %.4f", best_score)
    logger.info("=" * 60)

    return {
        "best_params": best_params,
        "best_score":  best_score,
        "cv_results":  {
            k: v.tolist() if hasattr(v, "tolist") else list(v)
            for k, v in searcher.cv_results_.items()
            if k.startswith(("mean_test", "std_test", "params"))
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Results export
# ═══════════════════════════════════════════════════════════════════════════

def _save_results(model_name: str, results: dict) -> None:
    """Write best_params and best_score to tuning_results/<model_name>.json."""
    out_path = TUNING_DIR / f"{model_name}.json"
    safe = {
        "best_params": results["best_params"],
        "best_score":  results["best_score"],
        "cv_results":  results.get("cv_results", {}),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, default=str)
    logger.info("Tuning results saved → %s", out_path)


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
        description="PathSnap hyperparameter tuning — RandomizedSearch / GridSearch over all models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples
        --------
          # Tune random_forest with default settings (20 random trials, 5 folds)
          python -m src.pathsnap.tuning --model random_forest

          # Tune decision_tree exhaustively (all param combinations)
          python -m src.pathsnap.tuning --model decision_tree --exhaustive

          # Tune hist_gradient_boosting with more trials and fewer folds
          python -m src.pathsnap.tuning --model hist_gradient_boosting --n-iter 30 --folds 3

          # Tune all models and save results
          python -m src.pathsnap.tuning --all --export

          # Tune gaussian_naive_bayes and save results
          python -m src.pathsnap.tuning --model gaussian_naive_bayes --export
        """),
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        default="random_forest",
        help="Model to tune (default: random_forest). Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tune all five models sequentially.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        metavar="N",
        help="Random configs to sample per model (RandomizedSearch). Default: 20.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="K",
        help="Trip-stratified CV folds. Default: 5.",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Use GridSearchCV (all combinations) instead of RandomizedSearchCV.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        help="Use Bayesian Optimization (Optuna TPE) instead of RandomizedSearch/GridSearch.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save best params to tuning_results/<model>.json after each search.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducibility. Default: 42.",
    )
    args = parser.parse_args(argv)

    models_to_tune = ALL_MODELS if args.all else [args.model]

    # ── Load data once ────────────────────────────────────────────────────────
    logger.info("Loading training data …")
    all_rows, all_labels, _adj = _load_training_data()

    if not all_rows:
        logger.error("No training data loaded. Check the Student folder path.")
        sys.exit(1)

    logger.info(
        "Loaded %d candidate rows, %d labeled observations.",
        len(all_rows), len(all_labels),
    )

    # ── Run tuning ────────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for model_name in models_to_tune:
        logger.info("━" * 60)
        logger.info("Tuning model: %s", model_name)
        logger.info("━" * 60)
        try:
            if args.bayes:
                from .tune_rf_bayes import run_bayesian_optimization
                best_params, best_score, _ = run_bayesian_optimization(
                    all_rows,
                    all_labels,
                    n_trials=args.n_iter,
                    n_splits=args.folds,
                    random_state=args.random_state,
                )
                results = {"best_params": best_params, "best_score": best_score, "cv_results": {}}
            else:
                results = tune_model(
                    model_name,
                    all_rows,
                    all_labels,
                    n_iter       = args.n_iter,
                    n_splits     = args.folds,
                    exhaustive   = args.exhaustive,
                    random_state = args.random_state,
                )
            all_results[model_name] = results

            if args.export:
                _save_results(model_name, results)

        except Exception as exc:
            logger.exception("Tuning failed for '%s': %s", model_name, exc)

    # ── Print summary ─────────────────────────────────────────────────────────
    if all_results:
        print("\n" + "═" * 70)
        print("  TUNING SUMMARY")
        print("═" * 70)
        print(f"{'Model':<28} {'Best CV Macro F1':>18}  Best Params")
        print("─" * 70)
        for name, r in all_results.items():
            score     = r["best_score"]
            params    = r["best_params"]
            score_str = f"{score:.4f}" if isinstance(score, float) and score == score else "  n/a"
            print(f"{name:<28} {score_str:>18}  {params}")
        print("═" * 70)

    logger.info("Tuning complete.")


if __name__ == "__main__":
    main()
