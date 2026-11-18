"""Train, score, and persist Gaussian Naive Bayes models on aligned training fixes."""

import json
import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from .candidates import Candidate

FEATURE_NAMES = ("distance_m", "heading_error_deg", "heading_available")


def candidate_features(candidate: Candidate) -> Tuple[float, float, int]:
    """Extract standard features from a Candidate for classification."""
    error = candidate.heading_error_deg
    return (
        float(candidate.distance_m),
        0.0 if error is None else float(error),
        int(error is not None),
    )


class GaussianNaiveBayesClassifier:
    """Gaussian Naive Bayes classifier with log-domain computation.

    Provides a robust, zero-dependency scikit-learn compatible estimator.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = np.array([0, 1])
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X_arr.shape

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Training requires both true and false candidates on aligned training fixes")

        self.class_priors_ = np.zeros(n_classes, dtype=float)
        self.means_ = np.zeros((n_classes, n_features), dtype=float)
        self.vars_ = np.zeros((n_classes, n_features), dtype=float)

        var_max = np.var(X_arr, axis=0).max() if n_samples > 1 else 1.0
        epsilon = self.var_smoothing * var_max

        for idx, c in enumerate(self.classes_):
            X_c = X_arr[y_arr == c]
            self.class_priors_[idx] = X_c.shape[0] / n_samples
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + epsilon

        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self.class_priors_ is None or self.means_ is None or self.vars_ is None:
            raise ValueError("Estimator must be fitted before predict_proba")

        log_probs = np.zeros((X_arr.shape[0], len(self.classes_)), dtype=float)
        for idx in range(len(self.classes_)):
            mean = self.means_[idx]
            var = self.vars_[idx]
            log_prior = math.log(max(self.class_priors_[idx], 1e-12))
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X_arr - mean) ** 2) / var, axis=1)
            log_probs[:, idx] = log_prior + log_lik

        # Stable softmax via log-sum-exp
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to a plain JSON-compatible dictionary."""
        return {
            "model_type": "GaussianNaiveBayesClassifier",
            "var_smoothing": self.var_smoothing,
            "classes": self.classes_.tolist() if self.classes_ is not None else [0, 1],
            "class_priors": self.class_priors_.tolist() if self.class_priors_ is not None else [],
            "means": self.means_.tolist() if self.means_ is not None else [],
            "vars": self.vars_.tolist() if self.vars_ is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianNaiveBayesClassifier":
        """Instantiate classifier from serialized dictionary."""
        model = cls(var_smoothing=data.get("var_smoothing", 1e-9))
        model.classes_ = np.array(data["classes"], dtype=int)
        model.class_priors_ = np.array(data["class_priors"], dtype=float)
        model.means_ = np.array(data["means"], dtype=float)
        model.vars_ = np.array(data["vars"], dtype=float)
        return model


def train_gaussian_naive_bayes(
    training_rows: Sequence[Tuple[str, Candidate]],
    labels: Dict[Tuple[str, int], int],
    var_smoothing: float = 1e-9,
):
    """Fit Gaussian Naive Bayes from (trip_id, Candidate) rows and labels mapping.

    Labels map (trip_id, observation_index) -> true_arc_id.
    """
    groups: Dict[Tuple[str, int], List[Candidate]] = {}
    for trip_id, candidate in training_rows:
        key = (trip_id, candidate.observation_index)
        if key in labels:
            groups.setdefault(key, []).append(candidate)

    features: List[Tuple[float, float, int]] = []
    targets: List[int] = []
    for key, candidates in groups.items():
        true_arc = labels[key]
        if not any(c.arc_id == true_arc for c in candidates):
            continue
        for candidate in candidates:
            features.append(candidate_features(candidate))
            targets.append(int(candidate.arc_id == true_arc))

    if set(targets) != {0, 1}:
        raise ValueError("Training requires both true and false candidates on aligned training fixes")

    # Native robust GaussianNaiveBayesClassifier
    model = GaussianNaiveBayesClassifier(var_smoothing=var_smoothing)
    return model.fit(features, targets)


def score_candidates(model, candidates: Sequence[Candidate]) -> List[float]:
    """Return positive-class (true arc) probabilities in original candidate order."""
    if not candidates:
        return []
    features = [candidate_features(candidate) for candidate in candidates]
    probs = model.predict_proba(features)
    # Return probability of class 1 (true arc)
    return probs[:, 1].tolist()


def save_model(model: Any, path: Path | str) -> Path:
    """Save model artifact to disk (.json or .pkl)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".json" and hasattr(model, "to_dict"):
        with target.open("w", encoding="utf-8") as f:
            json.dump(model.to_dict(), f, indent=2)
    else:
        with target.open("wb") as f:
            pickle.dump(model, f)
    return target


def load_model(path: Path | str) -> Any:
    """Load model artifact from disk (.json or .pkl)."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Model file not found: {target}")

    if target.suffix == ".json":
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("model_type") == "GaussianNaiveBayesClassifier":
            return GaussianNaiveBayesClassifier.from_dict(data)
    with target.open("rb") as f:
        return pickle.load(f)
