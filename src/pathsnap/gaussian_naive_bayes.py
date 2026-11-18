"""Train and score candidates using Gaussian Naive Bayes on aligned training fixes."""

import math
from typing import Dict, List, Sequence, Tuple
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

    # Try scikit-learn GaussianNB if installed, otherwise use native robust GaussianNaiveBayesClassifier
    try:
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB(var_smoothing=var_smoothing)
        return model.fit(features, targets)
    except ImportError:
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
