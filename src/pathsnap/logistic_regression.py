"""Train the logistic regression candidate classifier on aligned training fixes."""

from .candidates import Candidate


FEATURE_NAMES = ("distance_m", "heading_error_deg", "heading_available")


def candidate_features(candidate: Candidate):
    error = candidate.heading_error_deg
    return (candidate.distance_m, 0.0 if error is None else error, int(error is not None))


def train_logistic_regression(training_rows, labels):
    """Fit from (trip_id, Candidate) rows and {(trip_id, observation_index): arc_id}.

    Pass only training trips, or the training portion of a trip-grouped fold.
    Fixes whose labeled arc was missed are measured by the candidate scorer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = {}
    for trip_id, candidate in training_rows:
        key = trip_id, candidate.observation_index
        if key in labels:
            groups.setdefault(key, []).append(candidate)

    features, targets = [], []
    for key, candidates in groups.items():
        true_arc = labels[key]
        if not any(candidate.arc_id == true_arc for candidate in candidates):
            continue
        for candidate in candidates:
            features.append(candidate_features(candidate))
            targets.append(int(candidate.arc_id == true_arc))
    if set(targets) != {0, 1}:
        raise ValueError("Training requires both true and false candidates on aligned training fixes")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
    return model.fit(features, targets)


def score_candidates(model, candidates):
    """Return positive-class probabilities in the original candidate order."""
    features = [candidate_features(candidate) for candidate in candidates]
    return model.predict_proba(features)[:, 1].tolist() if features else []
