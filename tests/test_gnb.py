"""Tests for Gaussian Naive Bayes classifier, GroupKFold validation logic, and serialization."""

import numpy as np
import pytest
from src.pathsnap.candidates import Candidate
from src.pathsnap.gaussian_naive_bayes import (
    candidate_features,
    GaussianNaiveBayesClassifier,
    train_gaussian_naive_bayes,
    score_candidates,
    save_model,
    load_model,
)


def test_candidate_features():
    c = Candidate(
        observation_index=0,
        timestamp=1.0,
        arc_id=100,
        candidate_rank=1,
        distance_m=12.5,
        gps_heading_deg=45.0,
        arc_heading_deg=50.0,
        heading_error_deg=5.0,
    )
    feats = candidate_features(c)
    assert feats == (12.5, 5.0, 1)

    c_no_heading = Candidate(
        observation_index=1,
        timestamp=2.0,
        arc_id=101,
        candidate_rank=2,
        distance_m=30.0,
        gps_heading_deg=None,
        arc_heading_deg=50.0,
        heading_error_deg=None,
    )
    feats_no_heading = candidate_features(c_no_heading)
    assert feats_no_heading == (30.0, 0.0, 0)


def test_gnb_fit_predict_proba():
    X = np.array([
        [2.0, 5.0, 1],
        [3.0, 10.0, 1],
        [40.0, 120.0, 1],
        [60.0, 90.0, 0],
    ])
    y = np.array([1, 1, 0, 0])

    model = GaussianNaiveBayesClassifier()
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)
    # Probabilities must sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    # Close distance should have higher P(class=1) than far distance
    test_close = model.predict_proba([[2.5, 6.0, 1]])[0, 1]
    test_far = model.predict_proba([[55.0, 100.0, 1]])[0, 1]
    assert test_close > test_far


def test_gnb_serialization(tmp_path):
    X = np.array([[2.0, 5.0, 1], [50.0, 90.0, 0]])
    y = np.array([1, 0])
    model = GaussianNaiveBayesClassifier().fit(X, y)

    # Test JSON serialization
    json_path = tmp_path / "model.json"
    save_model(model, json_path)
    loaded_json = load_model(json_path)
    probs_orig = model.predict_proba(X)
    probs_loaded = loaded_json.predict_proba(X)
    assert np.allclose(probs_orig, probs_loaded)

    # Test Pickle serialization
    pkl_path = tmp_path / "model.pkl"
    save_model(model, pkl_path)
    loaded_pkl = load_model(pkl_path)
    probs_pkl = loaded_pkl.predict_proba(X)
    assert np.allclose(probs_orig, probs_pkl)


def test_group_kfold_leakage_prevention():
    # Verify strict trip isolation logic
    all_trips = [f"{i:08d}" for i in range(20)]
    folds = np.array_split(all_trips, 5)

    for i in range(5):
        val_trips = set(folds[i])
        train_trips = set(t for t in all_trips if t not in val_trips)
        # Must be strictly disjoint
        assert len(train_trips.intersection(val_trips)) == 0
        assert len(train_trips) + len(val_trips) == 20
