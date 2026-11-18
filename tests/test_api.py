"""Tests for FastAPI backend endpoints, request validation, and ground-truth privacy."""

import pytest
from fastapi import HTTPException
from api.main import (
    app,
    health_check,
    list_evaluation_trips,
    predict,
    find_trip_dir,
    HealthResponse,
    PredictionResponse,
)


def test_health_endpoints():
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "ok"
    assert res.service == "PathSnap API"
    assert res.model_loaded is True


def test_list_trips():
    trips = list_evaluation_trips(split="student_test")
    assert isinstance(trips, list)
    assert len(trips) == 10
    trip_ids = [t.trip_id for t in trips]
    assert "00000000" in trip_ids
    assert all(t.observation_count > 0 for t in trips)


def test_predict_endpoint_valid_trip():
    res = predict(trip_id="00000000", radius_m=75.0, top_k=5)
    assert isinstance(res, PredictionResponse)
    assert res.trip_id == "00000000"
    assert res.observation_count > 0
    assert len(res.predicted_route) == res.observation_count
    assert res.processing_time_ms > 0
    assert res.message == "Route reconstruction successful."

    # Ground truth privacy check: PredictionResponse model has no ground truth route sequence field
    response_dict = res.model_dump()
    assert "ground_truth" not in response_dict
    assert "true_route" not in response_dict
    assert "aligned_labels" not in response_dict

    # Accuracy was evaluated server-side since student_test/00000000 has reference route
    assert res.accuracy is not None
    assert 0.0 <= res.accuracy <= 1.0


def test_predict_endpoint_missing_trip():
    with pytest.raises(HTTPException) as exc_info:
        predict(trip_id="99999999")
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_predict_endpoint_invalid_trip_id_format():
    with pytest.raises(HTTPException) as exc_info:
        predict(trip_id="!@#$%^&*")
    assert exc_info.value.status_code == 400
    assert "invalid trip_id" in exc_info.value.detail.lower()
