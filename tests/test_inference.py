"""Tests for inference pipeline, model loading, and trip prediction."""

from pathlib import Path
import pytest
from src.pathsnap.inference import PathSnapPredictor, predict_trip


def test_predictor_missing_directory():
    predictor = PathSnapPredictor()
    with pytest.raises(FileNotFoundError):
        predictor.predict_trip("non_existent_trip_folder_xyz")


def test_predict_trip_on_sample_trip():
    # Use real test trip from student_test or train
    sample_trip = Path(r"D:\ML_Hackathon\Student folder\student_test\00000000")
    if not sample_trip.exists():
        pytest.skip("Dataset student_test/00000000 not found")

    result = predict_trip(sample_trip, radius_m=75.0, top_k=10)
    assert result["trip_id"] == "00000000"
    assert result["observation_count"] > 0

    # Length of predicted_arcs must match observation count
    assert len(result["predicted_arcs"]) == result["observation_count"]
    # Candidate scores length must match observation count
    assert len(result["candidate_scores"]) == result["observation_count"]
    # Quality report must be returned
    assert result["quality_report"] is not None
    assert result["quality_report"].total_observations == result["observation_count"]
