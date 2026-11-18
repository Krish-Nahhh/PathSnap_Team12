"""Tests for data preprocessing, coordinate validation, timestamp checks, and GPS gap detection."""

import math
import pytest
from src.pathsnap.preprocessing import (
    validate_coordinates,
    validate_and_audit_track,
    sanitize_track_for_candidate_generation,
    haversine_distance_m,
)


def test_coordinate_validation_normal():
    valid, flags = validate_coordinates(8.595, 49.844)
    assert valid is True
    assert len(flags) == 0


def test_coordinate_validation_out_of_bounds():
    # Longitude > 180
    valid, flags = validate_coordinates(185.0, 49.844)
    assert valid is False
    assert "OUT_OF_GLOBAL_BOUNDS" in flags

    # Latitude < -90
    valid, flags = validate_coordinates(8.595, -95.0)
    assert valid is False
    assert "OUT_OF_GLOBAL_BOUNDS" in flags


def test_coordinate_validation_non_finite():
    valid, flags = validate_coordinates(float("nan"), 49.844)
    assert valid is False
    assert "NON_FINITE_COORDINATE" in flags


def test_haversine_distance():
    # Distance between same points is 0
    assert haversine_distance_m(8.5, 49.8, 8.5, 49.8) == pytest.approx(0.0)
    # Known approximate distance: 1 degree latitude is ~111.19 km
    dist = haversine_distance_m(8.0, 49.0, 8.0, 50.0)
    assert 111_000 <= dist <= 112_000


def test_track_timestamp_anomalies_and_gaps():
    # Sample track with:
    # 0: normal
    # 1: normal (dt=1.0)
    # 2: duplicate timestamp (dt=0.0)
    # 3: backward timestamp (dt=-2.0)
    # 4: gap (dt=15.0)
    track = [
        (8.5950, 49.8440, 10.0),
        (8.5951, 49.8441, 11.0),
        (8.5952, 49.8442, 11.0),  # Duplicate
        (8.5953, 49.8443, 9.0),   # Backward
        (8.5955, 49.8445, 24.0),  # Gap of 15s
    ]
    report = validate_and_audit_track(track, trip_id="test_trip", gap_threshold_s=5.0)
    assert report.total_observations == 5
    assert report.valid_observations < 5
    assert len(report.anomalies) >= 3

    flags_found = {flag for a in report.anomalies for flag in a.flags}
    assert "DUPLICATE_TIMESTAMP" in flags_found
    assert "BACKWARD_TIMESTAMP" in flags_found
    assert "GPS_GAP" in flags_found
    assert len(report.gaps) >= 1
    assert any(g.gap_seconds == 15.0 for g in report.gaps)


def test_excessive_speed_and_implausible_jump():
    # Sudden jump of ~10 km in 1 second
    track = [
        (8.5950, 49.8440, 1.0),
        (8.7500, 49.9500, 2.0),
    ]
    report = validate_and_audit_track(track, max_speed_mps=50.0, max_jump_m=200.0)
    assert report.anomalous_observations >= 1
    flags = report.anomalies[0].flags
    assert "EXCESSIVE_SPEED" in flags or "IMPLAUSIBLE_JUMP" in flags


def test_track_sanitization_preserves_length_and_indices():
    track = [
        (8.5950, 49.8440, 1.0),
        (float("nan"), 49.8441, 2.0),  # Corrupted fix
        (8.5954, 49.8444, 3.0),
    ]
    report = validate_and_audit_track(track)
    sanitized = sanitize_track_for_candidate_generation(track, report)

    # Length must be strictly preserved
    assert len(sanitized) == len(track)
    # Middle NaN coordinate must be interpolated to finite float
    assert math.isfinite(sanitized[1][0])
    assert math.isclose(sanitized[1][0], (8.5950 + 8.5954) / 2.0, rel_tol=1e-3)
    assert sanitized[1][2] == 2.0
