import sys
from pathlib import Path

# Ensure src is on PYTHONPATH for imports
repo_root = Path(__file__).resolve().parents[2]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import json
import math
import numpy as np

from src.pathsnap.label_alignment import load_trip_data, detect_gps_gaps, align_trip_labels
from src.pathsnap.preprocessing import validate_and_audit_track


def test_trip_00000001_gap_detection():
    repo_root = Path(__file__).resolve().parents[2]
    trip_dir = repo_root / "Student folder" / "train" / "00000001"
    track, _, _, _ = load_trip_data(trip_dir)
    gaps = detect_gps_gaps(track, threshold_s=5.0)
    # Find a gap with duration close to 16 seconds
    gap_durations = [g.gap_seconds for g in gaps]
    assert any(abs(d - 16.0) < 0.5 for d in gap_durations), f"Expected ~16s gap, got {gap_durations}"
    # Verify that all observation indices are present after alignment
    from src.pathsnap.label_alignment import export_aligned_labels
    # Use the verification script function indirectly
    # Alignment should preserve every index
    _, nodes, arcs, route = load_trip_data(trip_dir)
    alignment = align_trip_labels(track, nodes, arcs, route)
    expected = set(range(len(track)))
    assert set(alignment.keys()) == expected


def test_verify_all_trips_runs_without_error():
    # import the verification function from the script
    from scripts.verify_alignment import verify_all_trips
    repo_root = Path(__file__).resolve().parents[2]
    train_dir = repo_root / "Student folder" / "train"
    output_csv = repo_root / "results" / "tmp_aligned.csv"
    # Ensure function runs (it will raise if mismatch)
    total = verify_all_trips(train_dir, output_csv)
    assert total > 0
