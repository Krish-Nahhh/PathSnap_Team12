import csv
import sys
from pathlib import Path
from src.pathsnap.label_alignment import load_trip_data, align_trip_labels, export_aligned_labels

def verify_all_trips(train_dir: Path, output_csv: Path):
    """Align all training trips and verify index coverage.

    Returns the total number of rows written.
    """
    trip_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    total_written = 0
    for trip_dir in trip_dirs:
        track, nodes, arcs, route = load_trip_data(trip_dir)
        if route is None:
            print(f"[WARN] No route for {trip_dir.name}, skipping", file=sys.stderr)
            continue
        alignment = align_trip_labels(track, nodes, arcs, route)
        # Verify that every observation index 0..len(track)-1 appears exactly once
        expected_indices = set(range(len(track)))
        aligned_indices = set(alignment.keys())
        if expected_indices != aligned_indices:
            missing = expected_indices - aligned_indices
            extra = aligned_indices - expected_indices
            raise AssertionError(
                f"Alignment index mismatch for {trip_dir.name}: missing {sorted(missing)}, extra {sorted(extra)}"
            )
        for obs_idx, true_arc in alignment.items():
            total_written += 1
    return total_written

if __name__ == "__main__":
    # Expected layout: <repo>/Student folder/train/<trip_id>/
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = repo_root / "Student folder" / "train"
    output_csv = repo_root / "results" / "aligned_labels_verification.csv"
    written = verify_all_trips(train_dir, output_csv)
    print(f"Verification completed, total rows written: {written}")
