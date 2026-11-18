import json
from pathlib import Path
import sys

# Ensure repository src is on PYTHONPATH for imports
repo_root = Path(__file__).resolve().parents[2]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from scripts.gnb_cv import cross_validate

def test_small_cv():
    # Use fixture directory with at least 5 trips
    fixture_dir = repo_root / "tests" / "fixtures"
    assert fixture_dir.is_dir(), f"Fixture directory missing: {fixture_dir}"
    # Verify we have at least 5 distinct trips
    trips = [p.name for p in fixture_dir.iterdir() if p.is_dir()]
    assert len(trips) >= 5, f"Need >=5 trips for 5‑fold CV, found {len(trips)}"
    # Run CV with 5 folds on this small dataset
    metrics = cross_validate(fixture_dir, n_splits=5)
    # Basic sanity checks
    assert isinstance(metrics, list) and len(metrics) == 5
    # Ensure each fold has disjoint train/val groups (GroupKFold guarantees this)
    # We'll re‑run the split to extract the groups and verify
    from sklearn.model_selection import GroupKFold
    import numpy as np
    # Re‑build training rows to get groups
    from scripts.gnb_cv import build_training_data
    training_rows, _, _ = build_training_data(fixture_dir)
    groups = [trip_id for (trip_id, _cand) in training_rows]
    X = np.arange(len(training_rows))
    y = np.zeros(len(training_rows))  # dummy y
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups):
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups), "Train/val groups overlap"
    # Ensure metric keys exist
    required_keys = {"fold", "top1_accuracy", "precision", "recall", "f1", "candidate_recall", "candidate_miss_rate", "log_loss"}
    for m in metrics:
        assert required_keys.issubset(m.keys()), f"Missing keys in metrics: {m}"
