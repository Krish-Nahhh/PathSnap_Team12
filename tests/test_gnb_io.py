import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.pathsnap.gaussian_naive_bayes import train_gaussian_naive_bayes, score_candidates, save_model, load_model
from src.pathsnap.candidates import Candidate

def test_model_io_roundtrip(tmp_path: Path):
    # Create a tiny synthetic dataset with two candidates per observation
    candidates = []
    for obs_idx in range(3):
        # true candidate
        true_cand = Candidate(
            observation_index=obs_idx,
            timestamp=0.0,
            arc_id=obs_idx,
            candidate_rank=1,
            distance_m=5.0,
            gps_heading_deg=0.0,
            arc_heading_deg=None,
            heading_error_deg=0.0,
        )
        # false candidate
        false_cand = Candidate(
            observation_index=obs_idx,
            timestamp=0.0,
            arc_id=obs_idx + 100,
            candidate_rank=2,
            distance_m=20.0,
            gps_heading_deg=None,
            arc_heading_deg=None,
            heading_error_deg=30.0,
        )
        candidates.extend([true_cand, false_cand])
    rows = [("trip1", c) for c in candidates]
    labels = {("trip1", c.observation_index): c.arc_id for c in candidates if c.arc_id == c.observation_index}
    model = train_gaussian_naive_bayes(rows, labels)
    # Score original candidates
    probs_orig = score_candidates(model, candidates)
    # Save to JSON and reload
    json_path = tmp_path / "model.json"
    save_model(model, json_path)
    loaded_json = load_model(json_path)
    probs_json = score_candidates(loaded_json, candidates)
    assert all(abs(a - b) < 1e-12 for a, b in zip(probs_orig, probs_json))
    # Save to pickle and reload
    pkl_path = tmp_path / "model.pkl"
    save_model(model, pkl_path)
    loaded_pkl = load_model(pkl_path)
    probs_pkl = score_candidates(loaded_pkl, candidates)
    assert all(abs(a - b) < 1e-12 for a, b in zip(probs_orig, probs_pkl))
