import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

from src.pathsnap.candidates import generate_candidates
from src.pathsnap.label_alignment import load_trip_data, align_trip_labels
from src.pathsnap.gaussian_naive_bayes import train_gaussian_naive_bayes, score_candidates

def build_training_data(train_dir: Path):
    """Return (training_rows, labels, trip_ids) for all trips in train_dir.
    training_rows: list of (trip_id, Candidate)
    labels: dict mapping (trip_id, observation_index) -> true_arc_id
    """
    training_rows = []
    labels = {}
    trip_ids = []
    for trip_dir in sorted([d for d in train_dir.iterdir() if d.is_dir()]):
        trip_id = trip_dir.name
        trip_ids.append(trip_id)
        track, nodes, arcs, route = load_trip_data(trip_dir)
        if route is None:
            continue
        alignment = align_trip_labels(track, nodes, arcs, route)
        for obs_idx, true_arc in alignment.items():
            labels[(trip_id, obs_idx)] = true_arc
        candidates_by_obs = list(generate_candidates(track, nodes, arcs, radius_m=75.0, top_k=15))
        for obs_idx, cand_list in candidates_by_obs:
            for cand in cand_list:
                training_rows.append((trip_id, cand))
    return training_rows, labels, trip_ids

def evaluate_fold(model, candidates, true_arc_id):
    if not candidates:
        return False, False
    probs = score_candidates(model, candidates)
    max_idx = int(np.argmax(probs))
    pred_arc = candidates[max_idx].arc_id
    top1_correct = (pred_arc == true_arc_id)
    candidate_present = any(c.arc_id == true_arc_id for c in candidates)
    return top1_correct, candidate_present

def cross_validate(train_dir: Path, n_splits: int = 5):
    training_rows, labels, _ = build_training_data(train_dir)
    groups = [trip_id for (trip_id, _cand) in training_rows]
    X = np.arange(len(training_rows))
    y = np.array([1 if cand.arc_id == labels[(trip_id, cand.observation_index)] else 0
                  for (trip_id, cand) in training_rows])
    gkf = GroupKFold(n_splits=n_splits)
    metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_subset = [training_rows[i] for i in train_idx]
        model = train_gaussian_naive_bayes(train_subset, labels)
        val_by_obs = defaultdict(list)
        for i in val_idx:
            trip_id, cand = training_rows[i]
            val_by_obs[(trip_id, cand.observation_index)].append(cand)
        tp = fp = fn = 0
        candidate_miss = 0
        total_obs = 0
        log_losses = []
        for (trip_id, obs_idx), cand_list in val_by_obs.items():
            true_arc = labels[(trip_id, obs_idx)]
            top1_correct, present = evaluate_fold(model, cand_list, true_arc)
            total_obs += 1
            if present:
                if top1_correct:
                    tp += 1
                else:
                    fn += 1
            else:
                candidate_miss += 1
                fn += 1
            probs = score_candidates(model, cand_list)
            p_true = 0.0
            for cand, prob in zip(cand_list, probs):
                if cand.arc_id == true_arc:
                    p_true = prob
                    break
            p_true = max(p_true, 1e-15)
            log_losses.append(-np.log(p_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        top1_acc = tp / total_obs if total_obs > 0 else 0.0
        candidate_recall = (total_obs - candidate_miss) / total_obs if total_obs > 0 else 0.0
        candidate_miss_rate = candidate_miss / total_obs if total_obs > 0 else 0.0
        avg_logloss = sum(log_losses) / len(log_losses) if log_losses else 0.0
        metrics.append({
            "fold": fold_idx,
            "top1_accuracy": top1_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidate_recall": candidate_recall,
            "candidate_miss_rate": candidate_miss_rate,
            "log_loss": avg_logloss,
        })
    return metrics

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    # Correct path: student data is sibling of repo root
    train_dir = repo_root.parent / "Student folder" / "train"
    metrics = cross_validate(train_dir)
    out_path = repo_root / "results" / "gnb_cv_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))