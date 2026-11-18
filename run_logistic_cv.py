"""Train and evaluate Manoj's logistic regression using whole-trip folds."""

import argparse
import csv
import gzip
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_data(repo):
    with (repo / "results/aligned_labels_train.csv").open(newline="", encoding="utf-8-sig") as source:
        labels = {
            (row["trip_id"], int(row["observation_index"])): int(row["true_arc_id"])
            for row in csv.DictReader(source)
            if row["split"] == "train" and row["true_arc_id"].strip()
        }
    trips = np.array([trip for trip, _ in labels])
    reference = np.array(list(labels.values()))
    observation_ids = {key: index for index, key in enumerate(labels)}
    # The generator writes candidates in observation order and breaks distance ties by arc ID.
    features, groups, arcs = [], [], []
    with gzip.open(repo / "data/candidate_handoff/train_candidates.csv.gz", "rt", newline="") as source:
        for row in csv.DictReader(source):
            observation = observation_ids.get((row["trip_id"], int(row["observation_index"])))
            if observation is None:
                continue
            error = row["heading_error_deg"]
            features.append((float(row["distance_m"]), float(error) if error else 0.0, bool(error)))
            groups.append(observation)
            arcs.append(int(row["arc_id"]))
    groups = np.array(groups)
    truth = np.array(arcs) == reference[groups]
    covered = np.zeros(len(trips), dtype=bool)
    covered[groups[truth]] = True
    return np.array(features), groups, truth, trips, covered


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    x, groups, truth, trips, covered = read_data(args.repo)
    rows = []
    for fold, (train, validation) in enumerate(GroupKFold(5).split(trips, groups=trips), 1):
        training_mask = np.zeros(len(trips), dtype=bool)
        training_mask[train] = True
        validation_mask = np.zeros(len(trips), dtype=bool)
        validation_mask[validation] = True
        fitting = (training_mask & covered)[groups]
        testing = validation_mask[groups]

        start = perf_counter()
        model = make_model().fit(x[fitting], truth[fitting])
        fit_seconds = perf_counter() - start
        start = perf_counter()
        scores = model.predict_proba(x[testing])[:, 1]
        inference_seconds = perf_counter() - start

        test_groups, test_truth = groups[testing], truth[testing]
        # Stable sorting keeps the original rank when scores tie.
        order = np.argsort(-scores, kind="stable")
        _, first = np.unique(test_groups[order], return_index=True)
        hits = int(test_truth[order[first]].sum())
        row = {
            "fold": fold,
            "validation_trip_ids": " ".join(np.unique(trips[validation])),
            "labeled_fixes": len(validation),
            "candidate_misses": int((~covered[validation]).sum()),
            "top1_hits": hits,
            "top1_accuracy": hits / len(validation),
            "candidate_macro_f1": f1_score(test_truth, scores > 0.5, average="macro"),
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
        }
        rows.append(row)
        print(f"Fold {fold}: Top-1 {row['top1_accuracy']:.2%}, macro F1 {row['candidate_macro_f1']:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "logistic_regression_cv.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model = make_model().fit(x[covered[groups]], truth[covered[groups]])
    joblib.dump(model, args.output / "logistic_regression.joblib")
    print(f"Top-1: {sum(row['top1_hits'] for row in rows) / len(trips):.2%}")
    print(f"Mean fold macro F1: {np.mean([row['candidate_macro_f1'] for row in rows]):.4f}")
    print(f"Candidate miss rate: {(~covered).mean():.2%}")


if __name__ == "__main__":
    main()
