"""Calibrate Viterbi transition penalties on trips held out from the emission model."""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from itertools import pairwise, product
from pathlib import Path

import numpy as np

from run_logistic_cv import make_model


def measure(trips, decode, penalties=None):
    correct = labeled = connected = transitions = 0
    for observations, reference, arcs, starts in trips:
        if penalties is None:
            prediction = [max(row, key=lambda pair: pair[1])[0] if row else None
                          for row in observations]
        else:
            hops, connector, disconnected = penalties
            prediction = decode(observations, arcs, max_connector_hops=hops,
                                connector_penalty=connector, disconnected_penalty=disconnected,
                                segment_starts=starts)
        if len(prediction) != len(reference):
            raise ValueError("Decoder changed the observation count")
        labeled += sum(arc is not None for arc in reference)
        correct += sum(expected is not None and actual == expected
                       for actual, expected in zip(prediction, reference, strict=True))
        for index, (previous, current) in enumerate(pairwise(prediction), 1):
            if index in starts or previous is None or current is None:
                continue
            transitions += 1
            connected += previous == current or arcs[previous][1] == arcs[current][0]
    return dict(top1_accuracy=correct / labeled, correct=correct, labeled_fixes=labeled,
                direct_continuity=connected / transitions if transitions else None,
                transitions=transitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.pathsnap.decoder import decode_route

    with (args.repo / "results/logistic_regression/logistic_regression_cv.csv").open(newline="") as source:
        folds = {int(row["fold"]): set(row["validation_trip_ids"].split()) for row in csv.DictReader(source)}
    tuning, evaluation = folds[2], folds[3]
    held_out = tuning | evaluation
    assert not tuning & evaluation

    labels = defaultdict(dict)
    with (args.repo / "results/aligned_labels_train.csv").open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row["split"] == "train" and row["true_arc_id"].strip():
                labels[row["trip_id"]][int(row["observation_index"])] = int(row["true_arc_id"])

    fit_features, fit_labels = [], []
    held_features = defaultdict(list)
    pending = []
    previous = None

    def save_observation():
        if previous is None:
            return
        trip, index = previous
        if trip in held_out:
            held_features[trip].extend((index, arc, features) for arc, features in pending)
        else:
            reference = labels[trip].get(index)
            if reference is not None and any(arc == reference for arc, _ in pending):
                fit_features.extend(features for _, features in pending)
                fit_labels.extend(arc == reference for arc, _ in pending)

    with gzip.open(args.repo / "data/candidate_handoff/train_candidates.csv.gz", "rt", newline="") as source:
        for row in csv.DictReader(source):
            key = row["trip_id"], int(row["observation_index"])
            if key != previous:
                save_observation()
                pending = []
                previous = key
            error = row["heading_error_deg"]
            pending.append((int(row["arc_id"]), (float(row["distance_m"]), float(error or 0), bool(error))))
        save_observation()

    model = make_model().fit(np.asarray(fit_features), np.asarray(fit_labels))
    trip_groups = {"tuning": [], "evaluation": []}
    for trip in sorted(held_out):
        base = args.train_root / trip / trip
        times = [float(line.split()[2]) for line in base.with_suffix(".track").read_text().splitlines()]
        arcs = [tuple(map(int, line.split())) for line in base.with_suffix(".arcs").read_text().splitlines()]
        starts = {i for i, (a, b) in enumerate(pairwise(times), 1) if not 0 < b - a <= 10}
        observations = [[] for _ in times]
        features = held_features[trip]
        if features:
            scores = model.predict_proba(np.asarray([row[2] for row in features]))[:, 1]
            for (index, arc, _), score in zip(features, scores, strict=True):
                observations[index].append((arc, float(score)))
        reference = [labels[trip].get(i) for i in range(len(times))]
        trip_groups["tuning" if trip in tuning else "evaluation"].append(
            (observations, reference, arcs, starts))

    trials = []
    for hops, connector, disconnected in product((1, 3), (0.25, 1.0), (2.0, 8.0)):
        if disconnected <= hops * connector:
            continue
        config = hops, connector, disconnected
        metrics = measure(trip_groups["tuning"], decode_route, config)
        trials.append((config, metrics))
        print(config, f"tuning Top-1 {metrics['top1_accuracy']:.2%}", flush=True)
    best, tuning_metrics = max(trials, key=lambda trial: (trial[1]["top1_accuracy"],
                                                        trial[1]["direct_continuity"]))
    config = dict(zip(("max_connector_hops", "connector_penalty", "disconnected_penalty"), best))
    report = dict(
        config=config, max_gap_seconds=10, tuning_trip_ids=sorted(tuning),
        evaluation_trip_ids=sorted(evaluation), emission_training_trip_ids=sorted(labels.keys() - held_out),
        tuning_classifier_only=measure(trip_groups["tuning"], decode_route),
        tuning_viterbi=tuning_metrics,
        trials=[dict(parameters=p, **m) for p, m in trials],
        evaluation_classifier_only=measure(trip_groups["evaluation"], decode_route),
        evaluation_default_viterbi=measure(trip_groups["evaluation"], decode_route, (3, 1.0, 8.0)),
        evaluation_tuned_viterbi=measure(trip_groups["evaluation"], decode_route, best),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k == "config" or k.startswith("evaluation_") and k != "evaluation_trip_ids"}, indent=2))


if __name__ == "__main__":
    main()
