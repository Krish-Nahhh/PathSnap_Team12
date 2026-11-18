"""Compare a fitted candidate classifier with and without Viterbi on student-test."""

import argparse
import csv
from difflib import SequenceMatcher
from itertools import groupby, pairwise
from pathlib import Path
from time import perf_counter

import joblib

from src.pathsnap.candidates import generate_candidates
from src.pathsnap.decoder import decode_route
from src.pathsnap.label_alignment import align_trip_labels, detect_gps_gaps, load_trip_data
from src.pathsnap.logistic_regression import score_candidates


def predict_trip(model, track, nodes, arcs):
    candidate_lists = [rows for _, rows in generate_candidates(track, nodes, arcs)]
    candidates = [candidate for rows in candidate_lists for candidate in rows]
    scores = iter(score_candidates(model, candidates))
    observations = [
        [(candidate.arc_id, next(scores)) for candidate in rows]
        for rows in candidate_lists
    ]
    top1 = [max(rows, key=lambda pair: pair[1])[0] if rows else None
            for rows in observations]
    gaps = detect_gps_gaps(track)
    start = perf_counter()
    decoded = decode_route(observations, arcs, segment_starts=(gap.end_index for gap in gaps))
    return top1, decoded, perf_counter() - start, len(gaps)


def measure(predictions, labels, reference, arcs):
    hits = sum(predictions[index] == arc for index, arc in labels.items())
    adjacent = [(a, b) for a, b in pairwise(predictions) if a is not None and b is not None]
    direct = sum(a == b or arcs[a][1] == arcs[b][0] for a, b in adjacent)
    sequence = [arc for arc, _ in groupby(predictions) if arc is not None]
    reference_sequence = [arc for arc, _ in groupby(reference)]
    return {
        "matched_fixes": hits,
        "fix_accuracy": hits / len(labels),
        "sequence_similarity": SequenceMatcher(
            None, reference_sequence, sequence, autojunk=False
        ).ratio(),
        "scored_adjacent_pairs": len(adjacent),
        "direct_pairs": direct,
        "direct_continuity": direct / len(adjacent) if adjacent else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_test", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = joblib.load(args.model)
    rows = []
    for trip in sorted(path for path in args.student_test.iterdir() if path.is_dir()):
        missing = [suffix for suffix in ("track", "nodes", "arcs", "route")
                   if not (trip / f"{trip.name}.{suffix}").is_file()]
        if missing:
            print(f"{trip.name}: skipped (missing {', '.join(missing)})")
            continue
        track, nodes, arcs, reference = load_trip_data(trip)
        top1, decoded, seconds, gaps = predict_trip(model, track, nodes, arcs)
        labels = align_trip_labels(track, nodes, arcs, reference)
        if not labels:
            print(f"{trip.name}: skipped (empty alignment)")
            continue
        for method, predictions in (("classifier_only", top1), ("viterbi", decoded)):
            rows.append({
                "trip_id": trip.name,
                "method": method,
                "labeled_fixes": len(labels),
                "timestamp_breaks": gaps,
                "unmatched_fixes": predictions.count(None),
                **measure(predictions, labels, reference, arcs),
                "decode_seconds": seconds if method == "viterbi" else 0.0,
            })
        print(f"{trip.name}: evaluated {len(labels)} fixes", flush=True)
    if not rows:
        parser.error("No student-test trips could be evaluated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
