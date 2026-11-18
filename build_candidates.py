"""Export PathSnap development candidates or score them against aligned labels."""

import argparse
import csv
from itertools import islice
from pathlib import Path

from src.pathsnap.candidates import Candidate, generate_candidates


def read_trip_file(trip, suffix, converters):
    path = trip / (trip.name + suffix)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            try:
                yield tuple(convert(field) for convert, field in zip(converters, fields, strict=True))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def export(args):
    trip = args.trip
    split = trip.parent.name
    if split not in {"train", "student_test"}:
        raise ValueError("Choose a trip directory under train or student_test")
    track = list(islice(read_trip_file(trip, ".track", (float, float, float)), args.limit))
    nodes = list(read_trip_file(trip, ".nodes", (float, float)))
    arcs = list(read_trip_file(trip, ".arcs", (int, int)))
    if any(not (0 <= start < len(nodes) and 0 <= end < len(nodes)) for start, end in arcs):
        raise ValueError(f"{trip}: arc endpoint is outside the node list")

    empty = total = 0
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("split", "trip_id", *Candidate._fields))
        for _, candidates in generate_candidates(track, nodes, arcs, radius_m=args.radius_m, top_k=args.top_k):
            empty += not candidates
            total += len(candidates)
            writer.writerows((split, trip.name, *candidate) for candidate in candidates)
    print(f"{total:,} candidate rows for {len(track):,} observations ({empty:,} without candidates)")


def read_keys(path, columns, required):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {', '.join(columns)}")
        for row in reader:
            try:
                yield (row["split"], row["trip_id"], int(row["observation_index"]),
                       int(row[required]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{reader.line_num}: invalid observation or arc ID") from error


def score(args):
    keys = ("split", "trip_id", "observation_index")
    candidates = set(read_keys(args.candidates, (*keys, "arc_id"), "arc_id"))
    labels = {}
    for *key, arc_id in read_keys(args.labels, (*keys, "true_arc_id"), "true_arc_id"):
        key = tuple(key)
        if key in labels and labels[key] != arc_id:
            raise ValueError(f"Conflicting labels for {key}")
        labels[key] = arc_id
    if not labels:
        raise ValueError("No aligned observation labels supplied")
    misses = sum((*key, arc_id) not in candidates for key, arc_id in labels.items())
    print(f"Candidate miss rate: {misses:,}/{len(labels):,} = {misses / len(labels):.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    build = commands.add_parser("generate")
    build.add_argument("trip", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--radius-m", type=float, default=75.0)
    build.add_argument("--top-k", type=int, default=10)
    build.add_argument("--limit", type=int, help="Export only the first N observations for a sample")
    build.set_defaults(run=export)
    coverage = commands.add_parser("score")
    coverage.add_argument("candidates", type=Path)
    coverage.add_argument("labels", type=Path)
    coverage.set_defaults(run=score)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
