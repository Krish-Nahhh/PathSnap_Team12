"""GPS-to-reference-arc alignment and GPS gap detection for PathSnap."""

import csv
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

from .candidates import Candidate, generate_candidates


class GPSGap(NamedTuple):
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    gap_seconds: float


def detect_gps_gaps(track: List[Tuple[float, float, float]], threshold_s: float = 5.0) -> List[GPSGap]:
    """Detect timestamp gaps or anomalies in GPS observations.

    Standard nominal sampling is 1.0 s. Gaps greater than threshold_s
    or non-positive time differences are flagged.
    """
    gaps: List[GPSGap] = []
    for i in range(1, len(track)):
        t_prev = track[i - 1][2]
        t_curr = track[i][2]
        dt = t_curr - t_prev
        if dt > threshold_s or dt <= 0:
            gaps.append(GPSGap(i - 1, i, t_prev, t_curr, dt))
    return gaps


def align_trip_labels(
    track: List[Tuple[float, float, float]],
    nodes: List[Tuple[float, float]],
    arcs: List[Tuple[int, int]],
    route: List[int],
    *,
    radius_m: float = 75.0,
    top_k: int = 15,
) -> Dict[int, int]:
    """Align GPS observations to the ground-truth reference route.

    Because the vehicle traverses the route in order, this aligns observations
    to route positions monotonically using dynamic programming (Viterbi alignment)
    while penalising large jumps and preferring candidate arcs close to the GPS fix.

    Returns:
        Mapping of observation_index -> true_arc_id.
    """
    if not track or not route:
        return {}

    # Map each arc_id in route to its index position(s) along the route sequence
    arc_to_positions: Dict[int, List[int]] = {}
    for pos, arc_id in enumerate(route):
        arc_to_positions.setdefault(arc_id, []).append(pos)

    candidates_by_obs = list(generate_candidates(track, nodes, arcs, radius_m=radius_m, top_k=top_k))

    # Initialize DP state at observation 0: pos -> (accumulated_cost, path_of_positions)
    dp: Dict[int, Tuple[float, List[int]]] = {}
    c0 = candidates_by_obs[0][1]
    for c in c0:
        if c.arc_id in arc_to_positions:
            for p in arc_to_positions[c.arc_id]:
                if p <= 15:  # Trip start constraint
                    cost = c.distance_m
                    if p not in dp or cost < dp[p][0]:
                        dp[p] = (cost, [p])

    if not dp:
        # Fallback if no candidate matches start of route
        dp[0] = (100.0, [0])

    for t in range(1, len(track)):
        dt = track[t][2] - track[t - 1][2]
        # Maximum route arc progress permitted: vehicle traverses at most ~4 arcs/s
        max_advance = max(4, int(max(dt, 1.0) * 4))

        cands = candidates_by_obs[t][1]
        c_dict: Dict[int, float] = {}
        for c in cands:
            if c.arc_id in arc_to_positions:
                for p in arc_to_positions[c.arc_id]:
                    if p not in c_dict or c.distance_m < c_dict[p]:
                        c_dict[p] = c.distance_m

        new_dp: Dict[int, Tuple[float, List[int]]] = {}
        if c_dict:
            for p, dist in c_dict.items():
                best_cost = float("inf")
                best_path: List[int] | None = None
                for prev_p, (prev_cost, prev_path) in dp.items():
                    step = p - prev_p
                    if 0 <= step <= max_advance:
                        cost = prev_cost + dist + step * 0.1
                        if cost < best_cost:
                            best_cost = cost
                            best_path = prev_path
                if best_path is not None:
                    new_dp[p] = (best_cost, best_path + [p])

        if not new_dp:
            # Carry forward best prior position if candidates missed or fix is noisy
            best_p = min(dp, key=lambda k: dp[k][0])
            prev_cost, prev_path = dp[best_p]
            new_dp[best_p] = (prev_cost + 100.0, prev_path + [best_p])

        # Beam search pruning to keep DP fast and prevent state explosion
        sorted_states = sorted(new_dp.items(), key=lambda item: item[1][0])[:30]
        dp = dict(sorted_states)

    best_final = min(dp.values(), key=lambda x: x[0])
    aligned_positions = best_final[1]
    return {obs_idx: route[p] for obs_idx, p in enumerate(aligned_positions)}


def load_trip_data(trip_dir: Path):
    """Read .track, .nodes, .arcs, and optional .route from a trip directory."""
    name = trip_dir.name
    with (trip_dir / f"{name}.track").open(encoding="utf-8") as f:
        track = [tuple(map(float, line.split())) for line in f]
    with (trip_dir / f"{name}.nodes").open(encoding="utf-8") as f:
        nodes = [tuple(map(float, line.split())) for line in f]
    with (trip_dir / f"{name}.arcs").open(encoding="utf-8") as f:
        arcs = [tuple(map(int, line.split())) for line in f]

    route_path = trip_dir / f"{name}.route"
    route = None
    if route_path.exists():
        with route_path.open(encoding="utf-8") as f:
            route = [int(line.strip()) for line in f]

    return track, nodes, arcs, route


def export_aligned_labels(
    trip_dirs: List[Path],
    output_csv: Path,
    split: str = "train",
) -> int:
    """Align labels for given trips and write out a CSV file.

    Format matches PathSnap candidate scorer expectations:
    split,trip_id,observation_index,true_arc_id
    """
    total_written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("split", "trip_id", "observation_index", "true_arc_id"))
        for trip_dir in trip_dirs:
            track, nodes, arcs, route = load_trip_data(trip_dir)
            if route is None:
                continue
            alignment = align_trip_labels(track, nodes, arcs, route)
            for obs_idx, true_arc in alignment.items():
                writer.writerow((split, trip_dir.name, obs_idx, true_arc))
                total_written += 1
    return total_written
