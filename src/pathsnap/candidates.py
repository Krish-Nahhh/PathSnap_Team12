"""Build distance-ranked candidates for directed road arcs."""

from collections import defaultdict
from heapq import nsmallest
from math import atan2, cos, degrees, floor, hypot, isfinite, pi, radians
from typing import NamedTuple


class Candidate(NamedTuple):
    observation_index: int
    timestamp: float
    arc_id: int
    candidate_rank: int
    distance_m: float
    gps_heading_deg: float | None
    arc_heading_deg: float | None
    heading_error_deg: float | None


def heading(east, north):
    return degrees(atan2(east, north)) % 360


def generate_candidates(track, nodes, arcs, *, radius_m=75.0, top_k=10):
    """Yield (observation index, candidate rows), including empty candidate lists.

    Track rows are (longitude, latitude, timestamp in seconds). Arc IDs and
    observation indices are zero-based source row numbers. Ranks start at one.
    """
    if not isfinite(radius_m) or radius_m <= 0 or top_k < 1:
        raise ValueError("radius_m and top_k must be positive")
    if not track:
        return

    origin_lon, origin_lat, _ = track[0]
    meters_per_degree = pi * 6_371_008.8 / 180
    longitude_scale = meters_per_degree * cos(radians(origin_lat))

    def project(lon, lat):
        lon_difference = (lon - origin_lon + 180) % 360 - 180
        return lon_difference * longitude_scale, (lat - origin_lat) * meters_per_degree

    def cells(low, high):
        return range(floor(low / radius_m), floor(high / radius_m) + 1)

    projected_nodes = [project(*node) for node in nodes]
    segments = []
    grid = defaultdict(list)
    long_segments = []
    for arc_id, (start, end) in enumerate(arcs):
        x1, y1 = projected_nodes[start]
        x2, y2 = projected_nodes[end]
        dx, dy = x2 - x1, y2 - y1
        length_squared = dx * dx + dy * dy
        segments.append((x1, y1, dx, dy, length_squared,
                         heading(dx, dy) if length_squared else None))
        columns = cells(min(x1, x2), max(x1, x2))
        rows = cells(min(y1, y2), max(y1, y2))
        if len(columns) * len(rows) > 256:
            long_segments.append(arc_id)
            continue
        for column in columns:
            for row in rows:
                grid[column, row].append(arc_id)

    previous = None
    for index, (lon, lat, timestamp) in enumerate(track):
        x, y = project(lon, lat)
        gps_heading = None
        if previous is not None:
            prev_x, prev_y, prev_time = previous
            gap = timestamp - prev_time
            east, north = x - prev_x, y - prev_y
            if 0 < gap <= 10 and hypot(east, north) >= max(3, gap):
                gps_heading = heading(east, north)
        previous = x, y, timestamp

        nearby = set(long_segments)
        for column in cells(x - radius_m, x + radius_m):
            for row in cells(y - radius_m, y + radius_m):
                nearby.update(grid.get((column, row), ()))

        distances = []
        for arc_id in nearby:
            x1, y1, dx, dy, length_squared, _ = segments[arc_id]
            offset_x, offset_y = x - x1, y - y1
            fraction = 0.0
            if length_squared:
                fraction = max(0.0, min(1.0, (offset_x * dx + offset_y * dy) / length_squared))
            distance = hypot(offset_x - fraction * dx, offset_y - fraction * dy)
            if distance <= radius_m:
                distances.append((distance, arc_id))

        candidates = []
        for rank, (distance, arc_id) in enumerate(nsmallest(top_k, distances), 1):
            arc_heading = segments[arc_id][-1]
            error = None
            if gps_heading is not None and arc_heading is not None:
                error = abs((gps_heading - arc_heading + 180) % 360 - 180)
            candidates.append(Candidate(index, timestamp, arc_id, rank, distance,
                                        gps_heading, arc_heading, error))
        yield index, candidates
