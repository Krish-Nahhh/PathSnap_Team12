"""Viterbi decoding of per-fix candidate scores on a directed road network."""

from collections import defaultdict, deque
from functools import lru_cache
from math import isfinite, log


def decode_route(observations, arcs, *, max_connector_hops=3,
                 connector_penalty=1.0, disconnected_penalty=8.0):
    """Return an arc ID or None per observation.

    Each observation is an iterable of (arc_id, positive-class probability).
    Arc endpoints use the input graph's original node IDs. Empty observations
    break the sequence. The resulting IDs are per-fix predictions.
    """
    if (max_connector_hops < 0 or not isfinite(connector_penalty)
            or not isfinite(disconnected_penalty) or connector_penalty < 0
            or disconnected_penalty <= max_connector_hops * connector_penalty):
        raise ValueError("Disconnections must cost more than any allowed connector")

    outgoing = defaultdict(list)
    for start, end in arcs:
        outgoing[start].append(end)

    @lru_cache(maxsize=4096)
    def reachable(start):
        hops = {start: 0}
        pending = deque([start])
        while pending:
            node = pending.popleft()
            if hops[node] == max_connector_hops:
                continue
            for neighbor in outgoing.get(node, ()):
                if neighbor not in hops:
                    hops[neighbor] = hops[node] + 1
                    pending.append(neighbor)
        return hops

    def transition(previous, current):
        if previous == current:
            return 0.0
        hops = reachable(arcs[previous][1]).get(arcs[current][0])
        return disconnected_penalty if hops is None else connector_penalty * hops

    route, history = [], []
    scores = {}

    def finish_run():
        if scores:
            arc = max(scores, key=scores.get)
            run = []
            for parents in reversed(history):
                run.append(arc)
                arc = parents[arc]
            route.extend(reversed(run))
            history.clear()

    for candidates in observations:
        next_scores, parents = {}, {}
        for arc, probability in candidates:
            if not 0 <= arc < len(arcs) or not 0 <= probability <= 1:
                raise ValueError("Expected a valid directed arc ID and probability in [0, 1]")
            if arc in next_scores:
                raise ValueError("Each observation must contain unique directed arc IDs")
            emission = log(max(probability, 1e-12))
            if scores:
                value, predecessor = max((score - transition(previous, arc), previous)
                                         for previous, score in scores.items())
                next_scores[arc] = value + emission
                parents[arc] = predecessor
            else:
                next_scores[arc] = emission
                parents[arc] = None
        if next_scores:
            history.append(parents)
        else:
            finish_run()
            route.append(None)
        scores = next_scores
    finish_run()
    return route
