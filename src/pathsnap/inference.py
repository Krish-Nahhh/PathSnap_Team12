'''Inference pipeline integrating preprocessing, candidate generation, GNB scoring, and decoding.'''

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from collections import defaultdict

from .candidates import Candidate, generate_candidates
from .gaussian_naive_bayes import load_model, score_candidates
from .label_alignment import load_trip_data
from .preprocessing import (
    TripQualityReport,
    sanitize_track_for_candidate_generation,
    validate_and_audit_track,
    detect_gps_gaps,
)


def _build_arc_adjacency(arcs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Return a mapping from arc_id to list of successor arc_ids.
    Two arcs are connected if the destination node of the first equals the source node of the second.
    """
    # Build node -> outgoing arcs map first
    outgoing: Dict[int, List[int]] = defaultdict(list)
    for arc_id, (src, dst) in enumerate(arcs):
        outgoing[src].append(arc_id)
    # Now map each arc to successors via its destination node
    adjacency: Dict[int, List[int]] = {}
    for arc_id, (src, dst) in enumerate(arcs):
        adjacency[arc_id] = outgoing.get(dst, [])
    return adjacency


def viterbi_decode(
    fixes_with_scores: List[List[Tuple[int, float]]],
    arcs: List[Tuple[int, int]],
    nodes: List[Tuple[float, float]],
    gaps: Optional[set] = None,
) -> List[Optional[int]]:
    """Perform Viterbi decoding over candidate arcs.

    * fixes_with_scores[t] = list of (arc_id, probability) for observation t.
    * arcs/nodes provide road graph for transition penalties.
    * gaps (optional) is a set of observation indices that start a new segment due to a GPS gap.
    """
    n_obs = len(fixes_with_scores)
    adjacency = _build_arc_adjacency(arcs)
    # Convert probabilities to log‑space for numerical stability
    log_probs = [
        [math.log(max(p, 1e-15)) for _, p in obs]
        for obs in fixes_with_scores
    ]

    # DP tables
    V: List[Dict[int, float]] = []  # best log‑prob for each candidate arc at time t
    back: List[Dict[int, Optional[int]]] = []  # predecessor arc_id

    for t in range(n_obs):
        V.append({})
        back.append({})
        candidates = fixes_with_scores[t]
        if not candidates:
            continue  # keep empty for this time step
        for idx, (arc_id, prob) in enumerate(candidates):
            emit = log_probs[t][idx]
            # Start of sequence or after a gap or previous obs empty
            if t == 0 or (gaps is not None and t in gaps) or not fixes_with_scores[t - 1]:
                V[t][arc_id] = emit
                back[t][arc_id] = None
                continue
            best_score = -float('inf')
            best_prev: Optional[int] = None
            for prev_arc_id, prev_score in V[t - 1].items():
                # Transition penalty: 0 if arcs are connected, else -1 (log‑prob ≈0.367)
                if prev_arc_id in adjacency and arc_id in adjacency[prev_arc_id]:
                    trans = 0.0
                else:
                    trans = -1.0
                score = prev_score + trans + emit
                if score > best_score:
                    best_score = score
                    best_prev = prev_arc_id
            V[t][arc_id] = best_score
            back[t][arc_id] = best_prev

    # Backtrack to retrieve best path
    path: List[Optional[int]] = [None] * n_obs
    # Find last observation that has candidates
    last_t = n_obs - 1
    while last_t >= 0 and not fixes_with_scores[last_t]:
        last_t -= 1
    if last_t < 0:
        return path
    # Choose best final arc
    best_final_arc = max(V[last_t], key=V[last_t].get)
    cur = best_final_arc
    for t in range(last_t, -1, -1):
        path[t] = cur
        cur = back[t].get(cur)
        if cur is None and t > 0:
            # No predecessor – stop early; earlier positions stay None
            break
    return path


class PathSnapPredictor:
    """End‑to‑end inference engine for PathSnap trips."""

    def __init__(self, model: Any = None, model_path: Optional[Path | str] = None, *, scorer=score_candidates, batch_scoring: bool = False):
        self.scorer = scorer
        self.batch_scoring = batch_scoring
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_model(model_path)
        else:
            default_path = Path("artifacts/gnb_model.json")
            if default_path.exists():
                self.model = load_model(default_path)
            else:
                self.model = None
        self.model_path = model_path or Path("artifacts/gnb_model.json")

    def score_candidate_groups(self, groups):
        model = self.model
        if model is None:
            return [[1.0 / (candidate.distance_m + 1e-3) for candidate in group] for group in groups]
        if not self.batch_scoring:
            return [self.scorer(model, group) if group else [] for group in groups]

        scores = iter(self.scorer(model, [candidate for group in groups for candidate in group]))
        return [[next(scores) for _ in group] for group in groups]

    def predict_trip(
        self,
        trip_dir: Path | str,
        *,
        radius_m: float = 75.0,
        top_k: int = 10,
        use_viterbi: bool = False,
    ) -> Dict[str, Any]:
        """Process a single trip and optionally run Viterbi decoding.

        Returns a dict containing greedy and (if requested) Viterbi predictions.
        """
        target_dir = Path(trip_dir)
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Trip directory not found: {target_dir}")

        track, nodes, arcs, _ = load_trip_data(target_dir)
        trip_id = target_dir.name
        n_obs = len(track)
        if n_obs == 0:
            empty_report = TripQualityReport(trip_id, 0, 0, 0)
            return {
                "trip_id": trip_id,
                "observation_count": 0,
                "predicted_arcs": [],
                "greedy_arcs": [],
                "candidate_scores": [],
                "quality_report": empty_report,
                "empty_candidate_count": 0,
            }

        # 1. Preprocess & audit
        quality_report = validate_and_audit_track(track, trip_id=trip_id)
        clean_track = sanitize_track_for_candidate_generation(track, quality_report)

        # 2. Generate candidates per observation
        candidates_by_obs = list(
            generate_candidates(clean_track, nodes, arcs, radius_m=radius_m, top_k=top_k)
        )

        # 3. Score candidates
        fixes_with_scores: List[List[Tuple[int, float]]] = []
        greedy_preds: List[Optional[int]] = []
        empty_cnt = 0
        groups = [
            candidates_by_obs[obs_idx][1] if obs_idx < len(candidates_by_obs) else []
            for obs_idx in range(n_obs)
        ]
        for cand_list, probs in zip(groups, self.score_candidate_groups(groups)):
            if not cand_list:
                empty_cnt += 1
                fixes_with_scores.append([])
                greedy_preds.append(None)
                continue
            pairs = [(c.arc_id, float(p)) for c, p in zip(cand_list, probs)]
            fixes_with_scores.append(pairs)
            # Greedy top‑1 selection
            best_arc = max(pairs, key=lambda it: it[1])[0]
            greedy_preds.append(best_arc)

        # 4. Optional Viterbi decoding
        if use_viterbi:
            # Identify start indices of gaps longer than the threshold (default 5 s)
            gap_indices = {
                g.start_index + 1
                for g in quality_report.gaps
                if g.gap_seconds > 5.0
            }
            viterbi_preds = viterbi_decode(fixes_with_scores, arcs, nodes, gaps=gap_indices)
        else:
            viterbi_preds = greedy_preds

        return {
            "trip_id": trip_id,
            "observation_count": n_obs,
            "predicted_arcs": viterbi_preds,
            "greedy_arcs": greedy_preds,
            "candidate_scores": fixes_with_scores,
            "quality_report": quality_report,
            "empty_candidate_count": empty_cnt,
        }


def predict_trip(
    trip_dir: Path | str,
    model: Any = None,
    *,
    radius_m: float = 75.0,
    top_k: int = 10,
    use_viterbi: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper for single‑trip prediction."""
    predictor = PathSnapPredictor(model=model)
    return predictor.predict_trip(
        trip_dir,
        radius_m=radius_m,
        top_k=top_k,
        use_viterbi=use_viterbi,
    )
