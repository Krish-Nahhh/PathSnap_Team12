"""Inference pipeline integrating preprocessing, candidate generation, GNB scoring, and decoding."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .candidates import Candidate, generate_candidates
from .gaussian_naive_bayes import load_model, score_candidates
from .label_alignment import load_trip_data
from .preprocessing import (
    TripQualityReport,
    sanitize_track_for_candidate_generation,
    validate_and_audit_track,
)


class PathSnapPredictor:
    """End-to-end inference engine for PathSnap trips."""

    def __init__(self, model: Any = None, model_path: Optional[Path | str] = None):
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

    def predict_trip(
        self,
        trip_dir: Path | str,
        *,
        radius_m: float = 75.0,
        top_k: int = 10,
        decoder_fn: Any = None,
    ) -> Dict[str, Any]:
        """Process a single trip through the complete prediction pipeline.

        Returns a dictionary containing:
            trip_id: str
            observation_count: int
            predicted_arcs: List[int | None] (strictly aligned to observation indices)
            candidate_scores: List[List[Tuple[int, float]]] ((arc_id, probability) per fix)
            quality_report: TripQualityReport
            empty_candidate_count: int
        """
        target_dir = Path(trip_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Trip directory not found: {target_dir}")

        track, nodes, arcs, _ = load_trip_data(target_dir)
        trip_id = target_dir.name
        n_obs = len(track)

        if n_obs == 0:
            return {
                "trip_id": trip_id,
                "observation_count": 0,
                "predicted_arcs": [],
                "candidate_scores": [],
                "quality_report": TripQualityReport(trip_id, 0, 0, 0),
                "empty_candidate_count": 0,
            }

        # Step 1: Preprocessing & Anomaly Auditing
        quality_report = validate_and_audit_track(track, trip_id=trip_id)
        clean_track = sanitize_track_for_candidate_generation(track, quality_report)

        # Step 2: Spatial Candidate Generation
        candidates_by_obs = list(
            generate_candidates(clean_track, nodes, arcs, radius_m=radius_m, top_k=top_k)
        )

        # Step 3: Probabilistic Candidate Scoring
        fixes_with_scores: List[List[Tuple[int, float]]] = []
        top1_predictions: List[Optional[int]] = []
        empty_count = 0

        for obs_idx in range(n_obs):
            # Candidate generator yields (obs_idx, candidate_list)
            cands = candidates_by_obs[obs_idx][1] if obs_idx < len(candidates_by_obs) else []
            if not cands:
                empty_count += 1
                fixes_with_scores.append([])
                top1_predictions.append(None)
                continue

            if self.model is not None:
                probs = score_candidates(self.model, cands)
            else:
                # Fallback to inverse distance if model is not yet loaded
                probs = [1.0 / (c.distance_m + 1e-3) for c in cands]

            pairs = [(c.arc_id, float(p)) for c, p in zip(cands, probs)]
            fixes_with_scores.append(pairs)

            # Top-1 candidate selection
            best_arc = max(pairs, key=lambda item: item[1])[0]
            top1_predictions.append(best_arc)

        # Step 4: Decoding
        if decoder_fn is not None and callable(decoder_fn):
            try:
                predicted_arcs = decoder_fn(fixes_with_scores, arcs, nodes)
            except Exception:
                predicted_arcs = top1_predictions
        else:
            predicted_arcs = top1_predictions

        return {
            "trip_id": trip_id,
            "observation_count": n_obs,
            "predicted_arcs": predicted_arcs,
            "candidate_scores": fixes_with_scores,
            "quality_report": quality_report,
            "empty_candidate_count": empty_count,
        }


def predict_trip(
    trip_dir: Path | str,
    model: Any = None,
    *,
    radius_m: float = 75.0,
    top_k: int = 10,
    decoder_fn: Any = None,
) -> Dict[str, Any]:
    """Functional convenience wrapper for single-trip prediction."""
    predictor = PathSnapPredictor(model=model)
    return predictor.predict_trip(
        trip_dir,
        radius_m=radius_m,
        top_k=top_k,
        decoder_fn=decoder_fn,
    )
