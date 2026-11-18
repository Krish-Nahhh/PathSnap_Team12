"""FastAPI backend service for PathSnap map-matching and route reconstruction.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pathsnap.inference import PathSnapPredictor, predict_trip as predict_trip_fn
from src.pathsnap.label_alignment import align_trip_labels, load_trip_data

# Environment and dataset directory paths
BASE_DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", r"D:\\ML_Hackathon\\Student folder"))
STUDENT_TEST_DIR = BASE_DATA_DIR / "student_test"
TRAIN_DIR = BASE_DATA_DIR / "train"

# Tokens for protected endpoints – must be set in the environment.
EVALUATOR_TOKEN = os.getenv("EVALUATOR_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if EVALUATOR_TOKEN is None or ADMIN_TOKEN is None:
    raise RuntimeError("EVALUATOR_TOKEN and ADMIN_TOKEN environment variables must be set before starting the server.")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed map-matching and route reconstruction backend service.",
    version="1.0.0",
)

# Enable CORS for Laptop 2 / frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor = PathSnapPredictor()

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "PathSnap API"
    model_loaded: bool

class TripInfo(BaseModel):
    trip_id: str
    observation_count: int
    has_reference_route: bool

class PredictionResponse(BaseModel):
    trip_id: str
    observation_count: int
    predicted_route: List[Optional[int]]
    prediction_method: str = Field(..., description="Either 'greedy' or 'viterbi'.")

    candidate_miss_count: int
    anomalous_observations_count: int
    processing_time_ms: float
    message: str

def find_trip_dir(trip_id: str) -> Path:
    """Locate trip folder in student_test first, then train as fallback.
    Strict validation of trip_id to prevent directory traversal.
    """
    sanitized_id = "".join(c for c in trip_id if c.isalnum() or c in ("-", "_"))
    if not sanitized_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trip_id format.",
        )
    candidates = [
        STUDENT_TEST_DIR / sanitized_id,
        TRAIN_DIR / sanitized_id,
        BASE_DATA_DIR / sanitized_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Trip '{trip_id}' not found on server.",
    )

@app.get("/", tags=["General"]) 
def root():
    return {
        "service": "PathSnap API",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "trips": "/trips",
    }

@app.get("/health", response_model=HealthResponse, tags=["General"]) 
@app.get("/status", response_model=HealthResponse, tags=["General"]) 
def health_check():
    return HealthResponse(
        status="ok",
        service="PathSnap API",
        model_loaded=(predictor.model is not None),
    )

@app.get("/trips", response_model=List[TripInfo], tags=["Evaluation"]) 
def list_evaluation_trips(split: str = Query("student_test", description="Dataset split to list")):
    """List available evaluation trips with observation counts and route availability."""
    target_dir = STUDENT_TEST_DIR if split == "student_test" else TRAIN_DIR
    if not target_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset directory not found: {target_dir}",
        )
    trip_list: List[TripInfo] = []
    for trip_folder in sorted(target_dir.iterdir()):
        if trip_folder.is_dir():
            t_id = trip_folder.name
            track_p = trip_folder / f"{t_id}.track"
            route_p = trip_folder / f"{t_id}.route"
            obs_count = 0
            if track_p.is_file():
                with track_p.open(encoding="utf-8") as f:
                    obs_count = sum(1 for _ in f)
            trip_list.append(
                TripInfo(
                    trip_id=t_id,
                    observation_count=obs_count,
                    has_reference_route=route_p.is_file(),
                )
            )
    return trip_list

@app.post("/predict/{trip_id}", response_model=PredictionResponse, tags=["Inference"]) 
def predict(
    trip_id: str,
    radius_m: float = Query(75.0, ge=10.0, le=500.0, description="Candidate search radius in meters"),
    top_k: int = Query(10, ge=1, le=50, description="Max candidates to consider per observation"),
    use_viterbi: bool = Query(False, description="Run Viterbi decoding instead of greedy."),
):
    """Process selected trip through the map‑matching pipeline and return predictions.
    Ground‑truth route labels remain strictly server‑side and are never exposed to the client.
    Accuracy is calculated server‑side if ground‑truth route is present.
    """
    trip_dir = find_trip_dir(trip_id)
    t0 = time.time()
    try:
        result = predictor.predict_trip(
            trip_dir, radius_m=radius_m, top_k=top_k, use_viterbi=use_viterbi
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline failed: {str(e)}",
        )
    elapsed_ms = (time.time() - t0) * 1000.0
    predicted = result["predicted_arcs"]
    n_obs = result["observation_count"]


    return PredictionResponse(
        trip_id=trip_id,
        observation_count=n_obs,
        predicted_route=predicted,
        prediction_method="viterbi" if use_viterbi else "greedy",
        candidate_miss_count=result["empty_candidate_count"],
        anomalous_observations_count=result["quality_report"].anomalous_observations,
        processing_time_ms=round(elapsed_ms, 2),
        message="Route reconstruction successful.",
    )

# --- Evaluator endpoint (protected) -------------------------------------------------
@app.get("/evaluation/{trip_id}", tags=["Evaluator"], summary="Return full evaluation data for UI (protected).")
def evaluation(
    trip_id: str,
    token: str = Query(..., description="Evaluator access token"),
):
    if token != EVALUATOR_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid evaluator token")
    trip_dir = find_trip_dir(trip_id)
    track, nodes, arcs, route = load_trip_data(trip_dir)
    # Candidate generation & scoring (same as predictor but without Viterbi)
    candidates_by_obs = list(
        generate_candidates(track, nodes, arcs, radius_m=75.0, top_k=15)
    )
    # Greedy predictions
    greedy_data = predictor.predict_trip(trip_dir, use_viterbi=False)
    greedy_preds = greedy_data["greedy_arcs"]
    # Viterbi predictions
    viterbi_data = predictor.predict_trip(trip_dir, use_viterbi=True)
    viterbi_preds = viterbi_data["predicted_arcs"]

    # Geometry for predicted arcs (simple start/end coordinates)
    geometry: Dict[int, Dict[str, Dict[str, float]]] = {}
    for arc_id in set(filter(None, greedy_preds + viterbi_preds)):
        src_node, dst_node = arcs[arc_id]
        geometry[arc_id] = {
            "start": {"lon": nodes[src_node][0], "lat": nodes[src_node][1]},
            "end": {"lon": nodes[dst_node][0], "lat": nodes[dst_node][1]},
        }

    # Build candidate score list per observation for UI
    candidate_scores = []
    for obs_idx, (idx, cand_list) in enumerate(candidates_by_obs):
        scores = []
        if cand_list:
            probs = score_candidates(predictor.model, cand_list)
            for c, p in zip(cand_list, probs):
                scores.append({"arc_id": c.arc_id, "distance_m": c.distance_m, "score": p})
        candidate_scores.append(scores)

    return {
        "track": [
            {"index": i, "lon": lon, "lat": lat, "timestamp": ts}
            for i, (lon, lat, ts) in enumerate(track)
        ],
        "candidate_scores": candidate_scores,
        "greedy_predictions": greedy_preds,
        "viterbi_predictions": viterbi_preds,
        "geometry": geometry,
        "ground_truth": route if route is not None else None,
    }

# --- Admin endpoint (protected) ----------------------------------------------------
@app.post("/admin/reload-model", tags=["Admin"], summary="Reload the GNB model (protected).")
def reload_model(token: str = Query(..., description="Admin token")):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    try:
        predictor.model = load_model(predictor.model_path)
        return {"status": "model reloaded"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}",
        )
