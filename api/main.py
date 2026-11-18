"""FastAPI backend service for PathSnap map-matching and route reconstruction."""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pathsnap.inference import PathSnapPredictor
from src.pathsnap.label_alignment import align_trip_labels, load_trip_data


# Environment and dataset directory paths
BASE_DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", r"D:\ML_Hackathon\Student folder"))
STUDENT_TEST_DIR = BASE_DATA_DIR / "student_test"
TRAIN_DIR = BASE_DATA_DIR / "train"

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
    accuracy: Optional[float] = Field(
        None, description="Server-side evaluated Top-1 accuracy (if reference route exists)"
    )
    candidate_miss_count: int
    anomalous_observations_count: int
    processing_time_ms: float
    message: str


def find_trip_dir(trip_id: str) -> Path:
    """Locate trip folder in student_test first, then train as fallback."""
    # Strict validation of trip_id to prevent directory traversal
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
        if candidate.exists() and candidate.is_dir():
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
    if not target_dir.exists():
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
            if track_p.exists():
                with track_p.open(encoding="utf-8") as f:
                    obs_count = sum(1 for _ in f)

            trip_list.append(
                TripInfo(
                    trip_id=t_id,
                    observation_count=obs_count,
                    has_reference_route=route_p.exists(),
                )
            )

    return trip_list


@app.post("/predict/{trip_id}", response_model=PredictionResponse, tags=["Inference"])
def predict(
    trip_id: str,
    radius_m: float = Query(75.0, ge=10.0, le=500.0, description="Candidate search radius in meters"),
    top_k: int = Query(10, ge=1, le=50, description="Max candidates to consider per observation"),
):
    """Process selected trip through the map-matching pipeline and return predictions.

    Ground-truth route labels remain strictly server-side and are never exposed to the client.
    Accuracy is calculated server-side if ground-truth route is present.
    """
    trip_dir = find_trip_dir(trip_id)
    t0 = time.time()

    try:
        result = predictor.predict_trip(trip_dir, radius_m=radius_m, top_k=top_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline failed: {str(e)}",
        )

    elapsed_ms = (time.time() - t0) * 1000.0
    predicted_arcs = result["predicted_arcs"]
    n_obs = result["observation_count"]

    # Server-side accuracy calculation (without exposing ground truth sequence)
    server_accuracy: Optional[float] = None
    track, nodes, arcs, route = load_trip_data(trip_dir)
    if route is not None and len(route) > 0 and n_obs > 0:
        # Align ground truth labels server-side
        ground_truth_alignment = align_trip_labels(track, nodes, arcs, route)
        correct = sum(
            1
            for obs_idx, pred_arc in enumerate(predicted_arcs)
            if pred_arc is not None and pred_arc == ground_truth_alignment.get(obs_idx)
        )
        server_accuracy = round(correct / n_obs, 4)

    return PredictionResponse(
        trip_id=trip_id,
        observation_count=n_obs,
        predicted_route=predicted_arcs,
        accuracy=server_accuracy,
        candidate_miss_count=result["empty_candidate_count"],
        anomalous_observations_count=result["quality_report"].anomalous_observations,
        processing_time_ms=round(elapsed_ms, 2),
        message="Route reconstruction successful.",
    )
