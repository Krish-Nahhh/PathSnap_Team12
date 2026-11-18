"""Data cleaning, validation, and anomaly detection for GPS trajectories.

Preserves original GPS observation indices and never silently drops rows.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Sequence, Tuple

from .label_alignment import GPSGap, detect_gps_gaps


# Bounds for geographic coordinates
MIN_LAT, MAX_LAT = -90.0, 90.0
MIN_LON, MAX_LON = -180.0, 180.0

# Regional bounding box for Darmstadt, Germany study region (with generous margin)
REGIONAL_MIN_LON, REGIONAL_MAX_LON = 7.0, 10.5
REGIONAL_MIN_LAT, REGIONAL_MAX_LAT = 48.5, 51.5

# Kinematic thresholds
DEFAULT_MAX_SPEED_MPS = 50.0  # ~180 km/h: plausible upper bound for vehicles in dataset
DEFAULT_MAX_JUMP_M = 250.0    # Plausible single-step jump for consecutive 1s fixes
DEFAULT_GAP_THRESHOLD_S = 5.0 # Nominally 1 Hz; >5s indicates significant GPS dropout


@dataclass
class ObservationAnomaly:
    """Record of an anomaly detected at a specific observation index."""
    observation_index: int
    timestamp: float
    longitude: float
    latitude: float
    flags: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TripQualityReport:
    """Detailed quality and integrity report for a trajectory."""
    trip_id: str
    total_observations: int
    valid_observations: int
    anomalous_observations: int
    gaps: List[GPSGap] = field(default_factory=list)
    anomalies: List[ObservationAnomaly] = field(default_factory=list)
    flag_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Return True if no anomalies or gaps were detected."""
        return len(self.anomalies) == 0 and len(self.gaps) == 0


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute physical great-circle distance between two GPS coordinates in meters."""
    radius_earth_m = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_earth_m * c


def validate_coordinates(lon: float, lat: float, check_regional: bool = False) -> Tuple[bool, List[str]]:
    """Validate that longitude and latitude are finite and within physical bounds."""
    flags: List[str] = []
    if not (math.isfinite(lon) and math.isfinite(lat)):
        flags.append("NON_FINITE_COORDINATE")
        return False, flags

    if not (MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT):
        flags.append("OUT_OF_GLOBAL_BOUNDS")

    if check_regional and not (REGIONAL_MIN_LON <= lon <= REGIONAL_MAX_LON and REGIONAL_MIN_LAT <= lat <= REGIONAL_MAX_LAT):
        flags.append("OUT_OF_REGIONAL_BOUNDS")

    return len(flags) == 0, flags


def validate_and_audit_track(
    track: Sequence[Tuple[float, float, float]],
    trip_id: str = "unknown",
    *,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_jump_m: float = DEFAULT_MAX_JUMP_M,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
    check_regional: bool = False,
) -> TripQualityReport:
    """Inspect and audit a GPS trajectory.

    Strictly preserves the original row indices (0 to len(track) - 1).
    Does NOT delete or drop any rows. Returns a comprehensive quality report.

    Track items: (longitude, latitude, timestamp_seconds).
    """
    total = len(track)
    if total == 0:
        return TripQualityReport(
            trip_id=trip_id,
            total_observations=0,
            valid_observations=0,
            anomalous_observations=0,
        )

    gaps = detect_gps_gaps(list(track), threshold_s=gap_threshold_s)
    anomalies: List[ObservationAnomaly] = []
    flag_counts: Dict[str, int] = {}

    def add_anomaly(idx: int, t: float, lon: float, lat: float, flags: List[str], details: Dict[str, Any]):
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
        anomalies.append(ObservationAnomaly(idx, t, lon, lat, flags, details))

    for idx, (lon, lat, timestamp) in enumerate(track):
        obs_flags: List[str] = []
        obs_details: Dict[str, Any] = {}

        # 1. Coordinate check
        is_coord_valid, coord_flags = validate_coordinates(lon, lat, check_regional=check_regional)
        if not is_coord_valid:
            obs_flags.extend(coord_flags)

        # 2. Timestamp & kinematic checks against previous observation
        if idx > 0:
            prev_lon, prev_lat, prev_time = track[idx - 1]
            dt = timestamp - prev_time

            if dt < 0:
                obs_flags.append("BACKWARD_TIMESTAMP")
                obs_details["time_difference_s"] = dt
            elif dt == 0:
                obs_flags.append("DUPLICATE_TIMESTAMP")
                obs_details["time_difference_s"] = dt
            elif dt > gap_threshold_s:
                obs_flags.append("GPS_GAP")
                obs_details["gap_duration_s"] = dt

            # Distance and speed
            if math.isfinite(lon) and math.isfinite(lat) and math.isfinite(prev_lon) and math.isfinite(prev_lat):
                dist_m = haversine_distance_m(prev_lon, prev_lat, lon, lat)
                obs_details["step_distance_m"] = dist_m

                if dt > 0:
                    speed = dist_m / dt
                    obs_details["speed_mps"] = speed
                    obs_details["speed_kmh"] = speed * 3.6
                    if speed > max_speed_mps:
                        obs_flags.append("EXCESSIVE_SPEED")
                elif dt == 0 and dist_m > 0:
                    obs_flags.append("INSTANTANEOUS_DISPLACEMENT")

                # Single-step physical jump check (only for standard 1s fixes)
                if dt <= 2.0 and dist_m > max_jump_m:
                    obs_flags.append("IMPLAUSIBLE_JUMP")

        if obs_flags:
            add_anomaly(idx, timestamp, lon, lat, obs_flags, obs_details)

    anomalous_indices = {a.observation_index for a in anomalies}
    valid_count = total - len(anomalous_indices)

    return TripQualityReport(
        trip_id=trip_id,
        total_observations=total,
        valid_observations=valid_count,
        anomalous_observations=len(anomalous_indices),
        gaps=gaps,
        anomalies=anomalies,
        flag_counts=flag_counts,
    )


def sanitize_track_for_candidate_generation(
    track: Sequence[Tuple[float, float, float]],
    report: TripQualityReport,
) -> List[Tuple[float, float, float]]:
    """Produce a sanitized track for spatial candidate generation while preserving exact indices.

    If an observation has non-finite or completely invalid coordinates, it is linearly
    interpolated from surrounding valid observations to prevent projection errors,
    guaranteeing that the returned list has identical length and indexing (0 to N-1).
    """
    cleaned = list(track)
    total = len(cleaned)
    if total == 0:
        return cleaned

    invalid_indices = {
        a.observation_index
        for a in report.anomalies
        if "NON_FINITE_COORDINATE" in a.flags or "OUT_OF_GLOBAL_BOUNDS" in a.flags
    }

    if not invalid_indices:
        return cleaned

    for idx in invalid_indices:
        # Find nearest preceding and succeeding valid fixes
        prev_idx = idx - 1
        while prev_idx >= 0 and prev_idx in invalid_indices:
            prev_idx -= 1
        next_idx = idx + 1
        while next_idx < total and next_idx in invalid_indices:
            next_idx += 1

        t_curr = cleaned[idx][2]
        if prev_idx >= 0 and next_idx < total:
            # Linear interpolation in time
            t0 = cleaned[prev_idx][2]
            t1 = cleaned[next_idx][2]
            alpha = (t_curr - t0) / (t1 - t0) if t1 > t0 else 0.5
            interp_lon = cleaned[prev_idx][0] + alpha * (cleaned[next_idx][0] - cleaned[prev_idx][0])
            interp_lat = cleaned[prev_idx][1] + alpha * (cleaned[next_idx][1] - cleaned[prev_idx][1])
            cleaned[idx] = (interp_lon, interp_lat, t_curr)
        elif prev_idx >= 0:
            cleaned[idx] = (cleaned[prev_idx][0], cleaned[prev_idx][1], t_curr)
        elif next_idx < total:
            cleaned[idx] = (cleaned[next_idx][0], cleaned[next_idx][1], t_curr)

    return cleaned
