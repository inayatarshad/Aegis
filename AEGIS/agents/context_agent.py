"""
agents/context_agent.py — Geo reasoning and mission history context.
"""

import math
from core.state import AEGISState, ContextResult
from core.config import RESTRICTED_ZONES
from utils.logger import get_logger

logger = get_logger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two GPS coordinates."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Simulated threat corridors: list of (lat, lon, radius_km, name)
THREAT_CORRIDORS = [
    (33.70, 73.05, 5.0, "Northern Corridor Alpha"),
    (33.66, 72.98, 4.0, "Western Approach Bravo"),
]


def context_agent(state: AEGISState) -> AEGISState:
    """
    Context Agent:
    - Checks GPS coordinates against restricted zones
    - Checks against known threat corridors
    - Computes a geo risk score
    - Infers mission phase from telemetry
    """
    state["agent_trace"].append("context_agent")
    telemetry = state.get("telemetry")

    if telemetry is None:
        state["errors"].append("Context: no telemetry available")
        return state

    lat, lon = telemetry.latitude, telemetry.longitude

    # Check restricted zones
    in_restricted = False
    zone_name = None
    min_zone_dist = float("inf")

    for name, zlat, zlon, radius in RESTRICTED_ZONES:
        dist = haversine_km(lat, lon, zlat, zlon)
        if dist < min_zone_dist:
            min_zone_dist = dist
        if dist <= radius:
            in_restricted = True
            zone_name = name
            break

    # Check threat corridors
    in_corridor = False
    for clat, clon, cradius, cname in THREAT_CORRIDORS:
        if haversine_km(lat, lon, clat, clon) <= cradius:
            in_corridor = True
            break

    # Geo risk score: weighted sum of factors
    proximity_score = max(0.0, 1.0 - (telemetry.proximity_to_restricted_km / 10.0))
    geo_risk = (
        0.40 * float(in_restricted) +
        0.25 * float(in_corridor) +
        0.20 * proximity_score +
        0.10 * float(telemetry.loiter_detected) +
        0.05 * float(not telemetry.iff_signal)
    )
    geo_risk = min(1.0, round(geo_risk, 4))

    # Infer mission phase from altitude + speed
    if telemetry.altitude_m < 50 and telemetry.speed_kmh < 20:
        mission_phase = "LOITER/SURVEILLANCE"
    elif telemetry.altitude_m > 500 and telemetry.speed_kmh > 80:
        mission_phase = "TRANSIT"
    elif telemetry.rapid_altitude_change:
        mission_phase = "EVASIVE_MANEUVER"
    else:
        mission_phase = "APPROACH"

    # Simulated: nearby historical incidents (would be DB lookup in production)
    historical_incidents = 2 if in_corridor else (1 if in_restricted else 0)

    state["context"] = ContextResult(
        in_restricted_zone=in_restricted,
        zone_name=zone_name,
        historical_incidents_nearby=historical_incidents,
        threat_corridor=in_corridor,
        geo_risk_score=geo_risk,
        mission_phase=mission_phase,
    )

    logger.info(
        f"[Context] Geo risk={geo_risk:.2f}, restricted={in_restricted}, "
        f"corridor={in_corridor}, phase={mission_phase}"
    )
    return state
