"""
agents/ingestion_agent.py — Parses and validates incoming drone telemetry.
"""

from core.state import AEGISState, DroneTelemetry
from utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = [
    "scenario_id", "timestamp", "latitude", "longitude",
    "altitude_m", "speed_kmh", "heading_deg", "flight_pattern_entropy",
    "proximity_to_restricted_km", "iff_signal", "estimated_wingspan_m",
    "loiter_detected", "rapid_altitude_change", "mission_narrative"
]


def _parse_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field_name} must be a boolean")


def ingestion_agent(state: AEGISState) -> AEGISState:
    """
    Ingestion Agent:
    - Validates schema of raw telemetry packet
    - Normalizes and parses into DroneTelemetry dataclass
    - Flags and logs any missing/malformed fields
    """
    state["agent_trace"].append("ingestion_agent")
    raw = state.get("raw_telemetry", {})

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        msg = f"Ingestion: missing fields {missing}"
        state["errors"].append(msg)
        logger.warning(msg)

    # Parse with fallback defaults for missing fields
    try:
        telemetry = DroneTelemetry(
            scenario_id=str(raw.get("scenario_id", "UNKNOWN")),
            timestamp=str(raw.get("timestamp", "")),
            latitude=float(raw.get("latitude", 0.0)),
            longitude=float(raw.get("longitude", 0.0)),
            altitude_m=float(raw.get("altitude_m", 0.0)),
            speed_kmh=float(raw.get("speed_kmh", 0.0)),
            heading_deg=float(raw.get("heading_deg", 0.0)),
            flight_pattern_entropy=float(raw.get("flight_pattern_entropy", 0.0)),
            proximity_to_restricted_km=float(raw.get("proximity_to_restricted_km", 99.0)),
            iff_signal=_parse_bool(raw.get("iff_signal", False), "iff_signal"),
            estimated_wingspan_m=float(raw.get("estimated_wingspan_m", 1.0)),
            loiter_detected=_parse_bool(
                raw.get("loiter_detected", False), "loiter_detected"
            ),
            rapid_altitude_change=_parse_bool(
                raw.get("rapid_altitude_change", False), "rapid_altitude_change"
            ),
            mission_narrative=str(raw.get("mission_narrative", "")),
        )
        state["telemetry"] = telemetry
        logger.info(f"[Ingestion] Parsed scenario {telemetry.scenario_id}")
    except Exception as e:
        state["errors"].append(f"Ingestion parse error: {e}")
        logger.error(f"[Ingestion] Parse failed: {e}")

    return state
