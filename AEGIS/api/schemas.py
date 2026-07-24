"""
api/schemas.py — Pydantic models for request/response validation
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any


class TelemetryInput(BaseModel):
    scenario_id: str
    timestamp: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude_m: float = Field(..., ge=0, le=10000)
    speed_kmh: float = Field(..., ge=0)
    heading_deg: float = Field(..., ge=0, le=360)
    flight_pattern_entropy: float = Field(..., ge=0.0, le=1.0)
    proximity_to_restricted_km: float = Field(..., ge=0.0)
    iff_signal: bool
    estimated_wingspan_m: float = Field(..., gt=0)
    loiter_detected: bool
    rapid_altitude_change: bool
    mission_narrative: str = Field(..., min_length=5)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scenario_id": "SC-2024-0042",
                "timestamp": "2024-11-14T14:32:11Z",
                "latitude": 33.6844,
                "longitude": 73.0479,
                "altitude_m": 45.0,
                "speed_kmh": 85.0,
                "heading_deg": 270.0,
                "flight_pattern_entropy": 0.78,
                "proximity_to_restricted_km": 1.7,
                "iff_signal": False,
                "estimated_wingspan_m": 1.2,
                "loiter_detected": True,
                "rapid_altitude_change": True,
                "mission_narrative": "Unknown drone penetrating restricted airspace with no IFF signal."
            }
        }
    )

class SALUTEReportSchema(BaseModel):
    size: str
    activity: str
    location: str
    unit: str
    time: str
    equipment: str


class PipelineResponse(BaseModel):
    scenario_id: str
    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    threat_level: str
    confidence: float
    class_probabilities: Dict[str, float]
    xai_summary: str
    top_xai_factors: List[str]
    attribution_method: Optional[str] = None
    attribution_values: Dict[str, float] = Field(default_factory=dict)
    shap_plot_path: Optional[str]
    doctrine_reference: str
    salute_report: Dict[str, str]
    report_text: str
    fused_risk_score: float
    conflict_flags: List[str]
    human_review_required: bool
    review_status: str = "NOT_REQUIRED"
    review_reason: str = ""
    review_required_by: List[str] = Field(default_factory=list)
    escalation_level: int
    escalation_level_name: str
    escalation_reason: str
    alert_dispatched: bool
    geo_risk_score: float
    in_restricted_zone: bool
    zone_name: Optional[str]
    mission_phase: str
    agent_trace: List[str]
    errors: List[str]
    processing_latency_ms: Optional[float]
    node_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pipeline_version: str = "unknown"


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str] = Field(default_factory=dict)
