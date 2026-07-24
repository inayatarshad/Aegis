"""
core/state.py — Shared pipeline state for AEGIS LangGraph StateGraph
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(str, Enum):
    BENIGN = "BENIGN"
    SUSPICIOUS = "SUSPICIOUS"
    HOSTILE = "HOSTILE"
    UNKNOWN = "UNKNOWN"


class EscalationLevel(int, Enum):
    NONE = 0
    MONITOR = 1
    REVIEW = 2
    ALERT = 3
    CRITICAL = 4


@dataclass
class DroneTelemetry:
    """Parsed drone telemetry packet."""
    scenario_id: str
    timestamp: str
    latitude: float
    longitude: float
    altitude_m: float
    speed_kmh: float
    heading_deg: float
    flight_pattern_entropy: float   # Shannon entropy of heading changes
    proximity_to_restricted_km: float
    iff_signal: bool                # Identification Friend or Foe
    estimated_wingspan_m: float
    loiter_detected: bool
    rapid_altitude_change: bool
    mission_narrative: str          # Operator-provided text context


@dataclass
class ClassificationResult:
    threat_level: ThreatLevel
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str


@dataclass
class XAIResult:
    attribution_values: Dict[str, float]  # feature -> change in predicted probability
    top_factors: List[str]              # human-readable top 3 factors
    explanation_text: str               # NL explanation
    attribution_method: str
    target_class: str
    plot_path: Optional[str] = None

    @property
    def shap_values(self) -> Dict[str, float]:
        """Backward-compatible alias for older API consumers."""
        return self.attribution_values


@dataclass
class RetrievalResult:
    query_used: str
    retrieved_docs: List[Dict[str, str]]   # [{text, source, score}]
    doctrine_reference: str
    retrieval_confidence: float
    requeried: bool = False             # True if self-correction triggered


@dataclass
class ContextResult:
    in_restricted_zone: bool
    zone_name: Optional[str]
    historical_incidents_nearby: int
    threat_corridor: bool
    geo_risk_score: float               # 0.0 - 1.0
    mission_phase: str


@dataclass
class SALUTEReport:
    size: str
    activity: str
    location: str
    unit: str
    time: str
    equipment: str


@dataclass
class FusionResult:
    consistency_check_passed: bool
    conflict_flags: List[str]           # any agent disagreements
    human_review_required: bool
    fused_risk_score: float             # 0.0 - 1.0


@dataclass
class ReviewDecision:
    status: str                         # NOT_REQUIRED / PENDING / APPROVED / REJECTED
    reason: str
    required_by: List[str]


class AEGISState(TypedDict):
    """Full pipeline state — mutated by each agent in the graph."""

    # Input
    raw_telemetry: Optional[Dict[str, Any]]

    # Agent outputs
    telemetry: Optional[DroneTelemetry]
    classification: Optional[ClassificationResult]
    xai: Optional[XAIResult]
    retrieval: Optional[RetrievalResult]
    context: Optional[ContextResult]
    fusion: Optional[FusionResult]
    review: Optional[ReviewDecision]
    salute_report: Optional[SALUTEReport]
    report_text: Optional[str]

    # Escalation
    escalation_level: EscalationLevel
    escalation_reason: Optional[str]
    alert_dispatched: bool

    # Pipeline metadata
    agent_trace: List[str]              # ordered list of agents that ran
    errors: List[str]                   # any non-fatal errors logged
    processing_start_ms: Optional[float]
    processing_end_ms: Optional[float]
    node_metrics: Dict[str, Dict[str, Any]]
    pipeline_version: str

    # Final output
    final_response: Optional[Dict[str, Any]]


def initial_state(raw_telemetry: Dict[str, Any]) -> AEGISState:
    """Factory: create a fresh state for a new scenario."""
    return AEGISState(
        raw_telemetry=raw_telemetry,
        telemetry=None,
        classification=None,
        xai=None,
        retrieval=None,
        context=None,
        fusion=None,
        review=None,
        salute_report=None,
        report_text=None,
        escalation_level=EscalationLevel.NONE,
        escalation_reason=None,
        alert_dispatched=False,
        agent_trace=[],
        errors=[],
        processing_start_ms=None,
        processing_end_ms=None,
        node_metrics={},
        pipeline_version="aegis-2.0.0",
        final_response=None,
    )
