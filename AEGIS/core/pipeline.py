"""
core/pipeline.py — LangGraph StateGraph orchestration for AEGIS
Defines the full 7-agent pipeline with conditional routing.
"""

import time
from langgraph.graph import StateGraph, END

from core.state import AEGISState, EscalationLevel, ThreatLevel, initial_state
from core.config import HOSTILE_THRESHOLD, SUSPICIOUS_THRESHOLD
from agents.ingestion_agent import ingestion_agent
from agents.classification_agent import classification_agent
from agents.xai_agent import xai_agent
from agents.retrieval_agent import retrieval_agent
from agents.context_agent import context_agent
from agents.fusion_agent import fusion_agent
from agents.report_agent import report_agent
from agents.escalation_agent import escalation_agent
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────

def route_after_classification(state: AEGISState) -> str:
    """
    After classification: always run XAI, retrieval, and context in parallel.
    LangGraph handles fan-out; we always proceed.
    """
    clf = state.get("classification")
    if clf is None:
        logger.warning("Classification result missing — routing to END with error")
        return "end_with_error"
    return "parallel_analysis"


def route_after_fusion(state: AEGISState) -> str:
    """
    After fusion: if human review required and threat is not HOSTILE,
    skip LLM report and go directly to escalation with flag.
    """
    fusion = state.get("fusion")
    if fusion and fusion.human_review_required:
        clf = state.get("classification")
        if clf and clf.threat_level != ThreatLevel.HOSTILE:
            # Flag for review but still generate report
            pass
    return "generate_report"


def route_after_escalation(state: AEGISState) -> str:
    return END


# ── Pipeline builder ──────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct and compile the AEGIS LangGraph pipeline."""

    graph = StateGraph(AEGISState)

    # Register nodes
    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("classification", classification_agent)
    graph.add_node("xai", xai_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("context", context_agent)
    graph.add_node("fusion", fusion_agent)
    graph.add_node("report", report_agent)
    graph.add_node("escalation", escalation_agent)

    # Entry point
    graph.set_entry_point("ingestion")

    # Linear flow: ingestion → classification
    graph.add_edge("ingestion", "classification")

    # Fan-out: classification → [xai, retrieval, context] (parallel)
    graph.add_conditional_edges(
        "classification",
        route_after_classification,
        {
            "parallel_analysis": "xai",
            "end_with_error": END,
        }
    )
    # Also fan out to retrieval and context from classification
    graph.add_edge("classification", "retrieval")
    graph.add_edge("classification", "context")

    # Fan-in: all three → fusion
    graph.add_edge("xai", "fusion")
    graph.add_edge("retrieval", "fusion")
    graph.add_edge("context", "fusion")

    # fusion → report → escalation → END
    graph.add_conditional_edges(
        "fusion",
        route_after_fusion,
        {"generate_report": "report"}
    )
    graph.add_edge("report", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────

def run_pipeline(raw_telemetry: dict) -> dict:
    """
    Run the full AEGIS pipeline on a telemetry input.

    Args:
        raw_telemetry: Dict matching DroneTelemetry fields.

    Returns:
        Final AEGISState as a dict with `final_response` populated.
    """
    pipeline = build_pipeline()
    state = initial_state(raw_telemetry)
    state["processing_start_ms"] = time.time() * 1000

    logger.info(f"[AEGIS] Pipeline started — scenario: {raw_telemetry.get('scenario_id', 'unknown')}")

    try:
        final_state = pipeline.invoke(state)
    except Exception as e:
        logger.error(f"[AEGIS] Pipeline failed: {e}")
        state["errors"].append(str(e))
        return dict(state)

    final_state["processing_end_ms"] = time.time() * 1000
    latency = final_state["processing_end_ms"] - final_state["processing_start_ms"]
    logger.info(f"[AEGIS] Pipeline complete — latency: {latency:.0f}ms — "
                f"threat: {final_state.get('classification', {})}")

    return dict(final_state)
