"""
core/pipeline.py — LangGraph StateGraph orchestration for AEGIS
Defines the full 7-agent pipeline with conditional routing.
"""

import time
from functools import wraps
from langgraph.graph import StateGraph, END

from core.state import AEGISState, initial_state
from agents.ingestion_agent import ingestion_agent
from agents.classification_agent import classification_agent
from agents.xai_agent import xai_agent
from agents.retrieval_agent import retrieval_agent
from agents.context_agent import context_agent
from agents.fusion_agent import fusion_agent
from agents.report_agent import report_agent
from agents.escalation_agent import escalation_agent
from agents.review_agent import review_agent
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────

def route_after_classification(state: AEGISState) -> str:
    """
    After classification: proceed to XAI if classification succeeded.
    """
    clf = state.get("classification")
    if clf is None:
        logger.warning("Classification result missing - routing to END with error")
        return "end_with_error"
    return "run_xai"


def route_after_fusion(state: AEGISState) -> str:
    """Route uncertain, conflicting, or degraded runs through human review."""
    fusion = state.get("fusion")
    classification = state.get("classification")
    if (
        (fusion and fusion.human_review_required)
        or (classification and classification.confidence < 0.60)
        or state.get("errors")
    ):
        return "request_review"
    return "generate_report"


def route_after_escalation(state: AEGISState) -> str:
    return END


def instrument(name, agent):
    """Record node duration and status without changing individual agent APIs."""
    @wraps(agent)
    def wrapped(state: AEGISState) -> AEGISState:
        started = time.perf_counter()
        errors_before = len(state.get("errors", []))
        result = agent(state)
        result.setdefault("node_metrics", {})[name] = {
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": (
                "degraded"
                if len(result.get("errors", [])) > errors_before
                else "ok"
            ),
        }
        return result
    return wrapped


# ── Pipeline builder ──────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct and compile the AEGIS LangGraph pipeline."""

    graph = StateGraph(AEGISState)

    # Register nodes
    graph.add_node("ingestion_step", instrument("ingestion_agent", ingestion_agent))
    graph.add_node("classification_step", instrument("classification_agent", classification_agent))
    graph.add_node("xai_step", instrument("xai_agent", xai_agent))
    graph.add_node("retrieval_step", instrument("retrieval_agent", retrieval_agent))
    graph.add_node("context_step", instrument("context_agent", context_agent))
    graph.add_node("fusion_step", instrument("fusion_agent", fusion_agent))
    graph.add_node("review_step", instrument("review_agent", review_agent))
    graph.add_node("report_step", instrument("report_agent", report_agent))
    graph.add_node("escalation_step", instrument("escalation_agent", escalation_agent))

    # Entry point
    graph.set_entry_point("ingestion_step")

    # Linear flow: ingestion → classification
    graph.add_edge("ingestion_step", "classification_step")

    # Sequential analysis: classification -> xai -> retrieval -> context
    graph.add_conditional_edges(
        "classification_step",
        route_after_classification,
        {
            "run_xai": "xai_step",
            "end_with_error": END,
        }
    )
    graph.add_edge("xai_step", "retrieval_step")
    graph.add_edge("retrieval_step", "context_step")
    graph.add_edge("context_step", "fusion_step")

    # fusion → report → escalation → END
    graph.add_conditional_edges(
        "fusion_step",
        route_after_fusion,
        {
            "generate_report": "report_step",
            "request_review": "review_step",
        }
    )
    graph.add_edge("review_step", "report_step")
    graph.add_edge("report_step", "escalation_step")
    graph.add_edge("escalation_step", END)

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
    if final_state.get("final_response"):
        final_state["final_response"]["processing_latency_ms"] = round(latency, 2)
        final_state["final_response"]["node_metrics"] = final_state.get("node_metrics", {})
        final_state["final_response"]["pipeline_version"] = final_state.get(
            "pipeline_version", "unknown"
        )
    logger.info(f"[AEGIS] Pipeline complete — latency: {latency:.0f}ms — "
                f"threat: {final_state.get('classification', {})}")

    return dict(final_state)
