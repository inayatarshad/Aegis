"""
agents/escalation_agent.py — Threat routing and alert dispatching.
"""

from core.state import AEGISState, EscalationLevel
from utils.logger import get_logger

logger = get_logger(__name__)


def _determine_escalation(clf, fusion, ctx) -> tuple[EscalationLevel, str]:
    """Determine escalation level and reason from multi-agent outputs."""

    if clf is None:
        return EscalationLevel.NONE, "No classification available"

    threat = clf.threat_level.value
    conf = clf.confidence
    fused = fusion.fused_risk_score if fusion else conf

    # Rule-based escalation
    if threat == "HOSTILE" and conf >= 0.75:
        level = EscalationLevel.ALERT
        reason = f"HOSTILE classification (conf={conf:.2f}) — immediate alert"
    elif threat == "HOSTILE" and conf >= 0.50:
        level = EscalationLevel.REVIEW
        reason = f"HOSTILE classification with moderate confidence ({conf:.2f})"
    elif threat == "SUSPICIOUS" and conf >= 0.70:
        level = EscalationLevel.REVIEW
        reason = f"High-confidence SUSPICIOUS ({conf:.2f})"
    elif threat == "SUSPICIOUS":
        level = EscalationLevel.MONITOR
        reason = f"SUSPICIOUS classification ({conf:.2f}) — monitoring"
    elif threat == "BENIGN" and fused > 0.55:
        # Fusion override: geo/context factors warrant monitoring despite benign class
        level = EscalationLevel.MONITOR
        reason = f"BENIGN class but elevated fused risk ({fused:.2f}) — monitoring"
    else:
        level = EscalationLevel.NONE
        reason = "BENIGN classification — no action required"

    # Boost if in restricted zone
    if ctx and ctx.in_restricted_zone and level.value < EscalationLevel.REVIEW.value:
        level = EscalationLevel.REVIEW
        reason += f" | Elevated: in restricted zone ({ctx.zone_name})"

    # Human review flag
    if fusion and fusion.human_review_required and level.value < EscalationLevel.REVIEW.value:
        level = EscalationLevel.REVIEW
        reason += " | Human review flagged by fusion agent"

    return level, reason


def escalation_agent(state: AEGISState) -> AEGISState:
    """
    Escalation Agent:
    - Determines escalation level from all prior outputs
    - Logs alert
    - Assembles final_response dict
    """
    state["agent_trace"].append("escalation_agent")

    clf = state.get("classification")
    fusion = state.get("fusion")
    ctx = state.get("context")
    tel = state.get("telemetry")
    xai = state.get("xai")
    ret = state.get("retrieval")
    salute = state.get("salute_report")
    review = state.get("review")

    level, reason = _determine_escalation(clf, fusion, ctx)
    state["escalation_level"] = level
    state["escalation_reason"] = reason

    if review and review.status == "PENDING":
        logger.warning("[Escalation] Recommendation held for human review — %s", reason)
        state["alert_dispatched"] = False
    elif level >= EscalationLevel.ALERT:
        logger.warning(f"[Escalation] ⚠️  ALERT — {reason}")
        state["alert_dispatched"] = True
    elif level >= EscalationLevel.REVIEW:
        logger.info(f"[Escalation] 🔶 REVIEW — {reason}")
    else:
        logger.info(f"[Escalation] ✅ {level.name} — {reason}")

    # Build final response payload
    state["final_response"] = {
        "scenario_id": tel.scenario_id if tel else "unknown",
        "timestamp": tel.timestamp if tel else "",
        "latitude": tel.latitude if tel else None,
        "longitude": tel.longitude if tel else None,
        "threat_level": clf.threat_level.value if clf else "UNKNOWN",
        "confidence": clf.confidence if clf else 0.0,
        "class_probabilities": clf.class_probabilities if clf else {},
        "xai_summary": xai.explanation_text if xai else "",
        "top_xai_factors": xai.top_factors if xai else [],
        "attribution_method": xai.attribution_method if xai else None,
        "attribution_values": xai.attribution_values if xai else {},
        "shap_plot_path": xai.plot_path if xai else None,
        "doctrine_reference": ret.doctrine_reference if ret else "",
        "salute_report": {
            "size": salute.size,
            "activity": salute.activity,
            "location": salute.location,
            "unit": salute.unit,
            "time": salute.time,
            "equipment": salute.equipment,
        } if salute else {},
        "report_text": state.get("report_text", ""),
        "fused_risk_score": fusion.fused_risk_score if fusion else 0.0,
        "conflict_flags": fusion.conflict_flags if fusion else [],
        "human_review_required": fusion.human_review_required if fusion else False,
        "review_status": review.status if review else "NOT_REQUIRED",
        "review_reason": review.reason if review else "",
        "review_required_by": review.required_by if review else [],
        "escalation_level": level.value,
        "escalation_level_name": level.name,
        "escalation_reason": reason,
        "alert_dispatched": state.get("alert_dispatched", False),
        "geo_risk_score": ctx.geo_risk_score if ctx else 0.0,
        "in_restricted_zone": ctx.in_restricted_zone if ctx else False,
        "zone_name": ctx.zone_name if ctx else None,
        "mission_phase": ctx.mission_phase if ctx else "UNKNOWN",
        "agent_trace": state.get("agent_trace", []),
        "errors": state.get("errors", []),
        "processing_latency_ms": (
            state.get("processing_end_ms", 0) - state.get("processing_start_ms", 0)
            if state.get("processing_end_ms") else None
        ),
    }

    return state
