"""
agents/fusion_agent.py — Aggregates all agent outputs into a unified risk assessment.
Runs consistency checks and flags conflicts between agents.
"""

from core.state import AEGISState, FusionResult, ThreatLevel
from utils.logger import get_logger

logger = get_logger(__name__)

# Weights for fused risk score
WEIGHTS = {
    "classification_confidence": 0.50,
    "geo_risk_score": 0.30,
    "loiter_bonus": 0.10,
    "no_iff_bonus": 0.10,
}


def fusion_agent(state: AEGISState) -> AEGISState:
    """
    Fusion Agent:
    - Combines classification, XAI, retrieval, and context outputs
    - Detects conflicts (e.g., low threat score but high geo risk)
    - Computes a unified fused_risk_score
    - Flags cases requiring human review
    """
    state["agent_trace"].append("fusion_agent")

    clf = state.get("classification")
    ctx = state.get("context")
    ret = state.get("retrieval")
    tel = state.get("telemetry")

    conflict_flags = []
    human_review_required = False

    # ── Consistency checks ────────────────────────────────────────────────
    if clf and ctx:
        # Check 1: Low threat but high geo risk
        if (clf.threat_level == ThreatLevel.BENIGN and
                ctx.geo_risk_score > 0.60):
            conflict_flags.append(
                "BENIGN classification but high geo risk score (>0.60)"
            )
            human_review_required = True

        # Check 2: No IFF but classified as benign
        if tel and not tel.iff_signal and clf.threat_level == ThreatLevel.BENIGN:
            conflict_flags.append(
                "No IFF transponder detected but threat classified as BENIGN"
            )

        # Check 3: In restricted zone but not flagged
        if (ctx.in_restricted_zone and
                clf.threat_level == ThreatLevel.BENIGN and
                clf.confidence < 0.80):
            conflict_flags.append(
                "Object in restricted zone with low classification confidence"
            )
            human_review_required = True

        # Check 4: Loiter + no IFF = suspicious minimum
        if (tel and tel.loiter_detected and not tel.iff_signal and
                clf.threat_level == ThreatLevel.BENIGN):
            conflict_flags.append(
                "Loitering + no IFF detected — minimum SUSPICIOUS expected"
            )

    if ret and ret.retrieval_confidence < 0.35:
        conflict_flags.append(
            "Retrieved doctrine has low semantic relevance; evidence is incomplete"
        )
        human_review_required = True

    # ── Fused risk score ──────────────────────────────────────────────────
    clf_conf = clf.confidence if clf else 0.0
    # For BENIGN, invert confidence (high benign confidence = low risk)
    if clf and clf.threat_level == ThreatLevel.BENIGN:
        clf_risk = 1.0 - clf_conf
    else:
        clf_risk = clf_conf

    geo_risk = ctx.geo_risk_score if ctx else 0.0
    loiter = float(tel.loiter_detected) if tel else 0.0
    no_iff = float(not tel.iff_signal) if tel else 0.0

    fused_score = (
        WEIGHTS["classification_confidence"] * clf_risk +
        WEIGHTS["geo_risk_score"] * geo_risk +
        WEIGHTS["loiter_bonus"] * loiter +
        WEIGHTS["no_iff_bonus"] * no_iff
    )
    fused_score = min(1.0, round(fused_score, 4))

    # Flag for human review if score is in ambiguous zone
    if 0.45 <= fused_score <= 0.60:
        human_review_required = True
        conflict_flags.append(
            f"Ambiguous fused risk score ({fused_score:.2f}) — human review recommended"
        )

    state["fusion"] = FusionResult(
        consistency_check_passed=len(conflict_flags) == 0,
        conflict_flags=conflict_flags,
        human_review_required=human_review_required,
        fused_risk_score=fused_score,
    )

    logger.info(
        f"[Fusion] fused_score={fused_score:.2f}, "
        f"conflicts={len(conflict_flags)}, review={human_review_required}"
    )
    return state
