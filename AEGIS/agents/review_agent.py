"""Human-in-the-loop review gate for uncertain or conflicting decisions."""

from core.state import AEGISState, ReviewDecision
from core.config import MIN_CLASSIFICATION_CONFIDENCE
from utils.logger import get_logger

logger = get_logger(__name__)


def review_agent(state: AEGISState) -> AEGISState:
    """Mark a decision as pending without pretending that approval occurred."""
    state["agent_trace"].append("review_agent")
    classification = state.get("classification")
    fusion = state.get("fusion")
    required_by: list[str] = []

    if classification and classification.confidence < MIN_CLASSIFICATION_CONFIDENCE:
        required_by.append("low_classification_confidence")
    if fusion and fusion.human_review_required:
        required_by.append("fusion_consistency_check")
    if state.get("errors"):
        required_by.append("degraded_pipeline")

    reason = (
        "Automated escalation is advisory until a human operator reviews the "
        "uncertainty, conflicts, and cited evidence."
    )
    state["review"] = ReviewDecision(
        status="PENDING",
        reason=reason,
        required_by=required_by or ["policy"],
    )
    logger.info("[Review] Human decision pending: %s", ", ".join(required_by))
    return state
