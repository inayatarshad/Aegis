"""Integration tests for routing, observability, and transparent fallbacks."""

from core.pipeline import run_pipeline
from tests.test_agents import SAMPLE_HOSTILE


def test_pipeline_returns_observability_and_latency():
    result = run_pipeline(SAMPLE_HOSTILE)
    response = result["final_response"]

    assert response["processing_latency_ms"] > 0
    assert response["pipeline_version"] == "aegis-2.0.0"
    assert response["node_metrics"]["classification_agent"]["duration_ms"] >= 0
    assert response["attribution_method"] == (
        "leave-one-feature-out probability attribution"
    )
    assert "mission_narrative" in response["attribution_values"]
    assert response["report_text"].startswith(
        "[DETERMINISTIC EVIDENCE-ONLY REPORT]"
    )


def test_pending_review_never_claims_alert_was_dispatched():
    uncertain = {
        **SAMPLE_HOSTILE,
        "mission_narrative": "Unclear aerial contact requiring verification.",
        "flight_pattern_entropy": 0.50,
        "proximity_to_restricted_km": 4.0,
    }
    response = run_pipeline(uncertain)["final_response"]
    if response["review_status"] == "PENDING":
        assert response["alert_dispatched"] is False
        assert "review_agent" in response["agent_trace"]

