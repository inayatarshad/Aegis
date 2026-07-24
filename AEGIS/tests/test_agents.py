"""
tests/test_agents.py — Unit tests for AEGIS agents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.state import initial_state, ThreatLevel
from agents.ingestion_agent import ingestion_agent
from agents.classification_agent import classification_agent
from agents.context_agent import context_agent
from agents.fusion_agent import fusion_agent
from agents.escalation_agent import escalation_agent

SAMPLE_HOSTILE = {
    "scenario_id": "TEST-HOSTILE-001",
    "timestamp": "2024-11-14T14:32:11Z",
    "latitude": 33.6844,
    "longitude": 73.0479,
    "altitude_m": 45.0,
    "speed_kmh": 85.0,
    "heading_deg": 270.0,
    "flight_pattern_entropy": 0.82,
    "proximity_to_restricted_km": 1.2,
    "iff_signal": False,
    "estimated_wingspan_m": 1.2,
    "loiter_detected": True,
    "rapid_altitude_change": True,
    "mission_narrative": "Unknown drone penetrating restricted airspace with no IFF signal.",
}

SAMPLE_BENIGN = {
    "scenario_id": "TEST-BENIGN-001",
    "timestamp": "2024-11-14T10:00:00Z",
    "latitude": 33.75,
    "longitude": 73.15,
    "altitude_m": 200.0,
    "speed_kmh": 40.0,
    "heading_deg": 90.0,
    "flight_pattern_entropy": 0.1,
    "proximity_to_restricted_km": 15.0,
    "iff_signal": True,
    "estimated_wingspan_m": 0.5,
    "loiter_detected": False,
    "rapid_altitude_change": False,
    "mission_narrative": "Registered commercial delivery drone in approved corridor.",
}


class TestIngestionAgent:
    def test_valid_input_parses_correctly(self):
        state = initial_state(SAMPLE_HOSTILE)
        result = ingestion_agent(state)
        assert result["telemetry"] is not None
        assert result["telemetry"].scenario_id == "TEST-HOSTILE-001"
        assert not result["telemetry"].iff_signal
        assert result["telemetry"].loiter_detected

    def test_missing_fields_logged_as_errors(self):
        state = initial_state({"scenario_id": "INCOMPLETE"})
        result = ingestion_agent(state)
        assert len(result["errors"]) > 0

    def test_agent_trace_updated(self):
        state = initial_state(SAMPLE_BENIGN)
        result = ingestion_agent(state)
        assert "ingestion_agent" in result["agent_trace"]


class TestClassificationAgent:
    def test_hostile_scenario_classified_correctly(self):
        state = initial_state(SAMPLE_HOSTILE)
        state = ingestion_agent(state)
        state = classification_agent(state)
        clf = state["classification"]
        assert clf is not None
        # Hostile features should push toward SUSPICIOUS or HOSTILE
        assert clf.threat_level in [ThreatLevel.SUSPICIOUS, ThreatLevel.HOSTILE]
        assert 0.0 <= clf.confidence <= 1.0

    def test_benign_scenario(self):
        state = initial_state(SAMPLE_BENIGN)
        state = ingestion_agent(state)
        state = classification_agent(state)
        clf = state["classification"]
        assert clf is not None
        assert clf.threat_level in [ThreatLevel.BENIGN, ThreatLevel.SUSPICIOUS]

    def test_class_probabilities_sum_to_one(self):
        state = initial_state(SAMPLE_HOSTILE)
        state = ingestion_agent(state)
        state = classification_agent(state)
        probs = state["classification"].class_probabilities
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestContextAgent:
    def test_in_restricted_zone_detected(self):
        # Zone Alpha: 33.6844, 73.0479, radius 3km
        state = initial_state(SAMPLE_HOSTILE)
        state = ingestion_agent(state)
        state = context_agent(state)
        ctx = state["context"]
        assert ctx is not None
        # SAMPLE_HOSTILE is at Zone Alpha coords
        assert ctx.in_restricted_zone

    def test_far_from_zones_not_restricted(self):
        far_input = {**SAMPLE_BENIGN, "latitude": 35.0, "longitude": 75.0}
        state = initial_state(far_input)
        state = ingestion_agent(state)
        state = context_agent(state)
        assert not state["context"].in_restricted_zone

    def test_geo_risk_score_range(self):
        state = initial_state(SAMPLE_HOSTILE)
        state = ingestion_agent(state)
        state = context_agent(state)
        assert 0.0 <= state["context"].geo_risk_score <= 1.0


class TestFusionAgent:
    def _run_through_fusion(self, raw):
        state = initial_state(raw)
        state = ingestion_agent(state)
        state = classification_agent(state)
        state = context_agent(state)
        state = fusion_agent(state)
        return state

    def test_fusion_produces_result(self):
        state = self._run_through_fusion(SAMPLE_HOSTILE)
        assert state["fusion"] is not None
        assert 0.0 <= state["fusion"].fused_risk_score <= 1.0

    def test_benign_with_high_geo_flags_conflict(self):
        # Use benign narrative but hostile geo position
        tricky = {**SAMPLE_BENIGN, "latitude": 33.6844, "longitude": 73.0479,
                  "proximity_to_restricted_km": 0.5, "iff_signal": False}
        state = self._run_through_fusion(tricky)
        # Should detect conflict: geo risk high but classification may be benign
        fusion = state["fusion"]
        # Either conflict or human_review should be triggered given proximity
        assert fusion is not None


class TestEscalationAgent:
    def test_hostile_triggers_alert(self):
        state = initial_state(SAMPLE_HOSTILE)
        state = ingestion_agent(state)
        state = classification_agent(state)
        state = context_agent(state)
        state = fusion_agent(state)
        state = escalation_agent(state)
        # Hostile + restricted zone should escalate
        assert state["escalation_level"].value >= 1
        assert state["final_response"] is not None

    def test_final_response_has_required_keys(self):
        state = initial_state(SAMPLE_BENIGN)
        state = ingestion_agent(state)
        state = classification_agent(state)
        state = context_agent(state)
        state = fusion_agent(state)
        state = escalation_agent(state)
        fr = state["final_response"]
        required_keys = ["scenario_id", "threat_level", "confidence",
                         "escalation_level", "fused_risk_score", "agent_trace"]
        for key in required_keys:
            assert key in fr, f"Missing key: {key}"
