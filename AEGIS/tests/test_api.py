"""API contract tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_reports_component_readiness():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"operational", "degraded"}
    assert set(payload["components"]) == {"classifier", "vectorstore", "llm"}


def test_invalid_coordinates_are_rejected():
    response = client.post(
        "/analyze",
        json={
            "scenario_id": "INVALID",
            "timestamp": "2026-01-01T00:00:00Z",
            "latitude": 120,
            "longitude": 0,
            "altitude_m": 1,
            "speed_kmh": 1,
            "heading_deg": 0,
            "flight_pattern_entropy": 0,
            "proximity_to_restricted_km": 1,
            "iff_signal": True,
            "estimated_wingspan_m": 1,
            "loiter_detected": False,
            "rapid_altitude_change": False,
            "mission_narrative": "Invalid coordinate test",
        },
    )
    assert response.status_code == 422

