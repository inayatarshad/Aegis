"""Reproducible classifier evaluation with calibration and stress cases.

This benchmark is intentionally labelled synthetic. It reports both the regular
demo set and a small hand-authored stress set with overlapping signals and
unseen language so portfolio claims remain auditable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.classification_agent import classification_agent
from agents.ingestion_agent import ingestion_agent
from core.config import BASE_DIR
from core.state import initial_state

LABELS = ["BENIGN", "SUSPICIOUS", "HOSTILE"]
RESULT_PATH = BASE_DIR / "evaluation" / "results.json"

STRESS_CASES = [
    {
        "scenario_id": "STRESS-001", "timestamp": "2026-01-01T00:00:00Z",
        "latitude": 33.70, "longitude": 73.05, "altitude_m": 65,
        "speed_kmh": 75, "heading_deg": 20, "flight_pattern_entropy": 0.58,
        "proximity_to_restricted_km": 2.8, "iff_signal": False,
        "estimated_wingspan_m": 0.9, "loiter_detected": False,
        "rapid_altitude_change": True,
        "mission_narrative": "Unidentified contact changed altitude near the boundary.",
        "ground_truth_label": "SUSPICIOUS",
    },
    {
        "scenario_id": "STRESS-002", "timestamp": "2026-01-01T00:01:00Z",
        "latitude": 33.60, "longitude": 73.20, "altitude_m": 80,
        "speed_kmh": 95, "heading_deg": 210, "flight_pattern_entropy": 0.78,
        "proximity_to_restricted_km": 1.0, "iff_signal": False,
        "estimated_wingspan_m": 1.4, "loiter_detected": True,
        "rapid_altitude_change": True,
        "mission_narrative": "Contact ignores identification requests and accelerates inbound.",
        "ground_truth_label": "HOSTILE",
    },
    {
        "scenario_id": "STRESS-003", "timestamp": "2026-01-01T00:02:00Z",
        "latitude": 33.80, "longitude": 73.30, "altitude_m": 110,
        "speed_kmh": 35, "heading_deg": 90, "flight_pattern_entropy": 0.12,
        "proximity_to_restricted_km": 8.0, "iff_signal": True,
        "estimated_wingspan_m": 0.6, "loiter_detected": False,
        "rapid_altitude_change": False,
        "mission_narrative": "Authorized infrastructure inspection following its filed route.",
        "ground_truth_label": "BENIGN",
    },
    {
        "scenario_id": "STRESS-004", "timestamp": "2026-01-01T00:03:00Z",
        "latitude": 33.68, "longitude": 73.04, "altitude_m": 160,
        "speed_kmh": 30, "heading_deg": 175, "flight_pattern_entropy": 0.25,
        "proximity_to_restricted_km": 1.2, "iff_signal": True,
        "estimated_wingspan_m": 0.5, "loiter_detected": False,
        "rapid_altitude_change": False,
        "mission_narrative": "Cleared media aircraft operating close to the venue perimeter.",
        "ground_truth_label": "BENIGN",
    },
    {
        "scenario_id": "STRESS-005", "timestamp": "2026-01-01T00:04:00Z",
        "latitude": 33.67, "longitude": 73.00, "altitude_m": 190,
        "speed_kmh": 45, "heading_deg": 300, "flight_pattern_entropy": 0.46,
        "proximity_to_restricted_km": 5.5, "iff_signal": False,
        "estimated_wingspan_m": 0.8, "loiter_detected": True,
        "rapid_altitude_change": False,
        "mission_narrative": "Unknown platform circles intermittently outside the protected area.",
        "ground_truth_label": "SUSPICIOUS",
    },
    {
        "scenario_id": "STRESS-006", "timestamp": "2026-01-01T00:05:00Z",
        "latitude": 33.69, "longitude": 73.06, "altitude_m": 35,
        "speed_kmh": 115, "heading_deg": 5, "flight_pattern_entropy": 0.84,
        "proximity_to_restricted_km": 0.4, "iff_signal": False,
        "estimated_wingspan_m": 1.8, "loiter_detected": False,
        "rapid_altitude_change": True,
        "mission_narrative": "Fast low contact continues toward the protected site without identity.",
        "ground_truth_label": "HOSTILE",
    },
]


def expected_calibration_error(confidences, correctness, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            result += selected.mean() * abs(
                correctness[selected].mean() - confidences[selected].mean()
            )
    return float(result)


def evaluate(rows):
    actual, predicted, confidences, latencies = [], [], [], []
    for row in rows:
        started = time.perf_counter()
        state = classification_agent(ingestion_agent(initial_state(row)))
        latencies.append((time.perf_counter() - started) * 1000)
        result = state["classification"]
        actual.append(row["ground_truth_label"])
        predicted.append(result.threat_level.value)
        confidences.append(result.confidence)

    confidence_array = np.asarray(confidences)
    correctness = np.asarray(actual) == np.asarray(predicted)
    return {
        "samples": len(rows),
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "classification_report": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=LABELS
        ).tolist(),
        "expected_calibration_error": round(
            expected_calibration_error(confidence_array, correctness), 4
        ),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
    }


def main():
    demo_path = BASE_DIR / "data" / "simulated" / "sample_scenarios.json"
    demo_rows = json.loads(demo_path.read_text(encoding="utf-8"))
    output = {
        "limitations": [
            "All data is synthetic.",
            "The demo set resembles the training generator and is not independent.",
            "The stress set is small and should not support real-world safety claims.",
        ],
        "demo_set": evaluate(demo_rows),
        "stress_set": evaluate(STRESS_CASES),
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
