"""
agents/classification_agent.py — Hybrid threat classifier.

Architecture:
  - Tabular branch: GradientBoostingClassifier on telemetry features
  - Text branch: DistilBERT sentence embeddings on mission_narrative
  - Fusion: concatenated vector → LogisticRegression head
  
For the simulated environment we train on-the-fly with synthetic data.
In production, swap in a pre-trained model loaded from disk.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from core.state import AEGISState, ClassificationResult, ThreatLevel
from core.config import (
    HOSTILE_THRESHOLD, SUSPICIOUS_THRESHOLD, EMBEDDING_MODEL
)
from utils.logger import get_logger
from models.threat_classifier import get_or_train_classifier

logger = get_logger(__name__)


def telemetry_to_features(telemetry) -> np.ndarray:
    """Extract numerical feature vector from DroneTelemetry."""
    return np.array([
        telemetry.altitude_m,
        telemetry.speed_kmh,
        telemetry.heading_deg,
        telemetry.flight_pattern_entropy,
        telemetry.proximity_to_restricted_km,
        float(telemetry.iff_signal),
        telemetry.estimated_wingspan_m,
        float(telemetry.loiter_detected),
        float(telemetry.rapid_altitude_change),
    ], dtype=np.float32)


def classification_agent(state: AEGISState) -> AEGISState:
    """
    Threat Classification Agent:
    - Extracts tabular features from telemetry
    - Embeds mission_narrative via sentence transformer
    - Fuses both branches to predict threat level + confidence
    """
    state["agent_trace"].append("classification_agent")
    telemetry = state.get("telemetry")
    if telemetry is None:
        state["errors"].append("Classification: no telemetry available")
        return state

    try:
        classifier, embedder = get_or_train_classifier()

        # Tabular features
        tab_features = telemetry_to_features(telemetry).reshape(1, -1)

        # Text embedding of mission narrative
        text_embedding = embedder.encode(
            [telemetry.mission_narrative], normalize_embeddings=True
        )  # shape (1, 384)

        # Fuse: concatenate tabular + text embedding
        fused = np.concatenate([tab_features, text_embedding], axis=1)

        # Predict
        proba = classifier.predict_proba(fused)[0]
        classes = classifier.classes_                  # ['BENIGN', 'HOSTILE', 'SUSPICIOUS']
        class_probs = {c: float(p) for c, p in zip(classes, proba)}

        # Determine threat level from probabilities
        hostile_prob = class_probs.get("HOSTILE", 0.0)
        suspicious_prob = class_probs.get("SUSPICIOUS", 0.0)

        if hostile_prob >= HOSTILE_THRESHOLD:
            threat_level = ThreatLevel.HOSTILE
            confidence = hostile_prob
        elif suspicious_prob >= SUSPICIOUS_THRESHOLD or hostile_prob >= 0.45:
            threat_level = ThreatLevel.SUSPICIOUS
            confidence = max(suspicious_prob, hostile_prob)
        else:
            threat_level = ThreatLevel.BENIGN
            confidence = class_probs.get("BENIGN", 0.0)

        state["classification"] = ClassificationResult(
            threat_level=threat_level,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
            model_version="aegis-clf-v1.0",
        )
        logger.info(
            f"[Classification] {telemetry.scenario_id} → "
            f"{threat_level.value} (conf={confidence:.2f})"
        )

    except Exception as e:
        state["errors"].append(f"Classification error: {e}")
        logger.error(f"[Classification] Error: {e}")
        # Fallback to UNKNOWN
        state["classification"] = ClassificationResult(
            threat_level=ThreatLevel.UNKNOWN,
            confidence=0.0,
            class_probabilities={},
            model_version="fallback",
        )

    return state
