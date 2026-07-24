"""Faithful local explanations for the fused telemetry + text classifier."""


import numpy as np

from core.state import AEGISState, XAIResult
from core.config import SHAP_OUTPUT_DIR
from agents.classification_agent import telemetry_to_features
from models.threat_classifier import get_or_train_classifier
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "altitude_m", "speed_kmh", "heading_deg", "flight_pattern_entropy",
    "proximity_to_restricted_km", "iff_signal", "estimated_wingspan_m",
    "loiter_detected", "rapid_altitude_change",
]

FEATURE_LABELS = {
    "altitude_m": "altitude",
    "speed_kmh": "speed",
    "heading_deg": "heading",
    "flight_pattern_entropy": "flight-pattern entropy",
    "proximity_to_restricted_km": "restricted-zone proximity",
    "iff_signal": "IFF status",
    "estimated_wingspan_m": "estimated wingspan",
    "loiter_detected": "loiter detection",
    "rapid_altitude_change": "rapid altitude change",
    "mission_narrative": "mission narrative",
}

# Values represent an ordinary, identified aircraft away from a restricted zone.
REFERENCE_TELEMETRY = np.array(
    [175.0, 40.0, 180.0, 0.20, 12.0, 1.0, 0.70, 0.0, 0.0],
    dtype=np.float32,
)
REFERENCE_NARRATIVE = "Registered aircraft operating normally in an approved corridor."
METHOD = "leave-one-feature-out probability attribution"


def _target_probability(classifier, fused: np.ndarray, target_class: str) -> float:
    class_index = list(classifier.classes_).index(target_class)
    return float(classifier.predict_proba(fused.reshape(1, -1))[0][class_index])


def _explain(classifier, embedder, telemetry, target_class: str) -> dict[str, float]:
    """Measure probability change when each input is replaced by a reference value."""
    tabular = telemetry_to_features(telemetry)
    text = embedder.encode(
        [telemetry.mission_narrative], normalize_embeddings=True
    )[0]
    fused = np.concatenate([tabular, text])
    observed_probability = _target_probability(classifier, fused, target_class)

    attributions: dict[str, float] = {}
    for index, feature_name in enumerate(FEATURE_NAMES):
        perturbed = tabular.copy()
        perturbed[index] = REFERENCE_TELEMETRY[index]
        perturbed_fused = np.concatenate([perturbed, text])
        attributions[feature_name] = round(
            observed_probability
            - _target_probability(classifier, perturbed_fused, target_class),
            6,
        )

    reference_text = embedder.encode(
        [REFERENCE_NARRATIVE], normalize_embeddings=True
    )[0]
    text_perturbed = np.concatenate([tabular, reference_text])
    attributions["mission_narrative"] = round(
        observed_probability
        - _target_probability(classifier, text_perturbed, target_class),
        6,
    )
    return attributions


def generate_nl_explanation(attributions: dict[str, float], threat_level: str) -> str:
    ranked = sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top = ranked[:3]
    total = sum(abs(value) for _, value in ranked) or 1.0
    factors = []
    for feature, value in top:
        direction = "supported" if value >= 0 else "opposed"
        share = abs(value) / total
        factors.append(
            f"{FEATURE_LABELS[feature]} ({direction} the result; "
            f"{abs(value):.1%} probability change, {share:.0%} of local impact)"
        )
    return (
        f"Local perturbation analysis for the {threat_level} prediction. "
        f"Strongest factors: {', '.join(factors)}."
    )


def xai_agent(state: AEGISState) -> AEGISState:
    state["agent_trace"].append("xai_agent")
    telemetry = state.get("telemetry")
    classification = state.get("classification")
    if telemetry is None or classification is None:
        state["errors"].append("Explanation: missing telemetry or classification")
        return state

    try:
        classifier, embedder = get_or_train_classifier()
        target_class = classification.threat_level.value
        values = _explain(classifier, embedder, telemetry, target_class)
        ranked = sorted(values, key=lambda name: abs(values[name]), reverse=True)
        top_factors = [FEATURE_LABELS[name] for name in ranked[:3]]
        explanation = generate_nl_explanation(values, target_class)

        plot_path = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ordered = sorted(values.items(), key=lambda item: item[1])
            labels = [FEATURE_LABELS[name] for name, _ in ordered]
            impacts = [value for _, value in ordered]
            colors = ["#ef4444" if value > 0 else "#38bdf8" for value in impacts]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(labels, impacts, color=colors)
            ax.axvline(0, color="#94a3b8", linewidth=0.8)
            ax.set_title(f"Local attribution for {target_class}")
            ax.set_xlabel("Change in target-class probability")
            plt.tight_layout()
            plot_path = str(
                SHAP_OUTPUT_DIR / f"{telemetry.scenario_id}_attribution.png"
            )
            plt.savefig(plot_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
        except Exception as plot_error:
            logger.warning("[Explanation] Plot generation failed: %s", plot_error)

        state["xai"] = XAIResult(
            attribution_values=values,
            top_factors=top_factors,
            explanation_text=explanation,
            attribution_method=METHOD,
            target_class=target_class,
            plot_path=plot_path,
        )
        logger.info("[Explanation] %s", explanation)
    except Exception as error:
        state["errors"].append(f"Explanation unavailable: {error}")
        logger.exception("[Explanation] Failed")
    return state
