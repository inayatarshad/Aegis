"""
agents/report_agent.py — LLM-powered SALUTE-format report generation.
Uses LLaMA 3.3 70B via Groq API.
"""

import json
from groq import Groq

from core.state import AEGISState, SALUTEReport
from core.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a military intelligence analyst AI assistant for an ISR (Intelligence, Surveillance & Reconnaissance) system. 
Your task is to generate structured SALUTE-format intelligence reports based on sensor data and AI analysis.

SALUTE format:
- Size: Description of the threat's physical dimensions/count
- Activity: What the threat is doing
- Location: Precise location description
- Unit: Identification/classification of the unit or object
- Time: Timestamp of observation
- Equipment: Technical description of equipment

Rules:
1. Be precise and concise — military reporting style
2. Use the data provided — do NOT hallucinate facts
3. Always end with a "Recommended Action" based on threat level and doctrine
4. Return a valid JSON object only — no markdown, no preamble
"""

REPORT_PROMPT_TEMPLATE = """
Based on the following ISR sensor analysis, generate a SALUTE report as JSON.

=== CLASSIFICATION RESULT ===
Threat Level: {threat_level}
Confidence: {confidence}
XAI Summary: {xai_summary}

=== TELEMETRY ===
Scenario ID: {scenario_id}
Timestamp: {timestamp}
GPS: {lat}, {lon}
Altitude: {altitude_m}m | Speed: {speed_kmh} km/h | Heading: {heading_deg}°
IFF Signal: {iff_signal}
Loiter Detected: {loiter_detected}
Rapid Altitude Change: {rapid_altitude_change}
Estimated Wingspan: {wingspan_m}m
Proximity to Restricted Zone: {proximity_km}km

=== MISSION CONTEXT ===
In Restricted Zone: {in_restricted} ({zone_name})
Threat Corridor: {in_corridor}
Geo Risk Score: {geo_risk}
Mission Phase: {mission_phase}
Mission Narrative: {narrative}

=== DOCTRINE REFERENCE ===
{doctrine_ref}

=== FUSION ANALYSIS ===
Fused Risk Score: {fused_score}
Human Review Required: {human_review}
Conflict Flags: {conflicts}

Generate a JSON with these exact keys:
{{
  "size": "...",
  "activity": "...",
  "location": "...",
  "unit": "...",
  "time": "...",
  "equipment": "...",
  "recommended_action": "..."
}}
"""


def report_agent(state: AEGISState) -> AEGISState:
    """
    Report Generation Agent:
    - Assembles structured prompt from all prior agent outputs
    - Calls LLaMA 3.3 70B via Groq
    - Parses LLM response into SALUTEReport
    - Stores formatted report text for dashboard display
    """
    state["agent_trace"].append("report_agent")

    clf = state.get("classification")
    tel = state.get("telemetry")
    xai = state.get("xai")
    ret = state.get("retrieval")
    ctx = state.get("context")
    fusion = state.get("fusion")

    if not all([clf, tel]):
        state["errors"].append("Report: missing classification or telemetry")
        return state

    # Build prompt
    prompt = REPORT_PROMPT_TEMPLATE.format(
        threat_level=clf.threat_level.value,
        confidence=f"{clf.confidence:.2f}",
        xai_summary=xai.explanation_text if xai else "N/A",
        scenario_id=tel.scenario_id,
        timestamp=tel.timestamp,
        lat=tel.latitude,
        lon=tel.longitude,
        altitude_m=tel.altitude_m,
        speed_kmh=tel.speed_kmh,
        heading_deg=tel.heading_deg,
        iff_signal="YES" if tel.iff_signal else "NO",
        loiter_detected="YES" if tel.loiter_detected else "NO",
        rapid_altitude_change="YES" if tel.rapid_altitude_change else "NO",
        wingspan_m=tel.estimated_wingspan_m,
        proximity_km=tel.proximity_to_restricted_km,
        in_restricted="YES" if (ctx and ctx.in_restricted_zone) else "NO",
        zone_name=(ctx.zone_name or "N/A") if ctx else "N/A",
        in_corridor="YES" if (ctx and ctx.threat_corridor) else "NO",
        geo_risk=f"{ctx.geo_risk_score:.2f}" if ctx else "N/A",
        mission_phase=ctx.mission_phase if ctx else "UNKNOWN",
        narrative=tel.mission_narrative,
        doctrine_ref=ret.doctrine_reference if ret else "N/A",
        fused_score=f"{fusion.fused_risk_score:.2f}" if fusion else "N/A",
        human_review="YES" if (fusion and fusion.human_review_required) else "NO",
        conflicts=("; ".join(fusion.conflict_flags) if fusion and fusion.conflict_flags else "None"),
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        raw_text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        report_dict = json.loads(raw_text)

        state["salute_report"] = SALUTEReport(
            size=report_dict.get("size", ""),
            activity=report_dict.get("activity", ""),
            location=report_dict.get("location", ""),
            unit=report_dict.get("unit", ""),
            time=report_dict.get("time", tel.timestamp),
            equipment=report_dict.get("equipment", ""),
        )

        # Format human-readable report text
        state["report_text"] = _format_report(state["salute_report"], report_dict, clf, xai, fusion)
        logger.info(f"[Report] SALUTE report generated for {tel.scenario_id}")

    except Exception as e:
        state["errors"].append(f"Report generation error: {e}")
        logger.error(f"[Report] Error: {e}")
        # Fallback: generate template-based report
        state["salute_report"] = _fallback_salute(tel, clf, ctx)
        state["report_text"] = f"[FALLBACK REPORT — LLM unavailable]\n{state['salute_report']}"

    return state


def _format_report(salute: SALUTEReport, report_dict: dict, clf, xai, fusion) -> str:
    lines = [
        "═" * 60,
        "          AEGIS INTELLIGENCE REPORT",
        "═" * 60,
        f"THREAT LEVEL  : {clf.threat_level.value} (confidence: {clf.confidence:.0%})",
        f"FUSED RISK    : {fusion.fused_risk_score:.2f}" if fusion else "",
        "─" * 60,
        "SALUTE REPORT",
        "─" * 60,
        f"S — SIZE      : {salute.size}",
        f"A — ACTIVITY  : {salute.activity}",
        f"L — LOCATION  : {salute.location}",
        f"U — UNIT      : {salute.unit}",
        f"T — TIME      : {salute.time}",
        f"E — EQUIPMENT : {salute.equipment}",
        "─" * 60,
        "XAI RATIONALE",
        "─" * 60,
        xai.explanation_text if xai else "N/A",
        "─" * 60,
        "RECOMMENDED ACTION",
        "─" * 60,
        report_dict.get("recommended_action", "Consult duty officer."),
        "═" * 60,
    ]
    return "\n".join(l for l in lines if l is not None)


def _fallback_salute(tel, clf, ctx) -> SALUTEReport:
    return SALUTEReport(
        size=f"Single UAV, ~{tel.estimated_wingspan_m}m wingspan",
        activity=f"{'Loitering' if tel.loiter_detected else 'Flying'} at {tel.altitude_m}m altitude",
        location=f"{tel.latitude:.4f}°N, {tel.longitude:.4f}°E",
        unit="Unknown — no IFF signal" if not tel.iff_signal else "Unknown",
        time=tel.timestamp,
        equipment=f"Small UAV, wingspan ~{tel.estimated_wingspan_m}m",
    )
