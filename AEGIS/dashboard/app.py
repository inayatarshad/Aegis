"""
dashboard/app.py — AEGIS Operator Dashboard (Streamlit)
Real-time ISR analysis interface with threat map, SALUTE viewer, and XAI panel.
"""

import streamlit as st
import json
import requests
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS — ISR Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("AEGIS_API_BASE", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .threat-hostile { color: #ff4444; font-weight: bold; font-size: 1.4em; }
    .threat-suspicious { color: #ffaa00; font-weight: bold; font-size: 1.4em; }
    .threat-benign { color: #00cc44; font-weight: bold; font-size: 1.4em; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    .salute-field { font-family: 'Courier New', monospace; font-size: 0.9em; }
    .agent-trace span {
        display: inline-block; background: #21262d;
        border: 1px solid #30363d; border-radius: 4px;
        padding: 2px 8px; margin: 2px; font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0d1117/e6edf3?text=AEGIS+ISR", width=200)
    st.markdown("---")
    st.markdown("### Input Mode")
    input_mode = st.radio(
        "Select input source:",
        ["Sample Scenarios", "Manual Input", "JSON Upload"]
    )

    st.markdown("---")
    st.markdown("### System Status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            health = r.json()
            st.success(f"API: {health.get('status', 'online').title()}")
            for component, status in health.get("components", {}).items():
                st.caption(f"{component}: {status}")
        else:
            st.error("API: Error")
    except requests.RequestException:
        st.warning("API: Offline (running in demo mode)")

    st.markdown("---")
    st.caption("AEGIS v1.0 | Inayat Arshad | PIEAS")


# ── Main title ────────────────────────────────────────────────────────────
st.title("🛡️ AEGIS — Agentic ISR Intelligence Dashboard")
st.caption("Multi-Agent AI System for Drone Threat Classification & SALUTE Report Generation")
st.markdown("---")


# ── Input Section ─────────────────────────────────────────────────────────
def get_sample_scenarios():
    try:
        r = requests.get(f"{API_BASE}/scenarios/sample", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    # Fallback: load from file
    p = Path("data/simulated/sample_scenarios.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


result = None

if input_mode == "Sample Scenarios":
    scenarios = get_sample_scenarios()
    if scenarios:
        scenario_ids = [s["scenario_id"] for s in scenarios]
        selected_id = st.selectbox("Select scenario:", scenario_ids)
        selected = next(s for s in scenarios if s["scenario_id"] == selected_id)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Ground Truth:** {selected.get('ground_truth_label', 'N/A')}")
        with col2:
            st.info(f"**GPS:** {selected['latitude']:.4f}°N, {selected['longitude']:.4f}°E")
        with col3:
            st.info(f"**Altitude:** {selected['altitude_m']}m | Speed: {selected['speed_kmh']} km/h")

        if st.button("🚀 Run AEGIS Analysis", type="primary", use_container_width=True):
            with st.spinner("Running 7-agent pipeline..."):
                try:
                    r = requests.post(f"{API_BASE}/analyze", json=selected, timeout=60)
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
    else:
        st.warning("No scenarios found. Run: `python data/simulated/generate_telemetry.py`")

elif input_mode == "Manual Input":
    with st.expander("Enter Drone Telemetry", expanded=True):
        c1, c2, c3 = st.columns(3)
        scenario_id = c1.text_input("Scenario ID", value="SC-MANUAL-001")
        timestamp = c2.text_input("Timestamp", value="2024-11-14T14:32:11Z")
        lat = c1.number_input("Latitude", value=33.6844, format="%.6f")
        lon = c2.number_input("Longitude", value=73.0479, format="%.6f")
        altitude = c1.number_input("Altitude (m)", value=45.0)
        speed = c2.number_input("Speed (km/h)", value=85.0)
        heading = c3.number_input("Heading (°)", value=270.0)
        entropy = c1.slider("Flight Pattern Entropy", 0.0, 1.0, 0.75)
        proximity = c2.number_input("Proximity to Restricted Zone (km)", value=1.7)
        iff = c3.checkbox("IFF Signal Present", value=False)
        wingspan = c1.number_input("Wingspan (m)", value=1.2)
        loiter = c2.checkbox("Loiter Detected", value=True)
        rapid_alt = c3.checkbox("Rapid Altitude Change", value=True)
        narrative = st.text_area("Mission Narrative", value="Unknown drone penetrating restricted airspace with no IFF signal.")

    if st.button("🚀 Run AEGIS Analysis", type="primary", use_container_width=True):
        payload = {
            "scenario_id": scenario_id, "timestamp": timestamp,
            "latitude": lat, "longitude": lon, "altitude_m": altitude,
            "speed_kmh": speed, "heading_deg": heading,
            "flight_pattern_entropy": entropy, "proximity_to_restricted_km": proximity,
            "iff_signal": iff, "estimated_wingspan_m": wingspan,
            "loiter_detected": loiter, "rapid_altitude_change": rapid_alt,
            "mission_narrative": narrative,
        }
        with st.spinner("Running 7-agent pipeline..."):
            try:
                r = requests.post(f"{API_BASE}/analyze", json=payload, timeout=60)
                result = r.json()
            except Exception as e:
                st.error(f"API error: {e}")

elif input_mode == "JSON Upload":
    uploaded = st.file_uploader("Upload telemetry JSON", type="json")
    if uploaded:
        payload = json.load(uploaded)
        st.json(payload)
        if st.button("🚀 Run AEGIS Analysis", type="primary"):
            with st.spinner("Running pipeline..."):
                try:
                    r = requests.post(f"{API_BASE}/analyze", json=payload, timeout=60)
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")


# ── Results Panel ─────────────────────────────────────────────────────────
if result:
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    threat = result.get("threat_level", "UNKNOWN")
    conf = result.get("confidence", 0)
    esc_level = result.get("escalation_level", 0)
    esc_name = result.get("escalation_level_name", "NONE")
    fused = result.get("fused_risk_score", 0)
    latency = result.get("processing_latency_ms")

    # ── Top metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    threat_color = {"HOSTILE": "threat-hostile", "SUSPICIOUS": "threat-suspicious",
                    "BENIGN": "threat-benign"}.get(threat, "")
    col1.markdown("**THREAT LEVEL**")
    col1.markdown(f'<p class="{threat_color}">{threat}</p>', unsafe_allow_html=True)

    col2.metric("Confidence", f"{conf:.1%}")
    col3.metric("Fused Risk Score", f"{fused:.2f}")
    col4.metric("Escalation", f"L{esc_level} — {esc_name}")

    if latency:
        st.caption(f"⏱ Pipeline latency: {latency:.0f}ms")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 SALUTE Report", "🧠 Model Explanation", "📡 Doctrine",
        "⚠️ Escalation", "🔍 Agent Trace"
    ])

    with tab1:
        st.markdown("#### SALUTE Intelligence Report")
        salute = result.get("salute_report", {})
        for key, label in [("size", "S — Size"), ("activity", "A — Activity"),
                            ("location", "L — Location"), ("unit", "U — Unit"),
                            ("time", "T — Time"), ("equipment", "E — Equipment")]:
            st.markdown(f"**{label}:** {salute.get(key, 'N/A')}")
        st.markdown("---")
        st.markdown("#### Full Report")
        st.code(result.get("report_text", ""), language=None)

    with tab2:
        st.markdown("#### Faithful local explanation")
        st.caption(
            f"Method: {result.get('attribution_method', 'unavailable')}. "
            "Values show the change in predicted-class probability when one "
            "input is replaced by a neutral reference."
        )
        st.info(result.get("xai_summary", "N/A"))
        st.markdown("#### Top Contributing Factors")
        for i, factor in enumerate(result.get("top_xai_factors", []), 1):
            st.markdown(f"{i}. {factor}")
        if result.get("shap_plot_path") and Path(result["shap_plot_path"]).exists():
            st.image(result["shap_plot_path"], caption="Local probability attribution")
        else:
            st.caption("Attribution plot unavailable")

    with tab3:
        st.markdown("#### Retrieved Doctrine Reference")
        st.success(result.get("doctrine_reference", "N/A"))

    with tab4:
        st.markdown("#### Escalation Decision")
        review_status = result.get("review_status", "NOT_REQUIRED")
        if review_status == "PENDING":
            st.warning(
                "Human decision pending. The escalation shown below is a "
                "recommendation, not an automatically dispatched action."
            )
            st.caption(result.get("review_reason", ""))
        esc_reason = result.get("escalation_reason", "")
        if esc_level >= 3:
            st.error(f"🚨 Level {esc_level}: {esc_reason}")
        elif esc_level >= 2:
            st.warning(f"🔶 Level {esc_level}: {esc_reason}")
        elif esc_level >= 1:
            st.info(f"👁 Level {esc_level}: {esc_reason}")
        else:
            st.success(f"✅ Level {esc_level}: {esc_reason}")

        if result.get("human_review_required"):
            st.warning("⚠️ Human review flagged by Fusion Agent")

        if result.get("conflict_flags"):
            st.markdown("**Conflict Flags:**")
            for flag in result["conflict_flags"]:
                st.markdown(f"- {flag}")

    with tab5:
        st.markdown("#### Agent Execution Trace")
        trace = result.get("agent_trace", [])
        trace_html = " → ".join([f"<span>{a}</span>" for a in trace])
        st.markdown(f'<div class="agent-trace">{trace_html}</div>', unsafe_allow_html=True)

        metrics = result.get("node_metrics", {})
        if metrics:
            st.markdown("#### Node observability")
            metric_rows = [
                {
                    "agent": agent,
                    "duration_ms": values.get("duration_ms"),
                    "status": values.get("status"),
                }
                for agent, values in metrics.items()
            ]
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)

        if result.get("errors"):
            st.markdown("**Pipeline Errors:**")
            for err in result["errors"]:
                st.error(err)

    # Full JSON
    with st.expander("🔧 Raw JSON Response"):
        st.json(result)
