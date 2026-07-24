"""
api/main.py — FastAPI REST backend for AEGIS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json

from api.schemas import TelemetryInput, PipelineResponse, HealthResponse
from core.pipeline import run_pipeline
from utils.logger import get_logger
from core.config import VECTORSTORE_DIR, GROQ_API_KEY, BASE_DIR
from models.threat_classifier import MODEL_PATH

logger = get_logger(__name__)

app = FastAPI(
    title="AEGIS — Agentic ISR Intelligence API",
    description=(
        "Multi-Agent AI system for real-time drone threat classification, "
        "SALUTE report generation, and explainable decision support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Report readiness instead of returning an unconditional green status."""
    model_ready = MODEL_PATH.exists()
    vectorstore_ready = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())
    components = {
        "classifier": "ready" if model_ready else "cold_start",
        "vectorstore": "ready" if vectorstore_ready else "not_seeded",
        "llm": "configured" if GROQ_API_KEY else "offline_fallback",
    }
    degraded = any(value in {"not_seeded"} for value in components.values())
    return HealthResponse(
        status="degraded" if degraded else "operational",
        version="2.0.0",
        components=components,
    )


@app.post("/analyze", response_model=PipelineResponse, tags=["ISR Analysis"])
def analyze_telemetry(payload: TelemetryInput):
    """
    Submit drone telemetry for full AEGIS pipeline analysis.
    Returns threat classification, SALUTE report, XAI explanation, and escalation decision.
    """
    logger.info(f"[API] /analyze — scenario: {payload.scenario_id}")
    try:
        result = run_pipeline(payload.model_dump())
        final = result.get("final_response", {})
        if not final:
            raise HTTPException(status_code=500, detail="Pipeline produced no output")
        return final
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["ISR Analysis"])
def analyze_batch(payloads: list[TelemetryInput]):
    """
    Submit multiple telemetry packets for batch processing.
    """
    results = []
    for payload in payloads:
        try:
            result = run_pipeline(payload.model_dump())
            results.append(result.get("final_response", {}))
        except Exception as e:
            results.append({"error": str(e), "scenario_id": payload.scenario_id})
    return {"results": results, "count": len(results)}


@app.get("/scenarios/sample", tags=["Demo"])
def get_sample_scenarios():
    """Return pre-generated sample scenarios for demo/testing."""
    scenarios_path = BASE_DIR / "data" / "simulated" / "sample_scenarios.json"
    if not scenarios_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample scenarios not found. Run: python data/simulated/generate_telemetry.py"
        )
    with open(scenarios_path) as f:
        return json.load(f)


@app.get("/scenarios/sample/{scenario_id}", tags=["Demo"])
def run_sample_scenario(scenario_id: str):
    """Run the AEGIS pipeline on a pre-generated sample scenario."""
    scenarios_path = BASE_DIR / "data" / "simulated" / "sample_scenarios.json"
    if not scenarios_path.exists():
        raise HTTPException(status_code=404, detail="Sample scenarios not found")

    with open(scenarios_path) as f:
        scenarios = json.load(f)

    scenario = next((s for s in scenarios if s["scenario_id"] == scenario_id), None)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")

    result = run_pipeline(scenario)
    return result.get("final_response", {})
