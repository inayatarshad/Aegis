"""
core/config.py — Global configuration for AEGIS
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent
RUNTIME_DIR = Path("/tmp/aegis") if os.getenv("VERCEL") else BASE_DIR

load_dotenv(BASE_DIR / ".env")
load_dotenv(WORKSPACE_DIR / ".env", override=True)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = RUNTIME_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DOCTRINE_DIR = BASE_DIR / "data" / "simulated" / "doctrine_docs"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
SHAP_OUTPUT_DIR = OUTPUTS_DIR / "shap"

for d in [VECTORSTORE_DIR, OUTPUTS_DIR, SHAP_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM Config ────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"      # Groq-hosted LLaMA 3.3 70B
LLM_TEMPERATURE = 0.1                       # low for deterministic reports
LLM_MAX_TOKENS = 1024

# ── Embedding Config ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "aegis_doctrine"
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.45              # below this → re-query
MIN_CLASSIFICATION_CONFIDENCE = 0.60

# ── Classification Thresholds ─────────────────────────────────────────────
HOSTILE_THRESHOLD = 0.75
SUSPICIOUS_THRESHOLD = 0.45

# ── Geo: Simulated Restricted Zones ───────────────────────────────────────
# Each zone: (name, lat, lon, radius_km)
RESTRICTED_ZONES = [
    ("Zone Alpha", 33.6844, 73.0479, 3.0),
    ("Zone Bravo", 33.7200, 73.1200, 2.0),
    ("Zone Charlie", 33.6500, 72.9800, 1.5),
]

# ── Escalation Rules ──────────────────────────────────────────────────────
# (threat_level, confidence_threshold) -> escalation_level
ESCALATION_RULES = {
    ("HOSTILE", 0.75): 3,
    ("HOSTILE", 0.50): 2,
    ("SUSPICIOUS", 0.70): 2,
    ("SUSPICIOUS", 0.45): 1,
    ("BENIGN", 0.0): 0,
}

# ── Self-Correction ───────────────────────────────────────────────────────
MAX_RETRIEVAL_RETRIES = 2
HALLUCINATION_GUARD_ENABLED = True

# ── API ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))
