"""
data/simulated/generate_telemetry.py
Generates synthetic drone scenarios and seeds the ChromaDB vector store
with simulated doctrine documents.

Run once before starting the application:
    python data/simulated/generate_telemetry.py
"""

import json
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.config import (
    VECTORSTORE_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, DATA_DIR
)

random.seed(42)
np.random.seed(42)

SCENARIOS_OUTPUT = DATA_DIR / "simulated" / "sample_scenarios.json"


# ── DOCTRINE DOCUMENTS ────────────────────────────────────────────────────

DOCTRINE_DOCS = [
    {
        "id": "roe-4-1",
        "source": "ROE Section 4.1",
        "text": (
            "ROE Section 4.1 — Identification Protocol: All aerial objects within 10km of "
            "a protected perimeter must be identified via IFF transponder. Unidentified objects "
            "should be classified as potentially hostile until verified. Notify the duty officer "
            "immediately upon detection of any unidentified aerial vehicle."
        )
    },
    {
        "id": "roe-4-2",
        "source": "ROE Section 4.2",
        "text": (
            "ROE Section 4.2 — Restricted Airspace Breach: Any unidentified aerial vehicle "
            "within 2km of the designated restricted perimeter is to be treated as a Level 3 threat. "
            "The duty officer must be notified within 60 seconds. Tracking protocol must be initiated "
            "and maintained until the vehicle is identified, exits restricted area, or is neutralized."
        )
    },
    {
        "id": "roe-4-3",
        "source": "ROE Section 4.3",
        "text": (
            "ROE Section 4.3 — Loitering Aerial Vehicles: Any aerial vehicle exhibiting loitering "
            "behavior (defined as remaining within a 500m radius for more than 5 minutes) near a "
            "protected installation without prior clearance is to be classified as SUSPICIOUS minimum. "
            "Initiate continuous tracking and request identification via standard challenge protocol."
        )
    },
    {
        "id": "roe-5-1",
        "source": "ROE Section 5.1",
        "text": (
            "ROE Section 5.1 — Hostile Classification Response: Upon confirmed HOSTILE classification "
            "by automated ISR systems (confidence >= 75%), Level 3 alert is to be dispatched to: "
            "(a) Duty Officer, (b) Quick Reaction Force, (c) Air Defense Command. "
            "Manual override remains with the duty officer at all times."
        )
    },
    {
        "id": "roe-5-2",
        "source": "ROE Section 5.2",
        "text": (
            "ROE Section 5.2 — Electronic Warfare Countermeasures: For confirmed hostile UAVs, "
            "electronic countermeasures may be deployed upon Level 3 alert authorization. "
            "RF jamming and GPS spoofing are authorized within the protected perimeter only. "
            "Documentation of all countermeasure deployment is mandatory."
        )
    },
    {
        "id": "tactics-uav-1",
        "source": "Tactical Handbook — UAV Threat Profiles",
        "text": (
            "Common hostile UAV profiles include: (1) High-entropy loitering — erratic flight pattern "
            "with multiple direction reversals indicating reconnaissance behavior; (2) Low-altitude "
            "fast approach — altitude below 50m combined with speed above 60 km/h indicating "
            "attack/delivery profile; (3) Swarm pattern — multiple small contacts within 500m "
            "radius indicating coordinated operation."
        )
    },
    {
        "id": "tactics-uav-2",
        "source": "Tactical Handbook — ISR Best Practices",
        "text": (
            "ISR best practices for UAV threat assessment: Always cross-reference automated AI "
            "classifications with geo-contextual data. A BENIGN classification in isolation may "
            "be insufficient if proximity to restricted zone is less than 3km. Human operator "
            "review is mandatory for all ambiguous threat scores (0.45–0.60 fused risk range)."
        )
    },
    {
        "id": "escalation-matrix",
        "source": "Escalation Matrix v2.3",
        "text": (
            "Escalation Matrix: Level 0 — BENIGN, log only. Level 1 — MONITOR, "
            "continuous tracking activated. Level 2 — REVIEW, duty officer notified, "
            "tracking team on standby. Level 3 — ALERT, QRF notified, countermeasures "
            "on standby. Level 4 — CRITICAL, all assets activated, immediate response authorized."
        )
    },
]


# ── SCENARIO GENERATION ───────────────────────────────────────────────────

def random_timestamp(base: datetime = None) -> str:
    if base is None:
        base = datetime(2024, 11, 14, 12, 0, 0)
    delta = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_scenarios(n: int = 20) -> list:
    scenarios = []
    for i in range(n):
        threat_class = random.choices(
            ["BENIGN", "SUSPICIOUS", "HOSTILE"],
            weights=[50, 30, 20]
        )[0]

        # Base GPS near Islamabad (Zone Alpha)
        base_lat, base_lon = 33.6844, 73.0479
        lat = base_lat + random.uniform(-0.1, 0.1)
        lon = base_lon + random.uniform(-0.1, 0.1)

        if threat_class == "BENIGN":
            proximity = random.uniform(5.0, 20.0)
            altitude = random.uniform(50, 400)
            speed = random.uniform(20, 60)
            entropy = random.uniform(0.0, 0.3)
            iff = True
            loiter = False
            rapid_alt = False
            wingspan = random.uniform(0.3, 0.8)
            narrative = random.choice([
                "Registered commercial delivery drone operating in approved corridor.",
                "Agricultural survey UAV with valid flight plan.",
                "Media drone with active clearance from ATC.",
            ])
        elif threat_class == "SUSPICIOUS":
            proximity = random.uniform(1.5, 6.0)
            altitude = random.uniform(30, 200)
            speed = random.uniform(15, 80)
            entropy = random.uniform(0.35, 0.65)
            iff = random.random() > 0.5
            loiter = random.random() > 0.45
            rapid_alt = random.random() > 0.6
            wingspan = random.uniform(0.5, 1.5)
            narrative = random.choice([
                "Unregistered UAV detected approaching from north without flight plan.",
                "Drone observed performing repeated passes over installation boundary.",
                "Aerial vehicle loitering in vicinity with no response to challenge.",
            ])
        else:  # HOSTILE
            proximity = random.uniform(0.1, 2.5)
            altitude = random.uniform(10, 80)
            speed = random.uniform(40, 130)
            entropy = random.uniform(0.65, 1.0)
            iff = False
            loiter = random.random() > 0.3
            rapid_alt = random.random() > 0.3
            wingspan = random.uniform(0.8, 2.5)
            narrative = random.choice([
                "Unknown drone penetrating restricted airspace with no IFF signal.",
                "Fast-moving UAV on direct approach vector to protected zone, no transponder.",
                "Hostile profile drone detected — erratic pattern, no identification possible.",
            ])

        scenario = {
            "scenario_id": f"SC-2024-{i+1:04d}",
            "timestamp": random_timestamp(),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "altitude_m": round(altitude, 1),
            "speed_kmh": round(speed, 1),
            "heading_deg": round(random.uniform(0, 360), 1),
            "flight_pattern_entropy": round(entropy, 4),
            "proximity_to_restricted_km": round(proximity, 2),
            "iff_signal": iff,
            "estimated_wingspan_m": round(wingspan, 2),
            "loiter_detected": loiter,
            "rapid_altitude_change": rapid_alt,
            "mission_narrative": narrative,
            "ground_truth_label": threat_class,  # for evaluation
        }
        scenarios.append(scenario)

    return scenarios


def seed_vectorstore():
    """Populate ChromaDB with doctrine documents."""
    print("Seeding ChromaDB with doctrine documents...")
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(CHROMA_COLLECTION)

    # Clear existing
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Cleared {len(existing['ids'])} existing documents")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in DOCTRINE_DOCS]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=[d["id"] for d in DOCTRINE_DOCS],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": d["source"]} for d in DOCTRINE_DOCS],
    )
    print(f"  Inserted {len(DOCTRINE_DOCS)} doctrine documents")


if __name__ == "__main__":
    print("=" * 50)
    print("AEGIS Data Generator")
    print("=" * 50)

    # 1. Seed vector store
    seed_vectorstore()

    # 2. Generate scenarios
    print("\nGenerating 20 sample scenarios...")
    scenarios = generate_scenarios(n=20)
    SCENARIOS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENARIOS_OUTPUT, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved {len(scenarios)} scenarios to {SCENARIOS_OUTPUT}")

    label_counts = {}
    for s in scenarios:
        lbl = s["ground_truth_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  Label distribution: {label_counts}")

    print("\n✅ Setup complete. You can now run the application.")
    print("   uvicorn api.main:app --reload")
    print("   streamlit run dashboard/app.py")
