"""
models/threat_classifier.py — Hybrid tabular+text fusion threat classifier.
Trains on synthetic data if no saved model exists.
In production: swap generate_synthetic_data() for real labeled ISR data.
"""

import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL, BASE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = BASE_DIR / "models" / "saved" / "threat_clf_v2.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

_cached_classifier = None
_cached_embedder = None


def generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic drone telemetry training data.
    Labels:
      BENIGN    — normal civil/commercial drone
      SUSPICIOUS — anomalous but unclear intent
      HOSTILE   — high-risk profile
    """
    np.random.seed(42)
    X_tab = []
    X_text = []
    y = []

    narratives = {
        "BENIGN": [
            "Routine delivery drone in designated corridor",
            "Registered commercial UAV performing survey",
            "Agricultural drone within permitted area",
            "Media drone with valid flight clearance",
        ],
        "SUSPICIOUS": [
            "Unregistered UAV detected near boundary",
            "Drone loitering with no flight plan filed",
            "Aerial vehicle approaching from restricted direction",
            "UAV observed performing repeated passes",
        ],
        "HOSTILE": [
            "Unknown drone penetrating restricted airspace",
            "Unidentified UAV with no IFF approaching perimeter",
            "Drone with erratic flight pattern near critical zone",
            "Fast-moving aerial object detected on threat corridor",
        ],
    }

    for _ in range(n_samples):
        label = np.random.choice(["BENIGN", "SUSPICIOUS", "HOSTILE"],
                                  p=[0.50, 0.30, 0.20])

        if label == "BENIGN":
            features = [
                np.random.uniform(50, 300),    # altitude
                np.random.uniform(20, 60),     # speed
                np.random.uniform(0, 360),     # heading
                np.random.uniform(0.0, 0.3),   # entropy (low)
                np.random.uniform(5.0, 20.0),  # proximity (far)
                1.0,                           # IFF present
                np.random.uniform(0.3, 0.8),   # wingspan
                0.0,                           # no loiter
                0.0,                           # no rapid alt change
            ]
        elif label == "SUSPICIOUS":
            features = [
                np.random.uniform(20, 200),
                np.random.uniform(10, 80),
                np.random.uniform(0, 360),
                np.random.uniform(0.3, 0.7),   # medium entropy
                np.random.uniform(1.0, 8.0),   # closer
                float(np.random.random() > 0.6),  # IFF sometimes absent
                np.random.uniform(0.5, 1.5),
                float(np.random.random() > 0.4),  # loiter sometimes
                float(np.random.random() > 0.6),
            ]
        else:  # HOSTILE
            features = [
                np.random.uniform(10, 100),    # low altitude
                np.random.uniform(30, 120),    # fast
                np.random.uniform(0, 360),
                np.random.uniform(0.6, 1.0),   # high entropy
                np.random.uniform(0.1, 3.0),   # very close
                0.0,                           # no IFF
                np.random.uniform(0.5, 2.0),
                float(np.random.random() > 0.3),
                float(np.random.random() > 0.3),
            ]

        X_tab.append(features)
        X_text.append(np.random.choice(narratives[label]))
        y.append(label)

    return np.array(X_tab), X_text, np.array(y)


def train_classifier():
    """Train hybrid classifier on synthetic data."""
    logger.info("[Model] Training threat classifier on synthetic data...")

    X_tab, X_text, y = generate_synthetic_data(n_samples=2000)

    # Text embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X_text_emb = embedder.encode(X_text, normalize_embeddings=True, show_progress_bar=False)

    # Fuse
    X_fused = np.concatenate([X_tab, X_text_emb], axis=1)

    # Train pipeline
    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.07,
                random_state=42,
            ),
            method="sigmoid",
            cv=5,
        ))
    ])
    clf_pipeline.fit(X_fused, y)

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf_pipeline, f)

    logger.info(f"[Model] Classifier saved to {MODEL_PATH}")
    return clf_pipeline, embedder


def get_or_train_classifier():
    """Load cached classifier, or train if not available."""
    global _cached_classifier, _cached_embedder

    if _cached_classifier is not None:
        return _cached_classifier, _cached_embedder

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _cached_classifier = pickle.load(f)
            _cached_embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("[Model] Loaded saved classifier from disk")
            return _cached_classifier, _cached_embedder
        except Exception as e:
            logger.warning(f"[Model] Failed to load saved model ({e}), retraining...")

    _cached_classifier, _cached_embedder = train_classifier()
    return _cached_classifier, _cached_embedder
