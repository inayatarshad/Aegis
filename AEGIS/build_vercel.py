"""Create immutable inference artifacts during the Vercel build."""

import os
from pathlib import Path

os.environ["AEGIS_BUILD"] = "1"
os.environ.setdefault("HF_HOME", str(Path("models") / ".huggingface"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from core.config import BUNDLED_EMBEDDING_DIR, EMBEDDING_MODEL_NAME  # noqa: E402


def main():
    if not BUNDLED_EMBEDDING_DIR.exists():
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(BUNDLED_EMBEDDING_DIR))

    from models.threat_classifier import MODEL_PATH, train_classifier

    if not MODEL_PATH.exists():
        train_classifier()


if __name__ == "__main__":
    main()

