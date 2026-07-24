"""
agents/retrieval_agent.py — ChromaDB RAG retrieval over doctrine documents.
Includes self-correction: re-queries with expanded terms if confidence is low.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.state import AEGISState, RetrievalResult, ThreatLevel
from core.config import (
    VECTORSTORE_DIR, CHROMA_COLLECTION,
    RETRIEVAL_TOP_K, RETRIEVAL_MIN_SCORE, EMBEDDING_MODEL
)
from utils.logger import get_logger

logger = get_logger(__name__)

_chroma_client = None
_embedder = None


def get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client.get_or_create_collection(CHROMA_COLLECTION)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _build_query(threat_level: ThreatLevel, telemetry) -> str:
    """Build a targeted query from threat context."""
    base = f"{threat_level.value} UAV threat response"
    extras = []
    if telemetry.proximity_to_restricted_km < 3.0:
        extras.append("restricted airspace violation")
    if telemetry.loiter_detected:
        extras.append("loitering UAV")
    if not telemetry.iff_signal:
        extras.append("unidentified aircraft no transponder")
    if telemetry.rapid_altitude_change:
        extras.append("aggressive maneuver")
    return base + (" " + " ".join(extras) if extras else "")


def _build_fallback_query(threat_level: ThreatLevel) -> str:
    """Broader fallback query for re-query on low confidence."""
    return f"UAV aerial threat rules of engagement escalation protocol {threat_level.value}"


def retrieval_agent(state: AEGISState) -> AEGISState:
    """
    Retrieval Agent:
    - Queries ChromaDB with threat-aware query
    - If retrieval confidence < threshold, re-queries (self-correction)
    - Extracts doctrine reference from top result
    """
    state["agent_trace"].append("retrieval_agent")
    classification = state.get("classification")
    telemetry = state.get("telemetry")

    if classification is None or telemetry is None:
        state["errors"].append("Retrieval: missing classification or telemetry")
        return state

    collection = get_chroma_collection()
    embedder = get_embedder()
    requeried = False

    def do_query(query: str):
        query_vec = embedder.encode([query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=query_vec,
            n_results=RETRIEVAL_TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        return results

    try:
        query = _build_query(classification.threat_level, telemetry)
        results = do_query(query)

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # For unit-normalized embeddings, cosine similarity = 1 - squared_L2 / 2.
        scores = [max(-1.0, min(1.0, 1.0 - (d / 2.0))) for d in distances]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Evidence-quality retry: retain the better result instead of blindly
        # replacing the first retrieval.
        if avg_score < RETRIEVAL_MIN_SCORE:
            logger.info(f"[Retrieval] Low confidence ({avg_score:.2f}), re-querying...")
            fallback_query = _build_fallback_query(classification.threat_level)
            retry_results = do_query(fallback_query)
            retry_docs = retry_results.get("documents", [[]])[0]
            retry_distances = retry_results.get("distances", [[]])[0]
            retry_metadatas = retry_results.get("metadatas", [[]])[0]
            retry_scores = [
                max(-1.0, min(1.0, 1.0 - (distance / 2.0)))
                for distance in retry_distances
            ]
            retry_avg = sum(retry_scores) / len(retry_scores) if retry_scores else 0.0
            if retry_avg > avg_score:
                docs, metadatas, scores = retry_docs, retry_metadatas, retry_scores
                avg_score = retry_avg
                query = fallback_query
            requeried = True

        retrieved_docs = []
        for doc, meta, score in zip(docs, metadatas, scores):
            retrieved_docs.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(score, 4),
            })

        # Doctrine reference: top result's source + first sentence
        doctrine_ref = ""
        if retrieved_docs:
            top = retrieved_docs[0]
            src = top["source"]
            snippet = top["text"][:200].split(".")[0]
            doctrine_ref = f"{src} — {snippet}"

        state["retrieval"] = RetrievalResult(
            query_used=query,
            retrieved_docs=retrieved_docs,
            doctrine_reference=doctrine_ref,
            retrieval_confidence=round(avg_score, 4),
            requeried=requeried,
        )
        logger.info(
            f"[Retrieval] Retrieved {len(retrieved_docs)} docs "
            f"(avg_score={avg_score:.2f}, requeried={requeried})"
        )

    except Exception as e:
        state["errors"].append(f"Retrieval error: {e}")
        logger.error(f"[Retrieval] Error: {e}")
        state["retrieval"] = RetrievalResult(
            query_used="",
            retrieved_docs=[],
            doctrine_reference="Doctrine retrieval unavailable.",
            retrieval_confidence=0.0,
        )

    return state
