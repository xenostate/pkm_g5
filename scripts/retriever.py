#!/usr/bin/env python3
"""
School Helper Retriever: embed a query, search a course's ChromaDB collection, return ranked results.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts.indexer import get_model, get_chroma_collection

# ── Config ──────────────────────────────────────────────────────────────────

THRESHOLD_HIGH = float(os.environ.get("THRESHOLD_HIGH", "0.75"))
THRESHOLD_MEDIUM = float(os.environ.get("THRESHOLD_MEDIUM", "0.5"))

# ── Retrieval ──────────────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = 5, model=None, course_id: str = "") -> dict:
    """
    Search ChromaDB for chunks matching the query within a course.

    Returns:
        {
            "confidence": "high" | "medium" | "low",
            "results": [
                {"chunk_text": str, "source": str, "title": str, "score": float, "doc_id": str}
            ]
        }
    """
    if model is None:
        model = get_model()

    query_embedding = model.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    collection = get_chroma_collection(course_id)

    # Check if collection has any documents
    if collection.count() == 0:
        return {"confidence": "low", "results": []}

    actual_k = min(top_k, collection.count())
    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    if chroma_results["ids"] and chroma_results["ids"][0]:
        for i, doc_id in enumerate(chroma_results["ids"][0]):
            distance = chroma_results["distances"][0][i]
            score = 1 - distance  # cosine distance -> similarity
            meta = chroma_results["metadatas"][0][i]
            results.append({
                "chunk_text": chroma_results["documents"][0][i],
                "source": meta.get("source", ""),
                "title": meta.get("title", "Untitled"),
                "score": round(score, 4),
                "doc_id": meta.get("doc_id", ""),
            })

    # Determine confidence
    if not results:
        confidence = "low"
    elif results[0]["score"] >= THRESHOLD_HIGH:
        confidence = "high"
    elif results[0]["score"] >= THRESHOLD_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"

    return {"confidence": confidence, "results": results}
