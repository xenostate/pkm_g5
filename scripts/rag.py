#!/usr/bin/env python3
"""
PKM RAG module: Q&A, summarization, knowledge connections, and knowledge base management.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI
from scripts.retriever import retrieve
from scripts.indexer import get_chroma_collection, get_model

# ── Config ──────────────────────────────────────────────────────────────────

RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"

# ── OpenAI client ──────────────────────────────────────────────────────────

_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# ── Knowledge Base JSON ────────────────────────────────────────────────────

_kb_lock = threading.Lock()


def _empty_kb() -> dict:
    return {
        "documents": {},
        "qa_history": [],
        "stats": {
            "total_documents": 0,
            "total_chunks": 0,
            "total_questions": 0,
        },
    }


def load_kb() -> dict:
    """Load knowledge base from disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _empty_kb()


def save_kb(kb: dict):
    """Save knowledge base to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def _update_stats(kb: dict):
    """Recalculate stats from current data."""
    kb["stats"]["total_documents"] = len(kb["documents"])
    kb["stats"]["total_chunks"] = sum(
        d.get("chunk_count", 0) for d in kb["documents"].values()
    )
    kb["stats"]["total_questions"] = len(kb["qa_history"])


def add_document_to_kb(kb: dict, doc_info: dict, summary: str = ""):
    """Add a document entry to the knowledge base."""
    with _kb_lock:
        kb["documents"][doc_info["doc_id"]] = {
            "id": doc_info["doc_id"],
            "title": doc_info["title"],
            "source_type": doc_info.get("source_type", "unknown"),
            "source": doc_info.get("source"),
            "added_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": doc_info.get("chunk_count", 0),
            "text_length": doc_info.get("text_length", 0),
            "summary": summary,
            "connections": [],
        }
        _update_stats(kb)
        save_kb(kb)


def remove_document_from_kb(kb: dict, doc_id: str):
    """Remove a document and its connections from the KB."""
    with _kb_lock:
        kb["documents"].pop(doc_id, None)
        # Remove connections referencing this doc
        for doc in kb["documents"].values():
            doc["connections"] = [
                c for c in doc.get("connections", []) if c.get("doc_id") != doc_id
            ]
        _update_stats(kb)
        save_kb(kb)


def add_qa_to_kb(kb: dict, question: str, answer: str, sources: list):
    """Add a Q&A entry to the knowledge base."""
    import uuid
    with _kb_lock:
        kb["qa_history"].append({
            "id": f"q_{uuid.uuid4().hex[:8]}",
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        _update_stats(kb)
        save_kb(kb)


# ── RAG Q&A ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable PKM (Personal Knowledge Management) assistant that answers questions using ONLY the provided source chunks from the user's document collection.

Rules:
1. Answer ONLY from the sources below. If the sources don't contain the answer, say "I don't have enough information from your documents to answer this."
2. Keep answers concise and factual.
3. Reference which document(s) your answer comes from.
4. Answer in the same language as the user's question.
5. If the user's question is a follow-up referencing a previous message, use the conversation history to understand their intent.
"""


def build_context(retrieval_result: dict) -> str:
    """Format retrieved chunks into numbered context block."""
    lines = []
    for i, r in enumerate(retrieval_result["results"], 1):
        lines.append(f"[{i}] (score: {r['score']}) {r['title']}")
        lines.append(r["chunk_text"])
        lines.append("")
    return "\n".join(lines)


def answer_question(query: str, conversation_history: list = None, model=None) -> dict:
    """
    RAG Q&A: retrieve relevant chunks, generate answer with OpenAI.

    Returns: {answer, sources, confidence}
    """
    retrieval = retrieve(query, top_k=TOP_K, model=model)
    context = build_context(retrieval)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    user_msg = f"""Source chunks from your documents:
{context}

Question: {query}"""

    messages.append({"role": "user", "content": user_msg})

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=RAG_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )

    answer = resp.choices[0].message.content

    sources = [
        {"title": r["title"], "doc_id": r["doc_id"], "score": r["score"]}
        for r in retrieval["results"]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": retrieval["confidence"],
    }


# ── Summarization ──────────────────────────────────────────────────────────

def summarize_document(text: str, title: str) -> str:
    """Generate a summary of a document using OpenAI."""
    # Truncate very long texts to ~8000 words to stay within context limits
    words = text.split()
    if len(words) > 8000:
        text = " ".join(words[:8000]) + "\n\n[...truncated for summarization]"

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise summarizer. Create a clear, structured summary of the document. Identify key concepts, main arguments, and conclusions. Use 3-5 paragraphs.",
            },
            {
                "role": "user",
                "content": f"Summarize the following document titled '{title}':\n\n{text}",
            },
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return resp.choices[0].message.content


# ── Knowledge Connections ──────────────────────────────────────────────────

def find_connections(doc_id: str, kb: dict, model=None) -> list[dict]:
    """
    Find connections between a document and all other documents.
    Uses embedding similarity between chunk centroids.
    Returns list of {doc_id, title, similarity, description}.
    """
    if model is None:
        model = get_model()

    collection = get_chroma_collection()

    # Get target document chunks
    target_results = collection.get(
        where={"doc_id": doc_id},
        include=["embeddings"],
    )
    if not target_results["ids"]:
        return []

    import numpy as np
    target_embeddings = np.array(target_results["embeddings"])
    target_centroid = target_embeddings.mean(axis=0)
    target_centroid = target_centroid / np.linalg.norm(target_centroid)

    connections = []
    target_title = kb["documents"].get(doc_id, {}).get("title", "")
    target_summary = kb["documents"].get(doc_id, {}).get("summary", "")

    for other_id, other_doc in kb["documents"].items():
        if other_id == doc_id:
            continue

        other_results = collection.get(
            where={"doc_id": other_id},
            include=["embeddings"],
        )
        if not other_results["ids"]:
            continue

        other_embeddings = np.array(other_results["embeddings"])
        other_centroid = other_embeddings.mean(axis=0)
        other_centroid = other_centroid / np.linalg.norm(other_centroid)

        similarity = float(np.dot(target_centroid, other_centroid))

        if similarity >= 0.3:
            connections.append({
                "doc_id": other_id,
                "title": other_doc["title"],
                "similarity": round(similarity, 4),
                "other_summary": other_doc.get("summary", ""),
            })

    # Sort by similarity descending
    connections.sort(key=lambda x: x["similarity"], reverse=True)

    # Use LLM to describe top connections
    client = get_openai_client()
    for conn in connections[:5]:  # Only describe top 5 to save API calls
        try:
            resp = client.chat.completions.create(
                model=RAG_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "In one concise sentence, describe how these two documents are related based on their summaries.",
                    },
                    {
                        "role": "user",
                        "content": f"Document 1: '{target_title}'\nSummary: {target_summary[:500]}\n\nDocument 2: '{conn['title']}'\nSummary: {conn['other_summary'][:500]}",
                    },
                ],
                temperature=0.1,
                max_tokens=100,
            )
            conn["description"] = resp.choices[0].message.content
        except Exception:
            conn["description"] = f"Related to {conn['title']} (similarity: {conn['similarity']})"

    # Clean up temp field
    for conn in connections:
        conn.pop("other_summary", None)

    return connections


def refresh_all_connections(kb: dict, model=None) -> dict:
    """Recompute connections for all documents. Returns updated KB."""
    for doc_id in list(kb["documents"].keys()):
        connections = find_connections(doc_id, kb, model=model)
        kb["documents"][doc_id]["connections"] = connections
    save_kb(kb)
    return kb
