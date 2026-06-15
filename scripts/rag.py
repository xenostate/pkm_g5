#!/usr/bin/env python3
"""
PKM RAG module: Q&A, summarization, knowledge connections, and knowledge base management.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI
from scripts.domain_context import domain_data_dir
from scripts.retriever import retrieve
from scripts.indexer import get_chroma_collection, get_model

# ── Config ──────────────────────────────────────────────────────────────────

RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-5.4-mini")
TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
LLM_CONTEXT_CHUNKS = int(os.environ.get("RAG_CONTEXT_CHUNKS", "8"))

def _kb_path() -> Path:
    data_dir = domain_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "knowledge_base.json"

# ── OpenAI client ──────────────────────────────────────────────────────────

_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# Reasoning models (GPT-5 family, o-series) reject `temperature` and `max_tokens`;
# they need `max_completion_tokens` and spend hidden reasoning tokens, so short
# output limits must be padded. This wrapper makes one call shape work everywhere.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def chat_complete(messages, *, model: str | None = None, temperature: float | None = None,
                  max_tokens: int | None = None, response_format: dict | None = None):
    """Model-agnostic chat completion. Translates params for reasoning models."""
    client = get_openai_client()
    model = model or RAG_MODEL
    kwargs: dict = {"model": model, "messages": messages}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if _is_reasoning_model(model):
        # no custom temperature; reserve headroom so reasoning doesn't eat the answer
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max(max_tokens, 2048) + 1024
        # 'none'/'low' minimize hidden reasoning for these fast extraction tasks
        # (gpt-5.4 rejects 'minimal'; older gpt-5 used it). 'low' is the safe floor.
        kwargs["reasoning_effort"] = "low"
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)


# ── Knowledge Base JSON ────────────────────────────────────────────────────

_kb_lock = threading.Lock()


def _empty_kb() -> dict:
    return {
        "documents": {},
        "links": [],
        "qa_history": [],
        "study_progress": {},
        "stats": {
            "total_documents": 0,
            "total_chunks": 0,
            "total_questions": 0,
        },
    }


def load_kb() -> dict:
    """Load knowledge base from disk."""
    kb_path = _kb_path()
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
            kb.setdefault("documents", {})
            kb.setdefault("qa_history", [])
            kb.setdefault("links", [])
            kb.setdefault("study_progress", {})
            kb.setdefault("stats", {
                "total_documents": 0,
                "total_chunks": 0,
                "total_questions": 0,
            })
            for doc in kb["documents"].values():
                doc.setdefault("concepts", [])
                doc.setdefault("connections", [])
                doc.setdefault("questions", [])
            return kb
    return _empty_kb()


def save_kb(kb: dict):
    """Save knowledge base to disk."""
    kb_path = _kb_path()
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def _update_stats(kb: dict):
    """Recalculate stats from current data."""
    kb["stats"]["total_documents"] = len(kb["documents"])
    kb["stats"]["total_chunks"] = sum(
        d.get("chunk_count", 0) for d in kb["documents"].values()
    )
    kb["stats"]["total_questions"] = len(kb["qa_history"])


def _safe_json_list(raw_text: str) -> list[str]:
    """Parse a JSON list from a model response and coerce it to clean strings."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

        if data is None:
            lines = []
            for line in raw_text.splitlines():
                cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                if cleaned:
                    lines.append(cleaned.strip("\"'"))
            data = lines

    if not isinstance(data, list):
        return []

    concepts = []
    seen = set()
    for item in data:
        if not isinstance(item, str):
            continue
        concept = item.strip()
        if not concept:
            continue
        normalized = concept.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        concepts.append(concept)
    return concepts[:10]


VALID_CATEGORIES = {"is-a", "part-of", "uses", "contrasts", "causes", "related"}
VALID_ENTITY_TYPES = {"concept", "person", "tech", "org", "event", "other"}


def parse_knowledge_response(raw_text: str) -> dict:
    """Validate an LLM knowledge-extraction response into {entities, triples}."""
    data = _safe_json_value(raw_text) or {}
    entities, triples = [], []
    for e in data.get("entities", []) if isinstance(data, dict) else []:
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].strip():
            etype = e.get("type") if e.get("type") in VALID_ENTITY_TYPES else "concept"
            entities.append({"name": e["name"].strip(), "type": etype})
    for t in data.get("triples", []) if isinstance(data, dict) else []:
        if not isinstance(t, dict):
            continue
        s, p, o = (t.get(k, "") for k in ("subject", "predicate", "object"))
        if all(isinstance(x, str) and x.strip() for x in (s, p, o)):
            cat = t.get("category") if t.get("category") in VALID_CATEGORIES else "related"
            triples.append({"subject": s.strip(), "predicate": p.strip(),
                            "object": o.strip(), "category": cat})
    return {"entities": entities[:15], "triples": triples[:20]}


def extract_knowledge(text: str, title: str = "", model_name: str | None = None) -> dict:
    """Extract typed entities and SPO triples from a document (one LLM call)."""
    excerpt = text[:3000]
    prompt = f"""Extract a small knowledge graph from this document.

Return ONLY JSON: {{"entities": [{{"name": "...", "type": "concept|person|tech|org|event|other"}}],
"triples": [{{"subject": "...", "predicate": "short verb phrase", "object": "...",
"category": "is-a|part-of|uses|contrasts|causes|related"}}]}}

Rules: 5-12 entities, 5-15 triples. Subjects/objects must be entity names.
Keep names short and canonical. Use the same language as the document.

Document title: {title}
Text:
{excerpt}"""
    resp = chat_complete(
        [{"role": "user", "content": prompt}],
        model=model_name or RAG_MODEL,
        temperature=0.1,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    return parse_knowledge_response(resp.choices[0].message.content or "{}")


def extract_concepts(text: str, model_name: str | None = None) -> list[str]:
    """Extract key concepts from a document (entity names from extract_knowledge)."""
    knowledge = extract_knowledge(text, model_name=model_name)
    return [e["name"] for e in knowledge["entities"]][:10]


def refresh_missing_concepts(kb: dict, get_chunks_fn, model_name: str | None = None) -> int:
    """Backfill concepts/entities/triples for documents that predate extraction."""
    updated = 0
    doc_ids_with_triples = {t.get("doc_id") for t in kb.get("triples", [])}
    with _kb_lock:
        for doc_id, doc in kb["documents"].items():
            has_concepts = bool(doc.get("concepts"))
            has_knowledge = doc_id in doc_ids_with_triples
            if has_concepts and has_knowledge:
                continue

            chunks = get_chunks_fn(doc_id)
            text = "\n\n".join(chunks[:4]).strip()
            if not text:
                text = doc.get("summary", "")
            if not text:
                continue

            try:
                knowledge = extract_knowledge(text, title=doc.get("title", ""), model_name=model_name)
            except Exception:
                continue
            if not knowledge["entities"] and not knowledge["triples"]:
                continue

            if not has_concepts:
                doc["concepts"] = [e["name"] for e in knowledge["entities"]][:10]
            merge_knowledge_into_kb(kb, doc_id, knowledge, doc.get("added_at"))
            updated += 1

        if updated:
            rebuild_concept_links(kb)
            save_kb(kb)

    return updated


def rebuild_concept_links(kb: dict):
    """Build concept-level links between documents from stored concepts."""
    links = []
    doc_ids = list(kb["documents"].keys())
    for from_doc_id in doc_ids:
        from_doc = kb["documents"][from_doc_id]
        from_concepts = {
            concept.strip().lower(): concept
            for concept in from_doc.get("concepts", [])
            if isinstance(concept, str) and concept.strip()
        }
        if not from_concepts:
            continue
        for to_doc_id in doc_ids:
            if from_doc_id == to_doc_id:
                continue
            to_doc = kb["documents"][to_doc_id]
            to_concepts = {
                concept.strip().lower(): concept
                for concept in to_doc.get("concepts", [])
                if isinstance(concept, str) and concept.strip()
            }
            shared_keys = sorted(set(from_concepts) & set(to_concepts))
            if not shared_keys:
                continue
            links.append({
                "from": from_doc_id,
                "to": to_doc_id,
                "concept": [from_concepts[key] for key in shared_keys],
            })
    kb["links"] = links


def add_document_to_kb(kb: dict, doc_info: dict, summary: str = "", concepts: list[str] | None = None,
                       knowledge: dict | None = None):
    """Add a document entry (and its extracted entities/triples) to the knowledge base."""
    with _kb_lock:
        now = datetime.now(timezone.utc).isoformat()
        doc_id = doc_info["doc_id"]
        kb["documents"][doc_id] = {
            "id": doc_id,
            "title": doc_info["title"],
            "source_type": doc_info.get("source_type", "unknown"),
            "source": doc_info.get("source"),
            "added_at": now,
            "chunk_count": doc_info.get("chunk_count", 0),
            "text_length": doc_info.get("text_length", 0),
            "summary": summary,
            "concepts": concepts or [],
            "connections": [],
            "questions": [],
        }
        if knowledge:
            merge_knowledge_into_kb(kb, doc_id, knowledge, now)
        rebuild_concept_links(kb)
        _update_stats(kb)
        save_kb(kb)


def merge_knowledge_into_kb(kb: dict, doc_id: str, knowledge: dict, created_at: str | None = None):
    """Merge extracted entities/triples for a document into the KB (idempotent per doc)."""
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    entities = kb.setdefault("entities", {})
    for ent in knowledge.get("entities", []):
        name = ent.get("name", "").strip()
        if not name:
            continue
        entry = entities.setdefault(name, {"type": ent.get("type", "concept"),
                                           "doc_ids": [], "created_at": created_at})
        if doc_id not in entry["doc_ids"]:
            entry["doc_ids"].append(doc_id)

    triples = kb.setdefault("triples", [])
    existing = {(t["subject"], t["predicate"], t["object"], t["doc_id"]) for t in triples}
    for t in knowledge.get("triples", []):
        key = (t["subject"], t["predicate"], t["object"], doc_id)
        if key in existing:
            continue
        triples.append({**t, "doc_id": doc_id, "created_at": created_at})


def remove_document_from_kb(kb: dict, doc_id: str):
    """Remove a document, its connections, and its entities/triples from the KB."""
    with _kb_lock:
        kb["documents"].pop(doc_id, None)
        # Remove connections referencing this doc
        for doc in kb["documents"].values():
            doc["connections"] = [
                c for c in doc.get("connections", []) if c.get("doc_id") != doc_id
            ]
        # Drop entity memberships and triples sourced from this doc
        entities = kb.get("entities", {})
        for name in list(entities):
            entities[name]["doc_ids"] = [d for d in entities[name].get("doc_ids", []) if d != doc_id]
            if not entities[name]["doc_ids"]:
                entities.pop(name)
        kb["triples"] = [t for t in kb.get("triples", []) if t.get("doc_id") != doc_id]
        rebuild_concept_links(kb)
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

def build_context(results: list[dict]) -> str:
    """Format retrieved chunks into numbered context block."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] (score: {r['score']}) {r['title']}")
        lines.append(r["chunk_text"])
        lines.append("")
    return "\n".join(lines)


def group_results_by_document(results: list[dict]) -> dict[str, list[dict]]:
    """Group retrieved chunks by doc_id while preserving retrieval order."""
    grouped: dict[str, list[dict]] = {}
    for result in results:
        doc_id = result.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append(result)
    return grouped


def _infer_related_reason(chunk_text: str, score: float) -> str:
    """Generate a compact explanation for why a document is related."""
    lowered = chunk_text.lower()
    if any(term in lowered for term in [" is defined as ", " refers to ", " definition", " means "]):
        return "definition"
    if any(term in lowered for term in ["example", "for instance", "such as"]):
        return "example"
    if any(term in lowered for term in ["because", "therefore", "in summary", "overview", "explains"]):
        return "expanded explanation"
    if score >= 0.8:
        return "strong concept overlap"
    return "matching concept"


def build_related_docs(retrieval_result: dict) -> list[dict]:
    """Summarize other retrieved documents that discuss the same concept."""
    grouped = group_results_by_document(retrieval_result["results"])
    ranked_docs = sorted(
        grouped.items(),
        key=lambda item: max(chunk["score"] for chunk in item[1]),
        reverse=True,
    )

    if not ranked_docs:
        return []

    primary_doc_id = ranked_docs[0][0]
    related_docs = []
    for doc_id, chunks in ranked_docs:
        if doc_id == primary_doc_id:
            continue
        top_chunk = max(chunks, key=lambda chunk: chunk["score"])
        related_docs.append({
            "doc": top_chunk.get("source") or top_chunk.get("title") or doc_id,
            "doc_id": doc_id,
            "reason": _infer_related_reason(top_chunk.get("chunk_text", ""), top_chunk.get("score", 0.0)),
        })
        if len(related_docs) == 3:
            break

    return related_docs


def extract_top_docs(retrieval_result: dict, limit: int = 3) -> list[str]:
    """Return the top document ids from grouped retrieval results."""
    grouped = group_results_by_document(retrieval_result["results"])
    ranked_docs = sorted(
        grouped.items(),
        key=lambda item: max(chunk["score"] for chunk in item[1]),
        reverse=True,
    )
    return [doc_id for doc_id, _chunks in ranked_docs[:limit]]


def build_graph_connections(kb: dict | None, top_docs: list[str]) -> list[dict]:
    """Return concept graph links connected to the top retrieved documents."""
    if not kb:
        return []

    graph_links = []
    for link in kb.get("links", []):
        if link.get("from") not in top_docs:
            continue
        from_doc = kb["documents"].get(link["from"], {})
        to_doc = kb["documents"].get(link["to"], {})
        graph_links.append({
            "from": link["from"],
            "from_title": from_doc.get("title", link["from"]),
            "to": link["to"],
            "to_title": to_doc.get("title", link["to"]),
            "concept": link.get("concept", []),
        })
    return graph_links


_GLOBAL_QUESTION_RE = re.compile(
    r"전체|전반|모든\s*(문서|내용|자료)|주제들|overall|themes?\b|all\s+(my\s+)?documents|across\s+(all\s+)?documents",
    re.IGNORECASE,
)


def is_global_question(query: str) -> bool:
    """Heuristic: does the question ask about the whole knowledge base?"""
    return len(query) <= 200 and bool(_GLOBAL_QUESTION_RE.search(query))


def answer_global_question(query: str, kb: dict, embed_fn=None) -> dict:
    """LazyGraphRAG-style global answer: summarize on demand from the concept
    communities instead of retrieving individual chunks."""
    from scripts.graph import build_graph_payload

    payload = build_graph_payload(kb, embed_fn=embed_fn)

    members: dict[int, list[dict]] = {}
    for node in payload["nodes"]:
        members.setdefault(node.get("community", -1), []).append(node)

    sections = []
    rep_docs: list[tuple[str, str]] = []  # (doc_id, title) one per community
    for comm in payload["communities"][:8]:
        nodes = members.get(comm["id"], [])
        if len(nodes) < 2:
            continue
        concepts = [n["label"] for n in sorted(nodes, key=lambda n: -n.get("centrality", 0))[:8]]
        doc_titles, doc_ids = [], []
        for n in nodes:
            for doc_id, title in zip(n.get("doc_ids", []), n.get("docs", [])):
                if title not in doc_titles:
                    doc_titles.append(title)
                    doc_ids.append(doc_id)
        sections.append(f"## {comm['label']}\nConcepts: {', '.join(concepts)}\n"
                        f"Documents: {', '.join(doc_titles[:5])}")
        if doc_ids:
            rep_docs.append((doc_ids[0], doc_titles[0]))

    summaries = []
    for doc_id, doc in list(kb.get("documents", {}).items())[:10]:
        if doc.get("summary"):
            summaries.append(f"[{doc.get('title', doc_id)}] {doc['summary'][:600]}")

    context = ("# Knowledge map (concept clusters)\n" + "\n\n".join(sections) +
               "\n\n# Document summaries\n" + "\n\n".join(summaries))[:9000]

    prompt = f"""You are a learning assistant. The user asks a question about their whole
knowledge base. Below is a map of their knowledge: concept clusters discovered
across documents, plus per-document summaries. Answer using this map —
describe the main themes, how clusters relate, and where coverage is thin.
Answer in the same language as the user's question.

{context}

Question:
{query}"""

    resp = chat_complete(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )

    seen = set()
    sources = []
    for doc_id, title in rep_docs:
        if doc_id not in seen:
            seen.add(doc_id)
            sources.append({"title": title, "doc_id": doc_id, "score": 1.0})

    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "confidence": "high" if sections else "low",
        "related_docs": [],
        "connections": [],
    }


def answer_question(query: str, conversation_history: list = None, model=None, kb: dict | None = None,
                    embed_fn=None) -> dict:
    """
    RAG Q&A: retrieve relevant chunks, generate answer with OpenAI.
    When kb and embed_fn are given, retrieval scores are fused with
    Personalized PageRank over the knowledge graph (multi-hop boost).

    Returns: {answer, sources, confidence, related_docs, connections}
    """
    # whole-knowledge-base questions answer from the concept map, not chunks
    if kb is not None and len(kb.get("documents", {})) >= 3 and is_global_question(query):
        try:
            return answer_global_question(query, kb, embed_fn=embed_fn)
        except Exception:
            pass  # fall back to chunk retrieval

    retrieval = retrieve(query, top_k=TOP_K, model=model)

    # graph fusion: rescore chunks by PPR document relevance (never fatal)
    if kb is not None and embed_fn is not None and retrieval["results"]:
        try:
            from scripts.graph import ppr_doc_scores
            graph_scores = ppr_doc_scores(kb, embed_fn, query)
            if graph_scores:
                for r in retrieval["results"]:
                    r["score"] = round(0.75 * r["score"] + 0.25 * graph_scores.get(r["doc_id"], 0.0), 4)
                retrieval["results"].sort(key=lambda r: r["score"], reverse=True)
        except Exception:
            pass

    related_docs = build_related_docs(retrieval)
    top_docs = extract_top_docs(retrieval)
    graph_connections = build_graph_connections(kb, top_docs)
    context_results = retrieval["results"][:LLM_CONTEXT_CHUNKS]
    context = build_context(context_results)

    messages = []

    if conversation_history:
        messages.extend(conversation_history)

    user_msg = f"""
You are a learning assistant.

Using the context below:
1. Answer the question clearly
2. Identify if the concept appears in multiple documents
3. Explain how the documents are related
4. Mention if one expands, contradicts, or builds on another
5. Use only the provided context. If the answer is not in the context, say "I don't have enough information from your documents to answer this."
6. Answer in the same language as the user's question

Context:
{context}

Question:
{query}
"""

    messages.append({"role": "user", "content": user_msg})

    resp = chat_complete(messages, temperature=0.1, max_tokens=2000)

    answer = resp.choices[0].message.content

    sources = [
        {"title": r["title"], "doc_id": r["doc_id"], "score": r["score"]}
        for r in retrieval["results"]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": retrieval["confidence"],
        "related_docs": related_docs,
        "connections": graph_connections,
    }


# ── Summarization ──────────────────────────────────────────────────────────

def summarize_document(text: str, title: str) -> str:
    """Generate a summary of a document using OpenAI."""
    # Truncate very long texts to ~8000 words to stay within context limits
    words = text.split()
    if len(words) > 8000:
        text = " ".join(words[:8000]) + "\n\n[...truncated for summarization]"

    resp = chat_complete(
        [
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

def find_connections(doc_id: str, kb: dict, model=None, describe_with_llm: bool = True) -> list[dict]:
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

    if describe_with_llm:
        for conn in connections[:5]:  # Only describe top 5 to save API calls
            try:
                resp = chat_complete(
                    [
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
    else:
        for conn in connections:
            conn["description"] = f"Related to {conn['title']} (similarity: {conn['similarity']})"

    # Clean up temp field
    for conn in connections:
        conn.pop("other_summary", None)

    return connections


def refresh_all_connections(kb: dict, model=None, describe_with_llm: bool = True) -> dict:
    """Recompute connections for all documents. Returns updated KB."""
    for doc_id in list(kb["documents"].keys()):
        connections = find_connections(doc_id, kb, model=model, describe_with_llm=describe_with_llm)
        kb["documents"][doc_id]["connections"] = connections
    save_kb(kb)
    return kb


def _safe_json_value(raw_text: str):
    """Best-effort JSON parse for arrays or objects returned by the model."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = raw_text.find(opener)
        end = raw_text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def generate_document_questions(doc_id: str, kb: dict, get_chunks_fn, question_type: str = "multiple_choice", model_name: str | None = None) -> list[dict]:
    """Generate study questions (multiple-choice or short-answer) for a document and persist them in the KB."""
    doc = kb["documents"].get(doc_id)
    if not doc:
        raise ValueError("Document not found.")

    summary = doc.get("summary", "")
    concepts = doc.get("concepts", [])
    chunks = get_chunks_fn(doc_id)
    excerpt = "\n\n".join(chunks[:3])[:3500]
    
    #- progress와 weak_topics(정답률 50% 미만)을 추가했습니다 - 김동윤
    progress = kb.get("study_progress", {}).get("default", {}).get("topics", {})
    if progress:
        sorted_topics = sorted(
            progress.items(),
            key=lambda x: x[1]["correct"] / max(x[1]["correct"] + x[1]["wrong"], 1)
        )
        weak_topics = [
            topic for topic, stats in sorted_topics
            if stats["correct"] / max(stats["correct"] + stats["wrong"], 1) < 0.5
        ][:3]
    else:
        weak_topics = []

    weak_hint = (
        f"\nThe student is struggling with these topics: {', '.join(weak_topics)}."
        f"\nGenerate at least 3 questions focused on these weak topics."
        if weak_topics else ""
    )

    if question_type == "short_answer":
        prompt = f"""
Create 6 short-answer study questions for one document.
{weak_hint} 
Return JSON only in this shape:
[
  {{
    "prompt": "Question text",
    "sample_answer": "Example ideal answer",
    "explanation": "Why this answer is correct",
    "topic": "single topic label",
    "difficulty": "easy|medium|hard"
  }}
]

Use the document summary, concepts, and excerpt below.
The questions should require explanation and reasoning, not simple memorization.
At least 2 questions should be medium or hard.

Title: {doc.get("title", "")}
Concepts: {json.dumps(concepts, ensure_ascii=False)}

Summary:
{summary[:2500]}

Excerpt:
{excerpt}
"""
    else:
        prompt = f"""
Create 6 multiple-choice study questions for one document.
{weak_hint}
Return JSON only in this shape:
[
  {{
    "prompt": "Question text",
    "options": ["A", "B", "C", "D"],
    "answer_index": 0,
    "explanation": "Why the answer is correct",
    "topic": "single topic label",
    "difficulty": "easy|medium|hard"
  }}
]

Use the document summary, concepts, and excerpt below. Make the questions varied.
At least 2 questions should be medium or hard.

Title: {doc.get("title", "")}
Concepts: {json.dumps(concepts, ensure_ascii=False)}

Summary:
{summary[:2500]}

Excerpt:
{excerpt}
"""

    resp = chat_complete(
        [{"role": "user", "content": prompt}],
        model=model_name or RAG_MODEL,
        temperature=0.3,
        max_tokens=1800,
    )
    content = resp.choices[0].message.content or "[]"
    parsed = _safe_json_value(content)

    questions = []
    if isinstance(parsed, list):
        for index, item in enumerate(parsed, 1):
            if not isinstance(item, dict):
                continue
            
            prompt_text = str(item.get("prompt", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            topic = str(item.get("topic", "")).strip() or (concepts[0] if concepts else "Core concept")
            difficulty = str(item.get("difficulty", "medium")).strip().lower()

            if question_type == "short_answer":
                sample_answer = str(item.get("sample_answer", "")).strip()

                if not prompt_text or not sample_answer:
                    continue

                questions.append({
                    "id": f"{doc_id}_q_{index}",
                    "type": "short_answer",
                    "prompt": prompt_text,
                    "sample_answer": sample_answer,
                    "explanation": explanation,
                    "topic": topic,
                    "difficulty": difficulty,
                })
            else:
                options = item.get("options")
                answer_index = item.get("answer_index")

                if not prompt_text or not isinstance(options, list) or len(options) < 2:
                    continue
                if not isinstance(answer_index, int) or answer_index < 0 or answer_index >= len(options):
                    continue

                questions.append({
                    "id": f"{doc_id}_q_{index}",
                    "type": "multiple_choice",
                    "prompt": prompt_text,
                    "options": [str(option).strip() for option in options[:4]],
                    "answer_index": answer_index,
                    "explanation": explanation,
                    "topic": topic,
                    "difficulty": difficulty,
                })
            

    if not questions:
        fallback_concepts = concepts[:6] or ["Core concept", "Main idea", "Application"]
        for index, concept in enumerate(fallback_concepts, 1):
            questions.append({
                "id": f"{doc_id}_q_{index}",
                "type": "multiple_choice",
                "prompt": f"Which idea best matches the role of '{concept}' in {doc.get('title', 'this document')}?",
                "options": [
                    f"It is a central idea discussed in the document.",
                    "It is unrelated to the document.",
                    "It is only a bibliography reference.",
                    "It appears only as a file format.",
                ],
                "answer_index": 0,
                "explanation": f"The document highlights {concept} as one of its key study topics.",
                "topic": concept,
                "difficulty": "easy" if index <= 2 else "medium",
            })

    with _kb_lock:
        kb["documents"][doc_id]["questions"] = questions
        save_kb(kb)

    return questions


def get_questions_by_document(kb: dict) -> list[dict]:
    """Return all study questions grouped by document."""
    grouped = []
    for doc_id, doc in kb["documents"].items():
        grouped.append({
            "doc_id": doc_id,
            "title": doc.get("title", doc_id),
            "question_count": len(doc.get("questions", [])),
            "questions": doc.get("questions", []),
            "concepts": doc.get("concepts", []),
        })
    return grouped


def _topic_stats(progress: dict, topic: str) -> dict:
    topic_key = topic.strip().lower() or "general"
    return progress.setdefault(topic_key, {"topic": topic, "correct": 0, "wrong": 0})


def record_question_result(
    kb: dict,
    session_id: str,
    doc_id: str,
    question_id: str,
    selected_index: int,
) -> dict:
    """Store quiz progress and return evaluation plus updated mastery."""
    doc = kb["documents"].get(doc_id)
    if not doc:
        raise ValueError("Document not found.")

    question = next((q for q in doc.get("questions", []) if q.get("id") == question_id), None)
    if not question:
        raise ValueError("Question not found.")

    is_correct = selected_index == question["answer_index"]

    with _kb_lock:
        session_progress = kb["study_progress"].setdefault(session_id, {"topics": {}, "history": []})
        topic_info = _topic_stats(session_progress["topics"], question.get("topic", "general"))
        if is_correct:
            topic_info["correct"] += 1
        else:
            topic_info["wrong"] += 1
        attempts = topic_info["correct"] + topic_info["wrong"]
        mastery = round(topic_info["correct"] / attempts, 2) if attempts else 0.0
        session_progress["history"].append({
            "type": "multiple_choice",
            "doc_id": doc_id,
            "question_id": question_id,
            "question_prompt": question.get("prompt", ""),
            "topic": question.get("topic", "general"),

            "selected_index": selected_index,
            "selected_answer": question.get("options", [])[selected_index],

            "correct_index": question.get("answer_index"),
            "correct_answer": question.get("options", [])[question.get("answer_index", 0)],

            "correct": is_correct,
            "explanation": question.get("explanation", ""),
            "mastery": mastery,

            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_kb(kb)

    

    return {
        "correct": is_correct,
        "answer_index": question["answer_index"],
        "explanation": question.get("explanation", ""),
        "topic": question.get("topic", "general"),
        "mastery": mastery,
    }


def pick_next_question(kb: dict, session_id: str, doc_id: str | None = None) -> dict | None:
    """Recommend the next question, biased toward weaker topics."""
    session_progress = kb.get("study_progress", {}).get(session_id, {"topics": {}, "history": []})
    history_counts = {}
    for item in session_progress.get("history", []):
        history_counts[item["question_id"]] = history_counts.get(item["question_id"], 0) + 1

    candidate_docs = (
        [(doc_id, kb["documents"].get(doc_id))] if doc_id
        else list(kb["documents"].items())
    )

    best_question = None
    best_score = None
    for current_doc_id, doc in candidate_docs:
        if not doc:
            continue
        for question in doc.get("questions", []):
            topic_key = question.get("topic", "general").strip().lower() or "general"
            topic_progress = session_progress.get("topics", {}).get(topic_key, {"correct": 0, "wrong": 0})
            attempts = topic_progress["correct"] + topic_progress["wrong"]
            mastery = topic_progress["correct"] / attempts if attempts else 0.0
            difficulty_bonus = {"easy": 0.0, "medium": 0.15, "hard": 0.3}.get(question.get("difficulty", "medium"), 0.15)
            revisit_penalty = history_counts.get(question["id"], 0) * 0.25
            score = (1.2 - mastery) + difficulty_bonus - revisit_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_question = {
                    "doc_id": current_doc_id,
                    "title": doc.get("title", current_doc_id),
                    **question,
                    "mastery": round(mastery, 2),
                }

    return best_question
