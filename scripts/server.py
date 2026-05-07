#!/usr/bin/env python3
"""
PKM API Server — FastAPI service for Personal Knowledge Management.

Endpoints:
    GET  /                          — serve dashboard
    POST /api/documents/upload-pdf  — upload & ingest PDF
    POST /api/documents/add-url     — fetch & ingest URL
    POST /api/documents/add-text    — ingest text note
    GET  /api/documents             — list all documents
    GET  /api/documents/{doc_id}    — document details
    DELETE /api/documents/{doc_id}  — delete document
    POST /api/chat                  — RAG Q&A
    GET  /api/chat/history          — Q&A history
    POST /api/search                — natural language search
    GET  /api/connections           — all document connections
    POST /api/connections/refresh   — recompute connections
    GET  /api/knowledge-base        — full KB JSON
    GET  /api/stats                 — stats
    GET  /health                    — status check

Start:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8090
"""

import asyncio
import datetime
import logging
import threading
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from sympy import content
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.indexer import get_model, get_chroma_collection, ingest_pdf, ingest_url, ingest_text, delete_document, list_documents, get_document_chunks
from scripts.retriever import retrieve
from scripts.rag import (
    load_kb, add_document_to_kb, remove_document_from_kb, add_qa_to_kb,
    answer_question, save_kb, summarize_document, extract_concepts, generate_document_questions,
    get_questions_by_document, pick_next_question, record_question_result,
    refresh_missing_concepts, refresh_all_connections,
    get_openai_client, RAG_MODEL, _safe_json_value,_kb_lock,
)

# ── Config ──────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pkm-server")

# ── Globals ─────────────────────────────────────────────────────────────────

embed_model = None
kb = None
start_time = 0.0

# Session history for conversation context
_session_history: dict[str, list[dict]] = {}
_session_lock = threading.Lock()
SESSION_HISTORY_LIMIT = 5


def _sync_knowledge_map():
    """Keep persisted knowledge-map data up to date after document changes."""
    global kb
    refresh_missing_concepts(kb, get_document_chunks)
    kb = refresh_all_connections(kb, embed_model, describe_with_llm=False)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, kb, start_time
    start_time = time.time()

    log.info("Loading embedding model...")
    embed_model = get_model()
    log.info("Embedding model ready.")

    log.info("Initializing ChromaDB...")
    get_chroma_collection()
    log.info("ChromaDB ready.")

    log.info("Loading knowledge base...")
    kb = load_kb()
    log.info(f"Knowledge base loaded: {kb['stats']['total_documents']} documents, {kb['stats']['total_questions']} Q&A entries.")
    if kb["documents"]:
        log.info("Syncing persisted knowledge map data...")
        await asyncio.to_thread(_sync_knowledge_map)
        log.info("Knowledge map sync complete.")

    yield

    log.info("Shutting down PKM server.")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="PKM Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class TextRequest(BaseModel):
    text: str
    title: str


class UrlRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class QuestionGenerateRequest(BaseModel):
    doc_id: str
    question_type: str = "multiple_choice"


class QuestionAnswerRequest(BaseModel):
    session_id: str = "default"
    doc_id: str
    question_id: str
    selected_index: int


class ShortAnswerRequest(BaseModel):
    session_id: str = "default"
    doc_id: str
    question_id: str
    answer_text: str


class QuestionNextRequest(BaseModel):
    session_id: str = "default"
    doc_id: str | None = None


# ── Routes: Static / Health ─────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    doc_count = get_chroma_collection().count()
    return {
        "status": "ok",
        "model_loaded": embed_model is not None,
        "documents": doc_count,
        "uptime_seconds": round(time.time() - start_time),
    }


# ── Routes: Document Management ─────────────────────────────────────────────

@app.post("/api/documents/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global kb
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are accepted."}, status_code=400)

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 20MB)."}, status_code=400)

    try:
        result = await asyncio.to_thread(ingest_pdf, file_bytes, file.filename, embed_model)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Auto-summarize
    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    try:
        concepts = await asyncio.to_thread(extract_concepts, result["full_text"])
    except Exception as e:
        log.warning(f"Concept extraction failed: {e}")
        concepts = []

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "pdf",
        "source": file.filename,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary, concepts=concepts)
    await asyncio.to_thread(_sync_knowledge_map)

    log.info(f"Ingested PDF: {file.filename} -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
        "concepts": concepts,
    }


@app.post("/api/documents/add-url")
async def add_url(req: UrlRequest):
    global kb
    try:
        result = await asyncio.to_thread(ingest_url, req.url, embed_model)
    except (ValueError, Exception) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    try:
        concepts = await asyncio.to_thread(extract_concepts, result["full_text"])
    except Exception as e:
        log.warning(f"Concept extraction failed: {e}")
        concepts = []

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "url",
        "source": req.url,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary, concepts=concepts)
    await asyncio.to_thread(_sync_knowledge_map)

    log.info(f"Ingested URL: {req.url} -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
        "concepts": concepts,
    }


@app.post("/api/documents/add-text")
async def add_text(req: TextRequest):
    global kb
    try:
        result = await asyncio.to_thread(ingest_text, req.text, req.title, embed_model)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    try:
        concepts = await asyncio.to_thread(extract_concepts, result["full_text"])
    except Exception as e:
        log.warning(f"Concept extraction failed: {e}")
        concepts = []

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "text",
        "source": None,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary, concepts=concepts)
    await asyncio.to_thread(_sync_knowledge_map)

    log.info(f"Ingested text: '{req.title}' -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
        "concepts": concepts,
    }


@app.get("/api/documents")
async def get_documents():
    global kb
    docs = []
    for doc_id, doc in kb["documents"].items():
        docs.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "source_type": doc["source_type"],
            "source": doc.get("source"),
            "added_at": doc.get("added_at"),
            "chunk_count": doc.get("chunk_count", 0),
            "summary": doc.get("summary", ""),
            "connection_count": len(doc.get("connections", [])),
        })
    return {"documents": docs}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    global kb
    doc = kb["documents"].get(doc_id)
    if not doc:
        return JSONResponse({"error": "Document not found."}, status_code=404)
    return doc


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    global kb
    if doc_id not in kb["documents"]:
        return JSONResponse({"error": "Document not found."}, status_code=404)

    await asyncio.to_thread(delete_document, doc_id)
    remove_document_from_kb(kb, doc_id)
    await asyncio.to_thread(_sync_knowledge_map)

    log.info(f"Deleted document: {doc_id}")
    return {"status": "deleted", "doc_id": doc_id}


# ── Routes: Chat / Q&A ──────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    global kb

    # Get/update session history
    with _session_lock:
        history = _session_history.get(req.session_id, [])

    result = await asyncio.to_thread(
        answer_question, req.message, history, embed_model, kb
    )

    # Update session history
    with _session_lock:
        if req.session_id not in _session_history:
            _session_history[req.session_id] = []
        _session_history[req.session_id].append({"role": "user", "content": req.message})
        _session_history[req.session_id].append({"role": "assistant", "content": result["answer"]})
        # Keep only last N exchanges
        if len(_session_history[req.session_id]) > SESSION_HISTORY_LIMIT * 2:
            _session_history[req.session_id] = _session_history[req.session_id][-(SESSION_HISTORY_LIMIT * 2):]

    # Save to KB
    add_qa_to_kb(kb, req.message, result["answer"], result["sources"])

    return result


@app.get("/api/chat/history")
async def chat_history():
    global kb
    return {"history": kb.get("qa_history", [])}


@app.get("/api/questions")
async def get_questions():
    global kb
    return {"documents": get_questions_by_document(kb)}


@app.post("/api/questions/generate")
async def generate_questions(req: QuestionGenerateRequest):
    global kb
    if req.doc_id not in kb["documents"]:
        return JSONResponse({"error": "Document not found."}, status_code=404)

    questions = await asyncio.to_thread(
        generate_document_questions,
        req.doc_id,
        kb,
        get_document_chunks,
        req.question_type
    )
    return {"doc_id": req.doc_id, "questions": questions}


@app.post("/api/questions/answer")
async def answer_question_card(req: QuestionAnswerRequest):
    global kb
    try:
        result = await asyncio.to_thread(
            record_question_result,
            kb,
            req.session_id,
            req.doc_id,
            req.question_id,
            req.selected_index,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    next_question = await asyncio.to_thread(pick_next_question, kb, req.session_id, req.doc_id)
    return {"result": result, "next_question": next_question}

@app.post("/api/questions/short-answer")
async def answer_short_question(req: ShortAnswerRequest):
    global kb

    doc = kb["documents"].get(req.doc_id)
    if not doc:
        return JSONResponse({"error": "Document not found."}, status_code=404)

    question = next(
        (q for q in doc.get("questions", []) if q.get("id") == req.question_id),
        None
    )

    if not question:
        return JSONResponse({"error": "Question not found."}, status_code=404)

    sample_answer = question.get("sample_answer", "")

    prompt = f"""
You are grading a student's short-answer response.

Question:
{question.get("prompt", "")}

Expected answer:
{sample_answer}

Student answer:
{req.answer_text}

Evaluate the student's answer.

Return JSON only in this format:
{{
  "score": 0-100,
  "feedback": "Detailed feedback",
  "mastery": 0.0-1.0
}}
"""

    client = get_openai_client()

    resp = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )

    content = resp.choices[0].message.content or "{}"
    parsed = _safe_json_value(content)

    if not isinstance(parsed, dict):
        parsed = {}

    result = {
        "score": parsed.get("score", 0),
        "feedback": parsed.get("feedback", ""),
        "mastery": parsed.get("mastery", 0.0),
        "sample_answer": sample_answer,
    }
    history_entry = {
        "question_id": req.question_id,
        "question_prompt": question.get("prompt", ""),
        "user_answer": req.answer_text,
        "score": result["score"],
        "feedback": result["feedback"],
        "timestamp": datetime.now().isoformat(),
    }
    with _kb_lock:
        sessions = kb.setdefault("question_sessions", {})
        session = sessions.setdefault(req.session_id, {})
        history = session.setdefault("short_answer_history", [])

        history.append(history_entry)

        save_kb(kb)

    return {"result": result}

@app.get("/api/questions/history")
async def get_answer_history(session_id: str = "default"):
    global kb

    short_history = (
        kb.get("question_sessions", {})
        .get(session_id, {})
        .get("short_answer_history", [])
    )

    multiple_history = (
        kb.get("study_progress", {})
        .get(session_id, {})
        .get("history", [])
    )

    history = short_history + multiple_history

    history.sort(
        key=lambda item: item.get("timestamp", ""),
        reverse=True
    )

    return {
        "session_id": session_id,
        "history": history
    }




@app.post("/api/questions/next")
async def next_question(req: QuestionNextRequest):
    global kb
    question = await asyncio.to_thread(pick_next_question, kb, req.session_id, req.doc_id)
    return {"question": question}


# ── Routes: Search ───────────────────────────────────────────────────────────

@app.post("/api/search")
async def search(req: SearchRequest):
    result = await asyncio.to_thread(retrieve, req.query, req.top_k, embed_model)
    return result


# ── Routes: Connections ──────────────────────────────────────────────────────

@app.get("/api/connections")
async def get_connections():
    global kb
    all_connections = []
    for doc_id, doc in kb["documents"].items():
        for conn in doc.get("connections", []):
            all_connections.append({
                "from_doc_id": doc_id,
                "from_title": doc["title"],
                "to_doc_id": conn["doc_id"],
                "to_title": conn["title"],
                "similarity": conn.get("similarity", 0),
                "description": conn.get("description", ""),
            })
    return {"connections": all_connections}


@app.post("/api/connections/refresh")
async def refresh_connections():
    global kb
    concept_updates = await asyncio.to_thread(refresh_missing_concepts, kb, get_document_chunks)
    kb = await asyncio.to_thread(refresh_all_connections, kb, embed_model, True)
    return {"status": "refreshed", "document_count": len(kb["documents"]), "concepts_backfilled": concept_updates}


# ── Routes: Knowledge Base / Stats ───────────────────────────────────────────

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    global kb
    return kb


@app.get("/api/stats")
async def get_stats():
    global kb
    return kb.get("stats", {})


# ── Mount static files (must be last) ───────────────────────────────────────

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
