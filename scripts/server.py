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
import logging
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.indexer import get_model, get_chroma_collection, ingest_pdf, ingest_url, ingest_text, delete_document, list_documents, get_document_chunks
from scripts.retriever import retrieve
from scripts.rag import (
    load_kb, save_kb, add_document_to_kb, remove_document_from_kb, add_qa_to_kb,
    answer_question, summarize_document, find_connections, refresh_all_connections,
)
from scripts.podcast import generate_podcast_script, synthesize_speech, PODCAST_DIR

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


class PodcastScriptRequest(BaseModel):
    doc_ids: list[str]
    topic: str = ""


class PodcastSynthesizeRequest(BaseModel):
    script: str


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

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "pdf",
        "source": file.filename,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested PDF: {file.filename} -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
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

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "url",
        "source": req.url,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested URL: {req.url} -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
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

    add_document_to_kb(kb, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "text",
        "source": None,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested text: '{req.title}' -> {result['chunk_count']} chunks")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
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
        answer_question, req.message, history, embed_model
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
    kb = await asyncio.to_thread(refresh_all_connections, kb, embed_model)
    return {"status": "refreshed", "document_count": len(kb["documents"])}


# ── Routes: Knowledge Base / Stats ───────────────────────────────────────────

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    global kb
    return kb


@app.get("/api/stats")
async def get_stats():
    global kb
    return kb.get("stats", {})


# ── Routes: Podcast ──────────────────────────────────────────────────────────

@app.post("/api/podcast/generate")
async def podcast_generate(req: PodcastScriptRequest):
    global kb
    if not req.doc_ids:
        return JSONResponse({"error": "Select at least one document."}, status_code=400)

    try:
        result = await asyncio.to_thread(generate_podcast_script, req.doc_ids, kb, req.topic)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info(f"Podcast script generated: {result['word_count']} words from {result['doc_count']} docs")
    return result


@app.post("/api/podcast/synthesize")
async def podcast_synthesize(req: PodcastSynthesizeRequest):
    if not req.script.strip():
        return JSONResponse({"error": "Script is empty."}, status_code=400)

    try:
        filename = await asyncio.to_thread(synthesize_speech, req.script)
    except ImportError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info(f"Podcast audio synthesized: {filename}")
    return {"filename": filename, "url": f"/podcasts/{filename}"}


# ── Mount static files (must be last) ───────────────────────────────────────

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
PODCAST_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/podcasts", StaticFiles(directory=PODCAST_DIR), name="podcasts")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
