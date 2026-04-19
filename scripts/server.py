#!/usr/bin/env python3
"""
School Helper API Server — FastAPI service with per-course knowledge management.

Endpoints:
    GET  /                                              — serve dashboard
    GET  /health                                        — status check

    Course management:
    GET  /api/courses                                   — list all courses
    POST /api/courses                                   — create course
    PUT  /api/courses/{course_id}                       — update course
    DELETE /api/courses/{course_id}                     — delete course

    Per-course resources:
    POST /api/courses/{cid}/documents/upload-pdf        — upload & ingest PDF
    POST /api/courses/{cid}/documents/add-url           — fetch & ingest URL
    POST /api/courses/{cid}/documents/add-text          — ingest text note
    GET  /api/courses/{cid}/documents                   — list documents
    GET  /api/courses/{cid}/documents/{doc_id}          — document details
    DELETE /api/courses/{cid}/documents/{doc_id}        — delete document
    POST /api/courses/{cid}/chat                        — RAG Q&A
    GET  /api/courses/{cid}/chat/history                — Q&A history
    POST /api/courses/{cid}/search                      — natural language search
    GET  /api/courses/{cid}/connections                  — document connections
    POST /api/courses/{cid}/connections/refresh          — recompute connections
    GET  /api/courses/{cid}/stats                        — course stats

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
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.indexer import (
    get_model, ingest_pdf, ingest_url, ingest_text,
    delete_document, delete_course_collection,
)
from scripts.retriever import retrieve
from scripts.rag import (
    load_kb, save_kb,
    create_course, delete_course, update_course, list_courses,
    add_document_to_kb, remove_document_from_kb, add_qa_to_kb,
    answer_question, summarize_document,
    find_connections, refresh_all_connections,
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Config ──────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("school-helper")

# ── Globals ─────────────────────────────────────────────────────────────────

embed_model = None
kb = None
start_time = 0.0

# Session history for conversation context (keyed by "course_id:session_id")
_session_history: dict[str, list[dict]] = {}
_session_lock = threading.Lock()
SESSION_HISTORY_LIMIT = 5


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_course_or_404(course_id: str):
    """Validate course exists, raise 404 if not."""
    if course_id not in kb["courses"]:
        raise HTTPException(status_code=404, detail=f"Course '{course_id}' not found.")
    return kb["courses"][course_id]


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, kb, start_time
    start_time = time.time()

    log.info("Loading embedding model...")
    embed_model = get_model()
    log.info("Embedding model ready.")

    log.info("Loading knowledge base...")
    kb = load_kb()
    course_count = len(kb.get("courses", {}))
    log.info(f"Knowledge base loaded: {course_count} courses.")

    yield

    log.info("Shutting down School Helper server.")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="School Helper", lifespan=lifespan)

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


class CourseCreateRequest(BaseModel):
    name: str
    color: str = "#6366f1"


class CourseUpdateRequest(BaseModel):
    name: str | None = None
    color: str | None = None



# ── Routes: Static / Health ─────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    course_count = len(kb.get("courses", {}))
    total_docs = sum(
        len(c.get("documents", {})) for c in kb.get("courses", {}).values()
    )
    return {
        "status": "ok",
        "model_loaded": embed_model is not None,
        "courses": course_count,
        "documents": total_docs,
        "uptime_seconds": round(time.time() - start_time),
    }


# ── Routes: Course Management ──────────────────────────────────────────────

@app.get("/api/courses")
async def get_courses():
    global kb
    return {"courses": list_courses(kb)}


@app.post("/api/courses")
async def create_new_course(req: CourseCreateRequest):
    global kb
    if not req.name.strip():
        return JSONResponse({"error": "Course name is required."}, status_code=400)
    course_id = create_course(kb, req.name.strip(), req.color)
    log.info(f"Created course: {req.name} ({course_id})")
    return {"course_id": course_id, "name": req.name.strip(), "color": req.color}


@app.put("/api/courses/{course_id}")
async def update_existing_course(course_id: str, req: CourseUpdateRequest):
    global kb
    _get_course_or_404(course_id)
    update_course(kb, course_id, name=req.name, color=req.color)
    log.info(f"Updated course: {course_id}")
    return {"status": "updated", "course_id": course_id}


@app.delete("/api/courses/{course_id}")
async def delete_existing_course(course_id: str):
    global kb
    _get_course_or_404(course_id)
    # Delete ChromaDB collection
    await asyncio.to_thread(delete_course_collection, course_id)
    # Delete from KB
    delete_course(kb, course_id)
    log.info(f"Deleted course: {course_id}")
    return {"status": "deleted", "course_id": course_id}


# ── Routes: Document Management (per-course) ──────────────────────────────

@app.post("/api/courses/{course_id}/documents/upload-pdf")
async def upload_pdf(course_id: str, file: UploadFile = File(...)):
    global kb
    _get_course_or_404(course_id)

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are accepted."}, status_code=400)

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 20MB)."}, status_code=400)

    try:
        result = await asyncio.to_thread(ingest_pdf, file_bytes, file.filename, embed_model, course_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Auto-summarize
    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    add_document_to_kb(kb, course_id, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "pdf",
        "source": file.filename,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested PDF: {file.filename} -> {result['chunk_count']} chunks (course: {course_id})")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
    }


@app.post("/api/courses/{course_id}/documents/add-url")
async def add_url(course_id: str, req: UrlRequest):
    global kb
    _get_course_or_404(course_id)

    try:
        result = await asyncio.to_thread(ingest_url, req.url, embed_model, course_id)
    except (ValueError, Exception) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    add_document_to_kb(kb, course_id, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "url",
        "source": req.url,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested URL: {req.url} -> {result['chunk_count']} chunks (course: {course_id})")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
    }


@app.post("/api/courses/{course_id}/documents/add-text")
async def add_text(course_id: str, req: TextRequest):
    global kb
    _get_course_or_404(course_id)

    try:
        result = await asyncio.to_thread(ingest_text, req.text, req.title, embed_model, course_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        summary = await asyncio.to_thread(summarize_document, result["full_text"], result["title"])
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = ""

    add_document_to_kb(kb, course_id, {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "source_type": "text",
        "source": None,
        "chunk_count": result["chunk_count"],
        "text_length": result["text_length"],
    }, summary=summary)

    log.info(f"Ingested text: '{req.title}' -> {result['chunk_count']} chunks (course: {course_id})")

    return {
        "doc_id": result["doc_id"],
        "title": result["title"],
        "chunk_count": result["chunk_count"],
        "summary": summary,
    }


@app.get("/api/courses/{course_id}/documents")
async def get_documents(course_id: str):
    global kb
    course = _get_course_or_404(course_id)
    docs = []
    for doc_id, doc in course["documents"].items():
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


@app.get("/api/courses/{course_id}/documents/{doc_id}")
async def get_document(course_id: str, doc_id: str):
    global kb
    course = _get_course_or_404(course_id)
    doc = course["documents"].get(doc_id)
    if not doc:
        return JSONResponse({"error": "Document not found."}, status_code=404)
    return doc


@app.delete("/api/courses/{course_id}/documents/{doc_id}")
async def remove_document(course_id: str, doc_id: str):
    global kb
    course = _get_course_or_404(course_id)
    if doc_id not in course["documents"]:
        return JSONResponse({"error": "Document not found."}, status_code=404)

    await asyncio.to_thread(delete_document, doc_id, course_id)
    remove_document_from_kb(kb, course_id, doc_id)

    log.info(f"Deleted document: {doc_id} (course: {course_id})")
    return {"status": "deleted", "doc_id": doc_id}


# ── Routes: Chat / Q&A (per-course) ───────────────────────────────────────

@app.post("/api/courses/{course_id}/chat")
async def chat(course_id: str, req: ChatRequest):
    global kb
    _get_course_or_404(course_id)

    # Key session history by course to prevent cross-course bleed
    session_key = f"{course_id}:{req.session_id}"

    # Get/update session history
    with _session_lock:
        history = _session_history.get(session_key, [])

    result = await asyncio.to_thread(
        answer_question, req.message, history, embed_model, course_id
    )

    # Update session history
    with _session_lock:
        if session_key not in _session_history:
            _session_history[session_key] = []
        _session_history[session_key].append({"role": "user", "content": req.message})
        _session_history[session_key].append({"role": "assistant", "content": result["answer"]})
        # Keep only last N exchanges
        if len(_session_history[session_key]) > SESSION_HISTORY_LIMIT * 2:
            _session_history[session_key] = _session_history[session_key][-(SESSION_HISTORY_LIMIT * 2):]

    # Save to KB
    add_qa_to_kb(kb, course_id, req.message, result["answer"], result["sources"])

    return result


@app.get("/api/courses/{course_id}/chat/history")
async def chat_history(course_id: str):
    global kb
    course = _get_course_or_404(course_id)
    return {"history": course.get("qa_history", [])}


# ── Routes: Search (per-course) ────────────────────────────────────────────

@app.post("/api/courses/{course_id}/search")
async def search(course_id: str, req: SearchRequest):
    _get_course_or_404(course_id)
    result = await asyncio.to_thread(retrieve, req.query, req.top_k, embed_model, course_id)
    return result


# ── Routes: Connections (per-course) ───────────────────────────────────────

@app.get("/api/courses/{course_id}/connections")
async def get_connections(course_id: str):
    global kb
    course = _get_course_or_404(course_id)
    all_connections = []
    for doc_id, doc in course["documents"].items():
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


@app.post("/api/courses/{course_id}/connections/refresh")
async def refresh_connections(course_id: str):
    global kb
    _get_course_or_404(course_id)
    kb = await asyncio.to_thread(refresh_all_connections, kb, course_id, embed_model)
    return {"status": "refreshed", "document_count": len(kb["courses"][course_id]["documents"])}


# ── Routes: Stats (per-course) ─────────────────────────────────────────────

@app.get("/api/courses/{course_id}/stats")
async def get_stats(course_id: str):
    global kb
    course = _get_course_or_404(course_id)
    return course.get("stats", {})


# ── Mount static files (must be last) ───────────────────────────────────────

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
