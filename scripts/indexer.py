#!/usr/bin/env python3
"""
PKM Indexer: ingest PDFs, URLs, and text notes into ChromaDB.

Provides functions to add, delete, and list documents with their embeddings.
"""

import hashlib
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import chromadb
import requests
from bs4 import BeautifulSoup, Comment
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────

EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── Embedding model (lazy-loaded) ──────────────────────────────────────────

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[embed] Loading model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ── ChromaDB ───────────────────────────────────────────────────────────────

_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="pkm_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ── HTML cleaning ──────────────────────────────────────────────────────────

STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "iframe"}

JUNK_CLASSES = re.compile(
    r"reflist|references|ref-list|mw-references|citation|navbox|sidebar|"
    r"infobox|mw-editsection|catlinks|printfooter|mw-jump-link|"
    r"noprint|metadata|hatnote|ambox|dmbox|tmbox|fmbox|ombox|"
    r"external[_-]links|see-also|authority-control|portal-bar",
    re.IGNORECASE,
)


def clean_html(html: str) -> tuple[str, str]:
    """Extract title and clean body text from raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    for sup in soup.find_all("sup", class_="reference"):
        sup.decompose()

    for tag in list(soup.find_all(True)):
        if not tag.parent:
            continue
        classes = " ".join(tag.get("class") or [])
        tag_id = tag.get("id") or ""
        if JUNK_CLASSES.search(classes) or JUNK_CLASSES.search(tag_id):
            tag.decompose()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers = []
        text_lines = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            if row.find("th") and not row.find("td"):
                headers = cell_texts
            elif headers and len(cell_texts) == len(headers):
                pairs = [f"{h}: {v}" for h, v in zip(headers, cell_texts) if v]
                if pairs:
                    text_lines.append(" | ".join(pairs))
            else:
                line = " | ".join(c for c in cell_texts if c)
                if line:
                    text_lines.append(line)
        table.replace_with(BeautifulSoup("\n".join(text_lines) + "\n", "lxml"))

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return title, text


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks respecting paragraph and sentence boundaries."""
    if not text.strip():
        return []

    paragraphs = re.split(r'\n\s*\n', text)

    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = para.split()
        if len(para_words) <= chunk_size:
            segments.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = []
            current_len = 0
            for sent in sentences:
                sent_words = sent.split()
                if current and current_len + len(sent_words) > chunk_size:
                    segments.append(" ".join(current))
                    current = []
                    current_len = 0
                current.extend(sent_words)
                current_len += len(sent_words)
            if current:
                segments.append(" ".join(current))

    if not segments:
        return []

    chunks = []
    current_words = []
    current_segs = []

    for seg in segments:
        seg_words = seg.split()

        if current_words and len(current_words) + len(seg_words) > chunk_size:
            chunks.append(" ".join(current_words))

            overlap_words = []
            overlap_segs = []
            for prev_seg in reversed(current_segs):
                pw = prev_seg.split()
                if len(overlap_words) + len(pw) <= overlap:
                    overlap_words = pw + overlap_words
                    overlap_segs.insert(0, prev_seg)
                else:
                    break
            current_words = overlap_words
            current_segs = overlap_segs

        current_words.extend(seg_words)
        current_segs.append(seg)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def _embed_and_store(doc_id: str, chunks: list[str], metadata_base: dict, model=None):
    """Embed chunks and store in ChromaDB. Returns chunk count."""
    if not chunks:
        return 0

    if model is None:
        model = get_model()

    collection = get_chroma_collection()

    texts_to_embed = [f"passage: {c}" for c in chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

    # ChromaDB batch limit is ~41666 but we batch at 40 for safety
    batch_size = 40
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_ids = [f"{doc_id}_chunk_{i}" for i in range(start, end)]
        batch_docs = chunks[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_metadatas = [
            {**metadata_base, "doc_id": doc_id, "chunk_index": i}
            for i in range(start, end)
        ]
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    return len(chunks)


# ── Ingestion functions ────────────────────────────────────────────────────

def ingest_pdf(file_bytes: bytes, filename: str, model=None) -> dict:
    """Ingest a PDF file. Returns {doc_id, title, chunk_count, text_length, full_text}."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise ValueError(f"No extractable text found in {filename}. The PDF may be scanned/image-based.")

    title = filename.rsplit(".", 1)[0] if "." in filename else filename
    doc_id = _generate_doc_id()
    chunks = chunk_text(full_text)

    chunk_count = _embed_and_store(
        doc_id, chunks,
        {"source_type": "pdf", "source": filename, "title": title},
        model=model,
    )

    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": chunk_count,
        "text_length": len(full_text),
        "full_text": full_text,
    }


def ingest_url(url: str, model=None) -> dict:
    """Ingest a single web page. Returns {doc_id, title, chunk_count, text_length, full_text}."""
    resp = requests.get(url, timeout=15, headers={"User-Agent": "PKM-Agent/1.0"})
    resp.raise_for_status()

    if "text/html" not in resp.headers.get("content-type", ""):
        raise ValueError(f"URL did not return HTML content: {url}")

    title, text = clean_html(resp.text)
    if len(text) < 50:
        raise ValueError(f"Page content too short (< 50 chars): {url}")

    if not title:
        title = url
    doc_id = _generate_doc_id()
    chunks = chunk_text(text)

    chunk_count = _embed_and_store(
        doc_id, chunks,
        {"source_type": "url", "source": url, "title": title},
        model=model,
    )

    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": chunk_count,
        "text_length": len(text),
        "full_text": text,
    }


def ingest_text(text: str, title: str, model=None) -> dict:
    """Ingest a plain text note. Returns {doc_id, title, chunk_count, text_length, full_text}."""
    if not text.strip():
        raise ValueError("Text content is empty.")

    doc_id = _generate_doc_id()
    chunks = chunk_text(text)

    chunk_count = _embed_and_store(
        doc_id, chunks,
        {"source_type": "text", "source": None, "title": title},
        model=model,
    )

    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": chunk_count,
        "text_length": len(text),
        "full_text": text,
    }


def delete_document(doc_id: str):
    """Delete all chunks for a document from ChromaDB."""
    collection = get_chroma_collection()
    # Get all chunk IDs for this document
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def list_documents() -> list[dict]:
    """List unique documents from ChromaDB metadata."""
    collection = get_chroma_collection()
    all_data = collection.get(include=["metadatas"])

    docs = {}
    for meta in all_data["metadatas"]:
        did = meta["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "title": meta.get("title", "Untitled"),
                "source_type": meta.get("source_type", "unknown"),
                "source": meta.get("source"),
                "chunk_count": 0,
            }
        docs[did]["chunk_count"] += 1

    return list(docs.values())


def get_document_chunks(doc_id: str) -> list[str]:
    """Get all chunk texts for a document, ordered by chunk_index."""
    collection = get_chroma_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )
    if not results["ids"]:
        return []

    # Sort by chunk_index
    paired = list(zip(results["metadatas"], results["documents"]))
    paired.sort(key=lambda x: x[0].get("chunk_index", 0))
    return [doc for _, doc in paired]
