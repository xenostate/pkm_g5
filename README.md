# PKM — Personal Knowledge Management

A local-first Personal Knowledge Management system powered by RAG (Retrieval-Augmented Generation). Upload PDFs, add URLs, or paste text notes — the system indexes everything, generates summaries, finds connections between documents, and lets you ask questions in natural language.

## Architecture

```
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │   PDFs /    │───▶│   FastAPI    │◀───│  Dashboard   │
  │  URLs / Text│    │   Server     │    │  (Frontend)  │
  └─────────────┘    └──────┬───────┘    └──────────────┘
                            │
                 ┌──────────┼──────────┐
                 │          │          │
          ┌──────┴──────┐ ┌┴────────┐ ┌┴──────────────┐
          │  ChromaDB   │ │ OpenAI  │ │ knowledge_    │
          │ (embeddings)│ │  API    │ │ base.json     │
          └─────────────┘ └─────────┘ └───────────────┘
```

**Pipeline:** Upload → Extract text → Chunk (500 words, overlapping) → Embed (`intfloat/multilingual-e5-base`, CPU) → Store in ChromaDB → Auto-summarize via OpenAI → Vector retrieval → LLM answer with sources

## Features

- **PDF ingestion** — drag & drop PDF files
- **URL ingestion** — fetch and index any web page
- **Text notes** — paste or type text directly
- **Automatic summarization** — every document is summarized on upload
- **Knowledge connections** — discover relationships between documents
- **Natural language Q&A** — ask questions, get answers with source attribution
- **Full-text search** — search across all your documents
- **Local vector store** — ChromaDB runs on CPU, no cloud needed
- **Knowledge base file** — all metadata, summaries, and Q&A history in `data/knowledge_base.json`

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** — for summaries, connections, and Q&A (uses `gpt-4o-mini`)

## Quick Start

### 1. Set up Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-proj-...your-openai-key
```

### 3. Start the server

```bash
./start.sh
```

Open `http://localhost:8090` in your browser.

## Usage

1. **Upload documents** — Go to the Documents tab, drag & drop PDFs, add URLs, or paste text
2. **Read summaries** — Check the Summaries tab for auto-generated document summaries
3. **Ask questions** — Use the Chat tab to ask questions about your documents
4. **Explore connections** — Click "Refresh Connections" in the Knowledge Map tab
5. **Search** — Use the search bar at the top to search across all documents

## Project Structure

```
.
├── .env.example             # Template for environment variables
├── start.sh                 # Server launcher
├── scripts/
│   ├── requirements.txt     # Python dependencies
│   ├── indexer.py           # Document ingestion (PDF, URL, text) + ChromaDB storage
│   ├── retriever.py         # Vector search via ChromaDB
│   ├── rag.py               # RAG Q&A, summarization, connections, KB management
│   └── server.py            # FastAPI server with all API endpoints
├── frontend/
│   ├── index.html           # Dashboard HTML
│   ├── styles.css           # Dashboard styles
│   └── app.js               # Dashboard JavaScript
└── data/                    # Auto-created at runtime
    ├── chroma_db/           # ChromaDB vector store
    └── knowledge_base.json  # Summaries, connections, Q&A history
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/health` | Status check |
| `POST` | `/api/documents/upload-pdf` | Upload PDF file |
| `POST` | `/api/documents/add-url` | Add URL |
| `POST` | `/api/documents/add-text` | Add text note |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Document details |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `POST` | `/api/chat` | Q&A with RAG |
| `GET` | `/api/chat/history` | Q&A history |
| `POST` | `/api/search` | Natural language search |
| `GET` | `/api/connections` | All document connections |
| `POST` | `/api/connections/refresh` | Recompute connections |
| `GET` | `/api/knowledge-base` | Full knowledge base JSON |
| `GET` | `/api/stats` | Statistics |

## Tuning

| Env Var | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | Embedding model (runs on CPU) |
| `RAG_MODEL` | `gpt-4o-mini` | OpenAI model for LLM features |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |

## Cost

- **Embeddings**: Free — runs locally on CPU
- **Vector store**: Free — ChromaDB stores locally
- **LLM**: OpenAI `gpt-4o-mini` — ~$0.001 per query, ~$0.002 per summary
# pkm_g5
