# SourceMind

**Multi-modal RAG over PDFs and YouTube videos with hybrid search, agentic query routing, streaming responses, and automated evaluation.**

Built as a portfolio project demonstrating production-grade RAG engineering — not a tutorial clone.

---

## Features

- **Multi-modal ingestion**: Upload PDFs (up to 100MB, 500+ pages) or paste YouTube URLs
- **Hierarchical chunking**: 300-token children for retrieval precision, 1500-token parents for LLM context
- **Hybrid search**: Dense (vector) + Sparse (BM25) fused via Reciprocal Rank Fusion
- **MMR reranking**: Diverse candidate selection, no redundant chunks
- **Multi-query + HyDE**: Auto-rephrasing and hypothetical document embeddings
- **LangGraph agentic router**: Classifies queries as simple / comparative / multi-hop, dispatches appropriate retrieval strategy
- **Streaming responses**: Server-Sent Events stream tokens to the React frontend in real time
- **Cross-document querying**: Global chat endpoint searches across every ingested source
- **RAGAS evaluation**: Auto-generated 10 QA pairs scored on faithfulness, answer relevancy, context recall, context precision
- **100% free model stack**: Groq Llama 3.1/3.3 + local HuggingFace embeddings + local Whisper

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | React 18, Vite, Tailwind, React Query, React Router, EventSource SSE |
| Backend | FastAPI, async PostgreSQL (asyncpg), Pydantic |
| Vector DB | Qdrant (Cosine, metadata filtered) |
| Relational DB | PostgreSQL 16 (parent chunks, BM25 corpus, eval results, chat history) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, 384 dims) |
| LLM | Groq API — Llama 3.3 70B (answers), Llama 3.1 8B (auxiliary) |
| Transcription | faster-whisper (`small`, local, CTranslate2 int8) |
| RAG | LangChain + a compiled LangGraph `StateGraph` router |
| Evaluation | RAGAS (Mistral evaluator LLM) |
| Infra | Docker Compose |

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- A free Groq API key from [console.groq.com](https://console.groq.com/keys)
- (Optional) A free Mistral API key from [console.mistral.ai](https://console.mistral.ai) — used as the RAGAS evaluator LLM; without it, evaluation falls back to Groq 8B

### 2. Clone and configure

```bash
git clone <your-repo-url>
cd sourcemind
cp .env.example .env
```

Edit `.env` and set your `GROQ_API_KEY`.

### 3. Start everything

```bash
docker compose up --build
```

First boot will pull Postgres, Qdrant, and download the HuggingFace embedder (~90MB) and Whisper model (~460MB for "small"). After that, restarts are fast thanks to persistent volumes.

### 4. Open the app

- Frontend: http://localhost:5173
- Backend docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

---

## Local Development (without Docker)

If you'd rather run things directly:

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run Postgres + Qdrant separately, e.g.:
#   docker run -p 5432:5432 -e POSTGRES_PASSWORD=sourcemind_secret postgres:16
#   docker run -p 6333:6333 qdrant/qdrant

# Init the DB schema once:
psql postgresql://sourcemind:sourcemind_secret@localhost:5432/sourcemind \
    -f ../scripts/init_db.sql

# Set DATABASE_URL to localhost in .env, then:
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/ingest/document` | Upload PDF (returns 202, runs in background) |
| POST | `/api/ingest/video` | Submit YouTube URL (returns 202) |
| GET | `/api/sources` | List all sources with eval scores |
| GET | `/api/sources/{id}` | Source detail + full RAGAS results |
| DELETE | `/api/sources/{id}` | Delete source + cascade vector/index cleanup |
| GET | `/api/chat/stream` | **SSE** per-source streaming chat |
| GET | `/api/chat/global/stream` | **SSE** cross-document chat |
| POST | `/api/evaluate/{id}` | Manually re-run RAGAS evaluation |
| GET | `/api/evaluate/{id}` | Get latest RAGAS results |

### SSE event format

```
data: {"type": "query_type", "value": "simple"}
data: {"type": "sub_queries", "value": ["..."]}      // comparative/multi-hop only
data: {"type": "citations", "value": [...]}
data: {"type": "token", "value": "..."}
data: {"type": "token", "value": "..."}
...
data: {"type": "done", "value": null}
```

---

## Architecture

```
React (Vite) ──REST + SSE──> FastAPI ─┬─> Document Pipeline ─┐
                                      │   (PyMuPDF + chunker) │
                                      │                       │─> embed (HF)
                                      └─> Video Pipeline ─────┤   + BM25 index
                                          (yt-dlp + Whisper)  │
                                                              ▼
                                          ┌── Qdrant (vectors + metadata) ──┐
                                          └── PostgreSQL (parents + corpus)─┘
                                                              ▲
                          /api/chat/stream                    │
                                  │                            │
                                  ▼                            │
                          LangGraph Router                     │
                          ├─ simple_rag      ──> retrieve() ───┤
                          ├─ comparative_rag ──> retrieve() x2─┤
                          └─ multi_hop_rag   ──> retrieve() x2─┘
                                  │
                                  ▼
                          Groq Llama 3.3 70B (streaming)
                                  │
                                  ▼
                          SSE → React EventSource → chat bubble
```

---

## Build Order (if working from scratch)

Already followed in this codebase, but for reference:

1. **Infrastructure** — Postgres + Qdrant via Docker Compose
2. **Document pipeline** — PyMuPDF → hierarchical chunks → embed → upsert
3. **RAG chain** — hybrid search + MMR + multi-query + HyDE
4. **Video pipeline** — yt-dlp → Whisper → segment chunking
5. **LangGraph router** — classifier + 3 retrieval paths
6. **FastAPI routes** — async ingestion via BackgroundTasks + SSE chat
7. **RAGAS evaluator** — QA generation + RAG runs + scoring
8. **React frontend** — Dashboard + Chat + Global chat + SSE hook
9. **Polish** — Docker Compose orchestration, README

---

## Free Model Strategy

| Task | Model |
|---|---|
| Final answer generation | Groq Llama 3.3 70B |
| Multi-query rephrasing | Groq Llama 3.1 8B |
| HyDE generation | Groq Llama 3.1 8B |
| Query classification | Groq Llama 3.1 8B |
| RAGAS evaluator LLM | Mistral `mistral-small-latest` (falls back to Groq 8B if no key) |
| Embeddings | local `all-MiniLM-L6-v2` (CPU) |
| Audio transcription | local faster-whisper `small` (CPU, int8) |

Groq free tier provides ~30K TPM for 8B and ~6K TPM for 70B — effectively unlimited for portfolio/demo use.

---

## Troubleshooting

**Whisper / sentence-transformers download stuck:** First boot pulls ~600MB of models. Be patient. Subsequent boots use the persisted Docker volumes (`whisper_models`, `hf_models`).

**Groq rate limit (429):** Wait a minute. Free tier is generous but not infinite. Drop multi-query rephrasings count from 3 to 2 in `core/rag_engine.py` if you hit limits often.

**PDF extraction returns empty pages:** PyMuPDF can't handle scanned PDFs without OCR. Run those through OCR first (Tesseract / `ocrmypdf`) before uploading.

**Qdrant collection mismatch error:** If you changed embedding dimensions, delete the volume: `docker compose down -v` then `docker compose up`.

---

## License

MIT — do what you want.
