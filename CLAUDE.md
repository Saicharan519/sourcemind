# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SourceMind is a multi-modal RAG platform: ingest PDFs or YouTube URLs, then chat with the content over SSE-streamed answers with citations (page numbers for docs, timestamp segments for videos). Backend is FastAPI (async), frontend is React/Vite, storage is Qdrant (vectors) + PostgreSQL (everything else). LLMs run on Groq (free tier); embeddings (`all-MiniLM-L6-v2`) and Whisper transcription run locally on CPU. See `README.md` and `sourcemind_prd.md` for the full design rationale.

## Running the project

The two databases run in Docker; the backend and frontend run directly on the host.

```bash
# Databases (start once; they persist across restarts)
docker start sourcemind-postgres sourcemind-qdrant   # if already created
# First-time creation + schema: see README.md "Local Development" and scripts/init_db.sql

# Backend (from backend/, venv activated)
python -m uvicorn main:app --reload --port 8000

# Frontend (from frontend/)
npm run dev        # Vite dev server on :5173
npm run build      # production build
```

- App: http://localhost:5173 · API docs: http://localhost:8000/docs · Qdrant: http://localhost:6333/dashboard
- `docker compose up --build` also runs the whole stack, but local dev normally uses only the two DB containers.

### Environment gotchas (these have bitten before)

- **`.env` must live in `backend/`, not the repo root.** `config.py` loads `env_file=".env"` relative to the current working directory, and the backend is always run from `backend/`. A root-only `.env` will silently leave `GROQ_API_KEY` empty.
- **Use `python -m uvicorn` / `python -m pip`, not the bare `uvicorn` / `pip` launchers,** if the venv was ever moved between folders. The `.exe` launcher stubs hardcode the original absolute path to `python.exe` and fail with "Unable to create process". `python -m ...` invokes the interpreter directly and always works. To fix permanently, recreate the venv in place.
- **`yt-dlp` goes stale.** YouTube changes formats often; an old pin produces `Requested format is not available`. `requirements.txt` uses `yt-dlp>=...` intentionally — keep it unpinned-upward and `pip install -U yt-dlp` when video ingest breaks.
- `--reload` watches source files only. After upgrading a package, restart uvicorn manually.

## Tests / lint

There is no test suite, linter config, or CI in this repo despite the PRD mentioning GitHub Actions. Manual verification is done through `http://localhost:8000/docs` and curl (SSE endpoints don't work well in Swagger — use `curl -N`). Don't claim tests pass; there are none to run.

## Architecture

### Request path for a chat query

`GET /api/chat/stream` (`api/routes/chat.py`) → `route_and_stream` (`core/agent_router.py`) → `retrieve` (`core/rag_engine.py`) → `hybrid_search` (`core/hybrid_search.py`). The route persists the user message, streams events, buffers tokens, and persists the assistant reply after the stream closes. Everything below the route yields dict events (`{"type": ..., "value": ...}`) that `utils/streaming.py` serializes into SSE frames.

### The agentic router (`core/agent_router.py`)

Despite the PRD calling it "LangGraph", this is **not** a compiled LangGraph graph — it's plain async generators. `classify_query` (Groq 8B) labels the question `simple | comparative | multi_hop`, then dispatches to `_simple_path`, `_comparative_path`, or `_multi_hop_path`. Comparative decomposes into 2 sub-queries retrieved in parallel then synthesized; multi-hop does retrieve → intermediate answer (8B) → follow-up query → second retrieve → final synthesis. All three converge on `stream_answer` (Groq 70B) for the final streamed answer.

### The retrieval pipeline (`core/rag_engine.py` + `core/hybrid_search.py`)

`retrieve()` expands the query with multi-query rephrasings + a HyDE hypothetical answer (both Groq 8B), runs `hybrid_search` for each variant in parallel, then dedupes by chunk id keeping the best RRF score. `hybrid_search` fuses **dense** (Qdrant cosine) and **sparse** (BM25) results via Reciprocal Rank Fusion, then MMR-reranks for diversity down to `MMR_FINAL_K` chunks.

Key structural facts:
- **Hierarchical chunking.** Child chunks (~300 tokens) are what's embedded and searched; parent chunks (~1500 tokens) live in Postgres. After retrieval, `hydrate_contexts()` swaps each child's content for its **parent** content (via `parent_chunk_id` in the Qdrant payload) before the LLM sees it. Video chunks have no parent and pass through unchanged.
- **Sparse vectors are NOT stored in Qdrant.** Contrary to the PRD's "sparse vectors enabled" claim, `db/qdrant.py` creates a dense-only collection. BM25 is a separate path: `core/bm25_index.py` pulls the text corpus from the Postgres `bm25_index` table and builds a fresh `BM25Okapi` index **per query**. Simple and fine at this scale; the place to change if retrieval gets slow.
- **Source isolation via metadata filter.** Every Qdrant/BM25 call filters by `source_id`. Passing `source_id=None` (the global-chat path, `api/routes/global_chat.py`) omits the filter and searches across all sources.

### Data layer

- `db/postgres.py` — a thin async `Database` classmethod wrapper over an asyncpg pool, plus typed helper functions (`create_source`, `insert_parent_chunks`, `fetch_bm25_corpus`, `insert_eval_result`, etc.). All raw SQL; no ORM despite SQLAlchemy being in requirements. NaN floats are scrubbed to `None` in several places because RAGAS can emit NaN.
- `db/qdrant.py` — `QdrantStore` singleton; `_ensure_collection` auto-creates the collection + `source_id`/`source_type` payload indexes on startup. Deletes cascade in Postgres (FK `ON DELETE CASCADE`) and are mirrored in Qdrant via `delete_by_source`.
- Point/chunk IDs are shared: the Qdrant point id is also the `chunk_id` in the `bm25_index` table, which is how RRF fuses the two result sets.

### Ingestion (async, non-blocking)

`api/routes/ingest.py` writes the uploaded PDF to `UPLOAD_DIR`, creates a `sources` row with `status='processing'`, returns `202` immediately, and runs the pipeline in a FastAPI `BackgroundTask`. `core/document_pipeline.py` (PyMuPDF → hierarchical chunk → embed → Qdrant upsert + BM25 rows → auto title/summary) and `core/video_pipeline.py` (yt-dlp via `utils/audio.py` → local Whisper → segment chunks) do the work and flip status to `ready`.

**RAGAS auto-eval is currently disabled.** The `evaluate_source(...)` calls in `_ingest_doc_and_eval` / `_ingest_video_and_eval` are commented out, so eval only runs when manually triggered via `POST /api/evaluate/{id}`. `RAGAS_NUM_QUESTIONS` defaults to 10 in `config.py` but the sample `.env` sets it to 3.

### Frontend (`frontend/src`)

React Router pages: Dashboard (upload + source list), ChatPage (`/chat/:sourceId`), GlobalChatPage (`/global`), SourceDetailPage. `api/client.js` holds the Axios instance (base `VITE_API_URL`, default `http://localhost:8000`) plus URL builders for the SSE endpoints. `hooks/useStreamingChat.js` consumes SSE. There is no Vite dev proxy — the browser calls `:8000` directly and CORS in `main.py` (`BACKEND_CORS_ORIGINS`) allows `:5173`.

## Model routing convention

Groq 70B (`GROQ_MODEL_MAIN`) is used **only** for final answer generation and comparative synthesis where quality matters. Everything auxiliary — classification, multi-query, HyDE, decomposition, follow-up queries, multi-hop intermediate answers, RAGAS QA generation, and the RAGAS evaluator LLM — uses the fast/cheap 8B (`GROQ_MODEL_FAST`). Preserve this split when adding LLM calls; it's what keeps the project within free-tier rate limits.
