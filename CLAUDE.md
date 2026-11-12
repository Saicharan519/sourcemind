# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SourceMind is a multi-modal RAG platform: ingest PDFs or YouTube URLs, then chat with the content over SSE-streamed answers with citations (page numbers for docs, timestamp segments for videos). Backend is FastAPI (async), frontend is React/Vite, storage is Qdrant (vectors) + PostgreSQL (everything else). LLMs run on Groq (free tier: `openai/gpt-oss-120b` main, `openai/gpt-oss-20b` fast); embeddings (`all-MiniLM-L6-v2`) and Whisper transcription run locally on CPU. See `README.md` for current behavior; `sourcemind_prd.md` is the original design doc and diverges from the code in places (noted at its top).

**SourceMind 2.0 is in progress** on branch `sourcemind-2.0`, executed phase by phase (evaluation benchmark → fixes → retrieval ablations → citations/abstention → isolation/security → MCP → observability → packaging). The pre-2.0 code is tagged `v1-as-shipped` and is ablation arm 1 — never move that tag. No metric goes into the README or resume unless a committed result file backs it.

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
- **Don't let `langchain-mistralai` float.** It's pinned to `0.2.4` (needs `langchain-core>=0.3.27,<0.4`). The current 1.x releases pull `langchain-core` 1.x, which breaks the whole langchain stack (groq, huggingface, text-splitters, langgraph). `langchain-core==0.3.27` and `langsmith==0.1.147` are pinned alongside it for the same reason.
- **faster-whisper is CPU/int8 by default.** `_whisper_model()` in `core/video_pipeline.py` keys off `EMBEDDING_DEVICE`; it only uses CUDA (float16) if that's `cuda`. GPU needs the CUDA 12 / cuDNN 9 runtime libs installed — parked for v2.
- `--reload` watches source files only. After upgrading a package, restart uvicorn manually.

## Tests / lint / evaluation

- **Unit tests + lint:** `python -m pytest eval/tests -q` and `python -m ruff check eval backend` (dev tools: `python -m pip install -r requirements-dev.txt` into the backend venv). CI (`.github/workflows/ci.yml`) runs these, validates every `eval/datasets/*.jsonl`, checks the frozen test-set hash, and builds the frontend.
- **Backend tests:** `cd backend && python -m pytest tests -q` (63 tests; fake LLM, no API calls). The isolation/MCP tests ingest two tiny PDFs into a throwaway `sourcemind_test` Postgres DB + Qdrant collection and skip if Docker isn't running. Not in CI yet (needs torch + DBs).
- **MCP server:** `backend/mcp_server.py` (stdio; tools in `core/source_tools.py`). `python mcp_demo_client.py "question"` runs it end to end. Keep `mcp==1.12.4` + `starlette==0.38.6`: mcp 2.x pulls starlette 1.x and breaks FastAPI 0.115. No `from __future__ import annotations` in `mcp_server.py` (FastMCP reads real annotation classes). Never print to stdout in code the MCP server calls without the stdout guard in `run_stdio()`.
- **Benchmark** (`eval/`, design in `docs/EVALUATION.md`): run everything from the repo root with the backend venv's python:
  ```bash
  python -m eval.corpus.download                               # fetch + hash-verify corpus PDFs
  python -m eval.run_eval --config eval/configs/<arm>.yaml     # ask every question (needs Docker DBs)
  python -m eval.score --arm <arm>                             # score (Mistral judge)
  python -m eval.report                                        # rebuild eval/REPORT.md
  ```
  The harness uses an **isolated** Postgres DB and Qdrant collection, both named `sourcemind_eval`, and never touches the app's `sourcemind` data. `eval/bootstrap.py` must run before any backend import: it copies keys from `backend/.env`, overrides DB/collection, and chdirs to `eval/.work` so the stale root `.env` is never read. An arm's `backend_dir` can point at `.worktrees/v1-as-shipped/backend` to measure old code.
- **Rules:** never edit `eval/datasets/test.jsonl` (sha256-locked, never used for tuning). Tune on `dev.jsonl` only. Any README or resume number must come from a committed `eval/results/*/scores.json`.
- Manual API checks still go through http://localhost:8000/docs and `curl -N` for SSE.

## Architecture

### Request path for a chat query

`GET /api/chat/stream` (`api/routes/chat.py`) → `route_and_stream` (`core/agent_router.py`) → `retrieve` (`core/rag_engine.py`) → `hybrid_search` (`core/hybrid_search.py`). The route persists the user message, streams events, buffers tokens, and persists the assistant reply after the stream closes. Everything below the route yields dict events (`{"type": ..., "value": ...}`) that `utils/streaming.py` serializes into SSE frames.

### The query router (`core/agent_router.py`)

This is a **real compiled LangGraph `StateGraph`** (`GRAPH = _build_graph()`, compiled once at import): `START → classify → conditional edge → {simple | comparative | multi_hop} → END`. `classify_query` (fast model) labels the question and drives the conditional edge. Comparative decomposes into 2 sub-queries retrieved in parallel then synthesized; multi-hop does retrieve → intermediate answer (fast model) → follow-up query → second retrieve → final synthesis. All three converge on `stream_grounded_answer` (main model): it emits `context` (all passages), streams tokens that cite passages as `[n]`, then emits `citations` (only cited, valid refs) and `evidence` (supported / insufficient / uncited). Empty retrieval returns `NOT_FOUND_ANSWER` without an LLM call. It is a router with fixed strategies — no tool loop — so don't describe it as an autonomous agent.

**Streaming out of graph nodes:** LangGraph nodes mutate state and return — they can't be async generators. To keep the token stream, each node pushes SSE event dicts into an `asyncio.Queue` via an `emit` callable carried in the graph state; `route_and_stream` runs `GRAPH.ainvoke()` as a background task and drains the queue, yielding the same events the API layer expects. Gotcha: LangGraph 0.2.34 requires every node to write ≥1 state channel, so `simple_node` returns `{"query_type": ...}` even though it has nothing else to update.

### The retrieval pipeline (`core/rag_engine.py` + `core/hybrid_search.py`)

`retrieve()` expands the query with multi-query rephrasings + a HyDE hypothetical answer (both fast model), runs `hybrid_search` for each variant in parallel, then dedupes by chunk id keeping the best RRF score. `hybrid_search` fuses **dense** (Qdrant cosine) and **sparse** (BM25) results via Reciprocal Rank Fusion, then MMR-reranks for diversity down to `MMR_FINAL_K` chunks.

Key structural facts:
- **Hierarchical chunking.** Child chunks (~300 **characters** — `length_function=len`, despite older docs saying tokens) are what's embedded and searched; parent chunks (~1500 characters) live in Postgres. After retrieval, `hydrate_contexts()` swaps each child's content for its **parent** content (via `parent_chunk_id` in the Qdrant payload) before the LLM sees it. Video chunks have no parent and pass through unchanged.
- **Sparse vectors are NOT stored in Qdrant.** Contrary to the PRD's "sparse vectors enabled" claim, `db/qdrant.py` creates a dense-only collection. BM25 is a separate path: `core/bm25_index.py` pulls the text corpus from the Postgres `bm25_index` table and builds a fresh `BM25Okapi` index **per query**. Simple and fine at this scale; the place to change if retrieval gets slow.
- **Source isolation via metadata filter.** Every Qdrant/BM25 call filters by `source_id`. Passing `source_id=None` (the global-chat path, `api/routes/global_chat.py`) omits the filter and searches across all sources.

### Data layer

- `db/postgres.py` — a thin async `Database` classmethod wrapper over an asyncpg pool, plus typed helper functions (`create_source`, `insert_parent_chunks`, `fetch_bm25_corpus`, `insert_eval_result`, etc.). All raw SQL; no ORM despite SQLAlchemy being in requirements. NaN floats are scrubbed to `None` in several places because RAGAS can emit NaN.
- `db/qdrant.py` — `QdrantStore` singleton; `_ensure_collection` auto-creates the collection + `source_id`/`source_type` payload indexes on startup. Deletes cascade in Postgres (FK `ON DELETE CASCADE`) and are mirrored in Qdrant via `delete_by_source`.
- Point/chunk IDs are shared: the Qdrant point id is also the `chunk_id` in the `bm25_index` table, which is how RRF fuses the two result sets.

### Ingestion (async, non-blocking)

`api/routes/ingest.py` writes the uploaded PDF to `UPLOAD_DIR`, creates a `sources` row with `status='processing'`, returns `202` immediately, and runs the pipeline in a FastAPI `BackgroundTask`. `core/document_pipeline.py` (PyMuPDF → hierarchical chunk → embed → Qdrant upsert + BM25 rows → auto title) and `core/video_pipeline.py` (yt-dlp via `utils/audio.py` → local **faster-whisper** transcription → segment chunks) do the work and flip status to `ready`.

**Blocking work is offloaded.** Whisper transcription, PyMuPDF extraction, and sentence-transformers embedding are synchronous CPU-bound calls — they run via `asyncio.to_thread(...)` so ingestion (and RAGAS) never freezes the async API server. Preserve this when adding CPU-heavy steps.

**Chat is English-only, so non-English sources are translated at ingestion** (`TRANSLATE_TO_ENGLISH`, default on). Videos use Whisper's built-in `task="translate"` (transcribe + translate to English in one pass, segment timings preserved). PDFs are language-detected with `langdetect`; if non-English, each page is translated via the fast model (bounded concurrency, page numbers kept) before chunking. This keeps the whole retrieval stack — MiniLM embeddings, the ASCII BM25 tokenizer, English prompts — unchanged. Citations therefore show the English translation, not the original text.

**RAGAS auto-eval runs after ingestion.** `_ingest_doc_and_eval` / `_ingest_video_and_eval` call `evaluate_source(...)` once the source reaches `ready` (skipped for `failed`). The evaluator LLM is **Mistral** (`_build_ragas_llm` in `core/evaluator.py`), which falls back to the Groq fast model only if `MISTRAL_API_KEY` is unset — this was the fix for Groq's low-TPM 429→NaN cascade. `ragas.evaluate()` itself is sync, so it's wrapped in `asyncio.to_thread`. `RAGAS_NUM_QUESTIONS` defaults to 10 in `config.py`; the sample `.env` uses 3. Its questions are generated from the source's *opening chunks* only, so treat the scores as a synthetic self-check, not a benchmark.

### Frontend (`frontend/src`)

React Router pages: Dashboard (upload + source list), ChatPage (`/chat/:sourceId`), GlobalChatPage (`/global`), SourceDetailPage. `api/client.js` holds the Axios instance (base `VITE_API_URL`, default `http://localhost:8000`) plus URL builders for the SSE endpoints. `hooks/useStreamingChat.js` consumes SSE. There is no Vite dev proxy — the browser calls `:8000` directly and CORS in `main.py` (`BACKEND_CORS_ORIGINS`) allows `:5173`.

## Model routing convention

`LLM_PROVIDER` (groq | deepseek | cloudflare, `core/llm_providers.py`) picks one provider; there is no automatic fallback by design. Within a provider, the main model (Groq: `openai/gpt-oss-120b`) is used **only** for final answer generation and comparative synthesis where quality matters. Everything auxiliary — classification, multi-query, HyDE, decomposition, follow-up queries, multi-hop intermediate answers, and RAGAS QA generation — uses the fast/cheap model (`GROQ_MODEL_FAST`, `openai/gpt-oss-20b`). Avoid Groq *preview* models (e.g. `qwen/qwen3.8-27b`) — they can be withdrawn at short notice, which would invalidate eval results. The one exception is the **RAGAS evaluator LLM, which is Mistral** (`MISTRAL_MODEL_EVAL`), moved off Groq to avoid rate-limit failures during scoring. Preserve this split when adding LLM calls; it's what keeps the project within free-tier limits.
