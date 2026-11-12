# SourceMind 2.0 — Roadmap (approved plan)

## Context

SourceMind (FastAPI + LangGraph + Qdrant + Postgres + React) is your flagship placement project for GenAI / AI-engineering roles. The architecture is sound, but a code review (confirmed line by line) found:
- **False claims.** Chunks are measured in characters, not tokens ([chunker.py:44](backend/core/chunker.py)). The "1.00 recall / 0.89 faithfulness" numbers come from questions generated off only the first chunks of each source ([evaluator.py:43-78](backend/core/evaluator.py)).
- **Retrieval bugs.** `retrieve()` re-sorts by the max RRF score across query variants, which throws away the MMR order ([rag_engine.py:123-135](backend/core/rag_engine.py)). BM25 is rebuilt for every search, up to 5 times per question. There is no reranker.
- **Security holes.** The video URL check is a substring test, so it is open to SSRF ([ingest.py:104](backend/api/routes/ingest.py)). Raw filenames are used in upload paths. `nocheckcertificate: True` ([audio.py:32](backend/utils/audio.py)). Chat history travels in GET query strings.
- **Missing pieces.** No "I don't know" answer, citations list every retrieved chunk, no tests, no CI, no LICENSE, unused deps, and the PRD author says "Tarun".

**Goal:** turn it from a demo into a *measured, defensible* system. Every resume number must come from a committed result file. Every component stays only if the benchmark shows it helps. Improve the existing repo; no rewrite.

**What we decided:**
- The test set is **AI-generated, cross-checked by a second model family (Mistral), and sample human-verified**. You won't hand-write questions.
- Simple auth only: no emailed OTP.
- ₹0 budget, so no paid hosting.
- Each phase ends in a state you can show and put on your resume, so the project is useful even if work stops early.

**How we execute:** one phase at a time. Each phase ends with its checks run, a summary shown to you, and a commit, before the next phase starts. Never all nine at once.

**Explicitly not building:** DOCX or OCR, knowledge graphs, multi-agent systems, fine-tuning, Redis or Celery, a TypeScript migration, a UI redesign, or a backend restructure (new modules are fine).

---

## Phase 0: Freeze and be honest

1. **Pin the LLMs.** The uncommitted diff disagrees with itself: `config.py` sets the fast model to `qwen/qwen3.8-27b`, `.env.example` says `openai/gpt-oss-20b`. Check Groq's current model list, pick one main and one fast model, and make `config.py` and `.env.example` match. Every eval results file will record the model IDs it ran with.
2. Commit, then **tag `v1-as-shipped`**. This is arm 1 of the ablation and must stay reproducible.
3. Repo hygiene:
   - Add an MIT `LICENSE`.
   - Remove `sqlalchemy` and `psycopg2-binary` from `backend/requirements.txt` (confirmed unused).
   - Fix the PRD author.
4. Make the README and PRD match the code:
   - Stop claiming chunk sizes in tokens; stop calling the router an "agent".
   - Remove the CI claim.
   - Relabel the RAGAS numbers as a "synthetic self-check".
   - Update `CLAUDE.md` to match.

**Done when:** the tag exists and every README claim matches the code.

---

## Phase 1: Measurement foundation (the core)

New top-level `eval/` folder:
```
eval/
  corpus/       manifest.yaml (url, license, sha256), download.py, transcripts/
  datasets/     dev.jsonl, test.jsonl, test.sha256, smoke.jsonl
  configs/      one YAML per experiment arm
  tools/        gen_questions.py, crosscheck.py, review_app.py, split.py
  metrics/      retrieval.py, answer.py, citation.py, abstention.py, cost.py
  run_eval.py   report.py   cache/   results/   REPORT.md
```

1. **Corpus.** 8–10 redistributable PDFs and 3 openly licensed lecture videos:
   - PDFs: arXiv CC-BY papers, NIST/government documents, OpenStax chapters, a policy document, a public report.
   - Videos: e.g. MIT OpenCourseWare.
   - Each file is listed in `manifest.yaml` with its license and SHA-256.
   - Video transcripts are committed so nobody needs to rerun Whisper.
2. **Question schema** (JSONL, one question per line). Labels use pages and time windows, never chunk IDs:
   ```json
   {"id":"t-017","category":"exact_id","question":"…","answerable":true,
    "gold_evidence":[{"doc":"nist.pdf","pages":[41,42]}],"gold_answer":"…"}
   ```
   - Categories: factoid, exact_id, paraphrase, comparative, multi_hop, unanswerable, follow_up.
   - Videos use `{"doc":…,"start":754,"end":812}`.
3. **Generation** (`gen_questions.py`):
   - Sample **random** chunks across each whole document and send them to the fast Groq model, following the model-routing rule.
   - Produce category-targeted questions, with the page or time window filled in automatically.
   - Unanswerable questions: plausible questions whose answer is absent from the document, plus cross-document negatives (a question about document B asked with retrieval restricted to document A).
4. **Cross-check** (`crosscheck.py`): Mistral, a different model family, checks each question for three things:
   - Is the answer actually on the cited page?
   - Is the question unambiguous?
   - For unanswerables, is the answer really absent?

   Failures are dropped or flagged.
5. **Your review** (`review_app.py`): a small local page showing the rendered PDF page (PyMuPDF, already a dependency) next to the question, with Keep / Edit / Reject buttons. You review:
   - every **flagged** question
   - **every unanswerable and multi-hop test question** (about 40), because that's where LLM-made questions fail most often and two models can share a blind spot
   - a random sample of about 25 of the rest

   Estimated total: about 2–3 hours. The README states: "AI-generated, cross-checked by a second model family, human-reviewed (all hard categories + sample)."
   - **Pilot first:** run steps 3–8 end to end on about 40 questions, fix any harness bugs, *then* scale to the full set.
6. **Split** (`split.py`):
   - About 140 test and 300+ dev questions, split evenly across categories.
   - The test set is locked by SHA-256 and never used for tuning.
   - A 20-question `smoke.jsonl` set uses a tiny fixture corpus, for CI.
7. **Harness** (`run_eval.py --config configs/X.yaml`):
   - Ingests the corpus, runs every question, caches every LLM call, and throttles requests to stay under Groq rate limits.
   - Writes `results/<arm>.json` with model IDs, git SHA and dataset hash.
   - `report.py` builds `REPORT.md`.
8. **Metrics** (deterministic wherever possible):
   - **Retrieval:** Recall@5/10, MRR and nDCG@10 by gold-page hit, plus recall at each stage (candidates → after fusion → after rerank).
   - **Answer:** exact-match or contains for factoid and ID questions. The Mistral judge (temperature 0) is used only for long answers, and is checked against about 30 labels from your sample review.
   - **Citations:** precision and recall.
   - **Abstention:** correct-refusal and false-refusal rates.
   - **Cost:** p50/p95 latency, tokens and cost per question, all measured on your own laptop.
   - **Statistics:** paired bootstrap confidence intervals.
9. **Failure attribution:** each miss is labeled retrieval, ranking, generation, or citation.
10. **CI starts now** (`.github/workflows/ci.yml`): ruff, pytest, frontend build, Docker build, and the smoke eval gate.
11. Produce the **"as shipped"** numbers by checking out `v1-as-shipped` and running the harness on it.

**Done when:** one command reproduces `REPORT.md`, and the as-shipped baseline is committed.

---

## Phase 2: Fix what's broken (every fix gets a test)

New `backend/tests/` (pytest, httpx, a fake LLM so tests are free and deterministic).

| Area | Fix |
|---|---|
| [chunker.py](backend/core/chunker.py) | Count tokens with the embedder's tokenizer and size children under the embedder's max length (MiniLM limit is ~256). Track a **page set** per chunk, not just one page. Test with a synthetic multi-page PDF. |
| [rag_engine.py](backend/core/rag_engine.py) `retrieve()` | Merge the per-variant lists with `reciprocal_rank_fusion` (reuse it from `hybrid_search.py`) instead of max-score re-sorting. |
| [bm25_index.py](backend/core/bm25_index.py) | Cache the BM25 index per source in memory and invalidate it on ingest or delete. |
| [ingest.py](backend/api/routes/ingest.py), [audio.py](backend/utils/audio.py) | New `utils/url_safety.py`: https only, exact host allow-list, DNS resolution that blocks private, loopback and link-local addresses, re-checked after redirects. Sanitize filenames (`uuid + safe ext`), check PDF magic bytes, cap size and page count. Remove `nocheckcertificate`. |
| [chat.py](backend/api/routes/chat.py), [global_chat.py](backend/api/routes/global_chat.py) | Switch to POST with a `conversation_id`; load history on the server. Add a `conversations` table in `scripts/init_db.sql`. |
| [main.py](backend/main.py) lifespan | On startup, mark stale `processing` sources as `failed`. |
| Logging | Replace `print` with `logging`, structured as JSON. |
| [evaluator.py](backend/core/evaluator.py) | Sample random chunks, not `ORDER BY chunk_index`, and label the output as a "synthetic self-check" in the API and UI. |

Run the harness to get the **"fixed baseline"** numbers. ✦ **First resume checkpoint.**

---

## Phase 3: Retrieval ablation ladder

Change one thing at a time, compare each arm against the previous one on the dev set, then confirm once on the test set.

Run in this order (highest expected value first):

| Arm | What changes |
|---|---|
| 1 | As shipped (`v1-as-shipped`) |
| 2 | Fixed baseline (end of Phase 2) |
| 3 | **Cross-encoder reranker** (small ONNX model, 20–30 candidates down to 5–8) vs MMR only |
| 4 | Embedder: MiniLM vs 1–2 newer models via FastEmbed/ONNX (may let us drop torch) |
| 5 | Query expansion: none / multi-query / HyDE / history-aware rewrite, each tested on its own |
| 6 | Chunking: parent/child vs flat, plus a sweep of child sizes |
| 7 | *Only if justified:* cached Postgres BM25 vs Qdrant-native sparse vectors (requires upgrading `qdrant-client` from 1.12) |
| 8 | *Optional:* lite config (low-RAM, no torch) |

Results are reported per metric (retrieval, answer, citation, abstention, cost). Per-category numbers are descriptive only, and small differences aren't claimed as wins.

**Decision rule:** keep a component only if its paired improvement is meaningful and its latency is acceptable. Otherwise remove it and record it in the README as a negative result. Retrieval is made configurable through `config.py` flags or eval YAML, so arms are config changes, not code forks. ✦ **Second resume checkpoint.**

---

## Phase 4: Query engine (citations, abstention, a real agent)

1. **Citations by ID.**
   - Number the chunks in the prompt (`format_context_for_llm`) and have the model cite IDs.
   - The server checks each cited ID against the retrieved, authorized set; `build_citations` keeps only valid, cited chunks.
   - The UI shows only cited chunks.
2. **Abstention.** Cheap signals (top rerank score, score margin, query-term coverage) feed a gate:
   - confident → answer
   - unsure → one LLM sufficiency check
   - insufficient → corrective retry (rewrite or decompose, at most 2 retries)
   - still insufficient → "Not found in your sources"

   Thresholds are calibrated on the dev set.
3. **Real tool-calling agent** (the "always-agentic" arm). Built **only after** steps 1–2 are working and measured. A LangGraph agent with the tools `search_sources`, `get_passage` and `compare_sources`, a step limit and a token budget.
4. **Headline experiment:** standard RAG vs always-agent vs the cascade. **All arms use the same models.** Correctness, refusals, latency and tokens are reported side by side, and the agent runs twice: unconstrained, and budget-matched to the cascade. That keeps the comparison fair without hiding the cascade's cost advantage.
5. **Plumbing.**
   - Pydantic structured outputs replace the hand-parsed JSON in [agent_router.py](backend/core/agent_router.py).
   - Upgrade langgraph, langchain and ragas in one dedicated step, with an eval run before and after.
   - Watch the `langchain-mistralai` pin.
   - Use native LangGraph streaming if the upgrade allows it; otherwise keep the queue approach.
   - The existing router stays only if the ablation shows it helps.

---

## Phase 5: Multi-user isolation and security

1. **Auth:** a `users` table, password hashing (argon2/bcrypt) and JWT. No email or OTP.
2. **Isolation.**
   - Add `user_id` to `sources` and to the Qdrant payload, with a payload index added in `QdrantStore._ensure_collection`.
   - Pre-filter every dense and sparse search by `user_id`.
   - Global chat becomes "all *my* sources".
   - Ownership mismatches return 404.
3. Simple per-user daily request cap, and a fallback provider (Mistral) when Groq returns 429.
4. **Prompt injection:** retrieved text is wrapped as untrusted data in the prompt, and poisoned-document cases are added to the eval.
5. **The 7 security tests** (pytest): cross-user retrieval, cross-user delete/read, SSRF, prompt injection, bad uploads, interrupted jobs, and citation authorization.

---

## Phase 6: MCP server

1. A new `backend/mcp_server/` using the official Python MCP SDK (version pinned; check the current spec before writing code). STDIO transport first.
2. Read-only tools that reuse the existing engine:
   - `list_sources`, `search_sources` (wraps `retrieve` and `hydrate_contexts`), `get_passage`
   - `ask_sources` second, rate-limited
3. Per-user tokens stored hashed and revocable, with the same `user_id` filtering as the web app. Tool descriptions are pinned in the repo. Output is structured JSON with passage text labeled untrusted.
4. Demo with MCP Inspector and a small Python client agent; Claude Desktop is a bonus.
5. Evals:
   - ~20 questions end-to-end through MCP
   - an injection-resistance test
   - a cross-user test

---

## Phase 7: Observability and frontend

1. **Traces.** A `traces` table recording step timings, tokens, cost, candidates per stage, gate signals and citations. The app and the eval harness write the same format.
2. **Frontend** (stays JS):
   - [useStreamingChat.js](frontend/src/hooks/useStreamingChat.js) switches from `EventSource` to fetch-streaming over POST.
   - Login screen.
   - Abstain state and confidence badge.
   - Cited chunks only, with click-through to the PDF page or a YouTube `&t=` link.
   - A collapsible "How I answered" trace panel.
   - An **Eval page** that renders the committed `results/` (ablation table, confidence intervals, latency and cost).
   - A retrieval debug page.

---

## Phase 8: Packaging (₹0)

1. `docker-compose.yml` for production (no `--reload`, no source mounts, built frontend) and `docker-compose.dev.yml` for development. Health checks, plus a seed command that loads the demo corpus.
2. A **static project page** (GitHub Pages):
   - architecture, results, limitations
   - a **replay demo**: about 10 pre-computed questions rendered from real traces, so it can never go down
3. A 2–3 minute demo video: ingest a source, get a cited answer, see a refusal, use the MCP client.
4. Final README built from `REPORT.md`, plus resume bullets filled with measured numbers only.

---

## Phase 9: Optional extras

Hosted demo (free tiers, read-only), a Postgres job queue (`FOR UPDATE SKIP LOCKED`), PDF table extraction as an ablation arm, and an Ollama local-model fallback with a Groq-vs-local comparison.

---

## Existing code to reuse (don't rewrite)

- `reciprocal_rank_fusion`, `mmr_rerank`, `hybrid_search` in [hybrid_search.py](backend/core/hybrid_search.py)
- `retrieve`, `hydrate_contexts`, `build_citations`, `format_context_for_llm`, `get_fast_llm` / `get_main_llm` in [rag_engine.py](backend/core/rag_engine.py)
- `extract_pdf_pages` and `ingest_document` in [document_pipeline.py](backend/core/document_pipeline.py); `ingest_video` and `transcribe_audio` in [video_pipeline.py](backend/core/video_pipeline.py). The eval harness ingests through these.
- `QdrantStore` ([qdrant.py](backend/db/qdrant.py)) and the `Database` helpers ([postgres.py](backend/db/postgres.py))
- `_build_ragas_llm` in [evaluator.py](backend/core/evaluator.py) for the Mistral client, used by the judge and the cross-check
- `format_sse` / `event_stream` in [streaming.py](backend/utils/streaming.py)

## Verification (each phase)

- `pytest backend/tests` passes, and the CI workflow is green on GitHub.
- `python eval/run_eval.py --config eval/configs/<arm>.yaml` reproduces the committed `results/<arm>.json`, with responses served from the cache.
- Manual check: run the stack and use the app in the browser preview (upload a source, ask a question, check citations and refusals).
- SSE contract check: `curl -N`.
- Phase 5: run the 7 security tests, including the SSRF probe `http://169.254.169.254/latest/meta-data/?x=youtube.com`, which must be rejected.
- Phase 6: MCP Inspector lists the tools and runs `search_sources` with a user token.
- Rule: no number goes into the README or resume unless a committed results file backs it.
