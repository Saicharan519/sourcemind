# Progress log

## Phase 0: Freeze and be honest (done 2026-11-12)

Branch `sourcemind-2.0`.

- `7febe8c`: pinned models to `openai/gpt-oss-120b` / `openai/gpt-oss-20b`; RAGAS runner throttled to one worker. Tagged **`v1-as-shipped`**.
- `13be429`:
  - README: removed "production-grade" and "agentic" claims, stated chunk sizes are characters, relabeled RAGAS as a synthetic self-check, added Current limitations.
  - PRD: author fixed, divergence note added.
  - Added an MIT LICENSE.
  - Removed `sqlalchemy` and `psycopg2-binary`.

Verified: the backend imports and starts with the pinned models.

## Phase 1: Measurement foundation (in progress)

Design: [EVALUATION.md](EVALUATION.md).

**Done**
- Corpus: 9 redistributable PDFs (414 pages) + 3 MIT OCW lectures. Licenses checked, PDFs hash-locked in `eval/corpus/manifest.yaml`, Whisper transcripts committed.
- Question pipeline: generator with verbatim-quote check and self-containment filter, cross-checker, review app (rendered PDF page + Keep/Edit/Reject), split script with a sha256-locked test set.
- Harness: isolated `sourcemind_eval` DB and collection, full-corpus ingest in ~2 min, the real `route_and_stream` chat path, resumable, flags degraded answers. Scorer (retrieval / answer / abstention / citation / latency, bootstrap CIs, failure attribution) and report generator.
- `v1-as-shipped` worktree at `.worktrees/v1-as-shipped` for measuring the old code.
- CI: ruff, 20 unit tests, dataset validation, frozen-test hash check, frontend build.
- Pilot set: 46 candidates → 43 passed a (same-family, pilot-only) cross-check → `eval/datasets/pilot.jsonl`.

**Bugs found while building**
1. `ingest_document` deletes its input PDF. The harness now ingests a copy.
2. Generated two-passage questions said "passage A/B". Prompts were fixed and a filter added.
3. The scorer was too harsh when a fact appears on two pages. Alternate pages are now accepted for short exact answers.
4. The backend silently drops HyDE / multi-query on rate limits. Now detected and retried.

**Blocked / open**
- **No independent checker key yet.** Add `DEEPSEEK_API_KEY` to `backend/.env`.
- **Groq free-tier quota** (200K tokens/day per model) ran out during the first pilot run. The pilot must be rerun after the quota resets, and a decision is needed on how to fund full benchmark runs.

## Release push (2026-11-12): local demo release

Scope cut to a stable local demo (no full agent, no auth). Commits `fe35185`, `76c0e7e` and the docs commit.

**Done and tested**
- Retrieval: RRF fusion across query variants; cached BM25 with fingerprint invalidation, built off the event loop; failed expansions surfaced as a `warning` event.
- Citations: numbered context, server-side validation, only cited passages shown, `evidence` status; gpt-oss `【n】` citations accepted (found live).
- Insufficient evidence: fixed "not found" answer; no LLM call on empty retrieval.
- Security: SSRF-safe video URLs, server-chosen upload names, PDF magic bytes, no `nocheckcertificate`, interrupted jobs marked failed on startup.
- Providers: `LLM_PROVIDER` groq / deepseek / cloudflare, no fallback. Groq and DeepSeek verified live (chat + streaming); Cloudflare unit-tested only (no keys).
- MCP server with 4 tools, verified over real stdio against the app DB.
- Bugs found and fixed: duplicate dashboard rows (a join on every eval run); the embedder loaded 5x on the first query; raw provider errors (with account ids) were shown in the chat.
- 63 backend tests. Browser check: cited answer, citation cards, "not in sources" badge, no console or server errors.

**Measured**
5-question Groq vs DeepSeek comparison (`eval/results/v2_*_compare5`):
- DeepSeek: about $0.004 total, 10.9 s median.
- Groq: 17.8 s median, with about 4x the output tokens.

**Not done**
- The full benchmark (pilot rerun, full set, as-shipped vs 2.0).
- Reranker ablation, corrective re-retrieval, auth, and backend tests in CI.
