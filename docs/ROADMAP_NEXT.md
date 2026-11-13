# Next phases (after the 2.0 local release)

The 2.0 release (merged to `main`) shipped validated citations, the insufficient-evidence answer, retrieval and security fixes, provider selection and the MCP server. This plan covers what is left, in four phases. Each phase ends in a complete, showable state, so work can stop after any of them.

**Status (2026-11-13):** all four phases are deferred to future work; nothing here is started. When resuming, start with Phase A. It needs a way to run about 300 Groq questions (free tier: about 50/day, so about 6 days; or Groq pay-as-you-go, about $1–2).

Every phase only *adds* to the working app: the current release keeps running unchanged until a phase's changes are merged, and each phase is tested before it lands.

---

## Phase A: Real numbers (benchmark)

The gap that remains: there is no measured "accuracy went from X to Y" yet.

1. Re-check the 43-question pilot with DeepSeek as the independent checker, run both arms on it, fix any harness issues.
2. Generate the full question set (~140 test + ~300 dev) and cross-check it with DeepSeek. The owner then reviews the flagged, unanswerable and multi-hop questions in `eval/tools/review_app.py` (about 2 hours).
3. Lock the test set (sha256), then run **v1-as-shipped vs 2.0** on it.
4. Score with a judge from a third model family (Groq `llama-3.3-70b-versatile`): the questions come from gpt-oss and are checked by DeepSeek. Then build `eval/REPORT.md` and put only those numbers in the README and resume.

**Provider for the runs:** the v1 code is Groq-only, and a fair comparison needs the same model in both arms. Groq's free tier allows about 50 questions/day (about 6 days for the test set), or about $1–2 on Groq pay-as-you-go for a one-day run. Every run pins one provider with no fallback, and records it in `run.json`.

**Done when:** `eval/REPORT.md` shows both arms with 95% confidence intervals, backed by committed `eval/results/*/scores.json`, and the README quotes them.

**Estimate:** 1–2 days, plus about 2 hours of review.

## Phase B: Answer quality (measured)

Every change is judged on `dev.jsonl` against the Phase A baseline, and confirmed once on the test set. A change that doesn't help is removed and reported as a negative result.

1. **Cross-encoder reranker** (small ONNX/sentence-transformers model, 20–30 candidates → 6), with vs without.
2. **Corrective re-search:** when `evidence` is insufficient, rewrite the query and retrieve once more before answering "not found". This targets the multi-hop miss seen in the 5-question check.
3. **Token-aware chunking** (sized to MiniLM's ~256-token limit) vs the current character sizes.
4. Retrieval settings switchable by config, so each arm is a YAML file, not a code fork.

**Done when:** an ablation table in `REPORT.md` shows each component's effect and the final configuration's test-set score.

**Estimate:** 2–3 days.

## Phase C: Multi-user and security (later)

1. Password + JWT login; a `user_id` on sources, the Qdrant payload, and every search filter.
2. Global chat becomes "all my sources"; another user's ids return 404.
3. Chat moves to POST with server-side conversation history (no history in GET query strings).
4. Per-user, revocable tokens for the MCP server.
5. Prompt-injection test documents, per-user daily request caps, JSON logging.
6. Backend tests in GitHub CI (with Postgres and Qdrant service containers).

**Done when:** the security suite (cross-user read/delete/retrieval, SSRF, injection, bad uploads, interrupted jobs, citation authorization) passes in CI.

**Estimate:** 2–3 days.

## Phase D: Agent, tracing, showcase (later)

1. A tool-calling agent over the same tools as the MCP server (`core/source_tools.py`), with a step limit and a token budget.
2. **Headline experiment:** fixed RAG vs the corrective pipeline vs the agent, with the same model and cost-matched. Compare accuracy, refusals, latency and tokens.
3. Per-answer traces, a "How I answered" panel, and an evaluation page in the UI.
4. A production Docker Compose setup, a GitHub Pages project page with a replay demo, and a 2–3 minute demo video.

**Done when:** the agent comparison is in `REPORT.md`, the project page is live, and the video is recorded.

**Estimate:** 3–4 days.

---

| Phase | Delivers | Time | API cost |
|---|---|---|---|
| A | Real before/after numbers | 1–2 days | $1–2 (or free over ~6 days) |
| B | Measured quality improvements | 2–3 days | < $1 |
| C | Auth, per-user isolation, CI | 2–3 days | $0 |
| D | Agent vs RAG result, showcase | 3–4 days | ~$1 |
