# Demo script (about 5 minutes)

## 0. Before the demo

```bash
docker start sourcemind-postgres sourcemind-qdrant
cd backend && python -m uvicorn main:app --port 8000      # terminal 1 (venv active)
cd frontend && npm run dev                                 # terminal 2
python -m eval.corpus.download                             # from the repo root, once: fetches the demo PDFs
```

- Check `http://localhost:8000/health`. It shows the active LLM provider.
- On Groq's free tier, a full afternoon of testing can use up the 200K tokens/day limit. For a live demo, `LLM_PROVIDER=deepseek` (about $0.001 per question) avoids that risk.

**Demo corpus** (public, redistributable; hash-locked in [`eval/corpus/manifest.yaml`](../eval/corpus/manifest.yaml), downloaded to `eval/corpus/files/`):

| File | What it is | License |
|---|---|---|
| `self_rag.pdf` | Self-RAG paper (arXiv 2310.11511) | CC BY 4.0 |
| `ragas.pdf` | Ragas paper (arXiv 2309.15217) | CC BY 4.0 |
| `nist_800_63b.pdf` | NIST SP 800-63B, Digital Identity Guidelines | US Government work |
| YouTube `TjZBTDzGeGg` | MIT 6.034 Lecture 1 (optional) | CC BY-NC-SA 4.0 |

Upload the three PDFs before the demo; each takes about 30–60 s to ingest. The video is optional: CPU transcription of a 50-minute lecture takes a while.

## 1. Ingestion (30 s)

Show the library: PDFs and videos, status, and the per-source RAGAS self-check. Say plainly that it is a self-check, not a benchmark.

## 2. A cited answer (1 min)

Open **self_rag.pdf** and ask:

> What is the accuracy of ChatGPT on PopQA?

- Expected: **29.3**, citing page **8**.
- Open the citations. The card number matches the `[n]` in the answer, and only the cited passage is listed, not every retrieved chunk.

Open **nist_800_63b.pdf** and ask:

> What is the typical minimum iteration count recommended for PBKDF2?

- Expected: **at least 10,000 iterations**, page **26**.

## 3. Comparison (1 min)

Open **ragas.pdf** and ask:

> How does the Ragas metric's agreement with human annotators for faithfulness compare to that of the GPT Score metric?

- The router labels the question **comparative**, splits it into two sub-queries, and then combines the answers.
- Expected: Ragas **0.95** vs GPT Score **0.72** (page 5).

## 4. "Not in sources" (30 s)

On any document, ask something it cannot answer:

> What was the company's annual revenue in 2022?

- Expected: "I could not find this in the provided sources." with a **not in sources** badge and **no citations**.
- Point out the design: when nothing relevant is retrieved, the LLM is never called.

## 5. Global chat (30 s)

Open **Global chat** and ask:

> Compare how Self-RAG and Ragas evaluate retrieval-augmented generation.

The citations come from both papers, each labeled with its title and page.

## 6. MCP (1 min)

```bash
cd backend
python mcp_demo_client.py "What is the accuracy of ChatGPT on PopQA?"
```

This launches the MCP server over stdio, the same way Claude Desktop does, and calls `list_sources`, `search_sources`, `get_passage` and `ask_sources`. Point out:
- passage text is returned as `untrusted_content`;
- the answer carries validated citations and an `evidence` status.

## 7. Engineering (if asked)

- `backend/tests/`: 63 tests, including an isolation test on real Postgres + Qdrant and a real MCP client session.
- `eval/`: benchmark harness with page-level gold labels, a frozen test split, and paired bootstrap CIs. [EVALUATION.md](EVALUATION.md) describes the method.
- Provider choice is explicit (`LLM_PROVIDER`), with no silent fallback. The 5-question Groq vs DeepSeek comparison is in `eval/results/v2_*_compare5/`.

## Verified answers

| Question | Answer | Page | Verified by |
|---|---|---|---|
| ChatGPT accuracy on PopQA | 29.3 | Self-RAG p. 8 | full pipeline, Groq and DeepSeek |
| PBKDF2 minimum iterations | 10,000 | 800-63B p. 26 | full pipeline, Groq and DeepSeek |
| Ragas vs GPT Score faithfulness | 0.95 vs 0.72 | Ragas p. 5 | full pipeline, Groq and DeepSeek |
| Textbook used in MIT 6.034 | not in the lecture | — | both providers refused (correct) |

The first three were checked through the full pipeline (`eval/results/v2_*_compare5`). The two cross-document questions (section 5) and the revenue question (section 4) were not run in that comparison. The revenue question was tried on a different document in the app.
