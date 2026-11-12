# How SourceMind is evaluated

This is the reference for the benchmark: what is measured, how, and what the numbers do and don't mean.

## 1. The corpus

Defined in [`eval/corpus/manifest.yaml`](../eval/corpus/manifest.yaml). Every file is redistributable and hash-locked.

| Key | Source | Type | License |
|---|---|---|---|
| `ragas` | RAGAS paper (arXiv 2309.15217v2) | research paper | CC BY 4.0 |
| `self_rag` | Self-RAG paper (arXiv 2310.11511v1) | research paper | CC BY 4.0 |
| `beir` | BEIR paper (arXiv 2104.08663v4) | research paper | CC BY-SA 4.0 |
| `nist_ai_rmf` | NIST AI 100-1, AI Risk Management Framework | standard | US Gov (public domain) |
| `nist_genai_profile` | NIST AI 600-1, Generative AI Profile | standard | US Gov (public domain) |
| `nist_csf` | NIST Cybersecurity Framework 2.0 | standard, many IDs (e.g. `PR.AA-01`) | US Gov (public domain) |
| `nist_800_63b` | NIST SP 800-63B, Digital Identity Guidelines | manual / rules | US Gov (public domain) |
| `census_income` | Census P60-282, Income in the US: 2023 | report, numbers and tables | US Gov (public domain) |
| `fed_fsr` | Federal Reserve Financial Stability Report, Apr 2024 | financial report | US federal publication |
| `mit_ai_lec1` | MIT 6.034 Lecture 1 (video) | lecture | CC BY-NC-SA 4.0 (MIT OCW) |
| `mit_linalg_lec1` | MIT 18.06 Lecture 1 (video) | lecture | CC BY-NC-SA 4.0 (MIT OCW) |
| `mit_algo_lec1` | MIT 6.006 Lecture 1 (video) | lecture | CC BY-NC-SA 4.0 (MIT OCW) |

That's 9 PDFs (414 pages) and 3 lectures (about 2.4 hours). The mix is deliberate:
- The papers overlap in topic (all RAG/retrieval), which makes retrieval harder.
- The NIST documents are full of exact identifiers, which tests keyword search.
- The reports are number-heavy.
- The lectures test timestamp citations.

```bash
python -m eval.corpus.download      # fetch PDFs + verify sha256
python -m eval.corpus.transcribe    # one-off: Whisper transcripts (committed)
```

## 2. The questions

### Schema
One JSON object per line ([`eval/datasets/schema.py`](../eval/datasets/schema.py)):

```json
{"id": "t-017", "category": "exact_id", "doc": "nist_csf",
 "question": "What does subcategory PR.AA-01 require?", "answerable": true,
 "gold_answer": "Identities and credentials for authorized users ... are managed",
 "gold_evidence": [{"doc": "nist_csf", "pages": [18]}]}
```

- `doc` is the source the question is asked against. Retrieval is scoped to it, as in the per-source chat.
- **Gold evidence is pages or time windows, never chunk IDs**, so the benchmark survives chunking changes.

### Categories

| Category | Tests |
|---|---|
| `factoid` | basic retrieval of one fact |
| `exact_id` | exact identifiers and values (keyword/BM25 strength) |
| `paraphrase` | the same question in different words (semantic strength) |
| `comparative` | two separate pages must both be found |
| `multi_hop` | two facts chained across two pages |
| `unanswerable` | the system must say "not found" (in-document trick questions, plus questions borrowed from an unrelated document) |
| `follow_up` | a second turn that depends on chat history |

### How they're made (AI-generated, cross-checked, human-reviewed)

1. **Generate** ([`gen_questions.py`](../eval/tools/gen_questions.py)): the fast Groq model (`gpt-oss-20b`) writes one question from a **random** page or transcript window. Table-of-contents and reference-list pages are skipped. Every answerable question must include a verbatim quote, and a quote that can't be found on the cited page is rejected automatically.
2. **Cross-check** ([`crosscheck.py`](../eval/tools/crosscheck.py)): Mistral (a different model family) checks:
   - support and correctness
   - ambiguity
   - self-containment
   - for two-page questions, that both pages are needed
   - for unanswerable questions, that none of the 8 most related pages (BM25) answers it

   Each question is marked pass, flag or fail.
3. **Human review** ([`review_app.py`](../eval/tools/review_app.py)): the rendered PDF page next to the question, with Keep / Edit / Reject. The reviewer sees every flagged question, every unanswerable and multi-hop question, and a random sample of 25 others.
4. **Split** ([`split.py`](../eval/tools/split.py)):
   - About 140 test questions (35 factoid, 20 exact_id, 20 paraphrase, 15 comparative, 15 multi_hop, 25 unanswerable, 10 follow_up) and the rest as dev.
   - The test set is locked by sha256 (`test.sha256`, checked in CI) and **never used for tuning**.
   - A pilot of about 40 questions validates the whole pipeline before scaling.

## 3. Running an experiment

```bash
python -m eval.run_eval --config eval/configs/<arm>.yaml   # asks every question
python -m eval.score --arm <arm>                            # scores it
python -m eval.report                                       # rebuilds eval/REPORT.md
```

- Each **arm** is a YAML config. It can point at a different code checkout (`backend_dir`, e.g. the `v1-as-shipped` worktree) and override settings.
- The harness rebuilds an **isolated** database and Qdrant collection (`sourcemind_eval`) from that arm's own schema and code, then ingests the corpus. Videos are ingested from the committed transcripts, so Whisper doesn't rerun.
- Questions go through the real chat path (`route_and_stream`), exactly as the API uses it.
- Output: `eval/results/<arm>/run.json` (models, git SHA, dataset hash, settings), `records.jsonl` (every answer) and `scores.json`. Results are committed; LLM caches are not.

## 4. Metrics

| Group | Metric | Definition |
|---|---|---|
| Retrieval | hit@k | a gold page/window is among the first k items given to the LLM |
| | MRR | mean of 1 / rank of the first gold item |
| | evidence recall@5 | share of the gold evidence items found in the top 5; matters for two-page questions |
| Answer | answer correct | short golds: normalized containment; otherwise, or when containment fails, a Mistral judge (correct / partial / incorrect, temperature 0, cached). Reported strictly (partial = wrong), plus a with-partial score |
| Abstention | correct refusal rate | unanswerable questions answered with "not found" |
| | false refusal rate | answerable questions wrongly refused |
| Citations | inline precision / recall | pages or timestamps the answer *writes* vs gold |
| | shown-card precision | citation cards the UI displays vs gold (v1 shows every retrieved chunk) |
| Cost | latency p50 / p95 | wall-clock per question on the developer's laptop, including Groq network time |

**Failure attribution** (wrong answers only):
- *retrieval*: gold never reached the LLM
- *generation*: gold was retrieved, but the answer was wrong or refused
- *citation*: the answer was correct, but its citations point elsewhere
- *hallucination*: an unanswerable question was answered

**Statistics:**
- 95% percentile-bootstrap intervals for every headline metric.
- Arms on the same dataset are compared with a paired bootstrap (per-question differences).
- Per-category numbers are descriptive only.
- With roughly 140 test questions, only differences of about 10 points or more are reliably detectable. Smaller gaps are reported as "no clear difference."

## 5. Known limitations

- The questions are LLM-generated. Cross-checking with a second model family plus targeted human review reduces errors but doesn't remove them, and both models can share blind spots.
- Gold evidence is the page the question was written from. If the same fact also appears on another page, retrieving that page counts as a miss (pessimistic).
- In v1, a child chunk's page is the page where the chunk *starts*, so a chunk crossing a page break can be scored as the earlier page.
- Token usage isn't recorded yet (the streamed answer path doesn't expose it); only latency is.
- Latency depends on Groq's free tier and network, so compare arms run close together in time.
