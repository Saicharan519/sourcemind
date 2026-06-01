# SourceMind — Product Requirements Document (PRD)
### Multi-Modal Document & Video Intelligence Platform

**Version:** 2.0  
**Author:** Tarun  
**Stack:** React + FastAPI + LangChain + Qdrant + Whisper + PostgreSQL + Docker  
**LLM:** Groq API (Llama 3.1 / 3.3 70B) — Free Tier  
**Embeddings:** HuggingFace all-MiniLM-L6-v2 — Local, Free  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Tech Stack — Full Breakdown](#4-tech-stack--full-breakdown)
5. [System Architecture](#5-system-architecture)
6. [Database Schema](#6-database-schema)
7. [Backend — API Specification](#7-backend--api-specification)
8. [Core Pipelines](#8-core-pipelines)
9. [RAG Engine](#9-rag-engine)
10. [Agentic Query Router (LangGraph)](#10-agentic-query-router-langgraph)
11. [RAG Evaluation Layer (RAGAS)](#11-rag-evaluation-layer-ragas)
12. [Frontend — React](#12-frontend--react)
13. [Folder Structure](#13-folder-structure)
14. [Environment Variables](#14-environment-variables)
15. [Docker Compose Setup](#15-docker-compose-setup)
16. [Build Order](#16-build-order)
17. [Free Model Strategy](#17-free-model-strategy)
18. [Resume Bullets](#18-resume-bullets)

---

## 1. Project Overview

**SourceMind** is a production-grade multi-modal RAG (Retrieval-Augmented Generation) platform. Users upload PDF documents or paste a YouTube URL — and then chat with that content using natural language. The system answers questions with cited sources: **page numbers** for documents, **segment references** for videos.

Beyond basic RAG, SourceMind includes:
- An **agentic query router** (LangGraph) that handles multi-hop and comparative questions
- An **automated evaluation layer** (RAGAS) that scores retrieval quality on any ingested source
- **Hybrid search** (BM25 + Vector + MMR) for superior retrieval quality
- **Streaming responses** via Server-Sent Events for real-time chat UX
- **Cross-document querying** to chat across all uploaded sources simultaneously
- **Async ingestion** via FastAPI BackgroundTasks so large files never block the API

This is not a tutorial clone. The engineering challenge is building RAG that actually works well on long documents and video transcripts — without retrieval quality degrading at scale.

---

## 2. Problem Statement

### Why basic RAG fails

Most RAG tutorials do this:
1. Split document into 500-token chunks
2. Embed all chunks
3. On query: retrieve top-5 chunks by cosine similarity
4. Feed to LLM

This breaks in practice for several reasons:

**Problem 1 — Short chunks lose context.**  
A chunk says "This approach was first proposed in 1987." Without surrounding context, the LLM has no idea what "this approach" refers to. Answer is wrong or hallucinated.

**Problem 2 — Query-document phrasing mismatch.**  
User asks "what is the attention mechanism?" but the book says "the scaled dot-product attention operation." The embeddings don't match. The right chunk is never retrieved.

**Problem 3 — Multiple sources bleed into each other.**  
User uploads a machine learning textbook and a cooking video. Asks "explain gradient descent." The retriever pulls chunks from the cooking video that contain words like "steps." Wrong context, confused LLM.

**Problem 4 — Redundant retrieved chunks.**  
Top-k similarity search returns near-duplicate chunks. The LLM gets 4 almost-identical passages instead of 4 diverse, relevant ones. Context window wasted.

**Problem 5 — Vector search misses exact keyword matches.**  
Pure semantic search fails on proper nouns, acronyms, and technical terms. "BERT" and "GPT-4o" have embedding representations but keyword search would catch them better.

**Problem 6 — Single-step retrieval fails on complex queries.**  
"Compare the conclusions of chapter 3 and chapter 7" requires two separate retrievals and a synthesis step. A single retrieval call will botch this.

**Problem 7 — No way to know if your RAG is actually good.**  
You build it, you test it manually, you have no numbers. You can't compare flat chunking vs hierarchical chunking quantitatively.

### SourceMind solves all seven:

| Problem | Solution |
|---|---|
| Short chunks lose context | Hierarchical parent-child chunking |
| Phrasing mismatch | HyDE + Multi-query retrieval |
| Cross-source bleed | Metadata-filtered Qdrant retrieval per source_id |
| Redundant chunks | MMR (Maximal Marginal Relevance) retrieval |
| Keyword miss by vector search | Hybrid search: BM25 + Vector combined |
| Complex multi-part queries | LangGraph agentic query router |
| No quality measurement | RAGAS automated evaluation pipeline |

---

## 3. Goals & Non-Goals

### Goals
- Upload any PDF (up to 100MB, 500+ pages) and chat with it with page citations
- Paste any YouTube URL and chat with the transcript with segment citations
- Multi-source support — manage multiple documents/videos, switch between them
- **Cross-document querying** — chat across all sources at once via global search
- Advanced RAG: hierarchical chunking, HyDE, multi-query retrieval, MMR, hybrid search
- Agentic query routing via LangGraph for comparative and multi-hop questions
- Automated RAGAS evaluation score displayed per source after ingestion
- **Streaming responses** — tokens stream to frontend via SSE in real time
- **Async ingestion** — large PDFs and videos processed in background, never block API
- Fully Dockerized — one `docker-compose up` command to run everything
- 100% free model stack — no paid API required for core functionality

### Non-Goals (for v1)
- User authentication / multi-user support
- Real-time collaboration
- Mobile app
- Support for non-YouTube video sources (local video files, Vimeo, etc.)
- Fine-tuning any models
- Production cloud deployment (this is a portfolio/demo project)
- Redis caching layer
- Celery/Dramatiq task queues

---

## 4. Tech Stack — Full Breakdown

### Frontend
| Tool | Version | Purpose |
|---|---|---|
| React | 18+ | UI framework |
| Vite | 5+ | Build tool, dev server |
| TailwindCSS | 3+ | Styling |
| Axios | 1.6+ | HTTP client for API calls |
| React Query (TanStack) | 5+ | Server state management, loading/error states |
| React Router | 6+ | Client-side routing |
| EventSource API | Native browser | Consuming SSE streaming responses |

### Backend
| Tool | Version | Purpose |
|---|---|---|
| FastAPI | 0.110+ | Async REST API + SSE streaming |
| Uvicorn | 0.29+ | ASGI server |
| Pydantic | 2+ | Request/response validation |
| asyncpg | 0.29+ | Async PostgreSQL driver |
| SQLAlchemy | 2+ | ORM for PostgreSQL |
| python-multipart | — | File upload handling |
| FastAPI BackgroundTasks | built-in | Async ingestion without blocking API |

### AI / ML
| Tool | Version | Purpose |
|---|---|---|
| LangChain | 0.2+ | RAG chain composition (LCEL) |
| LangChain-Community | 0.2+ | HuggingFace embeddings, BM25 integration |
| LangGraph | 0.1+ | Agentic query router state machine |
| langchain-groq | 0.1+ | Groq LLM integration |
| sentence-transformers | 3+ | all-MiniLM-L6-v2 local embeddings |
| openai-whisper | 20231117+ | Local speech-to-text transcription |
| ragas | 0.1+ | RAG evaluation metrics |
| PyMuPDF (fitz) | 1.24+ | PDF text extraction with page numbers |
| yt-dlp | 2024.4+ | YouTube audio download |
| pydub | 0.25+ | Audio processing |
| rank-bm25 | 0.2+ | BM25 sparse retrieval for hybrid search |

### Databases
| Tool | Purpose |
|---|---|
| Qdrant | Vector database — stores embeddings with metadata, supports hybrid search |
| PostgreSQL 16 | Relational DB — parent chunks, source metadata, eval results, BM25 index |

### DevOps
| Tool | Purpose |
|---|---|
| Docker | Containerization |
| Docker Compose | Multi-service orchestration |
| GitHub Actions | CI pipeline (lint + type check) |

### LLM — Free Stack
| Model | Provider | Used For | Free Tier |
|---|---|---|---|
| Llama 3.3 70B | Groq API | RAG answers, summarization, extraction | Yes — 6K TPM, 500K TPD |
| Llama 3.1 8B | Groq API | Multi-query rephrasing, HyDE generation | Yes — 30K TPM, higher RPD |
| all-MiniLM-L6-v2 | HuggingFace (local) | Embeddings | Completely free, runs locally |
| Whisper small | OpenAI Whisper (local) | Audio transcription | Completely free, runs locally |

> **Strategy:** Use Llama 3.1 8B (fast, generous limits) for all auxiliary LLM calls — multi-query rephrasing, HyDE, eval QA generation, RAGAS scoring. Use Llama 3.3 70B only for the final answer generation where quality matters most.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       REACT FRONTEND (Vite)                         │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │  Upload Panel  │  │ Source Sidebar │  │   Chat Interface       │ │
│  │  PDF / YouTube │  │ List + Switch  │  │   + Citations          │ │
│  └────────────────┘  └────────────────┘  │   + Streaming tokens   │ │
│                                          └────────────────────────┘ │
│  ┌──────────────────┐  ┌───────────────────────────────────────────┐ │
│  │  Global Search   │  │   RAGAS Eval Score Card (per source)      │ │
│  │  (cross-doc chat)│  └───────────────────────────────────────────┘ │
│  └──────────────────┘                                               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ REST API + SSE (HTTP)
┌─────────────────────────────▼───────────────────────────────────────┐
│                        FASTAPI BACKEND                              │
│                                                                     │
│  POST /api/ingest/document       POST /api/ingest/video             │
│  GET  /api/chat/stream           GET  /api/chat/global/stream       │
│  POST /api/evaluate/{id}         GET  /api/sources                  │
│  DELETE /api/sources/{id}                                           │
│                                                                     │
│  BackgroundTasks: ingestion runs async, API returns immediately     │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │                              │
┌──────────────▼──────────┐     ┌─────────────▼───────────┐
│   DOCUMENT PIPELINE     │     │    VIDEO PIPELINE        │
│                         │     │                          │
│  PyMuPDF                │     │  yt-dlp → audio          │
│  → text + page nums     │     │                          │
│                         │     │  Whisper (local)          │
│  Hierarchical Chunker   │     │  → segments with         │
│  → PARENT (1500 tokens) │     │    start/end positions   │
│  → CHILD  (300 tokens)  │     │                          │
│                         │     │  Segment Chunker         │
│  Parents → PostgreSQL   │     │  → groups (~400 words)   │
│  Children → Qdrant      │     │  → store segment range   │
└──────────────┬──────────┘     └─────────────┬────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                     SHARED INGESTION LAYER                          │
│                                                                     │
│  1. Embed child chunks → HuggingFace all-MiniLM-L6-v2 (local)      │
│  2. Build BM25 index over chunk text (rank-bm25)                    │
│  3. Attach metadata:                                                │
│     { source_id, source_type, filename/title,                       │
│       page_number (docs) | segment_start + segment_end (video) }    │
│  4. Upsert vectors → Qdrant collection "sourcemind"                 │
│  5. Auto-generate title + summary via Groq 8B                       │
│  6. Kick off RAGAS evaluation in background                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
          ┌───────────────────┴──────────────────┐
          │                                      │
┌─────────▼────────┐                   ┌─────────▼────────┐
│     QDRANT       │                   │   POSTGRESQL     │
│  Vector Store    │                   │  Relational DB   │
│                  │                   │                  │
│  Collection:     │                   │  - sources       │
│  "sourcemind"    │                   │  - parent_chunks │
│                  │                   │  - bm25_index    │
│  Dense vectors   │                   │  - eval_results  │
│  + sparse BM25   │                   │  - chat_history  │
│  per chunk       │                   └──────────────────┘
└─────────┬────────┘
          │
┌─────────▼──────────────────────────────────────────────────────────┐
│                    LANGGRAPH QUERY ROUTER                           │
│                  (on every /api/chat/stream request)               │
│                                                                     │
│  Node 1: CLASSIFIER                                                 │
│  → Classify: simple | comparative | multi_hop                       │
│                                                                     │
│  Node 2: SIMPLE PATH                                                │
│  → Hybrid search (BM25 + Vector, MMR rerank)                       │
│  → Multi-query (3 rephrasings) + HyDE                              │
│  → Metadata filter by source_id                                     │
│  → Parent fetch from PostgreSQL                                     │
│  → Groq 70B → SSE stream answer with citations                     │
│                                                                     │
│  Node 3: COMPARATIVE PATH                                           │
│  → Decompose into 2 sub-queries                                     │
│  → 2 independent hybrid retrievals                                  │
│  → Groq 70B synthesizes → SSE stream                               │
│                                                                     │
│  Node 4: MULTI-HOP PATH                                             │
│  → Step 1 retrieval → intermediate answer                           │
│  → Step 2 retrieval using step 1 answer as context                 │
│  → Final synthesis → SSE stream                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. Database Schema

### PostgreSQL Tables

```sql
-- Sources: every ingested document/video
CREATE TABLE sources (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type   VARCHAR(10) NOT NULL CHECK (source_type IN ('document', 'video')),
    title         TEXT NOT NULL,
    filename      TEXT,
    youtube_url   TEXT,
    page_count    INTEGER,
    duration_s    INTEGER,
    status        VARCHAR(20) DEFAULT 'processing',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Parent chunks: large context chunks for document RAG
CREATE TABLE parent_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id     UUID REFERENCES sources(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    page_start    INTEGER,
    page_end      INTEGER,
    chunk_index   INTEGER NOT NULL
);

-- BM25 index: sparse keyword index per chunk for hybrid search
CREATE TABLE bm25_index (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id     UUID REFERENCES sources(id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL,        -- matches Qdrant vector ID
    content       TEXT NOT NULL,        -- raw text for BM25 scoring
    chunk_index   INTEGER NOT NULL
);

-- Eval results: RAGAS scores per source
CREATE TABLE eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id           UUID REFERENCES sources(id) ON DELETE CASCADE,
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_recall      FLOAT,
    context_precision   FLOAT,
    overall_score       FLOAT,
    eval_questions      JSONB
);

-- Chat history: per-source conversation memory
CREATE TABLE chat_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID REFERENCES sources(id) ON DELETE CASCADE,
    role        VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### Qdrant Collection

```
Collection name: "sourcemind"
Vector size: 384 (all-MiniLM-L6-v2 output dimensions)
Distance metric: Cosine
Sparse vectors: enabled (for BM25 hybrid search)

Payload schema per vector:
{
  "source_id":        "uuid-string",
  "source_type":      "document" | "video",
  "content":          "chunk text content",
  "page_number":      12,              // documents only
  "segment_start":    142.5,           // video only (seconds)
  "segment_end":      198.2,           // video only (seconds)
  "parent_chunk_id":  "uuid-string",   // documents only
  "chunk_index":      7
}
```

---

## 7. Backend — API Specification

### POST `/api/ingest/document`
Upload a PDF. Returns immediately. Ingestion runs in background via FastAPI `BackgroundTasks`.

**Request:** `multipart/form-data`
```
file: <PDF file binary>
```

**Response:** `202 Accepted`
```json
{
  "source_id": "uuid",
  "status": "processing",
  "message": "Document ingestion started"
}
```

Poll `GET /api/sources/{id}` for status updates.

---

### POST `/api/ingest/video`
Provide a YouTube URL. Returns immediately. Pipeline runs in background.

**Request:** `application/json`
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=xxxx"
}
```

**Response:** `202 Accepted`
```json
{
  "source_id": "uuid",
  "status": "processing",
  "message": "Video ingestion started"
}
```

---

### GET `/api/sources`
List all ingested sources.

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "source_type": "document",
    "title": "Attention Is All You Need",
    "status": "ready",
    "page_count": 15,
    "eval_score": 0.82
  },
  {
    "id": "uuid",
    "source_type": "video",
    "title": "3Blue1Brown: Neural Networks",
    "status": "ready",
    "eval_score": 0.76
  }
]
```

---

### GET `/api/sources/{source_id}`
Get full source details including RAGAS scores.

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "source_type": "document",
  "title": "Attention Is All You Need",
  "status": "ready",
  "page_count": 15,
  "eval_results": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.85,
    "context_recall": 0.78,
    "context_precision": 0.83,
    "overall_score": 0.84
  }
}
```

---

### DELETE `/api/sources/{source_id}`
Delete source and all associated vectors, parent chunks, eval results.

**Response:** `200 OK`
```json
{ "message": "Source deleted successfully" }
```

Deletes from: PostgreSQL (cascade) + Qdrant (filter delete by source_id) + BM25 index.

---

### GET `/api/chat/stream`
**Streaming endpoint — Server-Sent Events (SSE).**  
Tokens stream to the client in real time as the LLM generates them.

**Query params:**
```
source_id: uuid
question:  string
history:   JSON-encoded array of {role, content} (optional)
```

**Response:** `text/event-stream`
```
data: {"type": "query_type", "value": "simple"}

data: {"type": "token", "value": "The "}
data: {"type": "token", "value": "attention "}
data: {"type": "token", "value": "mechanism "}
...
data: {"type": "citations", "value": [{"type": "document", "page_number": 4, "excerpt": "..."}]}
data: {"type": "done"}
```

Frontend uses the native `EventSource` API or a custom hook to consume this stream and append tokens to the chat bubble in real time.

---

### GET `/api/chat/global/stream`
**Cross-document streaming endpoint.**  
Queries across ALL ingested sources simultaneously. No source_id filter applied — retrieval spans the entire Qdrant collection.

**Query params:**
```
question: string
history:  JSON-encoded array of {role, content} (optional)
```

**Response:** `text/event-stream` — same format as `/api/chat/stream`, but citations include source title alongside page/segment reference.

```
data: {"type": "token", "value": "According "}
...
data: {"type": "citations", "value": [
  {"source_title": "Attention Is All You Need", "type": "document", "page_number": 4, "excerpt": "..."},
  {"source_title": "3Blue1Brown Neural Networks", "type": "video", "segment": "04:32", "excerpt": "..."}
]}
data: {"type": "done"}
```

---

### POST `/api/evaluate/{source_id}`
Manually trigger RAGAS evaluation (also auto-runs on ingestion).

**Response:** `200 OK`
```json
{
  "faithfulness": 0.91,
  "answer_relevancy": 0.85,
  "context_recall": 0.78,
  "context_precision": 0.83,
  "overall_score": 0.84,
  "test_questions": [
    {
      "question": "What problem does this paper solve?",
      "ground_truth": "...",
      "generated_answer": "...",
      "scores": { "faithfulness": 0.95, "answer_relevancy": 0.88 }
    }
  ]
}
```

---

## 8. Core Pipelines

### 8.1 Document Ingestion Pipeline

```
PDF Upload (via multipart/form-data)
    │
    ▼
FastAPI creates source record in PostgreSQL (status = "processing")
Returns 202 immediately
    │
    ▼  [BackgroundTask starts here]
PyMuPDF Text Extraction
→ Extract text page by page
→ Preserve page number metadata per page
    │
    ▼
Hierarchical Chunking
→ PARENT chunks: RecursiveTextSplitter(chunk_size=1500, overlap=200)
  → Store in PostgreSQL parent_chunks with page_start, page_end
→ CHILD chunks: RecursiveTextSplitter(chunk_size=300, overlap=30)
  → Each child stores parent_chunk_id
    │
    ▼
Embedding + BM25 Indexing (parallel)
→ Dense: HuggingFace all-MiniLM-L6-v2 embeds all child chunks
→ Sparse: BM25 tokenized index stored in PostgreSQL bm25_index table
    │
    ▼
Qdrant Upsert
→ Each vector: { dense_embedding, source_id, page_number, parent_chunk_id, content }
    │
    ▼
Auto-Summary + Title
→ Groq Llama 3.1 8B map-reduce summarization
→ Update sources.title, sources.status = "ready"
    │
    ▼
RAGAS Evaluation (BackgroundTask, runs after status = "ready")
→ See Section 11
```

### 8.2 Video Ingestion Pipeline

```
YouTube URL
    │
    ▼
FastAPI creates source record (status = "processing")
Returns 202 immediately
    │
    ▼  [BackgroundTask starts here]
yt-dlp Audio Download
→ Download best audio quality as WAV to /tmp
    │
    ▼
Whisper Transcription (local, "small" model)
→ Returns list of segments:
  [ { text: "...", start: 12.4, end: 18.9 }, ... ]
    │
    ▼
Segment-Aware Chunking
→ Group segments into chunks of ~400 words
→ Each chunk stores: { content, segment_start, segment_end }
→ Formatted as: "[segment 04:32] chunk text here..."
    │
    ▼
Embedding + BM25 Indexing (parallel)
→ Dense: HuggingFace all-MiniLM-L6-v2
→ Sparse: BM25 index in PostgreSQL bm25_index
    │
    ▼
Qdrant Upsert
→ Each vector: { dense_embedding, source_id, segment_start, segment_end, content }
    │
    ▼
Auto-Summary + Title → Groq 8B
→ Update sources.title, sources.status = "ready"
    │
    ▼
RAGAS Evaluation (BackgroundTask)
```

---

## 9. RAG Engine

The core retrieval engine. Called by the LangGraph router (Section 10) for every query.

### 9.1 Hybrid Search (BM25 + Vector)

Hybrid search combines two complementary signals:

**Dense retrieval (Vector search):**
- Embeds the query with all-MiniLM-L6-v2
- Retrieves top-k chunks by cosine similarity in Qdrant
- Good at: semantic similarity, paraphrases, conceptual matches

**Sparse retrieval (BM25):**
- Tokenizes query, scores against BM25 index in PostgreSQL
- Retrieves top-k chunks by BM25 score
- Good at: exact keyword matches, proper nouns, acronyms, technical terms

**Fusion:**
- Combine both result sets using Reciprocal Rank Fusion (RRF)
- RRF formula: `score = Σ 1 / (rank + k)` for each result across both lists
- Deduplicate by chunk ID
- Final ranked list passed to MMR

### 9.2 MMR Reranking (Maximal Marginal Relevance)

After hybrid fusion, MMR reranks to maximize diversity:
```
MMR score = λ * similarity(chunk, query) - (1-λ) * max_similarity(chunk, selected_chunks)
λ = 0.5  (balance between relevance and diversity)
fetch_k = 20 (candidates from hybrid search)
k = 6 (final chunks returned after MMR)
```

This prevents the LLM from receiving 6 near-identical chunks when one topic dominates retrieval.

### 9.3 Multi-Query + HyDE

**Multi-Query:**
```
User question: "what is attention mechanism?"
→ Groq Llama 3.1 8B generates 3 rephrasings:
  1. "how does the attention operation work in transformers?"
  2. "explain scaled dot-product attention"
  3. "what role does attention play in neural networks?"
→ Run hybrid search for each rephrasing
→ Union + deduplicate across all 4 result sets (original + 3 rephrasings)
→ Apply MMR on unioned set
```

**HyDE (Hypothetical Document Embeddings):**
```
User question: "what is attention mechanism?"
→ Groq Llama 3.1 8B generates a hypothetical answer paragraph
  (vocabulary and structure matching actual document text)
→ Embed the hypothetical answer (not the question)
→ Add to hybrid search union
```

### 9.4 Metadata Filtering

Every Qdrant search (both dense and sparse) includes a mandatory filter:
```python
filter = Filter(must=[
    FieldCondition(key="source_id", match=MatchValue(value=current_source_id))
])
```
For global cross-document queries (`/api/chat/global/stream`), this filter is **omitted entirely** — retrieval spans all vectors in the collection.

### 9.5 Parent Chunk Fetch (Documents only)

```
Retrieved child chunks (300 tokens)
→ Lookup parent_chunk_id for each
→ Fetch full parent chunk (1500 tokens) from PostgreSQL
→ Pass PARENT content to LLM (not child)
Result: precise retrieval, full context for the LLM
```

### 9.6 Streaming Answer Generation

```python
# FastAPI SSE endpoint
async def stream_answer(question, context, citations):
    async for token in groq_llm_70b.astream(prompt):
        yield f"data: {json.dumps({'type': 'token', 'value': token})}\n\n"
    yield f"data: {json.dumps({'type': 'citations', 'value': citations})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
```

Frontend EventSource receives tokens and appends them to the active chat bubble character by character.

### 9.7 Full Retrieval Pipeline Summary

```
Query
  │
  ├── Multi-query rephrasing (Groq 8B → 3 variants)
  ├── HyDE generation (Groq 8B → hypothetical answer)
  │
  ▼
Hybrid Search per query variant
  ├── Dense: Qdrant cosine similarity (+ metadata filter)
  └── Sparse: BM25 from PostgreSQL (+ source filter)
  │
  ▼
Reciprocal Rank Fusion across all variants
  │
  ▼
MMR Reranking (fetch_k=20, k=6, λ=0.5)
  │
  ▼
Parent Chunk Fetch (docs) / Segment chunks (video)
  │
  ▼
Groq Llama 3.3 70B → SSE stream → React frontend
```

---

## 10. Agentic Query Router (LangGraph)

Every `/api/chat/stream` request passes through this LangGraph state machine.

### State Definition

```python
class QueryState(TypedDict):
    question: str
    source_id: str              # empty string = global query
    query_type: str             # "simple" | "comparative" | "multi_hop"
    sub_queries: List[str]
    retrieved_contexts: List[str]
    intermediate_answers: List[str]
    final_answer: str
    citations: List[dict]
    chat_history: List[dict]
    stream_callback: Callable   # SSE writer passed through state
```

### Graph Nodes

**Node 1: `classify_query`**
```
Input: raw question
→ Groq Llama 3.1 8B classifies:
  - "simple": single factual question
  - "comparative": "compare X and Y", "difference between X and Y"
  - "multi_hop": "what did X say about Y, and how does that relate to Z"
Output: query_type
```

**Node 2: `simple_rag`**
```
→ Full hybrid search pipeline (Section 9)
→ Multi-query + HyDE + MMR + parent fetch
→ Groq 70B streams answer via SSE
Output: final_answer, citations
```

**Node 3: `comparative_rag`**
```
→ Decompose into 2 independent sub-queries
  e.g. "compare chapter 3 and chapter 7" →
       ["summarize chapter 3 conclusions", "summarize chapter 7 conclusions"]
→ Run 2 independent hybrid retrievals (parallel)
→ Feed both contexts to Groq 70B
→ Stream comparison synthesis via SSE
Output: final_answer, citations from both sub-retrievals
```

**Node 4: `multi_hop_rag`**
```
→ Step 1: Retrieve context for part 1 → get intermediate answer (Groq 8B)
→ Step 2: Form step 2 query using intermediate answer as context
→ Step 2: Retrieve context for step 2 query
→ Final: Groq 70B synthesizes both contexts → stream via SSE
Output: final_answer, citations from both hops
```

### Graph Edges

```
START → classify_query
classify_query → simple_rag       (query_type == "simple")
classify_query → comparative_rag  (query_type == "comparative")
classify_query → multi_hop_rag    (query_type == "multi_hop")
simple_rag      → END
comparative_rag → END
multi_hop_rag   → END
```

---

## 11. RAG Evaluation Layer (RAGAS)

Auto-runs as a BackgroundTask after every successful ingestion. Results stored in PostgreSQL.

### How it works

**Step 1 — QA Generation**
```
After ingestion (status = "ready"):
→ Sample 10 diverse chunks from the source
→ Groq Llama 3.1 8B generates 10 questions a real user might ask
→ For each question, generate ground truth answer from the chunk content
→ Store: [(question, context_chunk, ground_truth), ...]
```

**Step 2 — RAG Pipeline Run**
```
For each of the 10 questions:
→ Run the full RAG pipeline (hybrid search + MMR + HyDE + multi-query + parent fetch)
→ Collect: question, generated_answer, retrieved_contexts
```

**Step 3 — RAGAS Scoring**
```
Pass to RAGAS evaluator (using Groq 8B as evaluator LLM):

→ Faithfulness (0-1):
  Does the answer stick to the retrieved context?
  Catches hallucination — LLM making up facts not in the source.

→ Answer Relevancy (0-1):
  Does the answer actually address what was asked?
  Catches tangential or off-topic answers.

→ Context Recall (0-1):
  Did retrieval find the chunks actually needed to answer?
  Measures retrieval completeness.

→ Context Precision (0-1):
  Are retrieved chunks relevant, or is there noise?
  Measures retrieval signal-to-noise ratio.

→ Overall Score: weighted average of all four metrics
```

**Step 4 — Store + Display**
```
→ Store in PostgreSQL eval_results table
→ Available via GET /api/sources/{id}
→ Displayed as score card in React UI with per-metric progress bars
```

---

## 12. Frontend — React

### Pages / Views

**1. Dashboard**
- Upload area: drag-and-drop PDF + YouTube URL input + ingest button
- Ingestion progress: polling `/api/sources/{id}` until status = "ready"
- Source sidebar: all sources with status badges + eval score badges
- Global Search button: opens cross-document chat mode

**2. Chat View** (per source)
- Source title + type badge at top
- RAGAS score card: 4 metrics as progress bars
- Chat conversation (scrollable)
- User messages right (purple), assistant left (cyan)
- **Streaming**: tokens appear in real time as LLM generates
- Collapsible citation panel per assistant message
  - Documents: `📄 Page 12 — [excerpt]`
  - Videos: `🎬 Segment 04:32 — [excerpt]`
- Query type badge per answer: `Simple` / `Comparative` / `Multi-hop`
- Input box + send button

**3. Global Chat View**
- Same chat interface, no source selected
- Header: "Searching across all N sources"
- Citations include source title + page/segment reference
- Shows which sources contributed to each answer

**4. Source Detail**
- Full RAGAS evaluation report
- 10 test questions with per-question scores
- Re-run evaluation button

### Key Components

```
src/
├── components/
│   ├── UploadPanel.jsx           # PDF drag-drop + YouTube URL input
│   ├── SourceSidebar.jsx         # Source list with status + score badges
│   ├── ChatWindow.jsx            # Main chat interface
│   ├── ChatMessage.jsx           # Individual message bubble
│   ├── StreamingMessage.jsx      # Handles SSE token appending in real time
│   ├── CitationPanel.jsx         # Collapsible citations per answer
│   ├── EvalScoreCard.jsx         # RAGAS metrics display
│   ├── GlobalSearchButton.jsx    # Toggle cross-doc mode
│   ├── SourceBadge.jsx           # document / video type badge
│   └── StatusBadge.jsx           # processing / ready / failed
├── pages/
│   ├── Dashboard.jsx
│   ├── ChatPage.jsx
│   ├── GlobalChatPage.jsx
│   └── SourceDetailPage.jsx
├── hooks/
│   ├── useSources.js             # React Query — source list + polling
│   ├── useStreamingChat.js       # EventSource hook for SSE consumption
│   ├── useGlobalChat.js          # Cross-document SSE chat hook
│   └── useIngest.js              # Upload + poll for status
├── api/
│   └── client.js                 # Axios instance + all non-streaming API calls
└── App.jsx
```

### SSE Consumption Pattern (React)

```javascript
// useStreamingChat.js
const startStream = (sourceId, question, history) => {
  const params = new URLSearchParams({ source_id: sourceId, question, history: JSON.stringify(history) });
  const es = new EventSource(`/api/chat/stream?${params}`);

  es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'token') appendToken(data.value);
    if (data.type === 'citations') setCitations(data.value);
    if (data.type === 'done') es.close();
  };
};
```

### UI Design
- Dark theme: background `#0a0a0f`, surface `#111118`
- Accent: purple `#7c3aed` (user), cyan `#06b6d4` (assistant)
- Monospace font (JetBrains Mono) for source/citation content
- Sans-serif (Inter) for UI labels
- Pure Tailwind — no paid component library

---

## 13. Folder Structure

```
sourcemind/
├── docker-compose.yml
├── .env.example
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                      # FastAPI app entry, BackgroundTasks wiring
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py            # POST /api/ingest/document + /video
│   │   │   ├── chat.py              # GET /api/chat/stream (SSE)
│   │   │   ├── global_chat.py       # GET /api/chat/global/stream (SSE)
│   │   │   ├── sources.py           # GET/DELETE /api/sources
│   │   │   └── evaluate.py          # POST /api/evaluate/{id}
│   │   └── schemas.py               # Pydantic models
│   │
│   ├── core/
│   │   ├── document_pipeline.py     # PDF ingestion pipeline
│   │   ├── video_pipeline.py        # YouTube ingestion pipeline
│   │   ├── chunker.py               # Hierarchical + segment chunking
│   │   ├── embedder.py              # HuggingFace embedding wrapper
│   │   ├── bm25_index.py            # BM25 sparse index build + query
│   │   ├── hybrid_search.py         # RRF fusion of dense + sparse results
│   │   ├── rag_engine.py            # Full retrieval chain (hybrid+MMR+HyDE+multi-query)
│   │   ├── agent_router.py          # LangGraph state machine
│   │   └── evaluator.py             # RAGAS evaluation pipeline
│   │
│   ├── db/
│   │   ├── postgres.py              # asyncpg connection pool
│   │   ├── qdrant.py                # Qdrant client wrapper
│   │   ├── schema.sql               # PostgreSQL DDL
│   │   └── models.py                # SQLAlchemy models
│   │
│   └── utils/
│       ├── audio.py                 # yt-dlp + pydub
│       ├── streaming.py             # SSE response helpers
│       └── text.py                  # Text cleaning utilities
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       └── [see Section 12]
│
└── scripts/
    └── init_db.sql                  # Run once to create all tables
```

---

## 14. Environment Variables

```bash
# === GROQ API (free — get key at console.groq.com) ===
GROQ_API_KEY=gsk_xxxx

# === PostgreSQL ===
POSTGRES_USER=sourcemind
POSTGRES_PASSWORD=sourcemind_secret
POSTGRES_DB=sourcemind
DATABASE_URL=postgresql+asyncpg://sourcemind:sourcemind_secret@postgres:5432/sourcemind

# === Qdrant ===
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=sourcemind

# === Whisper ===
WHISPER_MODEL=small        # tiny | base | small | medium

# === Embeddings ===
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu       # change to "cuda" if GPU available

# === Models ===
GROQ_MODEL_MAIN=llama-3.3-70b-versatile    # final answer generation
GROQ_MODEL_FAST=llama-3.1-8b-instant       # multi-query, HyDE, eval, BM25

# === Hybrid Search ===
MMR_LAMBDA=0.5             # 0 = max diversity, 1 = max relevance
MMR_FETCH_K=20             # candidates before MMR reranking
MMR_FINAL_K=6              # final chunks passed to LLM
BM25_TOP_K=10              # sparse results before RRF fusion

# === App ===
MAX_PDF_SIZE_MB=100
MAX_VIDEO_DURATION_MIN=120
BACKEND_CORS_ORIGINS=http://localhost:5173
```

---

## 15. Docker Compose Setup

```yaml
version: "3.9"

services:

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: sourcemind
      POSTGRES_PASSWORD: sourcemind_secret
      POSTGRES_DB: sourcemind
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sourcemind"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:6333/health || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - whisper_models:/root/.cache/whisper
      - hf_models:/root/.cache/huggingface
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend/src:/app/src
    depends_on:
      - backend
    environment:
      VITE_API_URL: http://localhost:8000

volumes:
  postgres_data:
  qdrant_data:
  whisper_models:
  hf_models:
```

---

## 16. Build Order

**Follow this exactly. Do not skip ahead. Do not touch frontend until Phase 3 is producing good answers.**

```
PHASE 1 — INFRASTRUCTURE
  1. Write docker-compose.yml
  2. Run: docker-compose up postgres qdrant
  3. Run init_db.sql — verify all tables created
  4. Test Qdrant: curl http://localhost:6333/health
  5. Create Qdrant collection "sourcemind" (dense + sparse vectors enabled)
  6. Write db/postgres.py and db/qdrant.py connection wrappers
  7. Test: embed one sentence, upsert to Qdrant, query it back

PHASE 2 — DOCUMENT PIPELINE
  8. Write document_pipeline.py
     - PyMuPDF extraction (text + page numbers)
     - Flat chunking first → embed → upsert → verify retrieval works
  9. Upgrade to hierarchical chunking
     - Parent chunks (1500t) → PostgreSQL
     - Child chunks (300t) → Qdrant with parent_chunk_id
  10. Test parent fetch: after retrieval, confirm parent content is correct
  11. Build BM25 index: write bm25_index.py, index the same chunks into PostgreSQL

PHASE 3 — RAG CHAIN (core quality work)
  12. Write rag_engine.py
      - Basic dense-only chain: retrieval → prompt → Groq 70B answer
  13. Add BM25 sparse retrieval + RRF fusion (hybrid_search.py)
  14. Add MMR reranking on fused results
  15. Add multi-query: 3 rephrasings, union into hybrid search
  16. Add HyDE: hypothetical answer embedding added to union
  17. Add metadata filtering: verify zero cross-source bleed
  18. Test citations: page numbers in answers

  *** RAG MUST BE PRODUCING GOOD ANSWERS BEFORE PROCEEDING ***

PHASE 4 — VIDEO PIPELINE
  19. Write utils/audio.py: yt-dlp download test
  20. Write video_pipeline.py: Whisper transcription
  21. Segment-aware chunking: group segments, store positions
  22. Embed + upsert to Qdrant + BM25 index
  23. Test: query video source, confirm segment citations appear

PHASE 5 — STREAMING + FASTAPI ROUTES
  24. Write utils/streaming.py: SSE response helpers
  25. Write all API route files
  26. Wire ingest routes → pipeline → BackgroundTasks (non-blocking)
  27. Wire chat routes → SSE streaming (GET /api/chat/stream)
  28. Wire global chat route (GET /api/chat/global/stream, no source filter)
  29. Test: stream a response end-to-end with curl or Postman

PHASE 6 — LANGGRAPH AGENT
  30. Write agent_router.py
      - Classifier node first
      - Add simple_rag node (wraps rag_engine.py)
      - Add comparative_rag node
      - Add multi_hop_rag node
  31. Wire SSE stream callback through graph state
  32. Test each path with sample queries

PHASE 7 — RAGAS EVALUATOR
  33. Write evaluator.py
      - QA generation from source chunks
      - Run RAG pipeline on each question
      - RAGAS scoring (Groq 8B as evaluator LLM)
      - Store to eval_results
  34. Hook into ingestion as a second BackgroundTask (runs after status = "ready")
  35. Verify scores returned in GET /api/sources/{id}

PHASE 8 — REACT FRONTEND
  36. Vite + React + Tailwind setup
  37. api/client.js: all non-streaming API calls
  38. useStreamingChat.js: EventSource hook for SSE
  39. UploadPanel: drag-drop PDF + YouTube URL
  40. SourceSidebar: list, status badges, eval scores, global search button
  41. StreamingMessage: real-time token appending
  42. ChatWindow + CitationPanel
  43. GlobalChatPage: cross-document chat mode
  44. EvalScoreCard: per-metric progress bars
  45. Wire everything with React Query + streaming hooks

PHASE 9 — POLISH
  46. Error handling throughout backend (proper HTTP status codes)
  47. Loading/skeleton states in React
  48. Docker Compose: full stack in one command
  49. README.md: architecture diagram + setup guide
  50. GitHub Actions CI: lint + type check
```

---

## 17. Free Model Strategy

The entire project runs on free APIs and local models.

| Task | Model | Why |
|---|---|---|
| Final answer generation | Groq Llama 3.3 70B | Highest quality needed here |
| Multi-query rephrasing | Groq Llama 3.1 8B | Fast, generous limits |
| HyDE generation | Groq Llama 3.1 8B | Same reason |
| Comparative synthesis | Groq Llama 3.3 70B | Quality matters for synthesis |
| Multi-hop intermediate | Groq Llama 3.1 8B | Intermediate step, 8B is enough |
| Source title/summary | Groq Llama 3.1 8B | Doesn't need 70B |
| RAGAS QA generation | Groq Llama 3.1 8B | Runs in background |
| RAGAS evaluator LLM | Groq Llama 3.1 8B | RAGAS internal scoring |
| Embeddings | all-MiniLM-L6-v2 (local) | 100% free, CPU |
| Audio transcription | Whisper small (local) | 100% free, CPU |

**Rate limit reality check:**
Groq free tier gives Llama 3.1 8B ~30K TPM and Llama 3.3 70B ~6K TPM.
For a demo project this is effectively unlimited. A full PDF ingestion cycle (multi-query + HyDE + 10 RAGAS eval questions) uses roughly 50-70K tokens total — well within daily limits.

**Fallback:** Mistral API free tier (`mistral-small-latest` via `langchain-mistralai`) is a direct drop-in if Groq limits are hit.

---

## 18. Resume Bullets

Fill in `[X]` with actual measured numbers after building.

```
• Built SourceMind, a multi-modal RAG platform over PDFs and YouTube videos
  using hierarchical parent-child chunking (300-token children for retrieval
  precision, 1500-token parents passed to LLM for full context) combined with
  hybrid search (BM25 sparse + dense vector via Reciprocal Rank Fusion) and
  MMR reranking — eliminating both keyword-miss failures and redundant context.

• Engineered a LangGraph agentic query router classifying queries into
  simple/comparative/multi-hop categories — comparative queries decomposed
  into 2 parallel hybrid retrievals with LLM synthesis, multi-hop queries
  use iterative context chaining across 2 retrieval steps.

• Implemented real-time streaming responses via Server-Sent Events — FastAPI
  backend streams Groq LLM tokens as they generate, consumed by a React
  EventSource hook that appends tokens to the active chat bubble in real time,
  matching the UX of modern AI products.

• Added cross-document querying via a global chat endpoint that runs hybrid
  retrieval across all ingested sources simultaneously, with citations
  attributing each retrieved passage to its source document or video.

• Implemented automated RAGAS evaluation pipeline — auto-generates 10 QA
  pairs per ingested source, runs the full RAG pipeline, and scores
  faithfulness, answer relevancy, context recall, and context precision,
  providing quantitative retrieval quality measurement displayed per source.

• Deployed full-stack system (React + FastAPI + Qdrant + PostgreSQL) via
  Docker Compose with single-command startup; async ingestion via FastAPI
  BackgroundTasks; 100% free model stack (Groq Llama 3.1/3.3, local
  HuggingFace embeddings, local Whisper transcription).
```

---

*SourceMind PRD v2.0*  
*Stack: React · FastAPI · LangChain · LangGraph · Qdrant · PostgreSQL · Whisper · RAGAS · Groq · Docker*
