-- SourceMind PostgreSQL schema
-- Run once on first startup (auto-mounted into Docker Postgres init dir)

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sources: every ingested document/video
CREATE TABLE IF NOT EXISTS sources (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type   VARCHAR(10) NOT NULL CHECK (source_type IN ('document', 'video')),
    title         TEXT NOT NULL,
    filename      TEXT,
    youtube_url   TEXT,
    page_count    INTEGER,
    duration_s    INTEGER,
    status        VARCHAR(20) DEFAULT 'processing',
    error_message TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

-- Parent chunks: large context for document RAG
CREATE TABLE IF NOT EXISTS parent_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id     UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    page_start    INTEGER,
    page_end      INTEGER,
    chunk_index   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parent_source ON parent_chunks(source_id);

-- BM25 index: sparse text per child chunk for hybrid search
CREATE TABLE IF NOT EXISTS bm25_index (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id     UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL,
    content       TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bm25_source ON bm25_index(source_id);
CREATE INDEX IF NOT EXISTS idx_bm25_chunk_id ON bm25_index(chunk_id);

-- Eval results: RAGAS scores per source
CREATE TABLE IF NOT EXISTS eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id           UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_recall      FLOAT,
    context_precision   FLOAT,
    overall_score       FLOAT,
    eval_questions      JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_source ON eval_results(source_id);

-- Chat history: per-source conversation memory
CREATE TABLE IF NOT EXISTS chat_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID REFERENCES sources(id) ON DELETE CASCADE,
    role        VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_source ON chat_history(source_id);
