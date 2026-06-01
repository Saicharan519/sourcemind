"""Async PostgreSQL connection pool using asyncpg."""
from __future__ import annotations

import asyncpg
import json
import math
from typing import Any, Optional
from uuid import UUID

from config import settings


def _asyncpg_dsn() -> str:
    """Convert SQLAlchemy-style URL to asyncpg-compatible DSN."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


class Database:
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def connect(cls) -> None:
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                dsn=_asyncpg_dsn(),
                min_size=2,
                max_size=10,
                command_timeout=60,
            )

    @classmethod
    async def disconnect(cls) -> None:
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    def pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            raise RuntimeError("Database not initialized. Call Database.connect() first.")
        return cls._pool

    @classmethod
    async def fetch(cls, query: str, *args) -> list[asyncpg.Record]:
        async with cls.pool().acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args) -> Optional[asyncpg.Record]:
        async with cls.pool().acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        async with cls.pool().acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def executemany(cls, query: str, args: list[tuple]) -> None:
        async with cls.pool().acquire() as conn:
            await conn.executemany(query, args)


# -- Convenience helpers ---------------------------------------------------

async def create_source(
    source_type: str,
    title: str,
    filename: str | None = None,
    youtube_url: str | None = None,
) -> UUID:
    row = await Database.fetchrow(
        """
        INSERT INTO sources (source_type, title, filename, youtube_url, status)
        VALUES ($1, $2, $3, $4, 'processing')
        RETURNING id
        """,
        source_type, title, filename, youtube_url,
    )
    return row["id"]


async def update_source_status(
    source_id: UUID,
    status: str,
    error_message: str | None = None,
    page_count: int | None = None,
    duration_s: int | None = None,
    title: str | None = None,
) -> None:
    await Database.execute(
        """
        UPDATE sources
        SET status = $2,
            error_message = COALESCE($3, error_message),
            page_count = COALESCE($4, page_count),
            duration_s = COALESCE($5, duration_s),
            title = COALESCE($6, title)
        WHERE id = $1
        """,
        source_id, status, error_message, page_count, duration_s, title,
    )


async def get_source(source_id: UUID) -> Optional[dict[str, Any]]:
    row = await Database.fetchrow("SELECT * FROM sources WHERE id = $1", source_id)
    return dict(row) if row else None


async def list_sources() -> list[dict[str, Any]]:
    import math
    rows = await Database.fetch(
        """
        SELECT s.*, e.overall_score AS eval_score
        FROM sources s
        LEFT JOIN eval_results e ON e.source_id = s.id
        ORDER BY s.created_at DESC
        """
    )
    result = []
    for r in rows:
        d = dict(r)
        score = d.get("eval_score")
        if score is not None and isinstance(score, float) and math.isnan(score):
            d["eval_score"] = None
        result.append(d)
    return result


async def delete_source(source_id: UUID) -> None:
    """Cascade deletes parent_chunks, bm25_index, eval_results, chat_history."""
    await Database.execute("DELETE FROM sources WHERE id = $1", source_id)


async def insert_parent_chunks(source_id: UUID, parents: list[dict[str, Any]]) -> dict[int, UUID]:
    """
    Inserts parent chunks. Returns mapping from chunk_index -> parent UUID.
    `parents` items: { content, page_start, page_end, chunk_index }
    """
    if not parents:
        return {}
    rows = await Database.fetch(
        """
        INSERT INTO parent_chunks (source_id, content, page_start, page_end, chunk_index)
        SELECT $1, unnest($2::text[]), unnest($3::int[]), unnest($4::int[]), unnest($5::int[])
        RETURNING id, chunk_index
        """,
        source_id,
        [p["content"] for p in parents],
        [p.get("page_start") for p in parents],
        [p.get("page_end") for p in parents],
        [p["chunk_index"] for p in parents],
    )
    return {r["chunk_index"]: r["id"] for r in rows}


async def fetch_parent_chunks(parent_ids: list[UUID]) -> dict[UUID, dict[str, Any]]:
    if not parent_ids:
        return {}
    rows = await Database.fetch(
        "SELECT * FROM parent_chunks WHERE id = ANY($1::uuid[])",
        parent_ids,
    )
    return {r["id"]: dict(r) for r in rows}


async def insert_bm25_rows(source_id: UUID, rows: list[dict[str, Any]]) -> None:
    """rows items: { chunk_id, content, chunk_index }"""
    if not rows:
        return
    await Database.executemany(
        """
        INSERT INTO bm25_index (source_id, chunk_id, content, chunk_index)
        VALUES ($1, $2, $3, $4)
        """,
        [(source_id, r["chunk_id"], r["content"], r["chunk_index"]) for r in rows],
    )


async def fetch_bm25_corpus(source_id: UUID | None = None) -> list[dict[str, Any]]:
    """Pull BM25 corpus. If source_id is None, returns global corpus."""
    if source_id is None:
        rows = await Database.fetch("SELECT chunk_id, content, source_id FROM bm25_index")
    else:
        rows = await Database.fetch(
            "SELECT chunk_id, content, source_id FROM bm25_index WHERE source_id = $1",
            source_id,
        )
    return [dict(r) for r in rows]


async def insert_eval_result(source_id: UUID, result: dict[str, Any]) -> None:
    def safe(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return v

    await Database.execute(
        """
        INSERT INTO eval_results
            (source_id, faithfulness, answer_relevancy, context_recall,
             context_precision, overall_score, eval_questions)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        """,
        source_id,
        safe(result.get("faithfulness")),
        safe(result.get("answer_relevancy")),
        safe(result.get("context_recall")),
        safe(result.get("context_precision")),
        safe(result.get("overall_score")),
        json.dumps(result.get("eval_questions", [])),
    )


async def get_eval_result(source_id: UUID) -> Optional[dict[str, Any]]:
    row = await Database.fetchrow(
        "SELECT * FROM eval_results WHERE source_id = $1 ORDER BY created_at DESC LIMIT 1",
        source_id,
    )
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("eval_questions"), str):
        d["eval_questions"] = json.loads(d["eval_questions"])
    return d


async def insert_chat_message(source_id: UUID | None, role: str, content: str) -> None:
    await Database.execute(
        "INSERT INTO chat_history (source_id, role, content) VALUES ($1, $2, $3)",
        source_id, role, content,
    )
