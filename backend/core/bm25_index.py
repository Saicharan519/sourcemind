"""
BM25 sparse retrieval. The persistent corpus lives as plain text rows in
PostgreSQL (bm25_index table). A rank-bm25 index is built per source (or for the
global corpus) and cached in memory.

Cache invalidation: before each search a cheap (count, checksum) fingerprint of
the corpus is read from Postgres; the index is rebuilt only when it changed
(a source was ingested or deleted, by this or any other process). Building and
scoring are CPU-bound, so they run in a worker thread, off the event loop.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Okapi

from db.postgres import fetch_bm25_corpus, fetch_bm25_fingerprint

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# key: source_id (or None for global) -> (fingerprint, rows, BM25Okapi | None)
_CACHE: dict[UUID | None, tuple[tuple[int, int], list[dict[str, Any]], BM25Okapi | None]] = {}
_LOCKS: dict[UUID | None, asyncio.Lock] = {}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _build(rows: list[dict[str, Any]]) -> BM25Okapi | None:
    if not rows:
        return None
    return BM25Okapi([tokenize(r["content"]) for r in rows])


def clear_cache() -> None:
    _CACHE.clear()


async def _get_index(source_id: UUID | None):
    fp = await fetch_bm25_fingerprint(source_id)
    cached = _CACHE.get(source_id)
    if cached and cached[0] == fp:
        return cached[1], cached[2]
    lock = _LOCKS.setdefault(source_id, asyncio.Lock())
    async with lock:
        cached = _CACHE.get(source_id)
        if cached and cached[0] == fp:
            return cached[1], cached[2]
        rows = await fetch_bm25_corpus(source_id)
        index = await asyncio.to_thread(_build, rows)
        _CACHE[source_id] = (fp, rows, index)
        return rows, index


def _score(index: BM25Okapi, q_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
    scores = index.get_scores(q_tokens)
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]


async def bm25_search(query: str, source_id: UUID | None = None, top_k: int = 10) -> list[dict[str, Any]]:
    """Return [{chunk_id, score, content, source_id}] sorted by BM25 score desc."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    rows, index = await _get_index(source_id)
    if index is None:
        return []
    top = await asyncio.to_thread(_score, index, q_tokens, top_k)
    return [
        {
            "chunk_id": rows[i]["chunk_id"],
            "score": float(s),
            "content": rows[i]["content"],
            "source_id": rows[i]["source_id"],
        }
        for i, s in top
        if s > 0
    ]
