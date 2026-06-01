"""
BM25 sparse retrieval. The persistent index lives as plain text rows in
PostgreSQL (bm25_index table). At query time we hydrate a per-source (or
global) corpus, build a rank-bm25 index, and score.

This is intentionally simple. For a portfolio project, hydrating a small
corpus per query is fine. If scale becomes an issue, swap for Postgres
full-text-search or a dedicated search engine.
"""
from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Okapi

from db.postgres import fetch_bm25_corpus


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


async def bm25_search(query: str, source_id: UUID | None = None, top_k: int = 10) -> list[dict[str, Any]]:
    """Return [{chunk_id, score, content, source_id}] sorted by BM25 score desc."""
    corpus_rows = await fetch_bm25_corpus(source_id)
    if not corpus_rows:
        return []

    tokenized_corpus = [tokenize(r["content"]) for r in corpus_rows]
    bm25 = BM25Okapi(tokenized_corpus)

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "chunk_id": corpus_rows[i]["chunk_id"],
            "score": float(s),
            "content": corpus_rows[i]["content"],
            "source_id": corpus_rows[i]["source_id"],
        }
        for i, s in indexed
        if s > 0
    ]
