"""
Hybrid search:
  1. Dense retrieval via Qdrant (cosine similarity over MiniLM embeddings)
  2. Sparse retrieval via BM25 (rank-bm25 over Postgres corpus)
  3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF)
  4. Rerank the fused list with MMR (Maximal Marginal Relevance) for diversity

The returned chunks each have an `id` (the Qdrant point id used as chunk_id)
plus the original payload, so downstream code can fetch parent chunks if needed.
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import numpy as np

from config import settings
from core.bm25_index import bm25_search
from core.embedder import embed_text, embed_texts
from db.qdrant import QdrantStore


# -- RRF -------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion: score = sum(1 / (k + rank)) over each list.
    Each input list is sorted descending by relevance.
    Each result dict must have an `id` field.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            rid = str(item.get("id") or item.get("chunk_id"))
            if not rid:
                continue
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            if rid not in merged:
                merged[rid] = item

    fused = []
    for rid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        m = dict(merged[rid])
        m["rrf_score"] = score
        m["id"] = rid
        fused.append(m)
    return fused


# -- MMR -------------------------------------------------------------------

def mmr_rerank(
    query_vector: list[float],
    candidates: list[dict[str, Any]],
    candidate_vectors: list[list[float]],
    lambda_mult: float = 0.5,
    k: int = 6,
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance reranking.
    `candidate_vectors[i]` must align with `candidates[i]`.
    """
    if not candidates:
        return []

    qv = np.array(query_vector, dtype=np.float32)
    cv = np.array(candidate_vectors, dtype=np.float32)

    # similarity to query
    sim_to_query = cv @ qv  # cosine since vectors are normalized

    selected_idx: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected_idx) < k:
        if not selected_idx:
            best = max(remaining, key=lambda i: sim_to_query[i])
            selected_idx.append(best)
            remaining.remove(best)
            continue

        selected_vecs = cv[selected_idx]
        best_score = -1e9
        best_i = remaining[0]
        for i in remaining:
            div = float(np.max(cv[i] @ selected_vecs.T))
            score = lambda_mult * float(sim_to_query[i]) - (1 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_i = i
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]


# -- Main hybrid search ----------------------------------------------------

async def hybrid_search(
    query: str,
    source_id: UUID | None = None,
    fetch_k: int | None = None,
    final_k: int | None = None,
    lambda_mult: float | None = None,
) -> list[dict[str, Any]]:
    """
    Run dense + BM25 → RRF → MMR.
    Returns final_k chunks. Each chunk is a dict with at least:
      { id, payload, content, rrf_score }
    """
    fetch_k = fetch_k or settings.MMR_FETCH_K
    final_k = final_k or settings.MMR_FINAL_K
    lambda_mult = lambda_mult if lambda_mult is not None else settings.MMR_LAMBDA

    # Embedding is CPU-bound — keep it off the event loop.
    query_vec = await asyncio.to_thread(embed_text, query)

    # Dense — ask Qdrant for the stored vectors so we don't have to re-embed later.
    dense_results = await QdrantStore.search(
        query_vector=query_vec,
        source_id=source_id,
        top_k=settings.DENSE_TOP_K,
        with_vectors=True,
    )
    for r in dense_results:
        r["content"] = r["payload"].get("content", "")

    # Sparse (BM25) — these carry no Qdrant payload/vector yet.
    sparse_results = await bm25_search(query, source_id=source_id, top_k=settings.BM25_TOP_K)
    for r in sparse_results:
        r["id"] = r["chunk_id"]
        r.setdefault("payload", {})
        r.setdefault("vector", None)

    # RRF fuse (dense first, so shared chunks keep the dense payload+vector).
    fused = reciprocal_rank_fusion([dense_results, sparse_results])[:fetch_k]
    if not fused:
        return []

    # Backfill BM25-only hits: fetch their real payload + vector from Qdrant so
    # citations (source_type/title/page/segment), parent hydration, and MMR all
    # work for keyword-only matches instead of degrading to "unknown"/[Source].
    missing_ids = [
        c["id"] for c in fused
        if not c.get("payload") or c.get("vector") is None
    ]
    backfill = await QdrantStore.retrieve(missing_ids) if missing_ids else {}
    for c in fused:
        b = backfill.get(c["id"])
        if b:
            if not c.get("payload"):
                c["payload"] = b["payload"]
            if c.get("vector") is None:
                c["vector"] = b["vector"]
        if not c.get("content"):
            c["content"] = c.get("payload", {}).get("content", "")

    # Candidate vectors for MMR: prefer the stored vector; only embed the (rare)
    # leftovers that still lack one.
    to_embed_idx = [i for i, c in enumerate(fused) if c.get("vector") is None]
    if to_embed_idx:
        texts = [(fused[i].get("content") or " ") for i in to_embed_idx]
        embedded = await asyncio.to_thread(embed_texts, texts)
        for i, vec in zip(to_embed_idx, embedded):
            fused[i]["vector"] = vec

    cand_vecs = [c["vector"] for c in fused]

    reranked = mmr_rerank(
        query_vector=query_vec,
        candidates=fused,
        candidate_vectors=cand_vecs,
        lambda_mult=lambda_mult,
        k=final_k,
    )
    return reranked
