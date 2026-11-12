import asyncio

import pytest

from core import bm25_index, rag_engine
from core.hybrid_search import reciprocal_rank_fusion


def run(coro):
    return asyncio.run(coro)


def chunk(cid):
    return {"id": cid, "payload": {"source_id": "s1"}, "content": f"text {cid}"}


def test_rrf_rewards_agreement_across_lists():
    fused = reciprocal_rank_fusion([[chunk("a"), chunk("b")], [chunk("b"), chunk("c")]])
    assert [c["id"] for c in fused][0] == "b"
    assert {c["id"] for c in fused} == {"a", "b", "c"}


def test_retrieve_fuses_variant_lists(monkeypatch):
    lists = {
        "q": [chunk("a"), chunk("b"), chunk("c")],
        "r1": [chunk("b"), chunk("d")],
        "hyde": [chunk("b"), chunk("a")],
    }

    async def fake_search(q, source_id=None):
        return lists[q]

    async def fake_mq(q):
        return ["r1"]

    async def fake_hyde(q):
        return "hyde"

    monkeypatch.setattr(rag_engine, "hybrid_search", fake_search)
    monkeypatch.setattr(rag_engine, "generate_multi_queries", fake_mq)
    monkeypatch.setattr(rag_engine, "generate_hyde", fake_hyde)
    out = run(rag_engine.retrieve("q"))
    ids = [c["id"] for c in out]
    assert ids[0] == "b"                       # appears in all three lists
    assert len(ids) == len(set(ids))           # deduped
    assert set(ids) == {"a", "b", "c", "d"}


def test_retrieve_records_failed_expansions(monkeypatch):
    async def fake_search(q, source_id=None):
        return [chunk("a")]

    async def failed(q):
        return None

    monkeypatch.setattr(rag_engine, "hybrid_search", fake_search)
    monkeypatch.setattr(rag_engine, "generate_multi_queries", failed)
    monkeypatch.setattr(rag_engine, "generate_hyde", failed)
    diag = {}
    out = run(rag_engine.retrieve("q", diagnostics=diag))
    assert [c["id"] for c in out] == ["a"]
    assert sorted(diag["degraded"]) == ["hyde", "multi_query"]


def test_retrieve_passes_source_filter(monkeypatch):
    seen = []

    async def fake_search(q, source_id=None):
        seen.append(source_id)
        return []

    monkeypatch.setattr(rag_engine, "hybrid_search", fake_search)
    run(rag_engine.retrieve("q", source_id="SRC", use_multi_query=False, use_hyde=False))
    assert seen == ["SRC"]


def test_retrieve_raises_if_main_query_search_fails(monkeypatch):
    async def boom(q, source_id=None):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(rag_engine, "hybrid_search", boom)
    with pytest.raises(RuntimeError):
        run(rag_engine.retrieve("q", use_multi_query=False, use_hyde=False))


def test_bm25_cache_rebuilds_only_when_corpus_changes(monkeypatch):
    state = {"fp": (2, 7), "fetches": 0}
    rows = [
        {"chunk_id": "c1", "content": "retrieval augmented generation", "source_id": "s1"},
        {"chunk_id": "c2", "content": "cooking pasta at home", "source_id": "s1"},
        {"chunk_id": "x1", "content": "weather report for monday", "source_id": "s1"},
        {"chunk_id": "x2", "content": "football match highlights", "source_id": "s1"},
        {"chunk_id": "x3", "content": "stock market summary", "source_id": "s1"},
    ]

    async def fp(source_id=None):
        return state["fp"]

    async def corpus(source_id=None):
        state["fetches"] += 1
        return rows

    monkeypatch.setattr(bm25_index, "fetch_bm25_fingerprint", fp)
    monkeypatch.setattr(bm25_index, "fetch_bm25_corpus", corpus)
    bm25_index.clear_cache()

    r1 = run(bm25_index.bm25_search("retrieval generation", "s1"))
    r2 = run(bm25_index.bm25_search("pasta", "s1"))
    assert r1[0]["chunk_id"] == "c1" and r2[0]["chunk_id"] == "c2"
    assert state["fetches"] == 1               # second search used the cache

    state["fp"] = (3, 9)                       # a source was ingested
    rows.append({"chunk_id": "c3", "content": "pasta sauce recipe", "source_id": "s1"})
    run(bm25_index.bm25_search("pasta", "s1"))
    assert state["fetches"] == 2               # rebuilt after the change
