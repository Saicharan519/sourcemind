"""Source isolation on real data: searching one source never returns another's chunks."""
from core.bm25_index import bm25_search
from core.hybrid_search import hybrid_search
from core.rag_engine import hydrate_contexts


def sids(chunks):
    return {str(c.get("payload", {}).get("source_id") or c.get("source_id")) for c in chunks}


def test_ingest_produced_chunks(live):
    res = live.run(hybrid_search("zorblax enzymes photosynthesis", source_id=live.a))
    assert res, "no chunks retrieved"
    assert any("zorblax" in (c.get("content") or "").lower() for c in res)


def test_source_filter_isolates_dense_and_sparse(live):
    # Ask source A about source B's topic: nothing from B may come back.
    for q in ("quantum lattice tariff percent", "zorblax"):
        assert sids(live.run(hybrid_search(q, source_id=live.a))) <= {str(live.a)}
        assert sids(live.run(hybrid_search(q, source_id=live.b))) <= {str(live.b)}
        assert sids(live.run(bm25_search(q, source_id=live.a))) <= {str(live.a)}


def test_global_search_spans_sources(live):
    res = live.run(hybrid_search("zorblax enzymes and quantum lattice tariffs"))
    assert {str(live.a), str(live.b)} <= sids(res)


def test_parent_hydration_keeps_page_numbers(live):
    res = live.run(hydrate_contexts(live.run(hybrid_search("lattice exemption kilograms", source_id=live.b))))
    top = res[0]
    assert "200 kilograms" in top["content"]
    assert top["payload"]["page_number"] in (1, 2)
