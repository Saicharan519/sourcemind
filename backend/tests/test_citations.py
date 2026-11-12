"""Citation correctness and the insufficient-evidence path, through the real LangGraph."""
import asyncio

from core import agent_router, rag_engine
from core.rag_engine import (
    NOT_FOUND_ANSWER,
    build_citations,
    extract_citation_refs,
    format_context_for_llm,
    validate_citations,
)


def doc_chunk(cid, page, title="Paper A", sid="s1"):
    return {"id": cid, "content": f"content of {cid}",
            "payload": {"source_id": sid, "source_type": "document",
                        "source_title": title, "page_number": page}}


# -- pure functions -----------------------------------------------------------

def test_extract_refs_handles_common_formats():
    assert extract_citation_refs("A [1]. B [2][3]. C [1, 4]. D [5-7].") == [1, 2, 3, 4, 5, 6, 7]
    assert extract_citation_refs("no citations here") == []
    assert extract_citation_refs("year [2019] style [12]") == [2019, 12]  # caller validates range


def test_native_gpt_oss_citations_are_understood():
    # Found in the live Groq run: gpt-oss wrote "29.3 % 【6】" and was wrongly flagged uncited
    assert extract_citation_refs("ChatGPT reaches 29.3 %【6】.") == [6]
    assert extract_citation_refs("See 【2†L4-L9】 and 【3】") == [2, 3]
    assert rag_engine.normalize_citations("A【6】 B【2†L1-L3】") == "A[6] B[2]"
    assert validate_citations("Accuracy is 29.3【6】.", 6)["status"] == "supported"


def test_validate_drops_invented_refs():
    v = validate_citations("Fact one [2]. Fact two [9].", n_chunks=3)
    assert v == {"status": "supported", "valid": [2], "invalid": [9]}


def test_validate_classifies_abstention_and_uncited():
    assert validate_citations(NOT_FOUND_ANSWER, 4)["status"] == "insufficient"
    assert validate_citations("A confident answer with no sources.", 4)["status"] == "uncited"
    assert validate_citations("Only an invented one [7].", 4)["status"] == "uncited"


def test_build_citations_returns_only_cited_in_order():
    chunks = [doc_chunk("a", 1), doc_chunk("b", 5), doc_chunk("c", 9)]
    cards = build_citations(chunks, cited=[3, 1, 8])
    assert [(c["ref"], c["page_number"]) for c in cards] == [(3, 9), (1, 1)]
    assert len(build_citations(chunks)) == 3          # uncited form = full context


def test_context_is_numbered_with_titles_and_pages():
    ctx = format_context_for_llm([doc_chunk("a", 4, "NIST CSF"), doc_chunk("b", 12, "RAGAS")])
    assert "[1] NIST CSF - page 4" in ctx and "[2] RAGAS - page 12" in ctx


# -- full graph with a fake LLM ------------------------------------------------

class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    async def astream(self, messages):
        for word in self.answer.split(" "):
            yield FakeChunk(word + " ")

    async def ainvoke(self, prompt):
        return FakeChunk("simple")


def collect(monkeypatch, answer, chunks, classify="simple"):
    async def fake_classify(q):
        return classify

    async def fake_retrieve(q, source_id=None, diagnostics=None, **kw):
        return chunks

    async def fake_hydrate(c):
        return c

    monkeypatch.setattr(agent_router, "classify_query", fake_classify)
    monkeypatch.setattr(agent_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent_router, "hydrate_contexts", fake_hydrate)
    calls = {"llm": 0}

    def fake_main_llm(streaming=False):
        calls["llm"] += 1
        return FakeLLM(answer)

    monkeypatch.setattr(rag_engine, "get_main_llm", fake_main_llm)

    async def run():
        return [ev async for ev in agent_router.route_and_stream("q?", "s1", [])]

    events = asyncio.run(run())
    by_type = {}
    for ev in events:
        by_type.setdefault(ev["type"], []).append(ev["value"])
    return events, by_type, calls


def test_graph_emits_only_cited_passages(monkeypatch):
    chunks = [doc_chunk("a", 1), doc_chunk("b", 5), doc_chunk("c", 9)]
    events, ev, _ = collect(monkeypatch, "The answer is X [2]. Also Y [5].", chunks)
    assert len(ev["context"][0]) == 3                         # model saw 3 passages
    assert [c["ref"] for c in ev["citations"][0]] == [2]      # but cited only #2; #5 invented
    assert ev["citations"][0][0]["page_number"] == 5
    assert ev["evidence"][0]["status"] == "supported"
    assert ev["evidence"][0]["invalid_refs"] == [5]
    types = [e["type"] for e in events]
    assert types.index("citations") > max(i for i, t in enumerate(types) if t == "token")
    assert types[-1] == "done" and "error" not in types


def test_graph_abstains_without_llm_when_nothing_retrieved(monkeypatch):
    _, ev, calls = collect(monkeypatch, "should not be used", [])
    assert "".join(ev["token"]).strip() == NOT_FOUND_ANSWER
    assert ev["citations"] == [[]]
    assert ev["evidence"][0]["status"] == "insufficient"
    assert calls["llm"] == 0


def test_graph_model_refusal_shows_no_citations(monkeypatch):
    _, ev, _ = collect(monkeypatch, NOT_FOUND_ANSWER, [doc_chunk("a", 1)])
    assert ev["citations"] == [[]]
    assert ev["evidence"][0]["status"] == "insufficient"


def test_graph_reports_degraded_retrieval(monkeypatch):
    async def fake_classify(q):
        return "simple"

    async def search(q, source_id=None):
        return [doc_chunk("a", 1)]

    async def failed(q):
        return None

    async def fake_hydrate(c):
        return c

    monkeypatch.setattr(agent_router, "classify_query", fake_classify)
    monkeypatch.setattr(agent_router, "hydrate_contexts", fake_hydrate)
    monkeypatch.setattr(rag_engine, "hybrid_search", search)
    monkeypatch.setattr(rag_engine, "generate_multi_queries", failed)
    monkeypatch.setattr(rag_engine, "generate_hyde", failed)
    monkeypatch.setattr(rag_engine, "get_main_llm", lambda streaming=False: FakeLLM("X [1]."))

    async def run():
        return [ev async for ev in agent_router.route_and_stream("q?", "s1", [])]

    events = asyncio.run(run())
    warnings = [e["value"] for e in events if e["type"] == "warning"]
    assert warnings and sorted(warnings[0]["degraded"]) == ["hyde", "multi_query"]


def test_graph_error_is_reported_not_swallowed(monkeypatch):
    async def fake_classify(q):
        return "simple"

    async def boom(q, source_id=None, diagnostics=None, **kw):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(agent_router, "classify_query", fake_classify)
    monkeypatch.setattr(agent_router, "retrieve", boom)

    async def run():
        return [ev async for ev in agent_router.route_and_stream("q?", "s1", [])]

    events = asyncio.run(run())
    errors = [e["value"] for e in events if e["type"] == "error"]
    assert errors and "RuntimeError" in errors[0]
    assert events[-1]["type"] == "done"


def test_rate_limit_error_is_short_and_hides_account_details():
    err = RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit reached for model "
                       "`openai/gpt-oss-120b` in organization `org_01secret` on tokens per day'}}")
    msg = agent_router.user_facing_error(err)
    assert "rate limit" in msg and "org_01secret" not in msg
