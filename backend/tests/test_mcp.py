"""MCP tools on real ingested data (live fixture), and through a real MCP client session."""
import json

import pytest

from core import agent_router, rag_engine
from core import source_tools as tools


def test_list_sources(live):
    rows = live.run(tools.list_sources())
    ids = {r["source_id"] for r in rows}
    assert {str(live.a), str(live.b)} <= ids
    assert all(r["status"] == "ready" for r in rows)


def test_search_scoped_to_one_source(live):
    res = live.run(tools.search_sources("quantum lattice tariff", source_id=str(live.a)))
    assert res["passages"], "expected passages"
    assert {p["source_id"] for p in res["passages"]} == {str(live.a)}   # never B's text
    p = res["passages"][0]
    assert "untrusted_content" in p and p["page"] in (1, 2)


def test_search_all_sources_finds_the_right_one(live):
    res = live.run(tools.search_sources("standard quantum lattice tariff percent", top_k=3))
    assert res["passages"][0]["source_id"] == str(live.b)
    assert "17 percent" in res["passages"][0]["untrusted_content"]


def test_get_passage_by_page(live):
    res = live.run(tools.get_passage(str(live.b), page=2))
    assert "200 kilograms" in res["untrusted_content"] and res["page"] == 2


@pytest.mark.parametrize("call,msg", [
    (lambda: tools.search_sources("", None), "empty"),
    (lambda: tools.search_sources("x", "not-a-uuid"), "UUID"),
    (lambda: tools.search_sources("x", "00000000-0000-0000-0000-000000000000"), "No source"),
    (lambda: tools.get_passage("00000000-0000-0000-0000-000000000000", page=1), "No source"),
])
def test_bad_input_is_a_clear_tool_error(live, call, msg):
    with pytest.raises(tools.ToolError, match=msg):
        live.run(call())


def test_get_passage_requires_page_for_documents(live):
    with pytest.raises(tools.ToolError, match="page"):
        live.run(tools.get_passage(str(live.a)))


class _Msg:
    def __init__(self, c):
        self.content = c


class _FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    async def astream(self, messages):
        yield _Msg(self.answer)


def _no_llm(monkeypatch, answer):
    """Real retrieval on real data; only the LLM steps are faked."""
    async def simple(q):
        return "simple"

    async def none(q):
        return None

    monkeypatch.setattr(agent_router, "classify_query", simple)
    monkeypatch.setattr(rag_engine, "generate_multi_queries", none)
    monkeypatch.setattr(rag_engine, "generate_hyde", none)
    monkeypatch.setattr(rag_engine, "get_main_llm", lambda streaming=False: _FakeLLM(answer))


def test_ask_sources_returns_validated_citations(live, monkeypatch):
    _no_llm(monkeypatch, "The standard tariff is 17 percent [1]. See also [42].")
    res = live.run(tools.ask_sources("What is the standard quantum lattice tariff?", str(live.b)))
    assert res["error"] is None
    assert res["evidence"]["status"] == "supported" and res["evidence"]["invalid_refs"] == [42]
    assert [c["ref"] for c in res["citations"]] == [1]
    assert res["citations"][0]["source_id"] == str(live.b)


def test_ask_sources_not_found(live, monkeypatch):
    _no_llm(monkeypatch, rag_engine.NOT_FOUND_ANSWER)
    res = live.run(tools.ask_sources("Who won the 1998 world cup?", str(live.a)))
    assert res["answer"] == rag_engine.NOT_FOUND_ANSWER
    assert res["citations"] == [] and res["evidence"]["status"] == "insufficient"


def test_mcp_protocol_round_trip(live):
    """A real MCP client session (in-memory transport) against the real server object."""
    from mcp.shared.memory import create_connected_server_and_client_session

    import mcp_server

    async def session():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            res = await client.call_tool("search_sources",
                                         {"query": "zorblax enzyme rate", "source_id": str(live.a), "top_k": 2})
            bad = await client.call_tool("get_passage", {"source_id": "nope", "page": 1})
            listed_sources = await client.call_tool("list_sources", {})
            return names, res, bad, listed_sources

    names, res, bad, listed_sources = live.run(session())
    assert len(listed_sources.content) == 1
    src = json.loads(listed_sources.content[0].text)
    assert src["count"] >= 2 and {str(live.a), str(live.b)} <= {s["source_id"] for s in src["sources"]}
    assert names == {"list_sources", "search_sources", "get_passage", "ask_sources"}
    payload = json.loads(res.content[0].text)
    assert payload["passages"] and payload["passages"][0]["source_id"] == str(live.a)
    assert "42 molecules" in " ".join(p["untrusted_content"] for p in payload["passages"])
    assert "UUID" in json.loads(bad.content[0].text)["error"]


def test_list_sources_one_row_per_source_after_reevaluation(live):
    """Regression: each re-run evaluation added a row, and the dashboard listed the
    source once per evaluation."""
    from db.postgres import insert_eval_result

    for score in (0.5, 0.8):
        live.run(insert_eval_result(live.a, {"faithfulness": score, "answer_relevancy": score,
                                              "context_recall": score, "context_precision": score,
                                              "overall_score": score, "eval_questions": []}))
    rows = [r for r in live.run(tools.list_sources()) if r["source_id"] == str(live.a)]
    assert len(rows) == 1
