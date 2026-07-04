"""
Agentic query router — a real LangGraph ``StateGraph``.

Graph shape:

    START → classify ──(conditional on query_type)──┬─→ simple      → END
                                                    ├─→ comparative → END
                                                    └─→ multi_hop   → END

Streaming: LangGraph nodes mutate state and return; they cannot themselves be
async generators. To keep the token-by-token SSE stream, each node pushes events
into an ``asyncio.Queue`` via an ``emit`` callable carried in the graph state.
``route_and_stream`` runs the compiled graph as a background task and drains the
queue, yielding the same event dicts the API layer expects:

  {"type": "query_type", "value": ...}
  {"type": "sub_queries", "value": [...]}
  {"type": "citations", "value": [...]}
  {"type": "token", "value": "..."}
  {"type": "done", "value": None}
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from core.rag_engine import (
    build_citations,
    format_context_for_llm,
    get_fast_llm,
    hydrate_contexts,
    retrieve,
    stream_answer,
)


# -- Graph state -----------------------------------------------------------

EmitFn = Callable[[dict], Awaitable[None]]


class QueryState(TypedDict, total=False):
    question: str
    source_id: Optional[UUID]
    chat_history: list[dict[str, str]]
    emit: EmitFn            # SSE writer threaded through state
    query_type: str         # set by the classify node; drives routing
    sub_queries: list[str]


# -- Classification --------------------------------------------------------

CLASSIFY_PROMPT = """\
Classify the user's question into exactly one of:
- "simple": a single factual question
- "comparative": asks to compare or contrast 2+ things (uses words like "compare",
  "difference between", "versus", "vs")
- "multi_hop": requires 2 reasoning steps where step 2 depends on the answer to
  step 1 (uses words like "and then", "based on that", or chains: "what X said
  about Y, and how that relates to Z")

Return ONLY one of these JSON values (no other text):
{{"query_type": "simple"}}
{{"query_type": "comparative"}}
{{"query_type": "multi_hop"}}

Question: {question}"""


async def classify_query(question: str) -> str:
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(CLASSIFY_PROMPT.format(question=question))
        text = resp.content if hasattr(resp, "content") else str(resp)
        # be liberal in parsing — pluck the json blob
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(text[start : end + 1])
            qt = parsed.get("query_type", "simple")
            if qt in ("simple", "comparative", "multi_hop"):
                return qt
    except Exception as e:
        print(f"[classify_query] failed, defaulting to simple: {e}")
    return "simple"


# -- Decomposition for comparative queries ---------------------------------

DECOMPOSE_PROMPT = """\
The user asked a comparative question. Decompose it into exactly 2 independent
sub-questions, one per line, no numbering, no commentary. Each sub-question
should be answerable on its own.

Question: {question}

Sub-questions:"""


async def decompose_comparative(question: str) -> list[str]:
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(DECOMPOSE_PROMPT.format(question=question))
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [ln.strip(" -•").strip() for ln in text.splitlines() if ln.strip()]
        subs = [ln for ln in lines if len(ln) > 5][:2]
        return subs if len(subs) == 2 else [question, question]
    except Exception as e:
        print(f"[decompose_comparative] failed: {e}")
        return [question, question]


# -- Multi-hop step 2 query generation --------------------------------------

FOLLOWUP_PROMPT = """\
Given the original multi-step question and the intermediate answer from step 1,
generate the follow-up search query for step 2.
Return only the query string, no preamble.

Original question: {question}

Step 1 intermediate answer: {intermediate}

Step 2 search query:"""


async def make_followup_query(question: str, intermediate: str) -> str:
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(FOLLOWUP_PROMPT.format(
            question=question, intermediate=intermediate,
        ))
        text = resp.content if hasattr(resp, "content") else str(resp)
        return text.strip().splitlines()[0][:300] if text.strip() else question
    except Exception:
        return question


# -- Graph nodes -----------------------------------------------------------

async def classify_node(state: QueryState) -> dict[str, Any]:
    qt = await classify_query(state["question"])
    await state["emit"]({"type": "query_type", "value": qt})
    return {"query_type": qt}


async def simple_node(state: QueryState) -> dict[str, Any]:
    question = state["question"]
    source_id = state.get("source_id")
    emit = state["emit"]

    chunks = await retrieve(question, source_id=source_id)
    chunks = await hydrate_contexts(chunks)
    await emit({"type": "citations", "value": build_citations(chunks)})

    async for tok in stream_answer(question, chunks, state.get("chat_history")):
        await emit({"type": "token", "value": tok})
    # LangGraph requires each node to write at least one state channel.
    return {"query_type": state.get("query_type", "simple")}


async def comparative_node(state: QueryState) -> dict[str, Any]:
    question = state["question"]
    source_id = state.get("source_id")
    emit = state["emit"]

    subs = await decompose_comparative(question)
    await emit({"type": "sub_queries", "value": subs})

    chunk_lists = await asyncio.gather(
        *(retrieve(s, source_id=source_id) for s in subs),
        return_exceptions=True,
    )

    all_chunks: list[dict[str, Any]] = []
    for cl in chunk_lists:
        if isinstance(cl, Exception):
            continue
        all_chunks.extend(cl)

    # Dedupe by id
    seen: set[str] = set()
    unique = []
    for c in all_chunks:
        cid = str(c.get("id"))
        if cid not in seen:
            seen.add(cid)
            unique.append(c)

    hydrated = await hydrate_contexts(unique)
    await emit({"type": "citations", "value": build_citations(hydrated)})

    synth_question = (
        f"Compare the answers to the following sub-questions, and produce a "
        f"single synthesized answer to the original question.\n\n"
        f"Original: {question}\n"
        f"Sub-question 1: {subs[0]}\n"
        f"Sub-question 2: {subs[1]}"
    )
    async for tok in stream_answer(synth_question, hydrated, state.get("chat_history")):
        await emit({"type": "token", "value": tok})
    return {"sub_queries": subs}


async def multi_hop_node(state: QueryState) -> dict[str, Any]:
    question = state["question"]
    source_id = state.get("source_id")
    emit = state["emit"]

    # Step 1: retrieve + intermediate answer (Groq 8B, no stream)
    chunks_1 = await retrieve(question, source_id=source_id)
    chunks_1 = await hydrate_contexts(chunks_1)

    fast = get_fast_llm()
    interm_prompt = (
        f"Based ONLY on the following context, give a brief 2-3 sentence "
        f"intermediate answer to:\n\n{question}\n\nContext:\n"
        f"{format_context_for_llm(chunks_1)}"
    )
    interm_resp = await fast.ainvoke(interm_prompt)
    intermediate = interm_resp.content if hasattr(interm_resp, "content") else str(interm_resp)

    # Step 2: follow-up query
    followup = await make_followup_query(question, intermediate)
    await emit({"type": "sub_queries", "value": [question, followup]})

    chunks_2 = await retrieve(followup, source_id=source_id)
    chunks_2 = await hydrate_contexts(chunks_2)

    # combine + dedupe
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    for c in chunks_1 + chunks_2:
        cid = str(c.get("id"))
        if cid not in seen:
            seen.add(cid)
            combined.append(c)

    await emit({"type": "citations", "value": build_citations(combined)})

    final_question = (
        f"Original question: {question}\n\n"
        f"Step 1 intermediate finding: {intermediate}\n\n"
        f"Synthesize a final, well-grounded answer using BOTH the step-1 finding "
        f"and the additional context retrieved below."
    )
    async for tok in stream_answer(final_question, combined, state.get("chat_history")):
        await emit({"type": "token", "value": tok})
    return {"sub_queries": [question, followup]}


# -- Graph assembly --------------------------------------------------------

def _route(state: QueryState) -> str:
    return state.get("query_type", "simple")


def _build_graph():
    g = StateGraph(QueryState)
    g.add_node("classify", classify_node)
    g.add_node("simple", simple_node)
    g.add_node("comparative", comparative_node)
    g.add_node("multi_hop", multi_hop_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route,
        {"simple": "simple", "comparative": "comparative", "multi_hop": "multi_hop"},
    )
    g.add_edge("simple", END)
    g.add_edge("comparative", END)
    g.add_edge("multi_hop", END)
    return g.compile()


# Compiled once at import; reused across requests.
GRAPH = _build_graph()


# -- Public entry point ---------------------------------------------------

async def route_and_stream(
    question: str,
    source_id: UUID | None,
    chat_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict]:
    """
    Run the compiled LangGraph and yield SSE event dicts as nodes produce them.
    Nodes push events into a queue via `emit`; we drain it until the graph run
    completes.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def emit(event: dict) -> None:
        await queue.put(event)

    state: QueryState = {
        "question": question,
        "source_id": source_id,
        "chat_history": chat_history or [],
        "emit": emit,
    }

    async def _run() -> None:
        try:
            await GRAPH.ainvoke(state)
            await queue.put({"type": "done", "value": None})
        except Exception as e:
            print(f"[route_and_stream] graph run failed: {e}")
            await queue.put({"type": "error", "value": str(e)})
            await queue.put({"type": "done", "value": None})
        finally:
            await queue.put(_SENTINEL)

    task = asyncio.create_task(_run())
    try:
        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            yield event
    finally:
        await task
