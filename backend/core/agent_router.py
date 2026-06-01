"""
LangGraph agentic query router.

Every chat request runs through this state machine:
  - classify_query: simple | comparative | multi_hop
  - simple_rag:      standard retrieve + answer
  - comparative_rag: decompose into 2 sub-queries, retrieve both, synthesize
  - multi_hop_rag:   step-1 retrieve → intermediate answer → step-2 retrieve → final

Each path is an async generator that yields events:
  {"type": "query_type", "value": ...}
  {"type": "sub_queries", "value": [...]}
  {"type": "citations", "value": [...]}
  {"type": "token", "value": "..."}
  {"type": "done"}
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator
from uuid import UUID

from langchain_core.messages import HumanMessage

from core.rag_engine import (
    build_citations,
    format_context_for_llm,
    generate_answer_sync,
    get_fast_llm,
    hydrate_contexts,
    retrieve,
    stream_answer,
)


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


# -- Path implementations --------------------------------------------------

async def _emit(event_type: str, value: Any) -> dict:
    return {"type": event_type, "value": value}


async def _simple_path(
    question: str,
    source_id: UUID | None,
    chat_history: list[dict[str, str]] | None,
) -> AsyncIterator[dict]:
    yield {"type": "query_type", "value": "simple"}

    chunks = await retrieve(question, source_id=source_id)
    chunks = await hydrate_contexts(chunks)
    citations = build_citations(chunks)
    yield {"type": "citations", "value": citations}

    async for tok in stream_answer(question, chunks, chat_history):
        yield {"type": "token", "value": tok}
    yield {"type": "done", "value": None}


async def _comparative_path(
    question: str,
    source_id: UUID | None,
    chat_history: list[dict[str, str]] | None,
) -> AsyncIterator[dict]:
    yield {"type": "query_type", "value": "comparative"}

    subs = await decompose_comparative(question)
    yield {"type": "sub_queries", "value": subs}

    import asyncio
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
    seen = set()
    unique = []
    for c in all_chunks:
        cid = str(c.get("id"))
        if cid not in seen:
            seen.add(cid)
            unique.append(c)

    hydrated = await hydrate_contexts(unique)
    citations = build_citations(hydrated)
    yield {"type": "citations", "value": citations}

    synth_question = (
        f"Compare the answers to the following sub-questions, and produce a "
        f"single synthesized answer to the original question.\n\n"
        f"Original: {question}\n"
        f"Sub-question 1: {subs[0]}\n"
        f"Sub-question 2: {subs[1]}"
    )
    async for tok in stream_answer(synth_question, hydrated, chat_history):
        yield {"type": "token", "value": tok}
    yield {"type": "done", "value": None}


async def _multi_hop_path(
    question: str,
    source_id: UUID | None,
    chat_history: list[dict[str, str]] | None,
) -> AsyncIterator[dict]:
    yield {"type": "query_type", "value": "multi_hop"}

    # Step 1: retrieve + intermediate answer (Groq 8B, no stream)
    chunks_1 = await retrieve(question, source_id=source_id)
    chunks_1 = await hydrate_contexts(chunks_1)

    # quick intermediate answer using the fast model
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
    yield {"type": "sub_queries", "value": [question, followup]}

    chunks_2 = await retrieve(followup, source_id=source_id)
    chunks_2 = await hydrate_contexts(chunks_2)

    # combine + dedupe
    seen = set()
    combined: list[dict[str, Any]] = []
    for c in chunks_1 + chunks_2:
        cid = str(c.get("id"))
        if cid not in seen:
            seen.add(cid)
            combined.append(c)

    citations = build_citations(combined)
    yield {"type": "citations", "value": citations}

    final_question = (
        f"Original question: {question}\n\n"
        f"Step 1 intermediate finding: {intermediate}\n\n"
        f"Synthesize a final, well-grounded answer using BOTH the step-1 finding "
        f"and the additional context retrieved below."
    )
    async for tok in stream_answer(final_question, combined, chat_history):
        yield {"type": "token", "value": tok}
    yield {"type": "done", "value": None}


# -- Public entry point ---------------------------------------------------

async def route_and_stream(
    question: str,
    source_id: UUID | None,
    chat_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict]:
    """
    Main entry. Classify query, dispatch to the right path, yield events.
    """
    qt = await classify_query(question)
    if qt == "comparative":
        async for ev in _comparative_path(question, source_id, chat_history):
            yield ev
    elif qt == "multi_hop":
        async for ev in _multi_hop_path(question, source_id, chat_history):
            yield ev
    else:
        async for ev in _simple_path(question, source_id, chat_history):
            yield ev
