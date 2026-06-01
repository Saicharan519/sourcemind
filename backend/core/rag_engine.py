"""
RAG retrieval + answer generation.

Pipeline:
  1. Generate 3 multi-query rephrasings (Groq 8B)
  2. Generate a HyDE hypothetical answer (Groq 8B)
  3. Run hybrid_search for each variant + original query
  4. Union, dedupe, take the highest-scoring final_k chunks
  5. For document chunks: fetch parent chunks from Postgres
  6. Stream the final answer from Groq 70B token-by-token
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import settings
from core.hybrid_search import hybrid_search
from db.postgres import fetch_parent_chunks


# -- LLM clients -----------------------------------------------------------

def get_main_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model=settings.GROQ_MODEL_MAIN,
        api_key=settings.GROQ_API_KEY,
        temperature=0.2,
        streaming=streaming,
    )


def get_fast_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.GROQ_MODEL_FAST,
        api_key=settings.GROQ_API_KEY,
        temperature=0.3,
    )


# -- Multi-query + HyDE ----------------------------------------------------

MULTI_QUERY_PROMPT = """\
You generate alternative phrasings of a user question for document retrieval.
Given the question, return exactly 3 alternative rephrasings, one per line,
no numbering, no commentary. Each rephrasing should preserve the original
intent but use different wording or focus on a different angle.

Question: {question}

Alternative rephrasings:"""


HYDE_PROMPT = """\
You write a short hypothetical answer to the user's question, as it might
appear in a textbook or technical document. The answer must use the same
vocabulary and style as a real document on this topic. 3-5 sentences.

Question: {question}

Hypothetical answer:"""


async def generate_multi_queries(question: str) -> list[str]:
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(MULTI_QUERY_PROMPT.format(question=question))
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [ln.strip(" -•").strip() for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 5][:3]
    except Exception as e:
        print(f"[multi_query] failed: {e}")
        return []


async def generate_hyde(question: str) -> str:
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(HYDE_PROMPT.format(question=question))
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[hyde] failed: {e}")
        return ""


# -- Retrieval orchestration -----------------------------------------------

async def retrieve(
    question: str,
    source_id: UUID | None = None,
    use_multi_query: bool = True,
    use_hyde: bool = True,
) -> list[dict[str, Any]]:
    """
    Run multi-query + HyDE expanded hybrid search and return the unioned,
    deduped, MMR-reranked final chunks.
    """
    queries = [question]

    if use_multi_query or use_hyde:
        sub_tasks = []
        if use_multi_query:
            sub_tasks.append(generate_multi_queries(question))
        if use_hyde:
            sub_tasks.append(generate_hyde(question))
        results = await asyncio.gather(*sub_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, list):
                queries.extend(r)
            elif isinstance(r, str) and r.strip():
                queries.append(r)

    # Run hybrid search for each query in parallel
    search_tasks = [hybrid_search(q, source_id=source_id) for q in queries]
    all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Dedupe by chunk id, keeping highest rrf_score
    by_id: dict[str, dict[str, Any]] = {}
    for results in all_results:
        if isinstance(results, Exception):
            continue
        for chunk in results:
            cid = str(chunk.get("id"))
            existing = by_id.get(cid)
            if existing is None or chunk.get("rrf_score", 0) > existing.get("rrf_score", 0):
                by_id[cid] = chunk

    final = sorted(by_id.values(), key=lambda c: c.get("rrf_score", 0), reverse=True)
    return final[: settings.MMR_FINAL_K]


# -- Parent chunk hydration ------------------------------------------------

async def hydrate_contexts(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Replace child-chunk content with parent-chunk content for document sources.
    For video sources, keep the chunk content as-is.
    """
    parent_ids = []
    for c in chunks:
        pid = c.get("payload", {}).get("parent_chunk_id")
        if pid:
            parent_ids.append(UUID(pid))

    parents = await fetch_parent_chunks(parent_ids) if parent_ids else {}

    hydrated = []
    for c in chunks:
        payload = c.get("payload", {})
        pid = payload.get("parent_chunk_id")
        if pid and UUID(pid) in parents:
            p = parents[UUID(pid)]
            hydrated.append({
                **c,
                "content": p["content"],
                "page_start": p.get("page_start"),
                "page_end": p.get("page_end"),
            })
        else:
            # video chunk (or BM25-only hit with no parent linkage)
            hydrated.append({
                **c,
                "content": c.get("content") or payload.get("content", ""),
            })
    return hydrated


# -- Citation formatting ---------------------------------------------------

def format_seconds(secs: float | int | None) -> str:
    if secs is None:
        return "?:??"
    s = int(secs)
    return f"{s // 60:02d}:{s % 60:02d}"


def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for c in chunks:
        payload = c.get("payload", {})
        source_type = payload.get("source_type")
        excerpt = (c.get("content") or "")[:200].replace("\n", " ").strip()
        if source_type == "document":
            page = payload.get("page_number") or c.get("page_start")
            citations.append({
                "type": "document",
                "source_id": payload.get("source_id"),
                "source_title": payload.get("source_title"),
                "page_number": page,
                "excerpt": excerpt,
            })
        elif source_type == "video":
            citations.append({
                "type": "video",
                "source_id": payload.get("source_id"),
                "source_title": payload.get("source_title"),
                "segment": format_seconds(payload.get("segment_start")),
                "segment_start": payload.get("segment_start"),
                "segment_end": payload.get("segment_end"),
                "excerpt": excerpt,
            })
        else:
            citations.append({
                "type": "unknown",
                "source_id": payload.get("source_id"),
                "excerpt": excerpt,
            })
    return citations


# -- Answer generation -----------------------------------------------------

ANSWER_SYSTEM_PROMPT = """\
You are an expert research assistant. Answer the user's question based ONLY
on the context provided below. Be precise, concise, and direct.

Rules:
- If the answer is not in the context, say: "I could not find this in the source."
- For document sources, cite page numbers inline like: (page 12)
- For video sources, cite segment positions inline like: (segment 04:32)
- Do not fabricate page numbers or segments. Only use values present in the context.
- Do not mention "the context" or "the provided context" — speak naturally.
"""


def format_context_for_llm(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        payload = c.get("payload", {})
        source_type = payload.get("source_type")
        if source_type == "document":
            page = payload.get("page_number") or c.get("page_start")
            header = f"[Document — page {page}]" if page else "[Document]"
        elif source_type == "video":
            seg = format_seconds(payload.get("segment_start"))
            header = f"[Video — segment {seg}]"
        else:
            header = "[Source]"
        parts.append(f"{header}\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Stream the final answer token by token from Groq 70B."""
    context = format_context_for_llm(chunks)
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"

    messages = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]
    if chat_history:
        from langchain_core.messages import AIMessage
        for m in chat_history[-6:]:  # last 3 exchanges
            if m.get("role") == "user":
                messages.append(HumanMessage(content=m.get("content", "")))
            elif m.get("role") == "assistant":
                messages.append(AIMessage(content=m.get("content", "")))
    messages.append(HumanMessage(content=user_prompt))

    llm = get_main_llm(streaming=True)
    async for chunk in llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else ""
        if token:
            yield token


async def generate_answer_sync(
    question: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Non-streaming variant. Used by RAGAS and by the multi-hop intermediate step."""
    out = []
    async for tok in stream_answer(question, chunks, chat_history):
        out.append(tok)
    return "".join(out)
