"""
RAG retrieval + answer generation.

Pipeline:
  1. Generate 3 multi-query rephrasings (fast model)
  2. Generate a HyDE hypothetical answer (fast model)
  3. Run hybrid_search for each variant + original query
  4. Fuse the per-query ranked lists with RRF, keep final_k chunks
  5. For document chunks: fetch parent chunks from Postgres
  6. Stream the answer from the main model; it cites passages as [n]
  7. Validate the citations: only passages that were shown AND cited are
     returned; an answer with no supporting passage is flagged
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from core.hybrid_search import hybrid_search, reciprocal_rank_fusion
from core.llm_providers import make_llm
from db.postgres import fetch_parent_chunks


# -- LLM clients -----------------------------------------------------------
# The provider is chosen by settings.LLM_PROVIDER (see core/llm_providers.py).
# "main" = final answers / synthesis; "fast" = every auxiliary step.

def get_main_llm(streaming: bool = False):
    return make_llm("main", temperature=0.2, streaming=streaming)


def get_fast_llm():
    return make_llm("fast", temperature=0.3)


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


async def generate_multi_queries(question: str) -> list[str] | None:
    """Return up to 3 rephrasings, or None if the LLM call failed."""
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(MULTI_QUERY_PROMPT.format(question=question))
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [ln.strip(" -•").strip() for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 5][:3]
    except Exception as e:
        print(f"[multi_query] failed: {e}")
        return None


async def generate_hyde(question: str) -> str | None:
    """Return a hypothetical answer, or None if the LLM call failed."""
    llm = get_fast_llm()
    try:
        resp = await llm.ainvoke(HYDE_PROMPT.format(question=question))
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[hyde] failed: {e}")
        return None


# -- Retrieval orchestration -----------------------------------------------

async def retrieve(
    question: str,
    source_id: UUID | None = None,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run hybrid search for the question plus its multi-query / HyDE expansions,
    then merge the per-query ranked lists with Reciprocal Rank Fusion (so each
    list's own dense+BM25+MMR ordering is respected) and keep MMR_FINAL_K.

    If `diagnostics` is given, failed expansion steps are recorded in
    diagnostics["degraded"] so callers can surface them instead of silently
    answering with a weaker pipeline.
    """
    queries = [question]
    degraded: list[str] = []

    helpers = []
    if use_multi_query:
        helpers.append(("multi_query", generate_multi_queries(question)))
    if use_hyde:
        helpers.append(("hyde", generate_hyde(question)))
    if helpers:
        results = await asyncio.gather(*(h[1] for h in helpers), return_exceptions=True)
        for (name, _), r in zip(helpers, results):
            if r is None or isinstance(r, Exception):
                degraded.append(name)
            elif isinstance(r, list):
                queries.extend(r)
            elif isinstance(r, str) and r.strip():
                queries.append(r)

    all_results = await asyncio.gather(
        *(hybrid_search(q, source_id=source_id) for q in queries), return_exceptions=True
    )
    ranked_lists = []
    for q, res in zip(queries, all_results):
        if isinstance(res, Exception):
            if q is question:
                raise res  # the main query itself failed: a real error
            degraded.append("search")
            continue
        ranked_lists.append(res)

    if diagnostics is not None:
        diagnostics.setdefault("degraded", []).extend(degraded)
    fused = reciprocal_rank_fusion(ranked_lists)
    return fused[: settings.MMR_FINAL_K]


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


def _citation(c: dict[str, Any], ref: int | None = None) -> dict[str, Any]:
    payload = c.get("payload", {})
    source_type = payload.get("source_type")
    excerpt = (c.get("content") or "")[:200].replace("\n", " ").strip()
    base = {"ref": ref, "source_id": payload.get("source_id"),
            "source_title": payload.get("source_title"), "excerpt": excerpt}
    if source_type == "document":
        return {**base, "type": "document",
                "page_number": payload.get("page_number") or c.get("page_start")}
    if source_type == "video":
        return {**base, "type": "video",
                "segment": format_seconds(payload.get("segment_start")),
                "segment_start": payload.get("segment_start"),
                "segment_end": payload.get("segment_end")}
    return {**base, "type": "unknown"}


def build_citations(chunks: list[dict[str, Any]], cited: list[int] | None = None) -> list[dict[str, Any]]:
    """
    Citation cards. `ref` is the chunk's 1-based number in the LLM context.
    If `cited` is given, only those refs are returned, in citation order;
    otherwise every chunk is returned (used for the retrieval `context` event).
    """
    if cited is None:
        return [_citation(c, i) for i, c in enumerate(chunks, 1)]
    return [_citation(chunks[r - 1], r) for r in cited if 1 <= r <= len(chunks)]


# -- Answer generation -----------------------------------------------------

NOT_FOUND_ANSWER = "I could not find this in the provided sources."

ANSWER_SYSTEM_PROMPT = f"""You are an expert research assistant. Answer the user's question using ONLY the
numbered context passages below. Be precise, concise, and direct.

Rules:
- Cite every factual statement with the number of the passage it comes from, in
  square brackets, e.g. "The limit is 30 days [2]." or "[1][3]" for several.
- Only use passage numbers that appear in the context. Never invent citations.
- If the passages do not contain the answer, reply with exactly:
  "{NOT_FOUND_ANSWER}" and nothing else. Do not guess or use outside knowledge.
- If the passages only partly answer the question, answer that part (with
  citations) and say what is missing.
- Do not mention "the context" or "the passages" - speak naturally.
"""

_REF_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
# gpt-oss models sometimes cite in their native style: 【6】 or 【6†L3-L9】
_REF_FULLWIDTH = re.compile(r"【\s*(\d+(?:\s*[,–-]\s*\d+)*)\s*(?:†[^】]*)?】")


def normalize_citations(answer: str) -> str:
    """Rewrite native gpt-oss citations (【6】, 【6†L1-L4】) as [6] so users and the
    parser see one format."""
    return _REF_FULLWIDTH.sub(lambda m: f"[{m.group(1)}]", answer)


def extract_citation_refs(answer: str) -> list[int]:
    """All passage numbers cited as [n], [n, m], [n-m], [n][m] or 【n】, in first-seen order."""
    refs: list[int] = []
    for group in _REF_GROUP.findall(normalize_citations(answer)):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                lo, hi = (int(x) for x in re.split(r"\s*[–-]\s*", part)[:2])
                nums = range(lo, hi + 1) if 0 < hi - lo < 20 else [lo, hi]
            else:
                nums = [int(part)]
            for n in nums:
                if n not in refs:
                    refs.append(n)
    return refs


def is_not_found(answer: str) -> bool:
    a = " ".join(answer.lower().split())
    return "could not find this in the" in a or "could not find this information" in a


def validate_citations(answer: str, n_chunks: int) -> dict[str, Any]:
    """Split cited refs into valid (point at a passage that was shown) and invalid,
    and classify the answer: insufficient (the model abstained), supported (at least
    one valid citation) or uncited (an answer with no valid citation)."""
    refs = extract_citation_refs(answer)
    valid = [r for r in refs if 1 <= r <= n_chunks]
    invalid = [r for r in refs if not 1 <= r <= n_chunks]
    if is_not_found(answer) and not valid:
        status = "insufficient"
    elif valid:
        status = "supported"
    else:
        status = "uncited"
    return {"status": status, "valid": valid, "invalid": invalid}


def format_context_for_llm(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        payload = c.get("payload", {})
        source_type = payload.get("source_type")
        title = payload.get("source_title") or "Untitled"
        if source_type == "document":
            page = payload.get("page_number") or c.get("page_start")
            where = f"page {page}" if page else "document"
        elif source_type == "video":
            where = f"video segment {format_seconds(payload.get('segment_start'))}"
        else:
            where = "source"
        parts.append(f"[{i}] {title} - {where}\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Stream the final answer token by token from the main model."""
    if not chunks:
        yield NOT_FOUND_ANSWER
        return
    context = format_context_for_llm(chunks)
    user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"

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


async def stream_grounded_answer(
    emit,
    question: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    The one answer path shared by every query strategy:
      1. emit `context`  - every passage the model will see (for debugging / eval)
      2. stream `token`s - the answer, which cites passages as [n]
      3. emit `citations` - ONLY the passages the answer actually cited, validated
         against what was shown (invented numbers are dropped)
      4. emit `evidence` - supported / insufficient / uncited, plus invalid refs
    Returns the full answer text.
    """
    await emit({"type": "context", "value": build_citations(chunks)})
    parts: list[str] = []
    async for tok in stream_answer(question, chunks, chat_history):
        parts.append(tok)
        await emit({"type": "token", "value": tok})
    answer = "".join(parts)
    check = validate_citations(answer, len(chunks))
    cited = [] if check["status"] == "insufficient" else check["valid"]
    await emit({"type": "citations", "value": build_citations(chunks, cited)})
    await emit({"type": "evidence", "value": {"status": check["status"],
                                              "invalid_refs": check["invalid"],
                                              "n_context": len(chunks)}})
    return answer


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
