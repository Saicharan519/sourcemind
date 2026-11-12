"""
Source tools: the retrieval surface SourceMind exposes to other programs (the MCP
server today). Plain async functions over the same code the web app uses
(hybrid_search, retrieve, hydrate_contexts, route_and_stream), so they can be
unit-tested without MCP.

Every result that contains document text puts it under `untrusted_content`:
it comes from user-uploaded files and must be treated by the caller (often
another LLM) as data, never as instructions.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from qdrant_client.http import models as qm

from config import settings
from core.hybrid_search import hybrid_search
from core.rag_engine import format_seconds, hydrate_contexts, retrieve
from db.postgres import Database, get_source
from db.postgres import list_sources as _list_sources
from db.qdrant import QdrantStore

MAX_TOP_K = 10


class ToolError(ValueError):
    """A caller mistake (bad id, unknown source...). Message is safe to show."""


def _uuid(value: str, what: str = "source_id") -> UUID:
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        raise ToolError(f"{what} must be a UUID, got {value!r}") from None


async def _ready_source(source_id: str | None) -> UUID | None:
    if source_id in (None, ""):
        return None
    sid = _uuid(source_id)
    src = await get_source(sid)
    if not src:
        raise ToolError(f"No source with id {source_id}")
    if src["status"] != "ready":
        raise ToolError(f"Source {source_id} is not ready (status: {src['status']})")
    return sid


def _passage(c: dict[str, Any], rank: int) -> dict[str, Any]:
    p = c.get("payload", {})
    out = {
        "rank": rank,
        "source_id": p.get("source_id"),
        "source_title": p.get("source_title"),
        "source_type": p.get("source_type"),
        "untrusted_content": c.get("content") or p.get("content", ""),
    }
    if p.get("source_type") == "document":
        out["page"] = p.get("page_number") or c.get("page_start")
    elif p.get("source_type") == "video":
        out["segment"] = format_seconds(p.get("segment_start"))
        out["start_seconds"] = p.get("segment_start")
        out["end_seconds"] = p.get("segment_end")
    return out


async def list_sources() -> list[dict[str, Any]]:
    rows = await _list_sources()
    return [{
        "source_id": str(r["id"]),
        "title": r.get("title"),
        "type": r.get("source_type"),
        "status": r.get("status"),
        "pages": r.get("page_count"),
        "duration_seconds": r.get("duration_s"),
    } for r in rows]


async def search_sources(query: str, source_id: str | None = None, top_k: int = 6,
                         expand_query: bool = False) -> dict[str, Any]:
    """Hybrid (dense + BM25) search. expand_query=True adds the LLM multi-query/HyDE
    expansion the chat uses (costs LLM calls); False is free and deterministic."""
    query = (query or "").strip()
    if not query:
        raise ToolError("query must not be empty")
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    sid = await _ready_source(source_id)
    diag: dict[str, Any] = {}
    if expand_query:
        chunks = await retrieve(query, source_id=sid, diagnostics=diag)
    else:
        chunks = await hybrid_search(query, source_id=sid, final_k=top_k)
    chunks = await hydrate_contexts(chunks[:top_k])
    return {
        "query": query,
        "scope": str(sid) if sid else "all sources",
        "passages": [_passage(c, i) for i, c in enumerate(chunks, 1)],
        "degraded": sorted(set(diag.get("degraded", []))),
    }


async def get_passage(source_id: str, page: int | None = None,
                      at_seconds: float | None = None) -> dict[str, Any]:
    """Full text around a document page (parent chunks covering it) or a video
    timestamp (the transcript segment containing it)."""
    sid = await _ready_source(source_id)
    if sid is None:
        raise ToolError("source_id is required")
    src = await get_source(sid)

    if src["source_type"] == "document":
        if page is None:
            raise ToolError("page is required for a document source")
        rows = await Database.fetch(
            "SELECT content, page_start, page_end FROM parent_chunks "
            "WHERE source_id = $1 AND page_start <= $2 AND page_end >= $2 ORDER BY chunk_index",
            sid, int(page))
        if not rows:
            raise ToolError(f"No text found on page {page} (the document has {src.get('page_count')} pages)")
        return {"source_id": str(sid), "source_title": src["title"], "page": int(page),
                "untrusted_content": "\n\n".join(r["content"] for r in rows)}

    if at_seconds is None:
        raise ToolError("at_seconds is required for a video source")
    t = float(at_seconds)
    points, _ = await QdrantStore.client().scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="source_id", match=qm.MatchValue(value=str(sid))),
            qm.FieldCondition(key="segment_start", range=qm.Range(lte=t)),
            qm.FieldCondition(key="segment_end", range=qm.Range(gte=t)),
        ]),
        limit=3, with_payload=True,
    )
    if not points:
        raise ToolError(f"No transcript segment at {t:.0f}s")
    p = sorted(points, key=lambda x: x.payload.get("segment_start", 0))[0].payload
    return {"source_id": str(sid), "source_title": src["title"],
            "segment": format_seconds(p.get("segment_start")),
            "start_seconds": p.get("segment_start"), "end_seconds": p.get("segment_end"),
            "untrusted_content": p.get("content", "")}


async def ask_sources(question: str, source_id: str | None = None) -> dict[str, Any]:
    """The full SourceMind answer pipeline (router -> retrieval -> grounded answer),
    collected instead of streamed. Uses the configured LLM provider."""
    from core.agent_router import route_and_stream

    question = (question or "").strip()
    if not question:
        raise ToolError("question must not be empty")
    sid = await _ready_source(source_id)
    out: dict[str, Any] = {"question": question, "scope": str(sid) if sid else "all sources",
                           "answer": "", "citations": [], "evidence": None,
                           "query_type": None, "warnings": [], "error": None}
    parts: list[str] = []
    async for ev in route_and_stream(question, sid, []):
        t, v = ev.get("type"), ev.get("value")
        if t == "token":
            parts.append(v or "")
        elif t == "citations":
            out["citations"] = [{k: c.get(k) for k in ("ref", "source_id", "source_title", "type",
                                                       "page_number", "segment", "excerpt")
                                 if c.get(k) is not None} for c in v or []]
        elif t == "evidence":
            out["evidence"] = v
        elif t == "query_type":
            out["query_type"] = v
        elif t == "warning":
            out["warnings"].append(v)
        elif t == "error":
            out["error"] = str(v)
    from core.rag_engine import normalize_citations
    out["answer"] = normalize_citations("".join(parts)).strip()
    return out
