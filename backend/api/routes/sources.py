"""GET/DELETE sources."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from api.schemas import DeleteResponse, EvalResultModel, SourceDetail, SourceListItem
from db.postgres import delete_source, get_eval_result, get_source, list_sources
from db.qdrant import QdrantStore

router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("", response_model=list[SourceListItem])
async def get_all_sources():
    rows = await list_sources()
    return [SourceListItem(**r) for r in rows]


@router.get("/{source_id}", response_model=SourceDetail)
async def get_source_detail(source_id: UUID):
    src = await get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found.")
    eval_row = await get_eval_result(source_id)
    eval_model = EvalResultModel(**eval_row) if eval_row else None
    return SourceDetail(**src, eval_results=eval_model)


@router.delete("/{source_id}", response_model=DeleteResponse)
async def delete_source_route(source_id: UUID):
    src = await get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found.")
    # Postgres cascade handles parent_chunks, bm25_index, eval_results, chat_history
    await delete_source(source_id)
    # Qdrant vectors
    try:
        await QdrantStore.delete_by_source(source_id)
    except Exception as e:
        print(f"[delete_source] Qdrant cleanup failed: {e}")
    return DeleteResponse(message="Source deleted successfully.")
