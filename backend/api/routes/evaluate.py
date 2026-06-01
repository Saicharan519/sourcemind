"""Manual RAGAS re-evaluation."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import EvalResultModel
from core.evaluator import evaluate_source
from db.postgres import get_eval_result, get_source

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


@router.post("/{source_id}")
async def trigger_evaluation(source_id: UUID, background_tasks: BackgroundTasks):
    src = await get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found.")
    if src["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot evaluate — source status is '{src['status']}'.",
        )

    background_tasks.add_task(evaluate_source, source_id)
    return {"status": "evaluation_started", "source_id": str(source_id)}


@router.get("/{source_id}", response_model=EvalResultModel)
async def get_evaluation(source_id: UUID):
    row = await get_eval_result(source_id)
    if not row:
        raise HTTPException(status_code=404, detail="No evaluation found for this source.")
    return EvalResultModel(**row)
