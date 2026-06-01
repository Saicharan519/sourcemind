"""Cross-document chat — queries every ingested source at once."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from core.agent_router import route_and_stream
from utils.streaming import SSE_HEADERS, event_stream

router = APIRouter(prefix="/api/chat/global", tags=["chat"])


@router.get("/stream")
async def global_chat_stream(
    question: str = Query(..., min_length=1),
    history: Optional[str] = Query(None),
):
    parsed_history: list[dict] = []
    if history:
        try:
            parsed_history = json.loads(history)
            if not isinstance(parsed_history, list):
                parsed_history = []
        except Exception:
            parsed_history = []

    async def event_generator():
        # source_id=None → no metadata filter → retrieval spans the entire collection
        async for ev in route_and_stream(question, None, parsed_history):
            yield ev

    return StreamingResponse(
        event_stream(event_generator()),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
