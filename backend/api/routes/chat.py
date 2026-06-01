"""Per-source chat — SSE streaming."""
from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from core.agent_router import route_and_stream
from db.postgres import get_source, insert_chat_message
from utils.streaming import SSE_HEADERS, event_stream

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/stream")
async def chat_stream(
    source_id: UUID = Query(..., description="UUID of the source to chat with"),
    question: str = Query(..., min_length=1),
    history: Optional[str] = Query(None, description="JSON array of {role, content}"),
):
    src = await get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found.")
    if src["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Source is not ready (current status: {src['status']}).",
        )

    parsed_history: list[dict] = []
    if history:
        try:
            parsed_history = json.loads(history)
            if not isinstance(parsed_history, list):
                parsed_history = []
        except Exception:
            parsed_history = []

    # Persist user message
    try:
        await insert_chat_message(source_id, "user", question)
    except Exception as e:
        print(f"[chat] failed to persist user message: {e}")

    # Capture the streamed answer so we can persist it after the stream finishes
    answer_buffer: list[str] = []

    async def event_generator():
        async for ev in route_and_stream(question, source_id, parsed_history):
            if ev.get("type") == "token":
                answer_buffer.append(ev.get("value", ""))
            yield ev
        # After the stream completes, save the assistant's response
        full = "".join(answer_buffer)
        if full.strip():
            try:
                await insert_chat_message(source_id, "assistant", full)
            except Exception as e:
                print(f"[chat] failed to persist assistant message: {e}")

    return StreamingResponse(
        event_stream(event_generator()),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
