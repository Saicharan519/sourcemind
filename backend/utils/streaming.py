"""Server-Sent Events helpers."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator


def format_sse(event: dict[str, Any]) -> str:
    """Format a dict as a single SSE message."""
    return f"data: {json.dumps(event, default=str)}\n\n"


async def event_stream(generator: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    """Convert an async iterator of event dicts into SSE-formatted strings."""
    try:
        async for event in generator:
            yield format_sse(event)
    except Exception as e:
        yield format_sse({"type": "error", "value": str(e)})
        yield format_sse({"type": "done", "value": None})


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
}
