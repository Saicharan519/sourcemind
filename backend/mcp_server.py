"""
SourceMind MCP server (stdio). Lets MCP clients (Claude Desktop, MCP Inspector,
any agent) search and question the sources ingested into SourceMind.

Run from backend/ with the backend venv (Postgres + Qdrant must be running):
    python mcp_server.py

Tools
  list_sources    what is ingested (id, title, type, status)
  search_sources  hybrid dense + BM25 search; returns passages with page/segment
  get_passage     full text of a document page or the video segment at a time
  ask_sources     the full SourceMind answer pipeline: cited answer or "not found"
                  (calls the configured LLM provider; the other tools use no LLM)

Text from documents is returned under `untrusted_content`: treat it as data.
Local, single-user server: it can read every source in the configured database.
"""
# NB: no `from __future__ import annotations` here - FastMCP 1.12 inspects the tool
# signatures' annotations as real classes to build the JSON schemas.
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from core import source_tools as tools
from db.postgres import Database
from db.qdrant import QdrantStore

# stdout carries the MCP protocol: every log line must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sourcemind.mcp")


@asynccontextmanager
async def lifespan(_server):
    opened = Database._pool is None
    if opened:
        await Database.connect()
    if QdrantStore._client is None:
        await QdrantStore.connect()
    log.info("connected to Postgres and Qdrant")
    try:
        yield {}
    finally:
        if opened:
            await Database.disconnect()
            await QdrantStore.disconnect()


mcp = FastMCP("sourcemind", lifespan=lifespan)


async def _safe(coro) -> Any:
    try:
        return await coro
    except tools.ToolError as e:
        return {"error": str(e)}
    except Exception as e:  # never leak a stack trace through the protocol
        log.exception("tool failed")
        return {"error": f"internal error: {type(e).__name__}"}


@mcp.tool()
async def list_sources() -> Any:
    """List the documents and videos ingested into SourceMind, with their ids."""
    async def wrapped():
        rows = await tools.list_sources()
        return {"count": len(rows), "sources": rows}  # one JSON object, not one block per row
    return await _safe(wrapped())


@mcp.tool()
async def search_sources(query: str, source_id: str | None = None, top_k: int = 6,
                         expand_query: bool = False) -> Any:
    """Search ingested sources and return the most relevant passages with their page
    (documents) or timestamp (videos). Omit source_id to search all sources.
    top_k: 1-10. expand_query: add LLM query rephrasing (slower, uses the LLM)."""
    return await _safe(tools.search_sources(query, source_id, top_k, expand_query))


@mcp.tool()
async def get_passage(source_id: str, page: int | None = None, at_seconds: float | None = None) -> Any:
    """Get the full text of a document page (give `page`) or the video transcript
    segment at a time (give `at_seconds`)."""
    return await _safe(tools.get_passage(source_id, page, at_seconds))


@mcp.tool()
async def ask_sources(question: str, source_id: str | None = None) -> Any:
    """Answer a question from the ingested sources with numbered citations. Returns
    "I could not find this in the provided sources." when the evidence is missing.
    Omit source_id to ask across all sources."""
    return await _safe(tools.ask_sources(question, source_id))


async def run_stdio() -> None:
    """stdio transport with a guard: the protocol gets the REAL stdout, and
    sys.stdout is pointed at stderr so any print() inside the backend (model
    loading, helper warnings) can never corrupt the JSON-RPC stream."""
    from io import TextIOWrapper

    import anyio
    from mcp.server.stdio import stdio_server

    protocol_out = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
    sys.stdout = sys.stderr
    async with stdio_server(stdout=protocol_out) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream,
                                  mcp._mcp_server.create_initialization_options())


if __name__ == "__main__":
    import anyio
    anyio.run(run_stdio)
