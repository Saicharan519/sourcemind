"""
Minimal MCP client demo: launches the SourceMind MCP server over stdio (exactly
as Claude Desktop or MCP Inspector would) and calls its tools.

Run from backend/ with the backend venv (Postgres + Qdrant running):
    python mcp_demo_client.py "your question"            # all sources
    python mcp_demo_client.py "your question" --no-ask   # skip the LLM call
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


def show(title, result):
    print(f"\n=== {title} ===")
    data = json.loads(result.content[0].text) if result.content else None
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2500])
    return data


async def main(question: str, ask: bool) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "mcp_server.py")],
                                   cwd=str(HERE))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            sources = show("list_sources", await session.call_tool("list_sources", {}))
            hits = show("search_sources", await session.call_tool(
                "search_sources", {"query": question, "top_k": 3}))
            if hits and hits.get("passages"):
                top = hits["passages"][0]
                args = {"source_id": top["source_id"]}
                if top.get("page") is not None:
                    args["page"] = top["page"]
                else:
                    args["at_seconds"] = top.get("start_seconds", 0)
                show("get_passage (top hit)", await session.call_tool("get_passage", args))
            if ask:
                show("ask_sources", await session.call_tool("ask_sources", {"question": question}))
            print(f"\n{(sources or {}).get('count', 0)} sources in the database")


if __name__ == "__main__":
    q = next((a for a in sys.argv[1:] if not a.startswith("--")), "What is retrieval-augmented generation?")
    asyncio.run(main(q, ask="--no-ask" not in sys.argv))
