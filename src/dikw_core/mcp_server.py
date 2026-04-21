"""MCP server stub.

Phase 0 only checks that the ``mcp`` SDK imports and exposes a ``core.status``
tool so an MCP client can see that the server is reachable. Phases 1-3 will
add ``core.query``, ``doc.*``, ``wiki.*``, ``wisdom.*``, and ``admin.*`` tools
as they come online.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import api


async def build_server() -> Any:
    """Construct the ``mcp`` Server instance with the Phase-0 tool surface.

    Import of ``mcp.server`` is deferred so that importing this module never
    fails on environments without the SDK installed.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("dikw-core")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="core.status",
                description=(
                    "Return counts of documents, chunks, embeddings, links, and wisdom "
                    "items for the nearest dikw wiki."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory inside the wiki. Defaults to cwd.",
                        }
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name != "core.status":
            raise ValueError(f"unknown tool: {name}")
        path = Path(arguments.get("path", "."))
        counts = await api.status(path)
        return [TextContent(type="text", text=counts.model_dump_json(indent=2))]

    return server


def run_stdio() -> None:  # pragma: no cover - integration entry point
    import asyncio

    from mcp.server.stdio import stdio_server

    async def _main() -> None:
        server = await build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_main())
