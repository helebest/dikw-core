"""MCP server for dikw-core.

Phase 0-1 tools:
  * ``core.status`` — counts across DIKW layers.
  * ``core.query`` — natural-language question → cited answer.
  * ``admin.ingest`` — run the ingest pipeline.
  * ``doc.search`` — hybrid search returning ranked hits (no LLM call).
  * ``doc.read`` — return the body (or a chunk of it) for a given doc path or id.

Tool groups mirror the reference projects' shape, keeping room for
``wiki.*`` and ``wisdom.*`` in Phases 2 and 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import api
from .info.search import HybridSearcher
from .providers import build_embedder
from .schemas import Layer
from .storage import build_storage


async def build_server() -> Any:
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("dikw-core")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="core.status",
                description="Counts of documents, chunks, embeddings, links, and wisdom items.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Wiki-internal path."}
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="core.query",
                description=(
                    "Answer a natural-language question using the wiki as context, "
                    "returning an answer with inline citations and the raw citations list."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["q"],
                    "properties": {
                        "q": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="admin.ingest",
                description="Scan configured sources and update the D + I layers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "embed": {
                            "type": "boolean",
                            "default": True,
                            "description": "If false, skip embeddings (FTS only).",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="doc.search",
                description="Hybrid search (FTS + vectors, RRF) returning ranked hits. No LLM call.",
                inputSchema={
                    "type": "object",
                    "required": ["q"],
                    "properties": {
                        "q": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                        "layer": {
                            "type": "string",
                            "enum": ["source", "wiki", "wisdom"],
                            "description": "Restrict hits to a single DIKW layer.",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="doc.read",
                description="Return the full body of a document identified by its wiki-relative path.",
                inputSchema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "description": "Wiki-relative doc path."},
                        "wiki_path": {"type": "string", "description": "Directory inside the wiki."},
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "core.status":
            counts = await api.status(arguments.get("path", "."))
            return [TextContent(type="text", text=counts.model_dump_json(indent=2))]

        if name == "core.query":
            result = await api.query(
                arguments["q"],
                arguments.get("path", "."),
                limit=int(arguments.get("limit", 5)),
            )
            return [TextContent(type="text", text=result.model_dump_json(indent=2))]

        if name == "admin.ingest":
            path = arguments.get("path", ".")
            embedder = None
            if arguments.get("embed", True):
                cfg, _ = api.load_wiki(path)
                embedder = build_embedder(cfg.provider)
            report = await api.ingest(path, embedder=embedder)
            return [TextContent(type="text", text=json.dumps(report.__dict__, indent=2))]

        if name == "doc.search":
            wiki_path = arguments.get("path", ".")
            cfg, root = api.load_wiki(wiki_path)
            storage = build_storage(cfg.storage, root=root)
            await storage.connect()
            await storage.migrate()
            try:
                embedder = build_embedder(cfg.provider)
                searcher = HybridSearcher(
                    storage, embedder, embedding_model=cfg.provider.embedding_model
                )
                layer_arg = arguments.get("layer")
                layer = Layer(layer_arg) if layer_arg else None
                hits = await searcher.search(
                    arguments["q"],
                    limit=int(arguments.get("limit", 10)),
                    layer=layer,
                )
            finally:
                await storage.close()
            payload = [h.model_dump() for h in hits]
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        if name == "doc.read":
            doc_path = arguments["path"]
            wiki_path = arguments.get("wiki_path", ".")
            cfg, root = api.load_wiki(wiki_path)
            abs_path = (root / doc_path).resolve()
            if not abs_path.is_file():
                raise ValueError(f"not a file: {doc_path}")
            text = Path(abs_path).read_text(encoding="utf-8")
            return [TextContent(type="text", text=text)]

        raise ValueError(f"unknown tool: {name}")

    return server


def run_stdio() -> None:  # pragma: no cover - integration entry point
    import asyncio

    from mcp.server.stdio import stdio_server

    async def _main() -> None:
        server = await build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_main())
