"""MCP server for dikw-core.

Phase 0-3 tools:
  * ``core.status`` — counts across DIKW layers.
  * ``core.query`` — natural-language question -> cited answer (with wisdom).
  * ``admin.ingest`` — run the ingest pipeline.
  * ``admin.lint`` — run the K-layer hygiene checker.
  * ``doc.search`` — hybrid search returning ranked hits (no LLM call).
  * ``doc.read`` — return the body (or a chunk of it) for a given doc path or id.
  * ``wiki.synthesize`` — turn source docs into K-layer wiki pages.
  * ``wiki.list`` — list on-disk wiki pages with titles + types.
  * ``wiki.get`` — read a wiki page back with front-matter.
  * ``wisdom.distill`` — propose W-layer candidates from the K layer.
  * ``wisdom.list`` — list wisdom items filtered by status.
  * ``wisdom.approve`` — approve a candidate (candidate -> approved).
  * ``wisdom.reject`` — reject a candidate (-> archived).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import api
from .info.search import HybridSearcher
from .knowledge.wiki import read_page
from .providers import build_embedder
from .schemas import Layer, WisdomStatus
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
            Tool(
                name="wiki.synthesize",
                description="Turn source docs into K-layer wiki pages via the configured LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "all": {
                            "type": "boolean",
                            "default": False,
                            "description": "Re-synthesise every source, not just new ones.",
                        },
                        "embed": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wiki.list",
                description="List wiki pages (title + type + tags) for the nearest wiki.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wiki.get",
                description="Read a wiki page's front-matter and body by wiki-relative path.",
                inputSchema={
                    "type": "object",
                    "required": ["page_path"],
                    "properties": {
                        "page_path": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="admin.lint",
                description=(
                    "Report broken wikilinks, orphan pages, and duplicate titles "
                    "across the K layer."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wisdom.distill",
                description=(
                    "Propose W-layer candidates (principles / lessons / patterns) "
                    "from the current K-layer wiki. Each candidate must cite at "
                    "least two pieces of evidence."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "batch": {"type": "integer", "default": 8},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wisdom.list",
                description="List wisdom items filtered by status (candidate / approved / archived).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["candidate", "approved", "archived"],
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wisdom.approve",
                description="Approve a candidate wisdom item by id (W-xxxxxx).",
                inputSchema={
                    "type": "object",
                    "required": ["item_id"],
                    "properties": {
                        "path": {"type": "string"},
                        "item_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="wisdom.reject",
                description="Reject a candidate wisdom item — archives it and drops the candidate file.",
                inputSchema={
                    "type": "object",
                    "required": ["item_id"],
                    "properties": {
                        "path": {"type": "string"},
                        "item_id": {"type": "string"},
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
                    storage,
                    embedder,
                    embedding_model=cfg.provider.embedding_model,
                    rrf_k=cfg.retrieval.rrf_k,
                    bm25_weight=cfg.retrieval.bm25_weight,
                    vector_weight=cfg.retrieval.vector_weight,
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
            _cfg, root = api.load_wiki(wiki_path)
            abs_path = (root / doc_path).resolve()
            if not abs_path.is_file():
                raise ValueError(f"not a file: {doc_path}")
            text = Path(abs_path).read_text(encoding="utf-8")
            return [TextContent(type="text", text=text)]

        if name == "wiki.synthesize":
            wiki_path = arguments.get("path", ".")
            embedder = None
            if arguments.get("embed", True):
                cfg, _ = api.load_wiki(wiki_path)
                embedder = build_embedder(cfg.provider)
            synth_report = await api.synthesize(
                wiki_path,
                force_all=bool(arguments.get("all", False)),
                embedder=embedder,
            )
            return [
                TextContent(
                    type="text", text=json.dumps(synth_report.__dict__, indent=2)
                )
            ]

        if name == "wiki.list":
            wiki_path = arguments.get("path", ".")
            cfg, root = api.load_wiki(wiki_path)
            storage = build_storage(cfg.storage, root=root)
            await storage.connect()
            await storage.migrate()
            try:
                docs = list(await storage.list_documents(layer=Layer.WIKI, active=True))
            finally:
                await storage.close()
            list_payload: list[dict[str, Any]] = [
                {"path": d.path, "title": d.title, "mtime": d.mtime} for d in docs
            ]
            return [TextContent(type="text", text=json.dumps(list_payload, indent=2))]

        if name == "wiki.get":
            wiki_path = arguments.get("path", ".")
            _cfg, root = api.load_wiki(wiki_path)
            page = read_page(root, arguments["page_path"])
            page_payload: dict[str, Any] = {
                "path": page.path,
                "id": page.id,
                "type": page.type,
                "title": page.title,
                "tags": page.tags,
                "sources": page.sources,
                "created": page.created,
                "updated": page.updated,
                "body": page.body,
            }
            return [TextContent(type="text", text=json.dumps(page_payload, indent=2))]

        if name == "admin.lint":
            wiki_path = arguments.get("path", ".")
            lint_report = await api.lint(wiki_path)
            lint_payload: dict[str, Any] = {
                "ok": lint_report.ok,
                "summary": lint_report.by_kind(),
                "issues": [
                    {
                        "kind": i.kind,
                        "path": i.path,
                        "line": i.line,
                        "detail": i.detail,
                    }
                    for i in lint_report.issues
                ],
            }
            return [TextContent(type="text", text=json.dumps(lint_payload, indent=2))]

        if name == "wisdom.distill":
            wiki_path = arguments.get("path", ".")
            batch = int(arguments.get("batch", 8))
            distill_report = await api.distill(wiki_path, pages_per_call=batch)
            return [
                TextContent(
                    type="text", text=json.dumps(distill_report.__dict__, indent=2)
                )
            ]

        if name == "wisdom.list":
            wiki_path = arguments.get("path", ".")
            cfg, root = api.load_wiki(wiki_path)
            storage = build_storage(cfg.storage, root=root)
            await storage.connect()
            await storage.migrate()
            try:
                status_arg = arguments.get("status")
                status_filter = WisdomStatus(status_arg) if status_arg else None
                items = await storage.list_wisdom(status=status_filter)
            finally:
                await storage.close()
            wisdom_payload: list[dict[str, Any]] = [
                {
                    "item_id": i.item_id,
                    "kind": i.kind.value,
                    "status": i.status.value,
                    "title": i.title,
                    "confidence": i.confidence,
                    "approved_ts": i.approved_ts,
                }
                for i in items
            ]
            return [TextContent(type="text", text=json.dumps(wisdom_payload, indent=2))]

        if name == "wisdom.approve":
            wiki_path = arguments.get("path", ".")
            approve_result = await api.approve_wisdom(arguments["item_id"], wiki_path)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "item_id": approve_result.item_id,
                            "new_status": approve_result.new_status.value,
                        },
                        indent=2,
                    ),
                )
            ]

        if name == "wisdom.reject":
            wiki_path = arguments.get("path", ".")
            reject_result = await api.reject_wisdom(arguments["item_id"], wiki_path)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "item_id": reject_result.item_id,
                            "new_status": reject_result.new_status.value,
                        },
                        indent=2,
                    ),
                )
            ]

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
