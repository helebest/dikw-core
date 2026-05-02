"""Synchronous (millisecond-level) HTTP RPC endpoints.

Each route maps 1:1 to an ``api.py`` method and returns the engine's
DTO directly as JSON. Long-running ops (ingest / synth / distill /
query) are NOT here — they live behind ``/v1/{op}`` in
``routes_tasks.py`` and stream NDJSON.

The ``init`` endpoint is intentionally *not* implemented in Phase 2 —
the server's wiki root is bound at boot, so scaffolding a fresh wiki
through HTTP only makes sense once the upload pipeline (Phase 3)
exists. ``POST /v1/init`` will return 501 until then.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel

from .. import __version__, api
from ..info.search import HybridSearcher
from ..knowledge.lint import LintReport
from ..providers import build_embedder
from ..schemas import (
    ChunkRecord,
    DocumentRecord,
    Hit,
    Layer,
    StorageCounts,
    WisdomItem,
    WisdomStatus,
)
from ..wisdom.review import ReviewError, ReviewResult
from .errors import BadRequest, Conflict, NotFoundError
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


# Request bodies live at module scope so FastAPI's TypeAdapter can resolve
# their forward references — class bodies nested inside ``make_router``
# don't fully define until first call, which races the schema build.


class CheckRequest(BaseModel):
    llm_only: bool = False
    embed_only: bool = False


class WikiPageResponse(BaseModel):
    path: str
    body: str


class DocSearchRequest(BaseModel):
    q: str
    limit: int = 5
    layer: Layer | None = None
    # ``"hybrid"`` mirrors the CLI default (BM25 + dense fused via RRF);
    # ``"bm25"`` is the FTS-only escape hatch when the wiki has no active
    # embedding version (e.g. fresh init, filesystem backend) — the engine
    # would otherwise need a working embedding endpoint just to get a
    # text-only search response.
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    # ---- meta ---------------------------------------------------------

    @router.get("/info")
    async def info(request: Request) -> dict[str, Any]:
        rt: ServerRuntime = get_runtime(request.app)
        return {
            "engine_version": __version__,
            "wiki_root": str(rt.root),
            "storage_backend": rt.cfg.storage.backend,
            "providers": {
                "llm": rt.cfg.provider.llm,
                "llm_model": rt.cfg.provider.llm_model,
                "embedding": rt.cfg.provider.embedding,
                "embedding_model": rt.cfg.provider.embedding_model,
            },
            "auth_required": rt.auth.required,
        }

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        rt: ServerRuntime = get_runtime(request.app)
        # Storage migration ran in lifespan; reaching this point implies
        # the engine is wired. We deliberately don't probe providers here
        # (that's what ``/v1/check`` is for) so a flapping LLM endpoint
        # doesn't make the server look unhealthy.
        return {"status": "ready", "wiki_root": str(rt.root)}

    # ---- engine fast paths --------------------------------------------

    @router.get("/status", response_model=StorageCounts)
    async def status(request: Request) -> StorageCounts:
        rt: ServerRuntime = get_runtime(request.app)
        return await api.status(rt.root)

    @router.post("/check", response_model=api.CheckReport)
    async def check(
        request: Request,
        body: CheckRequest = Body(default_factory=CheckRequest),
    ) -> api.CheckReport:
        rt: ServerRuntime = get_runtime(request.app)
        if body.llm_only and body.embed_only:
            raise BadRequest("llm_only and embed_only are mutually exclusive")
        return await api.check_providers(
            rt.root, llm_only=body.llm_only, embed_only=body.embed_only
        )

    @router.post("/lint", response_model=LintReport)
    async def lint(request: Request) -> LintReport:
        rt: ServerRuntime = get_runtime(request.app)
        return await api.lint(rt.root)

    # ---- wiki + doc ---------------------------------------------------

    @router.get("/wiki/pages", response_model=list[DocumentRecord])
    async def list_wiki_pages(
        request: Request,
        active: bool | None = Query(default=True),
        since_ts: float | None = Query(default=None),
    ) -> list[DocumentRecord]:
        rt: ServerRuntime = get_runtime(request.app)
        # Each route opens its own storage handle for stateless safety;
        # the server's main storage handle stays connected for migrations
        # / status, but route-level reads use a fresh handle to keep one
        # transaction-per-request semantics.
        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
            docs = await storage.list_documents(
                layer=Layer.WIKI, active=active, since_ts=since_ts
            )
            return list(docs)
        finally:
            await storage.close()

    @router.get("/wiki/pages/{page_path:path}", response_model=WikiPageResponse)
    async def get_wiki_page(
        request: Request, page_path: str
    ) -> WikiPageResponse:
        rt: ServerRuntime = get_runtime(request.app)
        # Confine to the wiki tree — reject path-traversal attempts even
        # though FastAPI's path-converter already blocks the easy cases.
        abs_path = (rt.root / page_path).resolve()
        try:
            abs_path.relative_to(rt.root.resolve())
        except ValueError as e:
            raise BadRequest(
                f"page_path escapes the wiki root: {page_path!r}"
            ) from e
        if not abs_path.is_file():
            raise NotFoundError(f"wiki page not found: {page_path!r}")
        body = abs_path.read_text(encoding="utf-8")
        return WikiPageResponse(path=page_path, body=body)

    @router.post("/doc/search", response_model=list[Hit])
    async def doc_search(
        request: Request,
        body: DocSearchRequest = Body(...),
    ) -> list[Hit]:
        rt: ServerRuntime = get_runtime(request.app)
        cfg, _root, storage = await api._with_storage(rt.root)
        try:
            # Build an embedder only when the dense leg is actually
            # going to run AND the storage has an active text version.
            # This lets ``mode="bm25"`` work on a wiki that has never
            # been ingested (e.g. fresh init), and on backends without
            # embedding support (filesystem) — instead of failing on the
            # embedder factory before search even runs.
            embedder = None
            if body.mode in ("hybrid", "vector"):
                try:
                    embedder = build_embedder(cfg.provider)
                except Exception as e:
                    if body.mode == "vector":
                        raise BadRequest(
                            f"vector mode requires a working embedder: {e}",
                            code="embedder_unavailable",
                        ) from e
                    # hybrid → silently fall through with embedder=None;
                    # HybridSearcher treats missing embedder as BM25-only.
            searcher = HybridSearcher.from_config(
                storage,
                embedder,
                cfg.retrieval,
                embedding_model=cfg.provider.embedding_model,
            )
            return await searcher.search(
                body.q,
                limit=body.limit,
                layer=body.layer,
                mode=body.mode,
            )
        finally:
            await storage.close()

    @router.get("/doc/chunks/{chunk_id}", response_model=ChunkRecord)
    async def get_chunk(request: Request, chunk_id: int) -> ChunkRecord:
        rt: ServerRuntime = get_runtime(request.app)
        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
            chunk = await storage.get_chunk(chunk_id)
        finally:
            await storage.close()
        if chunk is None:
            raise NotFoundError(f"chunk_id {chunk_id} not found")
        return chunk

    # ---- wisdom -------------------------------------------------------

    @router.get("/wisdom", response_model=list[WisdomItem])
    async def list_wisdom(
        request: Request,
        status: WisdomStatus | None = Query(default=None),
        kind: str | None = Query(default=None),
    ) -> list[WisdomItem]:
        rt: ServerRuntime = get_runtime(request.app)
        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
            from ..schemas import WisdomKind

            kind_enum = WisdomKind(kind) if kind else None
            return await storage.list_wisdom(status=status, kind=kind_enum)
        finally:
            await storage.close()

    @router.post(
        "/wisdom/{item_id}/approve", response_model=ReviewResult
    )
    async def approve(
        request: Request, item_id: str
    ) -> ReviewResult:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            return await api.approve_wisdom(item_id, rt.root)
        except ReviewError as e:
            raise Conflict(str(e), code="review_conflict") from e

    @router.post(
        "/wisdom/{item_id}/reject", response_model=ReviewResult
    )
    async def reject(
        request: Request, item_id: str
    ) -> ReviewResult:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            return await api.reject_wisdom(item_id, rt.root)
        except ReviewError as e:
            raise Conflict(str(e), code="review_conflict") from e

    return router


# Re-export Path so nothing higher up needs a sibling import.
_ = Path

__all__ = ["make_router"]
