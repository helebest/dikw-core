"""Synchronous (millisecond-level) HTTP RPC endpoints.

Each route maps 1:1 to an ``api.py`` method and returns the engine's
DTO directly as JSON. Long-running ops (ingest / synth / distill /
query) are NOT here — they live behind ``/v1/{op}`` in
``routes_tasks.py`` and stream NDJSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel

from .. import __version__, api
from ..domains.info.search import HybridSearcher
from ..domains.knowledge.lint import LintReport
from ..domains.wisdom.review import ReviewError, ReviewResult
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
from .errors import BadRequest, Conflict, NotFoundError
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


# Request bodies live at module scope so FastAPI's TypeAdapter can resolve
# their forward references — class bodies nested inside ``make_router``
# don't fully define until first call, which races the schema build.


class CheckRequest(BaseModel):
    llm_only: bool = False
    embed_only: bool = False


class InitRequest(BaseModel):
    description: str | None = None


class InitResponse(BaseModel):
    root: str


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

    @router.post("/init", response_model=InitResponse)
    async def init_wiki_endpoint(
        request: Request,
        body: InitRequest = Body(default_factory=InitRequest),
    ) -> InitResponse:
        # ``DIKW_SERVER_DISABLE_INIT=1`` is the production lockdown — once
        # the server is bound to a real wiki, accepting a request that
        # rewrites scaffold files would only ever be a misconfiguration.
        if os.environ.get("DIKW_SERVER_DISABLE_INIT") == "1":
            raise Conflict(
                "init is disabled on this server (DIKW_SERVER_DISABLE_INIT=1)",
                code="init_disabled",
            )
        rt: ServerRuntime = get_runtime(request.app)
        try:
            root = api.init_wiki(rt.root, description=body.description)
        except FileExistsError as e:
            # Server-bound wiki was already scaffolded — re-init would
            # blow away whatever's on disk. Return 409 so the client can
            # branch (e.g., "wiki already exists, skip bootstrap").
            raise Conflict(
                f"wiki at {rt.root} is already initialised",
                code="wiki_already_initialised",
            ) from e
        return InitResponse(root=str(root))

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
        # Scope to the ``wiki/`` subtree (not the wiki root). Resolving
        # against ``rt.root`` would let an authenticated caller fetch
        # ``dikw.yml``, ``sources/...``, ``wisdom/...``, etc. through
        # this endpoint — page_path is meant to address K-layer pages
        # only. ``page_path`` may be ``"wiki/notes/foo.md"`` (matching
        # ``DocumentRecord.path``) or ``"notes/foo.md"`` (raw page-path
        # under ``wiki/``); we accept both by stripping a leading
        # ``"wiki/"`` before resolving.
        wiki_dir = (rt.root / "wiki").resolve()
        rel = page_path[5:] if page_path.startswith("wiki/") else page_path
        abs_path = (wiki_dir / rel).resolve()
        try:
            abs_path.relative_to(wiki_dir)
        except ValueError as e:
            raise BadRequest(
                f"page_path escapes the wiki/ subtree: {page_path!r}"
            ) from e
        if abs_path.suffix != ".md":
            raise BadRequest(
                f"only .md pages are addressable here: {page_path!r}"
            )
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
            # Pin the query embedder to the active text embed_version
            # (model + dim) — same anti-drift guard ``api.query`` uses.
            # If the operator just edited ``embedding_model`` /
            # ``embedding_dim`` in dikw.yml without re-ingesting, the
            # vec table still holds vectors from the OLD space; querying
            # with the new model would either dim-mismatch or rank
            # against an incompatible space.
            from ..storage.base import NotSupported

            text_version_id: int | None = None
            text_query_model = cfg.provider.embedding_model
            text_query_dim: int | None = None
            try:
                active_text = await storage.get_active_embed_version(modality="text")
            except NotSupported:
                active_text = None
            if active_text is not None and active_text.version_id is not None:
                text_version_id = active_text.version_id
                text_query_model = active_text.model
                text_query_dim = active_text.dim

            # Build an embedder only when the dense leg is actually
            # going to run AND the storage has an active text version.
            # This lets ``mode="bm25"`` work on a wiki that has never
            # been ingested (e.g. fresh init), and on backends without
            # embedding support (filesystem) — instead of failing on the
            # embedder factory before search even runs.
            embedder = None
            if body.mode in ("hybrid", "vector"):
                try:
                    embedder = build_embedder(
                        cfg.provider, dim_override=text_query_dim
                    )
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
                embedding_model=text_query_model,
                text_version_id=text_version_id,
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
        from ..schemas import WisdomKind

        # Validate ``kind`` BEFORE opening storage so a typo'd query
        # param surfaces as 4xx rather than crashing through the
        # exception handler as a 500.
        kind_enum: WisdomKind | None = None
        if kind:
            try:
                kind_enum = WisdomKind(kind)
            except ValueError as e:
                valid = ", ".join(k.value for k in WisdomKind)
                raise BadRequest(
                    f"unknown wisdom kind {kind!r} (valid: {valid})",
                    code="invalid_wisdom_kind",
                ) from e

        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
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
