"""Hybrid search: BM25 (FTS5) + vector(s) fused via Reciprocal Rank Fusion.

v1 has two operating modes:

* **Legacy 2-leg** (text embedder, no asset index) — BM25 over chunk
  text + vector search over chunk vectors, fused at chunk-level via RRF.
  Behavior identical to the original implementation.

* **Multimodal 3-leg** (multimodal embedder + asset version) — adds a
  third channel that runs ``vec_search_assets`` against the per-version
  asset vector table; matched assets promote their parent chunks (via
  the ``chunk_asset_refs`` reverse lookup) into the same RRF pool.
  Each returned Hit carries the assets that the chunk references so
  downstream consumers (CLI display, MCP schema, LLM synthesis) can
  render or cite them.

RRF is the right fusion choice — BM25 negative-log scores, cosine
distances on text vectors, and cosine distances on image vectors don't
normalize cleanly against each other, but their rank orderings do.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from ..providers import EmbeddingProvider, MultimodalEmbeddingProvider
from ..schemas import (
    AssetRecord,
    AssetVecHit,
    ChunkRecord,
    FTSHit,
    Layer,
    MultimodalInput,
    VecHit,
)
from ..storage.base import NotSupported, Storage


@dataclass(frozen=True)
class MultimodalSearch:
    """Wires the asset-vector retrieval channel into ``HybridSearcher``.

    All three fields are required to activate the channel — the embedder
    embeds the query into the multimodal vector space, the model is the
    name passed to the embedder, and the version_id selects which
    ``vec_assets_v<id>`` table to search.
    """

    embedder: MultimodalEmbeddingProvider
    model: str
    asset_version_id: int

RRF_K = 60

# FTS5 reserved query operators. Stripped from user queries because
# `_sanitize_fts` builds an OR-of-tokens expression itself; an unwary
# user word like "AND" would otherwise become syntax mid-query.
_FTS_RESERVED = frozenset({"AND", "OR", "NOT", "NEAR"})

# Which retrieval legs to fuse. ``hybrid`` is the historical default and
# what `dikw query` uses; ``bm25`` and ``vector`` exist so eval can
# ablate the contribution of each leg against public benchmarks.
RetrievalMode = Literal["bm25", "vector", "hybrid"]


class Hit(BaseModel):
    """One fused search result."""

    doc_id: str
    chunk_id: int | None = None
    score: float
    snippet: str | None = None
    path: str | None = None
    title: str | None = None
    asset_refs: list[AssetRecord] = Field(default_factory=list)


def reciprocal_rank_fusion(
    rank_lists: list[list[str]], *, k: int = RRF_K
) -> dict[str, float]:
    """Reciprocal Rank Fusion. Returns doc_id → fused score (higher = better)."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, key in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridSearcher:
    """Composes FTS + vector search(es) on top of a ``Storage`` backend.

    Pass a ``MultimodalSearch`` to activate the asset-vector retrieval
    leg; otherwise the searcher runs the FTS + (optional) text-vector
    legs only.
    """

    def __init__(
        self,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        *,
        embedding_model: str | None = None,
        multimodal: MultimodalSearch | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._mm = multimodal

    async def search(
        self,
        q: str,
        *,
        limit: int = 10,
        per_leg_limit: int = 40,
        layer: Layer | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[Hit]:
        if not q.strip():
            return []

        run_fts = mode in ("bm25", "hybrid")
        run_vec = mode in ("vector", "hybrid")
        text_vec_active = (
            self._embedder is not None and self._embedding_model is not None
        )
        if mode == "vector" and not (self._mm is not None or text_vec_active):
            raise ValueError(
                "mode='vector' requires either a MultimodalSearch or an "
                "(embedder, embedding_model) pair"
            )

        # Multimodal takes precedence so installations with both configured
        # drive all retrieval through the unified vector space.
        q_vec: list[float] | None = None
        if run_vec:
            if self._mm is not None:
                q_vec = await self._embed_query_multimodal(q)
            elif text_vec_active:
                q_vec = await self._embed_query_legacy(q)

        fts_task: asyncio.Task[list[FTSHit]] | None = None
        vec_task: asyncio.Task[list[VecHit]] | None = None
        asset_task: asyncio.Task[list[AssetVecHit]] | None = None
        if run_fts:
            fts_task = asyncio.create_task(
                self._storage.fts_search(_sanitize_fts(q), limit=per_leg_limit, layer=layer)
            )
        if q_vec is not None:
            vec_task = asyncio.create_task(
                self._storage.vec_search(q_vec, limit=per_leg_limit, layer=layer)
            )
            if self._mm is not None:
                asset_task = asyncio.create_task(
                    self._storage.vec_search_assets(
                        q_vec,
                        version_id=self._mm.asset_version_id,
                        limit=per_leg_limit,
                        layer=layer,
                    )
                )

        fts_hits: list[FTSHit] = await fts_task if fts_task is not None else []
        vec_hits: list[VecHit] = []
        if vec_task is not None:
            try:
                vec_hits = await vec_task
            except NotSupported:
                vec_hits = []
        asset_hits: list[AssetVecHit] = []
        if asset_task is not None:
            try:
                asset_hits = await asset_task
            except NotSupported:
                asset_hits = []

        # Asset hits promote the chunks that reference them. Rank a chunk
        # by its asset's rank in the asset-vec results.
        asset_doc_ranked: list[str] = []
        asset_chunk_lookup: dict[str, int] = {}  # doc_id → first promoted chunk_id
        if asset_hits:
            asset_ids_ordered = [h.asset_id for h in asset_hits]
            chunks_by_asset = await self._storage.chunks_referencing_assets(
                asset_ids_ordered
            )
            # Pre-fetch all promoted chunks in parallel — each asset hit
            # would otherwise trigger a serial get_chunk on the query path.
            promoted_chunk_ids = list(
                {cid for cids in chunks_by_asset.values() for cid in cids}
            )
            chunks = await asyncio.gather(
                *(self._storage.get_chunk(cid) for cid in promoted_chunk_ids)
            )
            chunk_by_id = {
                cid: c for cid, c in zip(promoted_chunk_ids, chunks, strict=True)
                if c is not None
            }
            seen_docs: set[str] = set()
            for asset_id in asset_ids_ordered:
                for cid in chunks_by_asset.get(asset_id, []):
                    chunk = chunk_by_id.get(cid)
                    if chunk is None or chunk.doc_id in seen_docs:
                        continue
                    asset_doc_ranked.append(chunk.doc_id)
                    asset_chunk_lookup[chunk.doc_id] = cid
                    seen_docs.add(chunk.doc_id)

        fts_ranked = [h.doc_id for h in fts_hits]
        vec_ranked = [h.doc_id for h in vec_hits]
        fused = reciprocal_rank_fusion(
            [fts_ranked, vec_ranked, asset_doc_ranked]
        )

        snippets: dict[str, str] = {
            h.doc_id: h.snippet or "" for h in fts_hits if h.snippet
        }
        chunk_lookup: dict[str, int] = {h.doc_id: h.chunk_id for h in vec_hits}
        # FTS hits expose their chunk_id (since the SQLite adapter aligns
        # documents_fts.rowid with chunks.chunk_id); use them as a fallback
        # so asset_refs can still be attached to FTS-only retrieved docs.
        for h in fts_hits:
            if h.chunk_id is not None:
                chunk_lookup.setdefault(h.doc_id, h.chunk_id)
        # Asset-promoted chunks fill in chunk_ids for docs neither vec nor
        # FTS surfaced.
        for doc_id, cid in asset_chunk_lookup.items():
            chunk_lookup.setdefault(doc_id, cid)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        # Batch-fetch chunk → asset_refs for every retrieved chunk so the
        # final Hit carries its referenced assets. Backends that don't
        # implement the asset bridge (filesystem / postgres) raise
        # NotSupported here — degrade to empty refs so the legacy
        # text-query path keeps working.
        retrieved_chunk_ids = [
            chunk_lookup[doc_id] for doc_id, _ in top if doc_id in chunk_lookup
        ]
        try:
            refs_by_chunk = await self._storage.chunk_asset_refs_for_chunks(
                retrieved_chunk_ids
            )
        except NotSupported:
            refs_by_chunk = {cid: [] for cid in retrieved_chunk_ids}
        # Materialize each unique asset_id once, in parallel — sequential
        # awaits would serialize one network/disk round-trip per asset.
        all_asset_ids = list(
            {r.asset_id for refs in refs_by_chunk.values() for r in refs}
        )
        try:
            fetched = await asyncio.gather(
                *(self._storage.get_asset(aid) for aid in all_asset_ids)
            )
        except NotSupported:
            fetched = []
            all_asset_ids = []
        assets_by_id: dict[str, AssetRecord] = {
            aid: a
            for aid, a in zip(all_asset_ids, fetched, strict=True)
            if a is not None
        }

        hits: list[Hit] = []
        for doc_id, score in top:
            hit_chunk_id: int | None = chunk_lookup.get(doc_id)
            snippet = snippets.get(doc_id) or await self._chunk_snippet(hit_chunk_id)
            doc = await self._storage.get_document(doc_id)
            asset_records: list[AssetRecord] = []
            if hit_chunk_id is not None:
                for r in refs_by_chunk.get(hit_chunk_id, []):
                    a = assets_by_id.get(r.asset_id)
                    if a is not None:
                        asset_records.append(a)
            hits.append(
                Hit(
                    doc_id=doc_id,
                    chunk_id=hit_chunk_id,
                    score=score,
                    snippet=snippet,
                    path=doc.path if doc else None,
                    title=doc.title if doc else None,
                    asset_refs=asset_records,
                )
            )
        return hits

    async def _embed_query_legacy(self, q: str) -> list[float] | None:
        assert self._embedder is not None
        assert self._embedding_model is not None
        vectors = await self._embedder.embed([q], model=self._embedding_model)
        return vectors[0] if vectors else None

    async def _embed_query_multimodal(self, q: str) -> list[float] | None:
        assert self._mm is not None
        vectors = await self._mm.embedder.embed(
            [MultimodalInput(text=q)], model=self._mm.model
        )
        return vectors[0] if vectors else None

    async def _chunk_snippet(self, chunk_id: int | None) -> str | None:
        if chunk_id is None:
            return None
        chunk: ChunkRecord | None = await self._storage.get_chunk(chunk_id)
        if chunk is None:
            return None
        # Return a compact preview rather than the whole chunk body.
        snippet = chunk.text.strip().replace("\n", " ")
        return snippet[:240] + ("…" if len(snippet) > 240 else "")


def _sanitize_fts(q: str) -> str:
    """Tokenize a natural-language query into a bag-of-words FTS5 expression.

    The Phase 1 implementation wrapped the whole query in quotes — a
    phrase query — which never matched on multi-word natural-language
    inputs and made BM25-only retrieval (and thus the FTS leg of hybrid
    RRF) return 0 hits in eval.

    Strategy:

    1. Replace anything that isn't a word character, whitespace, or a
       basic CJK ideograph with whitespace. Word characters (``\\w``)
       cover ASCII letters, digits, and underscore, so identifiers like
       ``expect_any`` survive intact; CJK pass-through keeps Chinese
       (e.g. CMTEB) queries from being stripped to nothing.
    2. Split on whitespace and drop FTS5 reserved tokens
       (``AND``/``OR``/``NOT``/``NEAR``) so a user word doesn't accidentally
       turn into an operator.
    3. Quote each token (``"<token>"``) — FTS5 phrase quotes around a
       single term are a no-op semantically but prevent column-qualifier
       interpretation for tokens that happen to contain a colon.
    4. Join with ``OR`` for bag-of-words BM25 retrieval — the same
       semantics published BEIR / CMTEB BM25 baselines use.

    Real CJK tokenization (jieba / character n-grams) is outside the
    Phase A retrieval-only scope; CJK queries with whitespace separators
    work, dense Chinese paragraphs as single tokens won't.
    """
    cleaned = re.sub(r"[^\w\s一-鿿]", " ", q)
    tokens = [
        t for t in cleaned.split() if t and t.upper() not in _FTS_RESERVED
    ]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
