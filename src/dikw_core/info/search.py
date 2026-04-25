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
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, Protocol

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
from ..storage.base import NotSupported, Storage, StorageError
from .tokenize import CJK_CHAR_CLASS, CjkTokenizer, preprocess_for_fts


class RetrievalConfigLike(Protocol):
    """Structural shape of ``config.RetrievalConfig`` for ``from_config``."""

    rrf_k: int
    bm25_weight: float
    vector_weight: float
    cjk_tokenizer: CjkTokenizer
    same_doc_penalty_alpha: float


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
    """One fused search result.

    ``chunk_id`` is non-optional: chunk-level fusion produces one hit per
    chunk, so every result is anchored to a concrete chunk. ``seq`` is
    the chunk's ordinal within its document — useful for disambiguating
    multiple hits from the same document in CLI / MCP output.
    """

    doc_id: str
    chunk_id: int
    seq: int | None = None
    score: float
    snippet: str | None = None
    path: str | None = None
    title: str | None = None
    asset_refs: list[AssetRecord] = Field(default_factory=list)


def apply_source_diversity_penalty(
    fused: dict[int, float],
    doc_id_by_chunk: dict[int, str],
    *,
    alpha: float,
) -> dict[int, float]:
    """Diminishing-returns demotion of repeat same-doc chunks.

    Walks ``fused`` in score-desc order. The 1st chunk seen from each
    ``doc_id`` is unpenalized (factor ``1.0``); the N-th chunk (N ≥ 2)
    from the same doc is scaled by ``1 / (1 + alpha * (N - 1))``.
    With ``alpha=0.3``: 1st = 1.0, 2nd ≈ 0.77, 3rd = 0.625, 4th ≈ 0.526.
    With ``alpha=0`` the function is the identity (no-op).

    Returns a new dict with the same key set; the caller re-sorts and
    slices top-K. Pure: no I/O, no side effects, no globals.
    """
    if alpha == 0.0:
        return dict(fused)
    per_doc_seen: dict[str, int] = {}
    out: dict[int, float] = {}
    for chunk_id, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        doc_id = doc_id_by_chunk.get(chunk_id)
        if doc_id is None:
            out[chunk_id] = score
            continue
        n_seen = per_doc_seen.get(doc_id, 0)
        out[chunk_id] = score / (1.0 + alpha * n_seen)
        per_doc_seen[doc_id] = n_seen + 1
    return out


def reciprocal_rank_fusion[K: Hashable](
    rank_lists: list[list[K]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[K, float]:
    """Reciprocal Rank Fusion. Returns key → fused score (higher = better).

    ``K`` is generic over any hashable identity — historically ``doc_id:
    str``, now also ``chunk_id: int`` once chunk-level fusion lands.

    ``weights`` lets the caller bias fusion toward a stronger leg — e.g.,
    when BM25 is measurably behind the dense leg on a given corpus,
    equal-weight RRF drags the combined ranking toward the weaker signal
    (observed on BEIR/SciFact: hybrid nDCG@10 0.736 < vector 0.773 at
    default k=60, equal weights). Setting ``weights=[0.5, 1.0]`` halves
    BM25's per-rank contribution while keeping every doc it found in the
    pool — better rank quality, no recall loss.

    ``None`` (the default) is equivalent to ``[1.0] * len(rank_lists)``
    — the behaviour before weighting landed, preserved bit-for-bit.
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError(
            f"weights length {len(weights)} must match rank_lists length "
            f"{len(rank_lists)}"
        )
    scores: dict[K, float] = {}
    for lst, w in zip(rank_lists, weights, strict=True):
        for rank, key in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
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
        rrf_k: int = RRF_K,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        cjk_tokenizer: CjkTokenizer = "none",
        same_doc_penalty_alpha: float = 0.3,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._mm = multimodal
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        # Must match the storage adapter's ingest-time tokenizer; a
        # mismatch silently drops CJK hits.
        self._cjk_tokenizer: CjkTokenizer = cjk_tokenizer
        self._same_doc_penalty_alpha = same_doc_penalty_alpha

    @classmethod
    def from_config(
        cls,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        cfg: RetrievalConfigLike,
        *,
        embedding_model: str | None = None,
        multimodal: MultimodalSearch | None = None,
    ) -> HybridSearcher:
        """Unpack a ``RetrievalConfig`` into the keyword kwargs.

        Centralises the knob mapping so adding a new knob is a
        one-file change. ``RetrievalConfigLike`` is any object with
        the listed attributes — pydantic ``RetrievalConfig`` qualifies.
        """
        return cls(
            storage,
            embedder,
            embedding_model=embedding_model,
            multimodal=multimodal,
            rrf_k=cfg.rrf_k,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
            cjk_tokenizer=cfg.cjk_tokenizer,
            same_doc_penalty_alpha=cfg.same_doc_penalty_alpha,
        )

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
                self._storage.fts_search(
                    _sanitize_fts(q, cjk_tokenizer=self._cjk_tokenizer),
                    limit=per_leg_limit,
                    layer=layer,
                )
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
            except StorageError as e:
                # Mid-migration scenario: a multimodal-mode query against
                # a legacy chunks_vec built with a different text-embed
                # dim. Degrade to BM25 + asset-vec leg rather than
                # killing the whole query.
                if "dim" not in str(e):
                    raise
                vec_hits = []
        asset_hits: list[AssetVecHit] = []
        if asset_task is not None:
            try:
                asset_hits = await asset_task
            except NotSupported:
                asset_hits = []

        # Asset hits promote the chunks that reference them. Each promoted
        # chunk enters the fusion pool directly — chunk-level fusion means
        # multiple chunks from the same asset's parent doc all compete on
        # their own merit.
        asset_chunk_ranked: list[int] = []
        asset_chunk_doc_ids: dict[int, str] = {}
        if asset_hits:
            asset_ids_ordered = [h.asset_id for h in asset_hits]
            chunks_by_asset = await self._storage.chunks_referencing_assets(
                asset_ids_ordered
            )
            promoted_chunk_ids = list(
                {cid for cids in chunks_by_asset.values() for cid in cids}
            )
            chunk_by_id = {
                c.chunk_id: c
                for c in await self._storage.get_chunks(promoted_chunk_ids)
                if c.chunk_id is not None
            }
            seen_chunks: set[int] = set()
            for asset_id in asset_ids_ordered:
                for cid in chunks_by_asset.get(asset_id, []):
                    chunk = chunk_by_id.get(cid)
                    if chunk is None or cid in seen_chunks:
                        continue
                    asset_chunk_ranked.append(cid)
                    asset_chunk_doc_ids[cid] = chunk.doc_id
                    seen_chunks.add(cid)

        # Build chunk-level rank lists. FTSHit.chunk_id is `int | None` in
        # the schema (legacy compat); every shipped adapter populates it,
        # but defensively skip any None to keep RRF keys homogeneous.
        fts_ranked = [h.chunk_id for h in fts_hits if h.chunk_id is not None]
        vec_ranked = [h.chunk_id for h in vec_hits]
        # Asset channel rides the vector weight — same family of signal
        # (semantic similarity in the multimodal space), distinct only in
        # what's embedded (chunk text vs asset bytes).
        fused = reciprocal_rank_fusion(
            [fts_ranked, vec_ranked, asset_chunk_ranked],
            k=self._rrf_k,
            weights=[
                self._bm25_weight,
                self._vector_weight,
                self._vector_weight,
            ],
        )

        # Per-chunk doc_id lookup, sourced from every leg that knows it.
        # Vec/asset legs always carry doc_id; FTS hits do too. The first
        # writer wins because every leg agrees on chunk_id -> doc_id.
        doc_id_by_chunk: dict[int, str] = {}
        for vh in vec_hits:
            doc_id_by_chunk.setdefault(vh.chunk_id, vh.doc_id)
        for fh in fts_hits:
            if fh.chunk_id is not None:
                doc_id_by_chunk.setdefault(fh.chunk_id, fh.doc_id)
        for cid, did in asset_chunk_doc_ids.items():
            doc_id_by_chunk.setdefault(cid, did)

        # Stage 3 source-diversity demotion. alpha=0 is a no-op; alpha>0
        # demotes later same-doc chunks via diminishing returns.
        adjusted = apply_source_diversity_penalty(
            fused, doc_id_by_chunk, alpha=self._same_doc_penalty_alpha
        )
        top = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        # Per-chunk snippet lookup from FTS hits (BM25's snippet() preview).
        snippets_by_chunk: dict[int, str] = {
            h.chunk_id: h.snippet or ""
            for h in fts_hits
            if h.chunk_id is not None and h.snippet
        }

        retrieved_chunk_ids = [cid for cid, _ in top]

        # Backends that don't implement the asset bridge raise NotSupported —
        # degrade to empty refs so the legacy text-query path keeps working.
        try:
            refs_by_chunk = await self._storage.chunk_asset_refs_for_chunks(
                retrieved_chunk_ids
            )
        except NotSupported:
            refs_by_chunk = {cid: [] for cid in retrieved_chunk_ids}
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

        # Batch-fetch the chunks (for snippet fallback + seq) and unique
        # parent docs (for path/title) — chunk-level fusion repeats
        # doc_ids across hits, so per-hit fetches would N+1 the storage.
        chunk_by_id_all: dict[int, ChunkRecord] = {
            c.chunk_id: c
            for c in await self._storage.get_chunks(retrieved_chunk_ids)
            if c.chunk_id is not None
        }
        unique_doc_ids = list({
            chunk_by_id_all[cid].doc_id
            for cid in retrieved_chunk_ids
            if cid in chunk_by_id_all
        })
        doc_by_id = {
            d.doc_id: d
            for d in await self._storage.get_documents(unique_doc_ids)
        }

        hits: list[Hit] = []
        for chunk_id, score in top:
            chunk = chunk_by_id_all.get(chunk_id)
            if chunk is None:
                # Race: chunk dropped between fusion and materialization.
                # Skip rather than emit a half-formed Hit (TODOS T4 covers
                # the principled "loud failure" path).
                continue
            doc = doc_by_id.get(chunk.doc_id)
            snippet = snippets_by_chunk.get(chunk_id)
            if not snippet:
                snippet = self._render_chunk_snippet(chunk)
            asset_records: list[AssetRecord] = []
            for r in refs_by_chunk.get(chunk_id, []):
                a = assets_by_id.get(r.asset_id)
                if a is not None:
                    asset_records.append(a)
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk_id,
                    seq=chunk.seq,
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

    @staticmethod
    def _render_chunk_snippet(chunk: ChunkRecord) -> str:
        """Compact one-line preview of a chunk's body for Hit.snippet."""
        snippet = chunk.text.strip().replace("\n", " ")
        return snippet[:240] + ("…" if len(snippet) > 240 else "")


def _sanitize_fts(q: str, *, cjk_tokenizer: CjkTokenizer = "none") -> str:
    """Tokenize a natural-language query into a bag-of-words FTS5 expression.

    The Phase 1 implementation wrapped the whole query in quotes — a
    phrase query — which never matched on multi-word natural-language
    inputs and made BM25-only retrieval (and thus the FTS leg of hybrid
    RRF) return 0 hits in eval.

    Strategy:

    0. If ``cjk_tokenizer != "none"``, pre-segment CJK runs (via the
       same ``preprocess_for_fts`` the ingest path uses) so that the
       whitespace-split below picks up word-level Chinese tokens.
       Symmetry with the indexed form is the whole point; see the
       ``SQLiteStorage.__init__`` note.
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
    """
    if cjk_tokenizer != "none":
        q = preprocess_for_fts(q, tokenizer=cjk_tokenizer)
    cleaned = re.sub(rf"[^\w\s{CJK_CHAR_CLASS}]", " ", q)
    tokens = [
        t for t in cleaned.split() if t and t.upper() not in _FTS_RESERVED
    ]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
