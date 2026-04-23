"""Hybrid search: BM25 (FTS5) + vector (sqlite-vec / pgvector) fused via RRF.

Phase 1 keeps this deliberately small:

* Run ``storage.fts_search`` and ``storage.vec_search`` in parallel.
* Fuse the two ranked lists with Reciprocal Rank Fusion (``k=60``), the
  value used by both reference projects and the original RRF paper.
* Return the top ``limit`` ``Hit``s, each with a representative snippet
  sourced from the FTS side when present, or the chunk body otherwise.

RRF is appropriate here because it ignores raw scores — BM25 negative-log
units and cosine distances don't normalize cleanly against each other, but
rank orderings do.

Query-expansion + the strong-signal short-circuit from ``qmd`` are named
extension points for Phase 2 and beyond; starting with plain RRF lets us
land a measurable baseline first.
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from pydantic import BaseModel

from ..providers import EmbeddingProvider
from ..schemas import ChunkRecord, FTSHit, Layer, VecHit
from ..storage.base import NotSupported, Storage

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


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion. Returns doc_id → fused score (higher = better).

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
    scores: dict[str, float] = {}
    for lst, w in zip(rank_lists, weights, strict=True):
        for rank, key in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
    return scores


class HybridSearcher:
    """Composes FTS + vector search on top of a ``Storage`` backend."""

    def __init__(
        self,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        *,
        embedding_model: str | None = None,
        rrf_k: int = RRF_K,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight

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
        if mode == "vector" and (
            self._embedder is None or self._embedding_model is None
        ):
            raise ValueError(
                "mode='vector' requires both embedder and embedding_model"
            )

        fts_task: asyncio.Task[list[FTSHit]] | None = None
        vec_task: asyncio.Task[list[VecHit]] | None = None
        if run_fts:
            fts_task = asyncio.create_task(
                self._storage.fts_search(_sanitize_fts(q), limit=per_leg_limit, layer=layer)
            )
        if run_vec and self._embedder is not None and self._embedding_model is not None:
            vec_task = asyncio.create_task(self._embed_and_search(q, per_leg_limit, layer))

        fts_hits: list[FTSHit] = []
        if fts_task is not None:
            fts_hits = await fts_task
        vec_hits: list[VecHit] = []
        if vec_task is not None:
            try:
                vec_hits = await vec_task
            except NotSupported:
                vec_hits = []

        fts_ranked = [h.doc_id for h in fts_hits]
        vec_ranked = [h.doc_id for h in vec_hits]
        fused = reciprocal_rank_fusion(
            [fts_ranked, vec_ranked],
            k=self._rrf_k,
            weights=[self._bm25_weight, self._vector_weight],
        )

        # index hits for fast snippet lookup
        snippets: dict[str, str] = {h.doc_id: h.snippet or "" for h in fts_hits if h.snippet}
        chunk_lookup: dict[str, int] = {h.doc_id: h.chunk_id for h in vec_hits}

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        hits: list[Hit] = []
        for doc_id, score in top:
            snippet = snippets.get(doc_id) or await self._chunk_snippet(
                chunk_lookup.get(doc_id)
            )
            doc = await self._storage.get_document(doc_id)
            hits.append(
                Hit(
                    doc_id=doc_id,
                    chunk_id=chunk_lookup.get(doc_id),
                    score=score,
                    snippet=snippet,
                    path=doc.path if doc else None,
                    title=doc.title if doc else None,
                )
            )
        return hits

    async def _embed_and_search(
        self, q: str, limit: int, layer: Layer | None
    ) -> list[VecHit]:
        assert self._embedder is not None
        assert self._embedding_model is not None
        vectors = await self._embedder.embed([q], model=self._embedding_model)
        if not vectors:
            return []
        return await self._storage.vec_search(vectors[0], limit=limit, layer=layer)

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
