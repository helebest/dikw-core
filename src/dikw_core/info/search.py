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

from pydantic import BaseModel

from ..providers import EmbeddingProvider
from ..schemas import ChunkRecord, Layer, VecHit
from ..storage.base import NotSupported, Storage

RRF_K = 60


class Hit(BaseModel):
    """One fused search result."""

    doc_id: str
    chunk_id: int | None = None
    score: float
    snippet: str | None = None
    path: str | None = None
    title: str | None = None


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
    """Composes FTS + vector search on top of a ``Storage`` backend."""

    def __init__(
        self,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        *,
        embedding_model: str | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._embedding_model = embedding_model

    async def search(
        self,
        q: str,
        *,
        limit: int = 10,
        per_leg_limit: int = 40,
        layer: Layer | None = None,
    ) -> list[Hit]:
        if not q.strip():
            return []

        fts_task = asyncio.create_task(
            self._storage.fts_search(_sanitize_fts(q), limit=per_leg_limit, layer=layer)
        )
        vec_task: asyncio.Task[list[VecHit]] | None = None
        if self._embedder is not None and self._embedding_model is not None:
            vec_task = asyncio.create_task(self._embed_and_search(q, per_leg_limit, layer))

        fts_hits = await fts_task
        vec_hits: list[VecHit] = []
        if vec_task is not None:
            try:
                vec_hits = await vec_task
            except NotSupported:
                vec_hits = []

        fts_ranked = [h.doc_id for h in fts_hits]
        vec_ranked = [h.doc_id for h in vec_hits]
        fused = reciprocal_rank_fusion([fts_ranked, vec_ranked])

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
    """FTS5 MATCH syntax treats certain punctuation specially.

    Quote the whole query as a phrase to avoid raw-user-input syntax errors
    (e.g. a trailing hyphen). Phase 2's query expansion can build structured
    MATCH expressions explicitly.
    """
    escaped = q.replace('"', '""').strip()
    return f'"{escaped}"' if escaped else ""
