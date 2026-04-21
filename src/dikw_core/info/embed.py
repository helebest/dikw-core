"""Embedding worker.

Given a list of persisted chunks (``chunk_id`` + text) and an
``EmbeddingProvider``, produce ``EmbeddingRow`` objects ready for
``Storage.upsert_embeddings``. Batching keeps HTTP round-trips low without
overwhelming providers' per-request input caps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..providers import EmbeddingProvider
from ..schemas import EmbeddingRow


@dataclass(frozen=True)
class ChunkToEmbed:
    chunk_id: int
    text: str


async def embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    batch_size: int = 64,
) -> list[EmbeddingRow]:
    """Embed ``chunks`` in fixed-size batches and return ``EmbeddingRow``s in order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows: list[EmbeddingRow] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        vectors = await provider.embed(texts, model=model)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding provider returned {len(vectors)} vectors for "
                f"{len(batch)} texts"
            )
        rows.extend(
            EmbeddingRow(chunk_id=c.chunk_id, model=model, embedding=v)
            for c, v in zip(batch, vectors, strict=True)
        )
    return rows
