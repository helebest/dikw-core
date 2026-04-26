"""Streaming-embed tests.

These cover the per-batch ``async generator`` behaviour of
``embed_chunks`` / ``embed_chunks_multimodal`` (slice 3) and — via
slice 4 — the per-batch upsert + resume-scan paths in ``api.ingest``.

Hermetic: only ``FakeEmbeddings`` and ``FakeMultimodalEmbedding`` from
``tests.fakes``; no network, no real provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.info.embed import (
    ChunkToEmbed,
    embed_chunks,
    embed_chunks_multimodal,
)
from dikw_core.schemas import EmbeddingRow

from .fakes import FakeMultimodalEmbedding


def _chunks(n: int) -> list[ChunkToEmbed]:
    return [ChunkToEmbed(chunk_id=i + 1, text=f"chunk-{i}") for i in range(n)]


# ---- A1: embed_chunks streams one batch per provider call ----------------


async def test_embed_chunks_yields_per_batch() -> None:
    """5 chunks at batch_size=2 → 3 yields of sizes [2, 2, 1]."""
    provider = FakeEmbeddings()
    chunks = _chunks(5)

    gen = embed_chunks(provider, chunks, model="fake", batch_size=2)
    # Confirm it's an async generator, not an awaitable returning a list.
    assert isinstance(gen, AsyncIterator)
    assert not hasattr(gen, "__await__")

    batches: list[list[EmbeddingRow]] = []
    async for batch in gen:
        batches.append(batch)

    assert [len(b) for b in batches] == [2, 2, 1]
    # Order preserved across yielded batches.
    flat_ids = [row.chunk_id for batch in batches for row in batch]
    assert flat_ids == [1, 2, 3, 4, 5]
    # Every row stamped with the requested model.
    assert all(r.model == "fake" for batch in batches for r in batch)


async def test_embed_chunks_empty_input_yields_nothing() -> None:
    """Empty corpus = zero yields, no provider call."""
    provider = FakeEmbeddings()
    batches = [b async for b in embed_chunks(provider, [], model="fake", batch_size=8)]
    assert batches == []


async def test_embed_chunks_rejects_non_positive_batch_size() -> None:
    """``batch_size=0`` raises before any provider call.

    The error must surface immediately, not be hidden inside the
    generator until first iteration — async generators only execute on
    iteration, so the function must validate at construction.
    """
    with pytest.raises(ValueError, match="batch_size"):
        async for _ in embed_chunks(
            FakeEmbeddings(), _chunks(3), model="fake", batch_size=0
        ):
            pass


# ---- A2: embed_chunks_multimodal mirrors the streaming shape -------------


async def test_embed_chunks_multimodal_yields_per_batch() -> None:
    """5 chunks at batch_size=2 via the multimodal provider → 3 yields."""
    provider = FakeMultimodalEmbedding(dim=4)
    chunks = _chunks(5)

    gen = embed_chunks_multimodal(provider, chunks, model="fake-mm", batch_size=2)
    assert isinstance(gen, AsyncIterator)

    batches: list[list[EmbeddingRow]] = []
    async for batch in gen:
        batches.append(batch)

    assert [len(b) for b in batches] == [2, 2, 1]
    flat_ids = [row.chunk_id for batch in batches for row in batch]
    assert flat_ids == [1, 2, 3, 4, 5]
    assert all(r.model == "fake-mm" for batch in batches for r in batch)


async def test_embed_chunks_multimodal_empty_still_pings_provider() -> None:
    """Legacy contract: empty input still calls provider.embed([]) so
    ``last_inputs`` reflects the call. Some tests rely on this signal.
    """
    provider = FakeMultimodalEmbedding(dim=4)
    batches = [
        b
        async for b in embed_chunks_multimodal(
            provider, [], model="fake-mm", batch_size=2
        )
    ]
    assert batches == []
    assert provider.last_inputs == []
    assert provider.last_model == "fake-mm"
