"""Streaming-embed tests.

These cover the per-batch ``async generator`` behaviour of
``embed_chunks`` / ``embed_chunks_multimodal`` (slice 3) and — via
slice 4 — the per-batch upsert + resume-scan paths in ``api.ingest``.

Hermetic: only ``FakeEmbeddings`` and ``FakeMultimodalEmbedding`` from
``tests.fakes``; no network, no real provider.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core import api
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


def _setup_multibatch_wiki(tmp_path: Path, num_docs: int = 6) -> Path:
    """Build a wiki with N small docs and embedding_batch_size=2.

    Yields multi-batch ingests for the streaming + resume-scan tests
    (default batch_size=64 only ever produces one batch on this corpus).
    """
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki, description="streaming-test wiki")
    sources = wiki / "sources" / "docs"
    sources.mkdir(parents=True, exist_ok=True)
    for i in range(num_docs):
        (sources / f"doc{i:02d}.md").write_text(
            f"# Doc {i}\n\nDeterministic body content for doc {i}.\n",
            encoding="utf-8",
        )
    cfg_path = wiki / "dikw.yml"
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(
        text.replace(
            "embedding_batch_size: 64",
            "embedding_batch_size: 2",
        ),
        encoding="utf-8",
    )
    return wiki


def _count_embed_meta_rows(wiki: Path) -> int:
    db = wiki / ".dikw" / "index.sqlite"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM embed_meta").fetchone()[0]
    finally:
        conn.close()


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


# ---- A3: api.ingest streams per-batch upserts ----------------------------


class _CountingEmbedder:
    """Wraps FakeEmbeddings; counts embed() calls (= batches).

    A simpler counting wrapper than the perf-suite version (slice 8);
    inlined here so streaming tests don't depend on the perf module.
    """

    def __init__(self, fail_after: int | None = None) -> None:
        self._inner = FakeEmbeddings()
        self.embed_calls = 0
        self.total_texts = 0
        self.fail_after = fail_after  # raise on (fail_after+1)-th call

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.embed_calls += 1
        self.total_texts += len(texts)
        if self.fail_after is not None and self.embed_calls > self.fail_after:
            raise RuntimeError(
                f"_CountingEmbedder simulated failure on call {self.embed_calls}"
            )
        return await self._inner.embed(texts, model=model)


async def test_api_ingest_streams_per_batch_upserts(tmp_path: Path) -> None:
    """``embedded`` count + embed_meta row count match across multiple batches.

    With 6 docs at batch_size=2 the streaming consumer must invoke the
    provider at least 3 times. The post-ingest embed_meta row count
    confirms each batch was upserted (not buffered until the end).
    """
    wiki = _setup_multibatch_wiki(tmp_path, num_docs=6)
    embedder = _CountingEmbedder()
    report = await api.ingest(wiki, embedder=embedder)
    assert report.embedded == 6
    assert embedder.embed_calls >= 3, (
        f"expected >=3 batches at batch_size=2, got {embedder.embed_calls}"
    )
    # Each successful batch upserts immediately — total embed_meta rows
    # should equal report.embedded (no buffering = no leftover state).
    assert _count_embed_meta_rows(wiki) == 6


# ---- A4: mid-flight failure persists prior batches -----------------------


async def test_api_ingest_persists_prior_batches_on_failure(tmp_path: Path) -> None:
    """When the embedder raises after batch 1, batch 0's vectors remain on disk.

    This is the core durability claim of streaming: bulk-then-write
    used to lose every API-dollar on mid-flight crash. After this PR,
    prior batches are committed before the next one even starts.
    """
    wiki = _setup_multibatch_wiki(tmp_path, num_docs=6)
    embedder = _CountingEmbedder(fail_after=1)  # batch 1 succeeds, batch 2 fails
    with pytest.raises(RuntimeError, match="simulated failure"):
        await api.ingest(wiki, embedder=embedder)
    # Exactly batch_size (2) rows must be on disk — not 0 (bulk-write
    # would have lost them) and not 6 (the failure cut it short).
    assert _count_embed_meta_rows(wiki) == 2


# ---- A5: resume scan picks up the missing tail ---------------------------


async def test_api_ingest_resume_scan_picks_up_missing_embeds(
    tmp_path: Path,
) -> None:
    """Re-running ingest after a crash finishes the unembedded chunks.

    The doc-level shortcut (existing.hash == parsed.hash) skips already-
    seen docs on retry, so without the resume scan the unembedded tail
    would be invisible. Resume scan surfaces those chunks via
    ``list_chunks_missing_embedding`` and pipes them into the same
    streaming loop.
    """
    wiki = _setup_multibatch_wiki(tmp_path, num_docs=6)
    # Cold attempt: dies after 1 successful batch. embed_meta = 2.
    failing = _CountingEmbedder(fail_after=1)
    with pytest.raises(RuntimeError):
        await api.ingest(wiki, embedder=failing)
    assert _count_embed_meta_rows(wiki) == 2

    # Retry with a non-failing embedder. The doc-level shortcut would
    # skip every doc on its own; the resume scan is what surfaces the
    # 4 missing chunks. Verify exactly 4 are embedded on the retry.
    fresh = _CountingEmbedder()
    report = await api.ingest(wiki, embedder=fresh)
    assert _count_embed_meta_rows(wiki) == 6
    assert fresh.total_texts == 4, (
        f"resume scan should have re-embedded only the missing tail "
        f"(4), got {fresh.total_texts}"
    )
    # Doc-level loop didn't add anything; resume scan did all the work.
    assert report.unchanged == 6
