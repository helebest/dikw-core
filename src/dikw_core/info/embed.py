"""Embedding worker.

Two pathways:

* **Legacy text** — ``embed_chunks(provider: EmbeddingProvider, ...)`` —
  the original v1 path used by installations that haven't enabled
  multimodal yet. Returns ``EmbeddingRow(chunk_id, model, embedding)``
  for ``Storage.upsert_embeddings`` against the legacy ``chunks_vec``
  table.

* **Multimodal** — ``embed_chunks_multimodal`` and ``embed_assets`` route
  through ``MultimodalEmbeddingProvider`` so chunks (text-only inputs)
  and asset binaries (image-only inputs) land in the *same* shared
  vector space. Asset embeddings are version-tagged and persist via
  ``Storage.upsert_asset_embeddings`` into the per-version
  ``vec_assets_v<id>`` virtual table.

Both pathways batch to keep HTTP round-trips low without overwhelming
providers' per-request input caps.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..data.backends.markdown import content_hash
from ..providers import EmbeddingProvider, MultimodalEmbeddingProvider
from ..schemas import (
    AssetEmbeddingRow,
    AssetRecord,
    CachedEmbeddingRow,
    EmbeddingRow,
    ImageContent,
    MultimodalInput,
)

if TYPE_CHECKING:
    from ..storage.base import Storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkToEmbed:
    chunk_id: int
    text: str


async def embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    storage: Storage | None = None,
    batch_size: int = 64,
) -> AsyncIterator[list[EmbeddingRow]]:
    """Embed ``chunks`` in fixed-size batches with optional content-hash cache.

    Streams one ``list[EmbeddingRow]`` per provider call so the caller
    can persist each batch immediately. Per-batch failure cost is bounded:
    prior batches are already on disk when the next API call raises.

    When ``storage`` is supplied, each batch first looks up
    ``sha256(chunk.text)`` in ``storage.embed_cache`` and routes hits
    around the provider. Misses are sent to the provider; their vectors
    are written to the cache after success. ``storage=None`` skips the
    cache entirely (no-op fast path for callers that don't have a wiki
    handy). Adapters that return ``NotSupported`` (filesystem) silently
    degrade to no-cache.

    NOTE: when many cache hits land in one batch, the residual miss
    batch sent to the provider can be small (under-utilizes per-request
    capacity). Acceptable tradeoff: correctness beats throughput at
    pre-alpha.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[start : start + batch_size]
        # INFO-level so a long-running ingest run shows progress without
        # needing a debug flag — debugging "did the embedder hang?"
        # without per-batch markers means restarting from scratch.
        logger.info(
            "embed batch %d/%d (chunks %d-%d)",
            batch_idx + 1,
            total_batches,
            start,
            start + len(batch) - 1,
        )
        yield await _embed_one_batch(
            provider, batch, model=model, storage=storage
        )


async def _embed_one_batch(
    provider: EmbeddingProvider,
    batch: Sequence[ChunkToEmbed],
    *,
    model: str,
    storage: Storage | None,
) -> list[EmbeddingRow]:
    """Embed one batch, consulting the cache when ``storage`` is given.

    Returns rows in input order regardless of how hits/misses interleave.
    """
    hashes = [content_hash(c.text) for c in batch]
    cached: dict[str, list[float]] = {}
    if storage is not None:
        try:
            cached = await storage.get_cached_embeddings(hashes, model=model)
        except Exception as exc:
            # Storage adapters declaring NotSupported (filesystem) or
            # raising on a transient cache fault must NOT abort the run
            # — degrade to no-cache and log once per batch.
            from ..storage.base import NotSupported

            if isinstance(exc, NotSupported):
                cached = {}
            else:
                logger.warning("embed_cache lookup failed: %s; falling back", exc)
                cached = {}

    miss_idx: list[int] = []
    miss_texts: list[str] = []
    for i, (chunk, h) in enumerate(zip(batch, hashes, strict=True)):
        if h not in cached:
            miss_idx.append(i)
            miss_texts.append(chunk.text)

    new_vectors: list[list[float]] = []
    if miss_texts:
        new_vectors = await provider.embed(miss_texts, model=model)
        if len(new_vectors) != len(miss_texts):
            raise RuntimeError(
                f"embedding provider returned {len(new_vectors)} vectors for "
                f"{len(miss_texts)} texts"
            )

    rows: list[EmbeddingRow] = []
    miss_iter = iter(zip(miss_idx, new_vectors, strict=True))
    next_miss = next(miss_iter, None)
    new_cache_rows: list[CachedEmbeddingRow] = []
    for i, (chunk, h) in enumerate(zip(batch, hashes, strict=True)):
        if next_miss is not None and i == next_miss[0]:
            vec = next_miss[1]
            next_miss = next(miss_iter, None)
            new_cache_rows.append(
                CachedEmbeddingRow(
                    content_hash=h, model=model, dim=len(vec), embedding=vec
                )
            )
        else:
            vec = cached[h]
        rows.append(EmbeddingRow(chunk_id=chunk.chunk_id, model=model, embedding=vec))

    if storage is not None and new_cache_rows:
        try:
            await storage.cache_embeddings(new_cache_rows)
        except Exception as exc:
            from ..storage.base import NotSupported

            if not isinstance(exc, NotSupported):
                logger.warning("embed_cache write failed: %s; ignoring", exc)
            # NotSupported is silent — filesystem etc. don't keep a cache.

    return rows


async def embed_chunks_multimodal(
    provider: MultimodalEmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    storage: Storage | None = None,
    batch_size: int = 16,
) -> AsyncIterator[list[EmbeddingRow]]:
    """Embed text chunks via the multimodal provider (text-only inputs).

    Streams one ``list[EmbeddingRow]`` per batch — same contract shape
    as ``embed_chunks``, including the optional content-hash cache.
    Uses the same ``EmbeddingRow`` output so chunk vectors can land in
    the same ``chunks_vec`` table — switching a corpus to a multimodal
    model in v1 is a config change at the factory level; the storage
    shape doesn't move.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunks:
        # Still notify the provider so its last_inputs reflects an empty
        # call (a legacy contract some tests rely on). Empty generator
        # otherwise.
        await provider.embed([], model=model)
        return
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        yield await _embed_one_batch_multimodal(
            provider, batch, model=model, storage=storage
        )


async def _embed_one_batch_multimodal(
    provider: MultimodalEmbeddingProvider,
    batch: Sequence[ChunkToEmbed],
    *,
    model: str,
    storage: Storage | None,
) -> list[EmbeddingRow]:
    """Multimodal-path counterpart of ``_embed_one_batch``.

    Same cache contract; only the provider call shape differs (wraps
    text in ``MultimodalInput``).
    """
    hashes = [content_hash(c.text) for c in batch]
    cached: dict[str, list[float]] = {}
    if storage is not None:
        try:
            cached = await storage.get_cached_embeddings(hashes, model=model)
        except Exception as exc:
            from ..storage.base import NotSupported

            if isinstance(exc, NotSupported):
                cached = {}
            else:
                logger.warning("embed_cache lookup failed: %s; falling back", exc)
                cached = {}

    miss_idx: list[int] = []
    miss_inputs: list[MultimodalInput] = []
    for i, (chunk, h) in enumerate(zip(batch, hashes, strict=True)):
        if h not in cached:
            miss_idx.append(i)
            miss_inputs.append(MultimodalInput(text=chunk.text))

    new_vectors: list[list[float]] = []
    if miss_inputs:
        new_vectors = await provider.embed(miss_inputs, model=model)
        if len(new_vectors) != len(miss_inputs):
            raise RuntimeError(
                f"multimodal provider returned {len(new_vectors)} vectors for "
                f"{len(miss_inputs)} chunk inputs"
            )

    rows: list[EmbeddingRow] = []
    new_cache_rows: list[CachedEmbeddingRow] = []
    miss_iter = iter(zip(miss_idx, new_vectors, strict=True))
    next_miss = next(miss_iter, None)
    for i, (chunk, h) in enumerate(zip(batch, hashes, strict=True)):
        if next_miss is not None and i == next_miss[0]:
            vec = next_miss[1]
            next_miss = next(miss_iter, None)
            new_cache_rows.append(
                CachedEmbeddingRow(
                    content_hash=h, model=model, dim=len(vec), embedding=vec
                )
            )
        else:
            vec = cached[h]
        rows.append(EmbeddingRow(chunk_id=chunk.chunk_id, model=model, embedding=vec))

    if storage is not None and new_cache_rows:
        try:
            await storage.cache_embeddings(new_cache_rows)
        except Exception as exc:
            from ..storage.base import NotSupported

            if not isinstance(exc, NotSupported):
                logger.warning("embed_cache write failed: %s; ignoring", exc)

    return rows


async def embed_assets(
    provider: MultimodalEmbeddingProvider,
    assets: Sequence[AssetRecord],
    *,
    project_root: Path,
    model: str,
    version_id: int,
    batch_size: int = 16,
) -> list[AssetEmbeddingRow]:
    """Embed asset binaries via the multimodal provider (image-only inputs).

    Reads each asset's bytes from ``project_root / asset.stored_path``.
    Assets whose binary is missing on disk are skipped with a WARNING log;
    the rest of the batch still embeds. SVG assets (``image/svg+xml``)
    are also skipped since v1 doesn't rasterize them; their AssetRecord
    survives in storage with no embedding row.

    Output rows are tagged with ``version_id`` so storage can route them
    into the correct ``vec_assets_v<id>`` table.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not assets:
        return []

    # Resolve and load bytes upfront so the provider call only sees ready
    # inputs. Skipped assets never enter the request payload.
    pending: list[tuple[AssetRecord, ImageContent]] = []
    for asset in assets:
        if asset.mime == "image/svg+xml":
            logger.info(
                "skipping SVG asset embedding (v1 doesn't rasterize): %s",
                asset.stored_path,
            )
            continue
        path = project_root / asset.stored_path
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.warning(
                "skipping asset embedding for %s (read failed: %s)",
                asset.stored_path,
                e,
            )
            continue
        pending.append((asset, ImageContent(bytes=data, mime=asset.mime)))

    rows: list[AssetEmbeddingRow] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        inputs = [MultimodalInput(images=[img]) for _, img in batch]
        vectors = await provider.embed(inputs, model=model)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"multimodal provider returned {len(vectors)} vectors for "
                f"{len(batch)} asset inputs"
            )
        rows.extend(
            AssetEmbeddingRow(
                asset_id=asset.asset_id, version_id=version_id, embedding=v
            )
            for (asset, _), v in zip(batch, vectors, strict=True)
        )
    return rows
