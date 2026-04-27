"""Embedding worker.

Two pathways:

* **Text** — ``embed_chunks(provider: EmbeddingProvider, ...)`` — the
  unimodal path. Returns ``EmbeddingRow(chunk_id, version_id, embedding)``
  for ``Storage.upsert_embeddings`` which routes into the per-version
  ``vec_chunks_v<id>`` table.

* **Multimodal** — ``embed_chunks_multimodal`` and ``embed_assets`` route
  through ``MultimodalEmbeddingProvider`` so chunks (text-only inputs)
  and asset binaries (image-only inputs) land in the *same* shared
  vector space, identified by their own multimodal ``version_id``.

Every embedding row carries a ``version_id`` from ``embed_versions``
so storage can isolate per-version vectors and the cache by identity.

Both pathways batch to keep HTTP round-trips low without overwhelming
providers' per-request input caps.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
from ..storage.base import NotSupported, Storage

logger = logging.getLogger(__name__)


async def _safe_cache_lookup(
    storage: Storage, hashes: Sequence[str], version_id: int
) -> dict[str, list[float]]:
    """Cache lookup that degrades to empty on NotSupported / transient faults."""
    try:
        return await storage.get_cached_embeddings(hashes, version_id=version_id)
    except NotSupported:
        return {}
    except Exception as exc:
        logger.warning("embed_cache lookup failed: %s; falling back", exc)
        return {}


async def _safe_cache_write(
    storage: Storage, rows: Sequence[CachedEmbeddingRow]
) -> None:
    """Cache write-back that degrades to no-op on NotSupported / transient faults."""
    try:
        await storage.cache_embeddings(rows)
    except NotSupported:
        return
    except Exception as exc:
        logger.warning("embed_cache write failed: %s; ignoring", exc)


@dataclass(frozen=True)
class ChunkToEmbed:
    chunk_id: int
    text: str


async def embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
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
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    async def _call(misses: list[ChunkToEmbed]) -> list[list[float]]:
        return await provider.embed([c.text for c in misses], model=model)

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
            batch,
            model=model,
            version_id=version_id,
            storage=storage,
            embed_misses=_call,
            error_template="embedding provider returned {got} vectors for {want} texts",
        )


async def embed_chunks_multimodal(
    provider: MultimodalEmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
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

    async def _call(misses: list[ChunkToEmbed]) -> list[list[float]]:
        return await provider.embed(
            [MultimodalInput(text=c.text) for c in misses], model=model
        )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        yield await _embed_one_batch(
            batch,
            model=model,
            version_id=version_id,
            storage=storage,
            embed_misses=_call,
            error_template="multimodal provider returned {got} vectors for {want} chunk inputs",
        )


async def _embed_one_batch(
    batch: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
    storage: Storage | None,
    embed_misses: Callable[[list[ChunkToEmbed]], Awaitable[list[list[float]]]],
    error_template: str,
) -> list[EmbeddingRow]:
    """Embed one batch, consulting the cache when ``storage`` is given.

    Returns rows in input order regardless of how hits/misses interleave.
    Within-batch duplicate hashes are sent to the provider once — vectors
    for the same content under the same version_id are deterministic
    by cache contract, so deduping saves a round-trip without changing
    semantics.
    """
    hashes = [content_hash(c.text) for c in batch]
    cached: dict[str, list[float]] = {}
    if storage is not None:
        cached = await _safe_cache_lookup(storage, hashes, version_id)

    miss_chunks: list[ChunkToEmbed] = []
    miss_hashes: list[str] = []
    queued: set[str] = set()
    for chunk, h in zip(batch, hashes, strict=True):
        if h in cached or h in queued:
            continue
        miss_chunks.append(chunk)
        miss_hashes.append(h)
        queued.add(h)

    new_vectors: list[list[float]] = []
    if miss_chunks:
        new_vectors = await embed_misses(miss_chunks)
        if len(new_vectors) != len(miss_chunks):
            raise RuntimeError(
                error_template.format(got=len(new_vectors), want=len(miss_chunks))
            )

    cached.update(zip(miss_hashes, new_vectors, strict=True))
    new_cache_rows = [
        CachedEmbeddingRow(
            content_hash=h, version_id=version_id, dim=len(v), embedding=v
        )
        for h, v in zip(miss_hashes, new_vectors, strict=True)
    ]

    rows = [
        EmbeddingRow(chunk_id=c.chunk_id, version_id=version_id, embedding=cached[h])
        for c, h in zip(batch, hashes, strict=True)
    ]

    if storage is not None and new_cache_rows:
        await _safe_cache_write(storage, new_cache_rows)

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
