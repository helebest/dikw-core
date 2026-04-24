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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..providers import EmbeddingProvider, MultimodalEmbeddingProvider
from ..schemas import (
    AssetEmbeddingRow,
    AssetRecord,
    EmbeddingRow,
    ImageContent,
    MultimodalInput,
)

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


async def embed_chunks_multimodal(
    provider: MultimodalEmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    batch_size: int = 16,
) -> list[EmbeddingRow]:
    """Embed text chunks via the multimodal provider (text-only inputs).

    Uses the same EmbeddingRow output shape as the legacy text pathway
    so chunk vectors can land in the same ``chunks_vec`` table — switching
    a corpus to a multimodal model in v1 is a config change at the
    factory level; the storage shape doesn't move.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunks:
        # Still notify the provider so its last_inputs reflects an empty call.
        await provider.embed([], model=model)
        return []
    rows: list[EmbeddingRow] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        inputs = [MultimodalInput(text=c.text) for c in batch]
        vectors = await provider.embed(inputs, model=model)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"multimodal provider returned {len(vectors)} vectors for "
                f"{len(batch)} chunk inputs"
            )
        rows.extend(
            EmbeddingRow(chunk_id=c.chunk_id, model=model, embedding=v)
            for c, v in zip(batch, vectors, strict=True)
        )
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
