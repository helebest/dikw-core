"""Focused tests for the embedding-version registry.

Cross-backend correctness lives in ``test_storage_contract.py``; this file
pins the SQLite-specific behaviours: each (provider, model, revision, dim,
normalize, distance) tuple is one identity, vec tables are dim-isolated
per version, and version transitions don't trash earlier data.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    EmbeddingVersion,
)
from dikw_core.storage.sqlite import SQLiteStorage


@pytest.fixture
async def storage(tmp_path: Path) -> AsyncIterator[SQLiteStorage]:
    s = SQLiteStorage(tmp_path / "v.sqlite")
    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        await s.close()


def _v(**overrides: object) -> EmbeddingVersion:
    base: dict[str, object] = {
        "provider": "prov",
        "model": "m",
        "revision": "",
        "dim": 4,
        "normalize": True,
        "distance": "cosine",
        "modality": "multimodal",
    }
    base.update(overrides)
    return EmbeddingVersion(**base)  # type: ignore[arg-type]


def _asset(asset_id: str) -> AssetRecord:
    import time

    return AssetRecord(
        asset_id=asset_id,
        hash=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=f"assets/{asset_id[:2]}/{asset_id[:8]}.png",
        original_paths=["x.png"],
        bytes=1,
        width=None,
        height=None,
        caption=None,
        caption_model=None,
        created_ts=time.time(),
    )


async def test_distinct_revision_creates_new_version(storage: SQLiteStorage) -> None:
    """Bumping ``revision`` (the user-facing escape hatch for silent weight
    refresh) must mint a new version row, even when every other field is
    identical."""
    a = await storage.upsert_embed_version(_v(revision=""))
    b = await storage.upsert_embed_version(_v(revision="2026-04"))
    assert a != b
    versions = await storage.list_embed_versions()
    assert len(versions) == 2


async def test_distinct_distance_creates_new_version(storage: SQLiteStorage) -> None:
    a = await storage.upsert_embed_version(_v(distance="cosine"))
    b = await storage.upsert_embed_version(_v(distance="dot"))
    assert a != b


async def test_distinct_normalize_creates_new_version(storage: SQLiteStorage) -> None:
    a = await storage.upsert_embed_version(_v(normalize=True))
    b = await storage.upsert_embed_version(_v(normalize=False))
    assert a != b


async def test_distinct_dim_creates_new_version(storage: SQLiteStorage) -> None:
    a = await storage.upsert_embed_version(_v(dim=768))
    b = await storage.upsert_embed_version(_v(dim=1024))
    assert a != b


async def test_text_and_multimodal_active_versions_independent(
    storage: SQLiteStorage,
) -> None:
    """Bumping the active multimodal version must not demote the active
    text version (different modality bucket)."""
    text = await storage.upsert_embed_version(_v(modality="text", model="t"))
    mm1 = await storage.upsert_embed_version(_v(modality="multimodal", model="m1"))
    mm2 = await storage.upsert_embed_version(_v(modality="multimodal", model="m2"))
    assert text != mm1 != mm2

    active_text = await storage.get_active_embed_version(modality="text")
    active_mm = await storage.get_active_embed_version(modality="multimodal")
    assert active_text is not None and active_text.version_id == text
    assert active_mm is not None and active_mm.version_id == mm2


async def test_per_version_vec_tables_are_dim_isolated(
    storage: SQLiteStorage,
) -> None:
    """Two coexisting versions with different dims each get their own
    sqlite-vec table; data written to one cannot collide with the other."""
    v_768 = await storage.upsert_embed_version(_v(model="x", dim=768))
    v_1024 = await storage.upsert_embed_version(_v(model="y", dim=1024))

    a = _asset("a" * 64)
    b = _asset("b" * 64)
    await storage.upsert_asset(a)
    await storage.upsert_asset(b)

    await storage.upsert_asset_embeddings(
        [AssetEmbeddingRow(asset_id=a.asset_id, version_id=v_768, embedding=[0.1] * 768)]
    )
    await storage.upsert_asset_embeddings(
        [AssetEmbeddingRow(asset_id=b.asset_id, version_id=v_1024, embedding=[0.2] * 1024)]
    )

    hits_768 = await storage.vec_search_assets(
        [0.1] * 768, version_id=v_768, limit=5
    )
    hits_1024 = await storage.vec_search_assets(
        [0.2] * 1024, version_id=v_1024, limit=5
    )
    assert len(hits_768) == 1 and hits_768[0].asset_id == a.asset_id
    assert len(hits_1024) == 1 and hits_1024[0].asset_id == b.asset_id


async def test_vec_search_assets_unknown_version_raises(
    storage: SQLiteStorage,
) -> None:
    """Querying an unregistered version_id must surface a clear failure
    rather than silently returning empty."""
    from dikw_core.storage.base import NotSupported

    with pytest.raises(NotSupported):
        await storage.vec_search_assets([0.0] * 4, version_id=999, limit=5)


async def test_vec_search_assets_no_embeddings_returns_empty(
    storage: SQLiteStorage,
) -> None:
    """A registered version with no upserted embeddings yet should return
    an empty hit list, not raise."""
    v = await storage.upsert_embed_version(_v(dim=4))
    hits = await storage.vec_search_assets(
        [1.0, 0.0, 0.0, 0.0], version_id=v, limit=5
    )
    assert hits == []
