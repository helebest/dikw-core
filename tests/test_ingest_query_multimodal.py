"""End-to-end ingest + query with the multimodal pipeline (Phase L).

Walks one synthetic markdown source that references a local image
through ingest → materialize → chunk → chunk_asset_refs → embed_assets →
storage upsert, then issues a text query and asserts the asset_refs
come back attached to the Hit. Uses an in-memory ``FakeMultimodalEmbedding``
so no real provider HTTP is touched.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import (
    AssetsConfig,
    DikwConfig,
    MultimodalEmbedConfig,
    SourceConfig,
    SQLiteStorageConfig,
    dump_config_yaml,
)
from tests.fakes import FakeMultimodalEmbedding


def _png_with_dims(w: int, h: int) -> bytes:
    """Synthetic PNG header sufficient for the dim-probe to read w/h."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", w, h)
        + bytes([8, 6, 0, 0, 0])
        + b"\x00\x00\x00\x00"
    )


@pytest.fixture
def project_with_image_doc(tmp_path: Path) -> Path:
    """Build a fresh dikw project root with a multimodal-enabled config
    and one source markdown that embeds a local PNG."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "diagrams").mkdir()
    (root / "sources" / "diagrams" / "arch.png").write_bytes(_png_with_dims(640, 480))
    (root / "sources" / "doc.md").write_text(
        "# Architecture\n\nSee the diagram: ![arch diagram](./diagrams/arch.png)\n\n"
        "It illustrates how alpha and beta interact.",
        encoding="utf-8",
    )

    cfg = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        assets=AssetsConfig(
            multimodal=MultimodalEmbedConfig(
                provider="gitee_multimodal",
                model="fake-mm-v1",
                dim=4,
                normalize=True,
                distance="cosine",
                batch=4,
            )
        ),
    )
    (root / "dikw.yml").write_text(dump_config_yaml(cfg), encoding="utf-8")
    (root / ".dikw").mkdir()
    return root


async def test_full_ingest_with_multimodal(project_with_image_doc: Path) -> None:
    """ingest() materializes the asset, embeds it via the mm provider,
    persists chunk_asset_refs, and bumps the asset counters in the
    IngestReport."""
    mm = FakeMultimodalEmbedding(dim=4)
    report = await api.ingest(project_with_image_doc, multimodal_embedder=mm)

    assert report.scanned == 1
    assert report.added == 1
    assert report.chunks >= 1
    assert report.embedded >= 1, "chunk vectors should also flow through mm"
    assert report.assets == 1, "the one local PNG should materialize"
    assert report.asset_embedded == 1, "the asset should be vectorized via mm"

    # The binary really landed under <root>/assets/ via the engine path scheme.
    assets_dir = project_with_image_doc / "assets"
    materialized = list(assets_dir.rglob("*.png"))
    assert len(materialized) == 1
    assert "arch" in materialized[0].name


async def test_query_returns_asset_refs_for_image_bearing_chunk(
    project_with_image_doc: Path,
) -> None:
    """After ingest, querying for a token in the chunk text should bring
    back a Hit whose asset_refs contains the image referenced by that
    chunk — exercises the Phase K Hit.asset_refs attachment via the
    real api.query glue."""
    mm = FakeMultimodalEmbedding(dim=4)
    await api.ingest(project_with_image_doc, multimodal_embedder=mm)

    # Use the search facade directly (the public api.query wraps an LLM
    # which we don't want to invoke here). The Phase L wiring is what
    # makes this work end-to-end through the SQLite storage on disk.
    from dikw_core.api import _with_storage
    from dikw_core.info.search import HybridSearcher

    cfg, _root, storage = await _with_storage(project_with_image_doc)
    try:
        active = await storage.get_active_embed_version(modality="multimodal")
        assert active is not None
        searcher = HybridSearcher(
            storage,
            embedder=None,
            multimodal_embedder=mm,
            multimodal_model=cfg.assets.multimodal.model,  # type: ignore[union-attr]
            asset_version_id=active.version_id,
        )
        hits = await searcher.search("alpha beta", limit=5)
        assert hits, "FTS leg should retrieve the chunk on 'alpha beta'"
        # The chunk that contains 'alpha and beta' is also the chunk that
        # holds the ![arch diagram](...) reference (one paragraph each in
        # this synthetic doc) — assert at least one hit carries the asset.
        hits_with_assets = [h for h in hits if h.asset_refs]
        assert hits_with_assets, (
            f"expected asset_refs on at least one hit, got {hits}"
        )
        asset = hits_with_assets[0].asset_refs[0]
        assert asset.mime == "image/png"
        assert asset.width == 640
        assert asset.height == 480
        assert asset.stored_path.startswith("assets/")
        assert "arch" in asset.stored_path
    finally:
        await storage.close()
