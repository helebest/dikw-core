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
from tests.fakes import FakeEmbeddings, FakeMultimodalEmbedding, make_provider_cfg


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
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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
    IngestReport. Chunk-text vectors flow through the text embedder
    (separate channel from the mm asset channel)."""
    text_embedder = FakeEmbeddings()
    mm = FakeMultimodalEmbedding(dim=4)
    report = await api.ingest(
        project_with_image_doc,
        embedder=text_embedder,
        multimodal_embedder=mm,
    )

    assert report.scanned == 1
    assert report.added == 1
    assert report.chunks >= 1
    assert report.embedded >= 1, "chunk vectors flow through the text channel"
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
    from dikw_core.info.search import HybridSearcher, MultimodalSearch

    cfg, _root, storage = await _with_storage(project_with_image_doc)
    try:
        active = await storage.get_active_embed_version(modality="multimodal")
        assert active is not None
        assert active.version_id is not None
        assert cfg.assets.multimodal is not None
        searcher = HybridSearcher(
            storage,
            embedder=None,
            multimodal=MultimodalSearch(
                embedder=mm,
                model=cfg.assets.multimodal.model,
                asset_version_id=active.version_id,
            ),
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
        assert asset.media_meta is not None
        assert asset.media_meta.width == 640
        assert asset.media_meta.height == 480
        assert asset.stored_path.startswith("assets/")
        assert "arch" in asset.stored_path
    finally:
        await storage.close()


async def test_asset_backfill_after_text_only_ingest(tmp_path: Path) -> None:
    """text-only ingest creates ``assets`` rows but no vectors. Re-ingest
    after enabling multimodal should backfill via
    ``Storage.list_assets_missing_embedding`` — every materialized asset
    ends up vectorized, not just newly-added ones.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "diagrams").mkdir()
    (root / "sources" / "diagrams" / "arch.png").write_bytes(_png_with_dims(640, 480))
    (root / "sources" / "doc.md").write_text(
        "# Architecture\n\nSee the diagram: ![arch](./diagrams/arch.png)\n",
        encoding="utf-8",
    )
    # Stage 1: text-only config (no assets.multimodal block).
    cfg_text_only = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
        assets=AssetsConfig(multimodal=None),
    )
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_text_only), encoding="utf-8")
    (root / ".dikw").mkdir()

    await api.ingest(root, embedder=FakeEmbeddings())

    # ``report.assets`` only counts assets fed into the mm-embed queue;
    # text-only ingest leaves it at 0 even though the binary has been
    # materialized. Inspect storage directly to confirm.
    counts_before = await api.status(root)
    assert counts_before.assets >= 1, (
        "asset should materialize on disk even without mm cfg"
    )
    assert counts_before.asset_embeddings == 0, (
        "no asset vectors should exist after text-only ingest"
    )

    # Stage 2: enable mm by rewriting dikw.yml in place.
    cfg_mm = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_mm), encoding="utf-8")

    mm = FakeMultimodalEmbedding(dim=4)
    report2 = await api.ingest(
        root, embedder=FakeEmbeddings(), multimodal_embedder=mm
    )
    assert report2.asset_embedded >= 1, (
        "backfill scan should pick up the existing asset and embed it"
    )
    assert mm.embed_calls >= 1, "the mm provider must actually run"


async def test_asset_backfill_excludes_unembeddable_mime(tmp_path: Path) -> None:
    """SVG (and any future no-vector mime) must NOT re-enter the
    backfill queue every ingest. ``embed_assets`` discards them
    without writing an ``asset_embed_meta`` row, so a naive resume
    scan would re-select them forever.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    # Inline SVG — referenced by ![alt](path); the image-detection
    # backend recognizes it via the ``<svg`` magic-byte sniff.
    (root / "sources" / "icon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
        encoding="utf-8",
    )
    (root / "sources" / "doc.md").write_text(
        "# Page\n\n![icon](./icon.svg)\n",
        encoding="utf-8",
    )
    cfg = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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

    mm = FakeMultimodalEmbedding(dim=4)
    await api.ingest(root, embedder=FakeEmbeddings(), multimodal_embedder=mm)
    calls_after_first = mm.embed_calls

    # Second ingest on the unchanged SVG-only corpus: backfill scan
    # finds the SVG (no meta row), but the caller filter must drop it
    # before requeuing — otherwise the mm provider gets called again
    # to no-op-skip the same asset.
    await api.ingest(root, embedder=FakeEmbeddings(), multimodal_embedder=mm)
    assert mm.embed_calls == calls_after_first, (
        "SVG asset should not re-enter the embed queue after first ingest"
    )


async def test_report_assets_counts_only_new_this_run(tmp_path: Path) -> None:
    """``IngestReport.assets`` documents "NEW assets materialized this
    run". Backfilled (historical) assets must not inflate this counter.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "diagrams").mkdir()
    (root / "sources" / "diagrams" / "arch.png").write_bytes(_png_with_dims(640, 480))
    (root / "sources" / "doc.md").write_text(
        "# Architecture\n\n![arch](./diagrams/arch.png)\n",
        encoding="utf-8",
    )
    # Stage 1: text-only ingest materializes the asset (no embed yet).
    cfg_text_only = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
        assets=AssetsConfig(multimodal=None),
    )
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_text_only), encoding="utf-8")
    (root / ".dikw").mkdir()
    await api.ingest(root, embedder=FakeEmbeddings())

    # Stage 2: enable mm + re-ingest the SAME corpus. The asset is
    # historical (was_new=False); only the backfill scan picks it up.
    # ``report.assets`` must be 0 — nothing was materialized this run.
    cfg_mm = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_mm), encoding="utf-8")
    report2 = await api.ingest(
        root, embedder=FakeEmbeddings(), multimodal_embedder=FakeMultimodalEmbedding(dim=4)
    )
    assert report2.assets == 0, (
        "no NEW assets materialized this run — historical asset is "
        "backfill, not new"
    )
    assert report2.asset_embedded == 1, (
        "the historical asset still gets embedded via the backfill path"
    )


async def test_asset_backfill_skips_orphan_assets(tmp_path: Path) -> None:
    """An asset whose only ``chunk_asset_refs`` entry was removed
    (md edited to drop the ![](path)) becomes unreachable from
    ``HybridSearcher``. Embedding it during backfill burns a provider
    call for a vector search can never surface.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "diagrams").mkdir()
    (root / "sources" / "diagrams" / "arch.png").write_bytes(_png_with_dims(640, 480))
    # Stage 1: text-only ingest with the image referenced.
    (root / "sources" / "doc.md").write_text(
        "# Page\n\n![arch](./diagrams/arch.png)\n",
        encoding="utf-8",
    )
    cfg_text_only = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
        assets=AssetsConfig(multimodal=None),
    )
    (root / "dikw.yml").write_text(
        dump_config_yaml(cfg_text_only), encoding="utf-8"
    )
    (root / ".dikw").mkdir()
    await api.ingest(root, embedder=FakeEmbeddings())

    # Stage 2: edit the md to remove the image reference. The asset
    # row stays in storage but ``chunk_asset_refs`` no longer points
    # to it (replace_chunks → DELETE old refs, insert empty for new
    # chunk text). The asset is now an orphan.
    (root / "sources" / "doc.md").write_text(
        "# Page\n\nNo image here anymore.\n",
        encoding="utf-8",
    )
    await api.ingest(root, embedder=FakeEmbeddings())

    # Stage 3: enable mm + re-ingest. The orphan must NOT enter the
    # backfill embed queue.
    cfg_mm = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_mm), encoding="utf-8")
    mm = FakeMultimodalEmbedding(dim=4)
    report = await api.ingest(root, embedder=FakeEmbeddings(), multimodal_embedder=mm)
    assert mm.embed_calls == 0, (
        "orphan asset (no live chunk reference) must not be embedded"
    )
    assert report.asset_embedded == 0


async def test_ingest_rejects_zero_batch_size_with_clear_error(
    tmp_path: Path,
) -> None:
    """Zero / negative batch_size must surface as a usable
    ``ValueError`` from the validation in ``_ceil_div`` rather than an
    opaque ``ZeroDivisionError`` from the progress-total computation.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "doc.md").write_text("# Page\n\nbody\n", encoding="utf-8")
    cfg = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(
            embedding_model="fake-text-v1", embedding_batch_size=0
        ),
        assets=AssetsConfig(multimodal=None),
    )
    (root / "dikw.yml").write_text(dump_config_yaml(cfg), encoding="utf-8")
    (root / ".dikw").mkdir()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        await api.ingest(root, embedder=FakeEmbeddings())


async def test_asset_backfill_skips_assets_with_missing_binary(
    tmp_path: Path,
) -> None:
    """An asset whose stored binary was deleted from disk after a
    prior ingest must NOT keep re-entering the backfill embed queue.
    ``embed_assets`` would log a WARN and skip without writing a meta
    row, so a naive resume scan would re-select + re-warn forever.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "sources").mkdir()
    (root / "sources" / "diagrams").mkdir()
    (root / "sources" / "diagrams" / "arch.png").write_bytes(_png_with_dims(32, 32))
    (root / "sources" / "doc.md").write_text(
        "# Page\n\n![arch](./diagrams/arch.png)\n",
        encoding="utf-8",
    )
    cfg = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
        assets=AssetsConfig(multimodal=None),
    )
    (root / "dikw.yml").write_text(dump_config_yaml(cfg), encoding="utf-8")
    (root / ".dikw").mkdir()
    # Stage 1: text-only ingest — asset row created, vault binary written.
    await api.ingest(root, embedder=FakeEmbeddings())

    # Find + delete the materialized binary inside the engine vault.
    # ``materialize_asset`` writes under ``<wiki>/assets/...`` keyed by
    # sha256; the asset row's ``stored_path`` points at it.
    vault_pngs = list((root / "assets").rglob("*.png"))
    assert vault_pngs, "expected the vault to hold the materialized PNG"
    for p in vault_pngs:
        p.unlink()

    # Stage 2: enable mm + ingest again. The orphan-on-disk asset row
    # must be filtered out of the backfill scan; mm provider stays idle.
    cfg_mm = DikwConfig(
        storage=SQLiteStorageConfig(path=".dikw/index.sqlite"),
        sources=[SourceConfig(path="./sources", pattern="**/*.md")],
        provider=make_provider_cfg(embedding_model="fake-text-v1"),
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
    (root / "dikw.yml").write_text(dump_config_yaml(cfg_mm), encoding="utf-8")
    mm = FakeMultimodalEmbedding(dim=4)
    report = await api.ingest(
        root, embedder=FakeEmbeddings(), multimodal_embedder=mm
    )
    assert mm.embed_calls == 0, (
        "asset whose binary is missing on disk must not enter the embed queue"
    )
    assert report.asset_embedded == 0


async def test_asset_backfill_idempotent_after_initial_embed(
    project_with_image_doc: Path,
) -> None:
    """Two consecutive mm-enabled ingests on an unchanged corpus —
    the second should NOT re-embed the asset (every asset already
    has an ``asset_embed_meta`` row for the active version).
    """
    mm = FakeMultimodalEmbedding(dim=4)
    report1 = await api.ingest(
        project_with_image_doc,
        embedder=FakeEmbeddings(),
        multimodal_embedder=mm,
    )
    assert report1.asset_embedded == 1
    calls_after_first = mm.embed_calls

    # Re-ingest unchanged corpus — chunker / asset materialize will
    # find no new work, AND the backfill scan should also see no
    # missing embeddings, so the mm provider stays untouched.
    report2 = await api.ingest(
        project_with_image_doc,
        embedder=FakeEmbeddings(),
        multimodal_embedder=mm,
    )
    assert report2.asset_embedded == 0, "no new asset work expected"
    assert mm.embed_calls == calls_after_first, (
        "backfill must skip already-embedded assets"
    )
