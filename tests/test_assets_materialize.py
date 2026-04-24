"""Tests for asset materialization.

Materialize_asset is the bridge between MarkdownBackend's symbolic
references and the engine-managed binary on disk. Contract:

  * Local files: read → sha256 → copy to assets/<h2>/<h8>-<name>.<ext>
    → probe dims → upsert AssetRecord. Idempotent by hash.
  * Remote URLs: skip (return None), no side effect.
  * Missing/unreadable files: skip (return None), no side effect.
  * Same hash, different referencing names: one row, original_paths
    accumulates each distinct reference.
"""

from __future__ import annotations

import struct
from collections.abc import Awaitable, Callable
from pathlib import Path

from dikw_core.data.assets import materialize_asset
from dikw_core.schemas import AssetKind, AssetRecord, AssetRef

# ---- Helpers -------------------------------------------------------------


def _make_fake_storage() -> tuple[
    dict[str, AssetRecord],
    Callable[[str], Awaitable[AssetRecord | None]],
    Callable[[AssetRecord], Awaitable[None]],
]:
    """Tiny in-memory get/upsert pair for testing the materialize pipeline
    without requiring the full Storage Protocol or a SQLite adapter."""
    store: dict[str, AssetRecord] = {}

    async def get(asset_id: str) -> AssetRecord | None:
        return store.get(asset_id)

    async def upsert(asset: AssetRecord) -> None:
        store[asset.asset_id] = asset

    return store, get, upsert


def _png_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable PNG: 8-byte signature + IHDR chunk header.
    Enough for the probe to extract width/height; not a renderable image."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_body = struct.pack(">II", width, height) + bytes([8, 6, 0, 0, 0])
    ihdr_crc = b"\x00\x00\x00\x00"  # probe doesn't validate CRC
    return sig + ihdr_len + ihdr_type + ihdr_body + ihdr_crc


def _jpeg_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable JPEG: SOI + JFIF APP0 + SOF0 with declared dims."""
    soi = b"\xff\xd8"
    # JFIF APP0: marker FFE0, length 16, "JFIF\0", v1.01, units 0, density 1x1, no thumb
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    # SOF0: marker FFC0, length 17, precision 8, height, width, 3 components
    sof0 = (
        b"\xff\xc0\x00\x11\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )
    return soi + app0 + sof0


def _gif_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable GIF89a header: 6 sig + 2 width LE + 2 height LE."""
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def _svg_bytes() -> bytes:
    return b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" />'


# ---- Tests ---------------------------------------------------------------


async def test_materialize_local_png_copies_and_records(tmp_path: Path) -> None:
    """Happy path: a local PNG referenced from a markdown file is copied
    into the engine vault and its metadata recorded."""
    project_root = tmp_path / "project"
    notes_dir = project_root / "notes"
    notes_dir.mkdir(parents=True)
    md = notes_dir / "doc.md"
    md.write_text("![arch](./diagrams/arch.png)", encoding="utf-8")
    (notes_dir / "diagrams").mkdir()
    png_bytes = _png_with_dims(640, 480)
    (notes_dir / "diagrams" / "arch.png").write_bytes(png_bytes)

    ref = AssetRef(
        original_path="./diagrams/arch.png",
        alt="arch",
        start=0,
        end=29,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.kind == AssetKind.IMAGE
    assert rec.mime == "image/png"
    assert rec.bytes == len(png_bytes)
    assert rec.width == 640
    assert rec.height == 480
    assert rec.original_paths == ["./diagrams/arch.png"]
    # File materialized at the engine path.
    on_disk = project_root / rec.stored_path
    assert on_disk.is_file()
    assert on_disk.read_bytes() == png_bytes
    # Asset row landed in fake storage.
    assert rec.asset_id in store


async def test_materialize_skips_remote_url(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")

    ref = AssetRef(
        original_path="https://example.com/img.png",
        alt="r",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_skips_missing_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")

    ref = AssetRef(
        original_path="./not-here.png",
        alt="m",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_dedup_by_hash_appends_original_paths(
    tmp_path: Path,
) -> None:
    """Same file content referenced from two paths → single AssetRecord,
    original_paths grows; binary copied only once."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    notes = project_root / "notes"
    notes.mkdir()

    png_bytes = _png_with_dims(10, 20)
    a_path = notes / "a.png"
    b_path = notes / "b.png"
    a_path.write_bytes(png_bytes)
    b_path.write_bytes(png_bytes)

    md = notes / "doc.md"
    md.write_text("placeholder", encoding="utf-8")

    store, get, upsert = _make_fake_storage()

    rec1 = await materialize_asset(
        AssetRef(
            original_path="a.png", alt="", start=0, end=10, syntax="markdown"
        ),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec1 = None if rec1 is None else rec1[0]
    assert rec1 is not None
    assert rec1.original_paths == ["a.png"]

    result2 = await materialize_asset(
        AssetRef(
            original_path="b.png", alt="", start=0, end=10, syntax="markdown"
        ),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert result2 is not None
    rec2, was_new_2 = result2
    # Second materialize hit the existing row — was_new flag tells the
    # caller (api.ingest) it doesn't need to re-embed this asset.
    assert was_new_2 is False
    assert rec2.asset_id == rec1.asset_id  # same content → same id
    assert rec2.original_paths == ["a.png", "b.png"]
    # Only one row.
    assert len(store) == 1


async def test_materialize_dedup_idempotent_on_same_path(
    tmp_path: Path,
) -> None:
    """Re-materializing the exact same reference must not duplicate the
    path string in original_paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    img = project_root / "img.png"
    img.write_bytes(_png_with_dims(1, 1))

    ref = AssetRef(
        original_path="img.png", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()

    r1 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    r2 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert r1 is not None and r2 is not None
    _rec1, was_new_1 = r1
    rec2, was_new_2 = r2
    assert was_new_1 is True
    assert was_new_2 is False
    assert rec2.original_paths == ["img.png"]


async def test_materialize_jpeg_dimensions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    jpg_bytes = _jpeg_with_dims(800, 600)
    (project_root / "photo.jpg").write_bytes(jpg_bytes)

    ref = AssetRef(
        original_path="photo.jpg", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/jpeg"
    assert rec.width == 800
    assert rec.height == 600


async def test_materialize_gif_dimensions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    gif_bytes = _gif_with_dims(50, 40)
    (project_root / "loop.gif").write_bytes(gif_bytes)

    ref = AssetRef(
        original_path="loop.gif", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/gif"
    assert rec.width == 50
    assert rec.height == 40


async def test_materialize_svg_records_no_dims(tmp_path: Path) -> None:
    """SVG is not vectorized in v1 (it's text-based); the materialize step
    still copies it and records mime, but width/height stay None."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    (project_root / "icon.svg").write_bytes(_svg_bytes())

    ref = AssetRef(
        original_path="icon.svg", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/svg+xml"
    assert rec.width is None
    assert rec.height is None


async def test_materialize_unsupported_format_skipped(tmp_path: Path) -> None:
    """Random non-image bytes with a non-image extension → skip."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    (project_root / "data.xyz").write_bytes(b"random binary content here")

    ref = AssetRef(
        original_path="data.xyz", alt="", start=0, end=10, syntax="markdown"
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_uses_engine_vault_path_layout(tmp_path: Path) -> None:
    """The stored_path column must match assets/<h2>/<h8>-<sanitized>.<ext>."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    png = _png_with_dims(1, 1)
    (project_root / "diagram with spaces.png").write_bytes(png)

    ref = AssetRef(
        original_path="diagram with spaces.png",
        alt="",
        start=0,
        end=10,
        syntax="markdown",
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    parts = rec.stored_path.split("/")
    assert parts[0] == "assets"
    assert len(parts[1]) == 2  # h2
    assert parts[2].startswith(rec.hash[:8] + "-")
    assert parts[2].endswith(".png")
    # Sanitized: spaces collapsed to '-'.
    assert "diagram-with-spaces" in parts[2]
