"""Tests for the Layer-3 render helper.

``render_chunk`` is the on-demand path-substitution that produces a
self-contained Markdown rendering of a chunk: each image reference's
original path is rewritten to the engine-managed ``stored_path`` so the
output renders correctly regardless of where the source binary
originally lived.

Contract:

  * Pure function — never mutates inputs or touches storage.
  * Substitutes by chunk-relative ``(start_in_chunk, end_in_chunk)`` —
    never re-parses the markdown, so handles arbitrary syntax variants
    (wikilink dimensions, alt edge cases) uniformly.
  * Refs without a matching asset in the ``assets`` map are left
    untouched (defensive — logs a warning at the call site, not here).
  * Output is identical to ``chunk.text`` when ``refs`` is empty.
"""

from __future__ import annotations

import time
from pathlib import Path

from dikw_core.domains.info.render import render_chunk
from dikw_core.schemas import (
    AssetKind,
    AssetRecord,
    ChunkAssetRef,
    ChunkRecord,
)


def _asset(asset_id: str, stored_path: str) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=stored_path,
        original_paths=["original.png"],
        bytes=1,
        created_ts=time.time(),
    )


def _chunk(text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=1, doc_id="d1", seq=0, start=0, end=len(text), text=text
    )


def test_render_no_refs_returns_text_unchanged() -> None:
    chunk = _chunk("Just plain text, no refs.")
    assert render_chunk(chunk, refs=[], assets={}) == "Just plain text, no refs."


def test_render_single_markdown_image_substituted() -> None:
    text = "Pre ![arch](./original/arch.png) post"
    # Markdown ref span: from `![` (idx 4) through `)` (idx 33).
    chunk = _chunk(text)
    a = _asset("a1", "assets/aa/aabbccdd-arch.png")
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="arch",
            start_in_chunk=4,
            end_in_chunk=4 + len("![arch](./original/arch.png)"),
        )
    ]
    out = render_chunk(chunk, refs=refs, assets={"a1": a})
    assert out == "Pre ![arch](assets/aa/aabbccdd-arch.png) post"


def test_render_obsidian_wikilink_with_dimension_substituted() -> None:
    text = "See ![[arch.png|400]] for details"
    chunk = _chunk(text)
    a = _asset("a1", "assets/cc/ccddeeff-arch.png")
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="400",
            start_in_chunk=4,
            end_in_chunk=4 + len("![[arch.png|400]]"),
        )
    ]
    out = render_chunk(chunk, refs=refs, assets={"a1": a})
    # Output uses standard markdown form so any consumer renders it; the
    # Obsidian dimension alias is preserved as alt.
    assert out == "See ![400](assets/cc/ccddeeff-arch.png) for details"


def test_render_multiple_refs_in_order() -> None:
    text = "![a](./a.png) middle ![b](./b.png) end"
    chunk = _chunk(text)
    a = _asset("a1", "assets/00/00000001-a.png")
    b = _asset("a2", "assets/00/00000002-b.png")
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="a",
            start_in_chunk=0,
            end_in_chunk=len("![a](./a.png)"),
        ),
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a2",
            ord=1,
            alt="b",
            start_in_chunk=text.index("![b]"),
            end_in_chunk=text.index("![b]") + len("![b](./b.png)"),
        ),
    ]
    out = render_chunk(chunk, refs=refs, assets={"a1": a, "a2": b})
    assert out == (
        "![a](assets/00/00000001-a.png) middle ![b](assets/00/00000002-b.png) end"
    )


def test_render_unknown_asset_left_intact() -> None:
    """If an asset_id can't be resolved (missing from the assets map), the
    original reference text stays in place — defensive degradation."""
    text = "![arch](./arch.png)"
    chunk = _chunk(text)
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="missing",
            ord=0,
            alt="arch",
            start_in_chunk=0,
            end_in_chunk=len(text),
        )
    ]
    assert render_chunk(chunk, refs=refs, assets={}) == text


def test_render_with_project_root_emits_absolute_paths() -> None:
    """When project_root is given, stored_path is resolved against it so the
    rendered Markdown is portable to any consumer (e.g. an LLM prompt)."""
    text = "![arch](./arch.png)"
    chunk = _chunk(text)
    a = _asset("a1", "assets/aa/aabbccdd-arch.png")
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="arch",
            start_in_chunk=0,
            end_in_chunk=len(text),
        )
    ]
    out = render_chunk(
        chunk, refs=refs, assets={"a1": a}, project_root=Path("/tmp/vault")
    )
    assert out == "![arch](/tmp/vault/assets/aa/aabbccdd-arch.png)"


def test_render_does_not_mutate_input() -> None:
    text = "![arch](./arch.png)"
    chunk = _chunk(text)
    a = _asset("a1", "assets/aa/aabbccdd-arch.png")
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="arch",
            start_in_chunk=0,
            end_in_chunk=len(text),
        )
    ]
    _ = render_chunk(chunk, refs=refs, assets={"a1": a})
    # Original chunk and asset unchanged.
    assert chunk.text == text
    assert a.stored_path == "assets/aa/aabbccdd-arch.png"


def test_render_handles_unsorted_refs() -> None:
    """Defensive: refs may arrive in arbitrary order; render must still
    produce a coherent output by sorting on start_in_chunk before
    substitution."""
    text = "![a](./a.png) and ![b](./b.png)"
    chunk = _chunk(text)
    a = _asset("a1", "assets/01/01-a.png")
    b = _asset("a2", "assets/02/02-b.png")
    # Pass refs out of order on purpose.
    refs = [
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a2",
            ord=1,
            alt="b",
            start_in_chunk=text.index("![b]"),
            end_in_chunk=text.index("![b]") + len("![b](./b.png)"),
        ),
        ChunkAssetRef(
            chunk_id=1,
            asset_id="a1",
            ord=0,
            alt="a",
            start_in_chunk=0,
            end_in_chunk=len("![a](./a.png)"),
        ),
    ]
    out = render_chunk(chunk, refs=refs, assets={"a1": a, "a2": b})
    assert out == "![a](assets/01/01-a.png) and ![b](assets/02/02-b.png)"
