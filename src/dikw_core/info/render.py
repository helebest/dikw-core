"""Layer-3 chunk rendering: substitute original image paths with engine paths.

After ingest, ``chunks.text`` keeps the user-written reference syntax
verbatim — ``![arch](./diagrams/arch.png)`` — so source-fidelity is
preserved (Layer 1). The structural mapping from "this chunk character
range references this asset" lives in ``chunk_asset_refs`` (Layer 2).

When a downstream consumer (CLI, MCP tool, LLM prompt builder, web
preview) needs a self-contained renderable Markdown snippet, it calls
``render_chunk`` to produce one with each original reference rewritten
to the engine-managed ``stored_path``. The substitution is
position-based (``start_in_chunk`` / ``end_in_chunk``) — no regex
re-parsing — so it handles every reference syntax variant uniformly,
including Obsidian's ``![[file|400]]`` dimension alias which becomes
standard ``![400](assets/...)`` in the output.

This module is pure: no I/O, no storage handle, no asset bytes loaded.
The asset metadata it needs is passed in as a ``Mapping[str,
AssetRecord]`` keyed by ``asset_id``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..schemas import AssetRecord, ChunkAssetRef, ChunkRecord


def render_chunk(
    chunk: ChunkRecord,
    *,
    refs: Sequence[ChunkAssetRef],
    assets: Mapping[str, AssetRecord],
    project_root: Path | None = None,
) -> str:
    """Return ``chunk.text`` with every reference in ``refs`` substituted
    for an engine-managed Markdown image link.

    Refs whose ``asset_id`` isn't in ``assets`` are left in place verbatim
    (the original reference text from ``chunk.text`` survives) — defensive
    degradation; the call site should log if it wants visibility into
    missing-asset cases.

    When ``project_root`` is given, output paths are absolute
    (``project_root / stored_path``) so the rendered Markdown is portable
    independent of the consumer's working directory; otherwise the
    relative ``stored_path`` (``assets/<h2>/<h8>-<name>.<ext>``) is used.

    The output reference uses standard ``![alt](path)`` Markdown form
    even when the source was an Obsidian wikilink — the wikilink alias
    becomes the alt text, so any consumer that renders Markdown displays
    the image correctly.
    """
    if not refs:
        return chunk.text

    # Refs may arrive in arbitrary order; sort by start so substitution
    # never overlaps. Skipping refs whose asset isn't resolvable lets the
    # original reference text survive (defensive).
    spans: list[tuple[int, int, str]] = []
    for r in sorted(refs, key=lambda x: x.start_in_chunk):
        asset = assets.get(r.asset_id)
        if asset is None:
            continue
        path = (
            str(project_root / asset.stored_path)
            if project_root is not None
            else asset.stored_path
        )
        replacement = f"![{r.alt}]({path})"
        spans.append((r.start_in_chunk, r.end_in_chunk, replacement))

    if not spans:
        return chunk.text

    # Splice in the substitutions.
    out_parts: list[str] = []
    cursor = 0
    text = chunk.text
    for start, end, replacement in spans:
        out_parts.append(text[cursor:start])
        out_parts.append(replacement)
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts)


__all__ = ["render_chunk"]
