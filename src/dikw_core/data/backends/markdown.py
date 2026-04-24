"""Markdown source backend.

Parses a markdown file into a ``ParsedDocument``: front-matter dict, body
text (with front-matter stripped), title, and a stable content hash.

Title resolution order:
1. ``title:`` in front-matter
2. First ATX heading (``# Title``) in the body
3. File stem (``my-note`` -> ``my-note``)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import frontmatter

from ...schemas import AssetRef
from .base import ParsedDocument

# Backwards-compatible alias for existing callers.
ParsedMarkdown = ParsedDocument

_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Standard markdown image: ![alt](path) with optional "title" attribute.
# Path may contain spaces (Obsidian-style ``![](My Diagram.png)``); the
# lookahead pins the lazy match at the position where the title or the
# closing paren begins, so the path captures everything in between
# without swallowing the title or trailing whitespace.
_IMG_MD = re.compile(
    r"!\[([^\]]*)\]\(\s*([^)\n]+?)"
    r"(?=\s+\"[^\"\n]*\"\s*\)|\s*\))"
    r"(?:\s+\"[^\"\n]*\")?\s*\)"
)

# Obsidian image embed: ![[file]] with optional |alias (caption or display
# dimension like ``150`` / ``150x100``). Both the path part and the alias
# part are inert against ``]`` and ``|`` so neighbouring embeds don't bleed.
_IMG_WIKILINK = re.compile(r"!\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")


def content_hash(body: str) -> str:
    """SHA-256 of the raw body; stable across runs so D-layer rows are idempotent."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _first_heading(body: str) -> str | None:
    m = _ATX_HEADING.search(body)
    return m.group(1).strip() if m else None


def extract_image_refs(body: str) -> list[AssetRef]:
    """Find every image embed in ``body`` and return them in source order.

    Both the standard markdown ``![alt](path)`` form (with optional
    ``"title"`` attribute) and the Obsidian wikilink ``![[file|alias]]``
    form are recognized. ``start`` / ``end`` are character offsets that
    cover the literal reference syntax so the chunker can treat them as
    atomic spans and downstream consumers can rewrite the reference in
    place without touching surrounding prose.

    Remote URLs are still emitted as ``AssetRef`` here; the materialize
    layer is what actually decides to skip non-local references.
    """
    refs: list[AssetRef] = []
    for m in _IMG_MD.finditer(body):
        refs.append(
            AssetRef(
                original_path=m.group(2),
                alt=m.group(1) or "",
                start=m.start(),
                end=m.end(),
                syntax="markdown",
            )
        )
    for m in _IMG_WIKILINK.finditer(body):
        refs.append(
            AssetRef(
                original_path=m.group(1),
                alt=m.group(2) or "",
                start=m.start(),
                end=m.end(),
                syntax="wikilink",
            )
        )
    refs.sort(key=lambda r: r.start)
    return refs


def parse_text(*, path: str, text: str, mtime: float) -> ParsedDocument:
    """Parse raw markdown text. Exposed so callers can test without filesystem I/O."""
    post = frontmatter.loads(text)
    body = post.content
    fm: dict[str, Any] = dict(post.metadata)

    title = fm.get("title") or _first_heading(body) or Path(path).stem

    return ParsedDocument(
        path=path,
        title=str(title),
        body=body,
        frontmatter=fm,
        hash=content_hash(body),
        mtime=mtime,
        asset_refs=extract_image_refs(body),
    )


def parse_file(path: Path, *, rel_path: str | None = None) -> ParsedDocument:
    """Read ``path`` and return a ``ParsedDocument``. ``rel_path`` becomes the D-layer path."""
    text = path.read_text(encoding="utf-8")
    mtime = path.stat().st_mtime
    return parse_text(path=rel_path or str(path), text=text, mtime=mtime)


class MarkdownBackend:
    """``SourceBackend`` impl for .md / .markdown files."""

    extensions: tuple[str, ...] = (".md", ".markdown")

    def parse(self, path: Path, *, rel_path: str) -> ParsedDocument:
        return parse_file(path, rel_path=rel_path)
