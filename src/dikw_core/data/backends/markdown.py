"""Markdown source backend.

Parses a markdown file into a ``ParsedMarkdown`` record: front-matter dict,
body text (with front-matter stripped), title, and a stable content hash.

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
from pydantic import BaseModel, Field

_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


class ParsedMarkdown(BaseModel):
    """In-memory representation of a parsed markdown file."""

    path: str
    title: str
    body: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    hash: str
    mtime: float


def content_hash(body: str) -> str:
    """SHA-256 of the raw body; stable across runs so D-layer rows are idempotent."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _first_heading(body: str) -> str | None:
    m = _ATX_HEADING.search(body)
    return m.group(1).strip() if m else None


def parse_text(*, path: str, text: str, mtime: float) -> ParsedMarkdown:
    """Parse raw markdown text. Exposed so callers can test without filesystem I/O."""
    post = frontmatter.loads(text)
    body = post.content
    fm: dict[str, Any] = dict(post.metadata)

    title = fm.get("title") or _first_heading(body) or Path(path).stem

    return ParsedMarkdown(
        path=path,
        title=str(title),
        body=body,
        frontmatter=fm,
        hash=content_hash(body),
        mtime=mtime,
    )


def parse_file(path: Path, *, rel_path: str | None = None) -> ParsedMarkdown:
    """Read ``path`` and return a ``ParsedMarkdown``. ``rel_path`` becomes the D-layer path."""
    text = path.read_text(encoding="utf-8")
    mtime = path.stat().st_mtime
    return parse_text(path=rel_path or str(path), text=text, mtime=mtime)
