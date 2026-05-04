"""HTML source backend.

Uses stdlib ``html.parser`` so the engine gains HTML ingestion without a new
dependency. The parser:

* extracts ``<title>`` (falls back to first ``<h1>``, then the file stem);
* walks the tree converting block-level tags into newline-separated plain
  text and dropping ``<script>``/``<style>`` contents;
* normalises whitespace so diffs stay stable across re-ingests.

This is deliberately lightweight — good enough for notes saved from a
browser. Complex documents (multi-column PDFs, scraped SPA pages) would
warrant a proper reader in a later backend.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .base import ParsedDocument
from .markdown import content_hash

_WHITESPACE = re.compile(r"[ \t]+")
_BLANKLINES = re.compile(r"\n{3,}")

_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "aside",
    "nav", "main", "ul", "ol", "li", "pre", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "thead", "tbody", "tfoot",
    "hr", "br",
}
_SKIP_TAGS = {"script", "style", "noscript", "template"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title: str | None = None
        self._first_h1: str | None = None
        self._in_h1 = False
        self._h1_buffer: list[str] = []

    # ---- HTMLParser hooks ----------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if t == "title":
            self._in_title = True
            return
        if t in _HEADING_TAGS and t == "h1" and self._first_h1 is None:
            self._in_h1 = True
        if t == "br":
            self._parts.append("\n")
        elif t in _HEADING_TAGS:
            self._parts.append(f"\n\n{'#' * int(t[1])} ")
        elif t == "li":
            self._parts.append("\n- ")
        elif t in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if t == "title":
            self._in_title = False
            return
        if t == "h1" and self._in_h1:
            self._in_h1 = False
            if self._first_h1 is None:
                self._first_h1 = "".join(self._h1_buffer).strip() or None
            self._h1_buffer.clear()
        if t in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title = ((self.title or "") + data).strip()
            return
        if self._in_h1:
            self._h1_buffer.append(data)
        self._parts.append(data)

    # ---- consumer-facing accessors -------------------------------------

    def body(self) -> str:
        raw = "".join(self._parts)
        # collapse runs of inline whitespace, then runs of blank lines
        lines = [_WHITESPACE.sub(" ", line).strip() for line in raw.splitlines()]
        text = "\n".join(lines)
        text = _BLANKLINES.sub("\n\n", text).strip()
        return text + ("\n" if text else "")

    def resolved_title(self, stem: str) -> str:
        if self.title:
            return self.title
        if self._first_h1:
            return self._first_h1
        return stem


class HtmlBackend:
    """``SourceBackend`` impl for .html / .htm files."""

    extensions: tuple[str, ...] = (".html", ".htm")

    def parse(self, path: Path, *, rel_path: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(raw)
        extractor.close()
        body = extractor.body()
        title = extractor.resolved_title(Path(rel_path).stem)
        mtime = path.stat().st_mtime
        return ParsedDocument(
            path=rel_path,
            title=title,
            body=body,
            frontmatter={},
            hash=content_hash(body),
            mtime=mtime,
        )
