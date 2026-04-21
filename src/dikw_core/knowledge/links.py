"""Link graph for K (and W) layers.

Parses three kinds of links out of markdown bodies:

* ``[[Target]]`` / ``[[Target|alias]]`` / ``[[Target#anchor]]`` — Obsidian
  wikilinks. Target resolution looks first for an exact path match, then
  for a unique title match across K-layer pages.
* ``[text](relative/path.md)`` — standard Markdown links. URLs and
  fragment-only references are classified as ``url`` or dropped.
* Bare URLs in the body — captured as ``url`` links with no target
  resolution.

Parsing is deliberately forgiving: we want best-effort discovery, not a CSS
parser. The output is a list of ``LinkRecord``s ready for
``Storage.upsert_link`` and a list of ``UnresolvedLink``s that ``lint``
surfaces to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import LinkRecord, LinkType

_WIKILINK = re.compile(r"\[\[([^\]\|\n]+?)(?:\|([^\]\n]+?))?\]\]")
_MD_LINK = re.compile(r"(?<!\!)\[([^\]\n]+?)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL = re.compile(r"(?<![\[\(])\b(https?://[^\s\)]+)")


@dataclass(frozen=True)
class UnresolvedLink:
    """A wikilink whose target could not be resolved to a known path."""

    src_doc_id: str
    target_text: str  # raw text between ``[[ ... ]]``
    line: int


@dataclass(frozen=True)
class ParsedLink:
    """Intermediate structure before storage-path resolution."""

    kind: LinkType
    target: str          # for wikilinks: page title; for md: href; for url: the URL
    anchor: str | None
    line: int
    raw: str             # the matched substring (debug aid)


def parse_links(body: str) -> list[ParsedLink]:
    """Return every link discovered in ``body`` in source order."""
    results: list[ParsedLink] = []

    # Build a line-offset index so each match can report a 1-based line number.
    line_starts = _line_starts(body)

    # Wikilinks first so wikilink substrings inside body aren't mis-read as md links
    # (wikilink syntax uses `[[` which wouldn't match `(...)` anyway, so order is
    # mostly cosmetic — but we still keep one source-ordered list).
    matches = sorted(
        [("wikilink", m) for m in _WIKILINK.finditer(body)]
        + [("md", m) for m in _MD_LINK.finditer(body)]
        + [("url", m) for m in _URL.finditer(body)],
        key=lambda item: item[1].start(),
    )

    for kind, m in matches:
        line = _offset_to_line(m.start(), line_starts)
        if kind == "wikilink":
            raw_target = m.group(1).strip()
            target, anchor = _split_anchor(raw_target)
            results.append(
                ParsedLink(
                    kind=LinkType.WIKILINK,
                    target=target,
                    anchor=anchor,
                    line=line,
                    raw=m.group(0),
                )
            )
        elif kind == "md":
            href = m.group(2).strip()
            # Skip mailto and other pseudo-schemes quietly.
            if href.startswith(("mailto:", "tel:", "#")):
                continue
            if href.startswith(("http://", "https://")):
                target, anchor = _split_anchor(href)
                results.append(
                    ParsedLink(
                        kind=LinkType.URL,
                        target=target,
                        anchor=anchor,
                        line=line,
                        raw=m.group(0),
                    )
                )
            else:
                target, anchor = _split_anchor(href)
                results.append(
                    ParsedLink(
                        kind=LinkType.MARKDOWN,
                        target=target,
                        anchor=anchor,
                        line=line,
                        raw=m.group(0),
                    )
                )
        else:  # url
            target, anchor = _split_anchor(m.group(1))
            results.append(
                ParsedLink(
                    kind=LinkType.URL,
                    target=target,
                    anchor=anchor,
                    line=line,
                    raw=m.group(0),
                )
            )

    return results


def resolve_links(
    src_doc_id: str,
    links: list[ParsedLink],
    *,
    title_to_path: dict[str, str],
) -> tuple[list[LinkRecord], list[UnresolvedLink]]:
    """Turn ``ParsedLink``s into storage records.

    ``title_to_path`` maps a K/W page title to its wiki-relative path so
    wikilinks can be resolved deterministically when the title is unique.
    URLs and Markdown links get their target copied verbatim.
    """
    resolved: list[LinkRecord] = []
    unresolved: list[UnresolvedLink] = []

    for link in links:
        if link.kind is LinkType.WIKILINK:
            path = title_to_path.get(link.target) or title_to_path.get(link.target.lower())
            if path is None:
                unresolved.append(
                    UnresolvedLink(
                        src_doc_id=src_doc_id, target_text=link.raw, line=link.line
                    )
                )
                continue
            resolved.append(
                LinkRecord(
                    src_doc_id=src_doc_id,
                    dst_path=path,
                    link_type=LinkType.WIKILINK,
                    anchor=link.anchor,
                    line=link.line,
                )
            )
        else:
            resolved.append(
                LinkRecord(
                    src_doc_id=src_doc_id,
                    dst_path=link.target,
                    link_type=link.kind,
                    anchor=link.anchor,
                    line=link.line,
                )
            )

    return resolved, unresolved


def _split_anchor(s: str) -> tuple[str, str | None]:
    if "#" not in s:
        return s, None
    head, _, anchor = s.partition("#")
    return head, anchor or None


def _line_starts(body: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(body):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line(offset: int, line_starts: list[int]) -> int:
    # binary search would be faster but we rarely have huge docs in K layer
    line = 1
    for start in line_starts:
        if start > offset:
            return line - 1
        line += 1
    return len(line_starts)
