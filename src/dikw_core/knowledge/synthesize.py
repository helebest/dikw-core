"""Synthesize K-layer wiki pages from D-layer source documents.

The LLM emits one ``<page>`` XML block per source; a small parser turns
that into a ``WikiPage`` the engine persists. Keeping the LLM output
format explicit (XML tags wrapping front-matter + markdown) avoids JSON
escaping pain and remains easy to unit-test with a ``FakeLLM``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

from .wiki import WikiPage, build_page, now_iso

_PAGE_BLOCK = re.compile(
    r"<page\s+([^>]+?)>\s*(.*?)\s*</page>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.DOTALL)
_ATX_TITLE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", flags=re.MULTILINE)


class SynthesisError(RuntimeError):
    """The LLM response didn't contain a usable ``<page>`` block."""


@dataclass(frozen=True)
class SynthesisOutcome:
    page: WikiPage
    source_path: str


def parse_synthesis_response(
    raw: str, *, source_path: str
) -> WikiPage:
    """Extract a ``WikiPage`` from the LLM's ``<page>`` block."""
    m = _PAGE_BLOCK.search(raw)
    if m is None:
        raise SynthesisError(
            f"no <page>…</page> block found in LLM response for {source_path}"
        )

    attrs = dict(_ATTR.findall(m.group(1)))
    inner = m.group(2)

    type_ = attrs.get("type", "note").strip().lower()
    if type_ not in ("concept", "entity", "note"):
        type_ = "note"

    fm_match = _FRONTMATTER.match(inner)
    if fm_match is None:
        frontmatter_yaml: dict[str, Any] = {}
        body = inner.strip()
    else:
        try:
            parsed_fm = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            raise SynthesisError(f"invalid YAML front-matter from LLM: {e}") from e
        if not isinstance(parsed_fm, dict):
            raise SynthesisError("front-matter must be a YAML mapping")
        frontmatter_yaml = parsed_fm
        body = fm_match.group(2).lstrip("\n")

    title_match = _ATX_TITLE.search(body)
    if title_match is None:
        raise SynthesisError("no ATX `# Title` found in page body")
    title = title_match.group(1).strip()

    tags = frontmatter_yaml.pop("tags", [])
    if not isinstance(tags, list):
        tags = []

    path = attrs.get("path") or None
    return build_page(
        title=title,
        body=body.rstrip() + "\n",
        type_=type_,
        tags=[str(t) for t in tags],
        sources=[source_path],
        path=path,
        extras={k: v for k, v in frontmatter_yaml.items() if k not in {"tags"}},
    )


def build_prompt(template: str, *, source_path: str, source_body: str) -> str:
    return template.format(source_path=source_path, source_body=source_body)


def touch(page: WikiPage) -> WikiPage:
    """Return a copy of ``page`` with ``updated`` bumped to now."""
    # WikiPage is frozen; dataclasses.replace would work but keeps import cost low.
    from dataclasses import replace

    return replace(page, updated=now_iso())
