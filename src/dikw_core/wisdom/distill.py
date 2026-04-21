"""Wisdom-layer distillation.

Parses the LLM's response to ``prompts/distill.md`` into a list of
``WisdomCandidate`` records. The LLM emits zero or more ``<wisdom>`` blocks;
each block carries a kind, YAML front-matter with ``confidence`` and an
``evidence`` list, and a markdown body whose first heading is the title.

The "≥2 evidence items" invariant is enforced here — blocks that don't
meet the bar are discarded with a reason, and the caller logs or surfaces
the discard count.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..knowledge.wiki import slugify
from ..schemas import WisdomEvidence, WisdomKind

_WISDOM_BLOCK = re.compile(
    r"<wisdom\s+([^>]*?)>\s*(.*?)\s*</wisdom>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.DOTALL)
_ATX_TITLE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", flags=re.MULTILINE)

MIN_EVIDENCE = 2


class DistillParseError(RuntimeError):
    """Raised when no valid ``<wisdom>`` block exists in the LLM response."""


@dataclass(frozen=True)
class WisdomCandidate:
    """An un-persisted wisdom proposal produced by the LLM."""

    item_id: str
    kind: WisdomKind
    title: str
    body: str
    confidence: float
    evidence: list[WisdomEvidence]


@dataclass
class DistillParseResult:
    """Outcome of parsing a single LLM response."""

    candidates: list[WisdomCandidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (title, reason)

    @property
    def ok(self) -> bool:
        return bool(self.candidates)


def make_wisdom_id(kind: str, title: str) -> str:
    digest = hashlib.blake2b(f"{kind}:{title}".encode(), digest_size=6).hexdigest()
    return f"W-{digest}"


def parse_distill_response(raw: str) -> DistillParseResult:
    """Extract every valid ``<wisdom>`` block from ``raw``."""
    result = DistillParseResult()
    blocks = list(_WISDOM_BLOCK.finditer(raw))
    if not blocks:
        return result

    for m in blocks:
        attrs = dict(_ATTR.findall(m.group(1)))
        inner = m.group(2)

        kind_str = (attrs.get("kind") or "").strip().lower()
        if kind_str not in {k.value for k in WisdomKind}:
            result.rejected.append(("<unknown>", f"invalid kind: {kind_str!r}"))
            continue
        kind = WisdomKind(kind_str)

        fm_match = _FRONTMATTER.match(inner)
        if fm_match is None:
            result.rejected.append(("<unknown>", "missing YAML front-matter"))
            continue
        try:
            fm: dict[str, Any] = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            result.rejected.append(("<unknown>", f"invalid YAML: {e}"))
            continue
        if not isinstance(fm, dict):
            result.rejected.append(("<unknown>", "front-matter must be a mapping"))
            continue
        body = fm_match.group(2).lstrip("\n")

        title_match = _ATX_TITLE.search(body)
        if title_match is None:
            result.rejected.append(("<unknown>", "missing `# Title` heading"))
            continue
        title = title_match.group(1).strip()

        raw_evidence = fm.get("evidence") or []
        if not isinstance(raw_evidence, list):
            result.rejected.append((title, "evidence must be a list"))
            continue

        evidence: list[WisdomEvidence] = []
        for ev in raw_evidence:
            if not isinstance(ev, dict):
                continue
            doc_path = str(ev.get("doc", "")).strip()
            excerpt = str(ev.get("excerpt", "")).strip()
            line = ev.get("line")
            if not doc_path or not excerpt:
                continue
            evidence.append(
                WisdomEvidence(
                    doc_id=doc_path,  # resolved to a real doc_id by caller
                    excerpt=excerpt[:1024],
                    line=int(line) if isinstance(line, int) else None,
                )
            )

        if len(evidence) < MIN_EVIDENCE:
            result.rejected.append(
                (title, f"only {len(evidence)} evidence entries (need ≥ {MIN_EVIDENCE})")
            )
            continue

        try:
            confidence = float(fm.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        result.candidates.append(
            WisdomCandidate(
                item_id=make_wisdom_id(kind.value, title),
                kind=kind,
                title=title,
                body=body.rstrip() + "\n",
                confidence=confidence,
                evidence=evidence,
            )
        )

    return result


def candidate_path(kind: WisdomKind, title: str) -> str:
    """Return the wiki-relative path where a candidate is persisted."""
    return f"wisdom/_candidates/{kind.value}-{slugify(title)}.md"
