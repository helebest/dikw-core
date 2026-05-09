"""Fixer for ``broken_wikilink`` issues.

PR1 path: pure heuristic. Normalize every K-layer page title (lower +
strip puncs + collapse whitespace) and the broken target the lint
scanner recovered, fuzzy-match via :class:`difflib.SequenceMatcher`,
and propose an in-place ``[[link]]`` rewrite when the ratio crosses
:data:`HEURISTIC_RATIO_THRESHOLD`.

A miss returns ``None`` so the orchestrator records ``"skipped"`` and
moves on. PR2 plugs an LLM-stub-page fallback into the same miss
branch; the fixer's contract — return ``FixProposal`` or ``None`` —
does not change.
"""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher
from typing import Any

import frontmatter

from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    extract_broken_target,
    file_sha256,
)

#: A page rewrites the broken link only when the closest existing title
#: clears this similarity ratio. Calibrated empirically: 0.85 catches
#: case / whitespace / punctuation drift while rejecting mid-word
#: substrings (a 3-character broken target hits 0.85 against many
#: titles by chance, so we also gate on ``len(target) >= 4``).
HEURISTIC_RATIO_THRESHOLD = 0.85
_MIN_TARGET_LEN = 4

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def _normalize_title(s: str) -> str:
    lowered = s.lower()
    no_puncts = _NON_ALNUM.sub(" ", lowered)
    return _WS.sub(" ", no_puncts).strip()


class BrokenWikilinkFixer:
    kind: LintKind = "broken_wikilink"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        target = extract_broken_target(issue.detail)
        if not target or len(_normalize_title(target)) < _MIN_TARGET_LEN:
            return None

        # Rank candidate titles by similarity, excluding the source page itself
        # so a self-referential broken link doesn't collapse onto its own title.
        target_norm = _normalize_title(target)
        best_title: str | None = None
        best_ratio = 0.0
        for page in ctx.all_pages:
            if page.path == issue.path:
                continue
            title = page.title
            if not title:
                continue
            ratio = SequenceMatcher(
                None, target_norm, _normalize_title(title)
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_title = title

        if best_title is None or best_ratio < HEURISTIC_RATIO_THRESHOLD:
            return None

        abs_path = (ctx.wiki_root / issue.path).resolve()
        if not abs_path.is_file():
            # Source file vanished between lint scan and fix proposal —
            # skip rather than synthesising an op against missing state.
            return None

        post = frontmatter.load(str(abs_path))
        body = post.content
        old = f"[[{target}]]"
        new = f"[[{best_title}]]"
        if old not in body:
            # The on-disk body uses an alias / anchor / casing the
            # detail string can't reproduce. Skip in PR1 — PR2 will
            # add a regex-aware replacement covering ``[[t|alias]]``.
            return None
        new_body = body.replace(old, new)

        op = FixOperation(
            kind="update_page",
            path=issue.path,
            new_frontmatter=dict(post.metadata),
            new_body=new_body,
            expected_hash=file_sha256(abs_path),
        )
        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            issue_kind=issue.kind,
            issue_path=issue.path,
            issue_detail=issue.detail,
            issue_line=issue.line,
            operations=[op],
            rationale=(
                f"fuzzy match — '{target}' → '{best_title}' "
                f"(ratio {best_ratio:.2f})"
            ),
            source="heuristic",
        )
