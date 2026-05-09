"""Tests for the per-rule lint fixers.

PR1 ships only ``broken_wikilink`` with a heuristic-only path: fuzzy match
the broken target against existing K-layer page titles and propose a
``[[link]]`` rewrite when the match is confident enough; otherwise the
fixer returns ``None`` so the orchestrator skips the issue (LLM stub
fallback lands in PR2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint import LintIssue
from dikw_core.domains.knowledge.lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
)
from dikw_core.domains.knowledge.lint_fixers.broken_wikilink import (
    BrokenWikilinkFixer,
)
from dikw_core.domains.knowledge.wiki import WikiPage, build_page

from .fakes import FakeLLM


def _make_page(title: str, body: str) -> WikiPage:
    return build_page(title=title, body=body, type_="concept")


def _ctx(*, pages: list[WikiPage], wiki_root: Path) -> FixerContext:
    return FixerContext(
        storage=None,  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        embedding=None,
        wiki_root=wiki_root,
        all_pages=pages,
    )


@pytest.mark.asyncio
async def test_fuzzy_match_above_threshold_proposes_update_page(
    tmp_path: Path,
) -> None:
    """A broken ``[[foo bar]]`` should rewrite to the existing ``Foo Bar``
    page when normalized titles match within the 0.85 ratio band."""
    wiki_root = tmp_path
    target_page = _make_page("Foo Bar", "# Foo Bar\nbody\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[foo  bar]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    assert isinstance(proposal, FixProposal)
    assert proposal.source == "heuristic"
    assert proposal.issue_path == src_page.path
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert isinstance(op, FixOperation)
    assert op.kind == "update_page"
    assert op.path == src_page.path
    assert op.expected_hash is not None  # we will hash the on-disk file
    assert op.new_body is not None
    # Rewrite should land an exact-title link.
    assert "[[Foo Bar]]" in op.new_body
    assert "[[foo  bar]]" not in op.new_body


@pytest.mark.asyncio
async def test_fuzzy_match_miss_returns_none_in_pr1(tmp_path: Path) -> None:
    """When no existing title is close enough, PR1 returns ``None``
    (LLM stub fallback ships in PR2)."""
    wiki_root = tmp_path
    other = _make_page("Completely Different", "# Completely Different\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[xyz123abc]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[other, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_match_excludes_self_page(tmp_path: Path) -> None:
    """A page must not propose a link to itself even if the broken
    target normalizes to the page's own title."""
    wiki_root = tmp_path
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[source page]].\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[source page]].\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[source page]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_extracts_target_from_detail_when_issue_lacks_explicit_target(
    tmp_path: Path,
) -> None:
    """The lint scanner formats detail as ``[[<target>]] has no matching ...``;
    the fixer must parse that to recover the broken target text."""
    wiki_root = tmp_path
    target_page = _make_page("Karpathy Rule", "# Karpathy Rule\nbody\n")
    src_page = _make_page(
        "Source",
        "# Source\n\nApply [[karpathy rule]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source\n---\n\n"
        "# Source\n\nApply [[karpathy rule]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[karpathy rule]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is not None
    assert "[[Karpathy Rule]]" in proposal.operations[0].new_body  # type: ignore[arg-type]


class _NullReporter:
    """Minimal ``ProgressReporter`` for fixer-unit tests."""

    def __init__(self) -> None:
        from dikw_core.progress import CancelToken

        self._token = CancelToken()

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        return self._token
