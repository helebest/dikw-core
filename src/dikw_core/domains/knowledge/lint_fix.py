"""Wiki-lint fix-proposal subsystem.

`run_lint` (in :mod:`lint`) reports four classes of K-layer hygiene issues
but never proposes how to fix them. This module adds a ``propose`` /
``apply`` pair so each lint issue can become a structured, reviewable,
applicable repair plan.

The contract is:

* :class:`Fixer` — Protocol implemented per ``LintKind``. Given a
  :class:`LintIssue` and a :class:`FixerContext`, returns a
  :class:`FixProposal` (or ``None`` if the issue isn't fixable).
* :func:`run_lint_propose` — orchestrator. Single task, serial loop:
  one ``ProgressEvent`` per issue, one ``LogEvent`` per skipped issue.
  Fixer-level failures don't fail the whole task — they accumulate in
  :attr:`FixProposalReport.skipped` and the loop moves on.
* :func:`run_lint_apply` — executor. Reads a :class:`FixProposalReport`
  produced earlier, optionally filters by ``pick`` / ``skip``, validates
  each :attr:`FixOperation.expected_hash` against the on-disk file
  bytes (concurrent-edit guard), then mutates ``wiki/`` via
  :func:`wiki.write_page` / unlink. Outgoing-link reconciliation rides
  the existing ``storage.replace_links_from`` machinery (PR #66).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ...providers.base import EmbeddingProvider, LLMProvider
from ...schemas import Layer, LinkRecord, LinkType
from ...storage.base import Storage
from .links import parse_links, resolve_links
from .lint import LintKind
from .wiki import WikiPage, now_iso, write_page


class FixOperation(BaseModel):
    """One page-level mutation in a fix proposal.

    Page-level (not line-level) is the right granularity here: the four
    rule kinds disagree on what "fix" means (rewrite a link, split into
    N pages, merge two pages, inject inbound links). A uniform
    "create / update / delete the whole page" vocabulary covers all
    cases without inventing a per-rule DSL — the cost is that even a
    one-line rewrite ships the full new body, which keeps apply trivial
    and lets an editor diff the proposal exactly like a normal commit.

    ``expected_hash`` is the sha256 of the file bytes the fixer
    observed; apply fails this op if the file has changed underneath
    (concurrent edit) and records the mismatch in ``ApplyReport.skipped``.
    Always ``None`` for ``create_page`` (the file shouldn't exist yet).
    """

    kind: Literal["create_page", "update_page", "delete_page"]
    path: str
    new_frontmatter: dict[str, Any] | None = None
    new_body: str | None = None
    expected_hash: str | None = None


class FixProposal(BaseModel):
    """One repair proposal targeting a single :class:`LintIssue`.

    The four issue fields are denormalised copies of the source
    :class:`LintIssue` (which is a frozen dataclass in :mod:`lint`,
    not a pydantic model). Embedding the values directly keeps the
    proposal record self-contained when the source lint result has
    long since been discarded — propose tasks ship to ``tasks.result``
    JSON dicts and apply tasks read them back days later.
    """

    proposal_id: str
    issue_kind: LintKind
    issue_path: str
    issue_detail: str
    issue_line: int | None = None
    operations: list[FixOperation]
    rationale: str
    source: Literal["heuristic", "llm"]


class FixProposalReport(BaseModel):
    proposals: list[FixProposal] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)


class ApplyReport(BaseModel):
    applied: list[FixOperation] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    wiki_paths_changed: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class FixerContext:
    """Per-task context handed to every fixer.

    ``storage`` and ``llm`` are optional because heuristic-only fixers
    (the PR1 broken_wikilink path) never touch them. Fixers that *do*
    need them (orphan_page, duplicate_title, non_atomic_page in later
    PRs) raise their own ``ValueError`` if asked to run without one.
    ``all_pages`` is pre-loaded by the orchestrator so each fixer does
    not re-read the K-layer page list."""

    storage: Storage | None
    llm: LLMProvider | None
    embedding: EmbeddingProvider | None
    wiki_root: Path
    all_pages: list[WikiPage]


class Fixer(Protocol):
    kind: LintKind

    async def propose(
        self,
        issue: Any,  # LintIssue (dataclass, no pydantic for the source type)
        ctx: FixerContext,
        reporter: Any,  # ProgressReporter
    ) -> FixProposal | None: ...


# Helpers shared across fixers ------------------------------------------------


_BROKEN_TARGET_RE = re.compile(r"\[\[([^\]]+)\]\]")


def extract_broken_target(detail: str) -> str | None:
    """Pull the ``[[<target>]]`` substring out of a lint ``detail`` string.

    The lint scanner formats every ``broken_wikilink`` issue as
    ``"[[<target>]] has no matching wiki page"``; we re-extract the
    target rather than re-parse the body so a fixer can act on the
    issue without reloading the source file.
    """
    m = _BROKEN_TARGET_RE.match(detail.strip())
    return m.group(1).strip() if m else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run_lint_propose(
    *,
    report: Any,  # LintReport — typed as Any to avoid circular import shenanigans
    rule: LintKind | None,
    limit: int,
    ctx: FixerContext,
    reporter: Any,  # ProgressReporter
    registry: dict[LintKind, Fixer] | None = None,
) -> FixProposalReport:
    """Single-task serial orchestrator: dispatch each lint issue to its
    registered :class:`Fixer`, collect proposals, accumulate skips.

    Failures inside one fixer never fail the whole task — they land in
    :attr:`FixProposalReport.skipped` so the apply step (or a human)
    can decide what to do. Cancellation is checked at the top of every
    iteration so a user clicking "stop" mid-loop bails cooperatively.
    """
    if registry is None:
        # Local import to avoid a top-of-module import cycle: the
        # ``lint_fixers`` package imports symbols from this module.
        from .lint_fixers import FIXER_REGISTRY

        registry = FIXER_REGISTRY

    issues = list(report.issues)
    if rule is not None:
        issues = [i for i in issues if i.kind == rule]
    issues = issues[:limit]
    total = len(issues)

    proposals: list[FixProposal] = []
    skipped: list[dict[str, Any]] = []

    for idx, issue in enumerate(issues):
        reporter.cancel_token().raise_if_cancelled()
        await reporter.progress(
            phase="lint_propose",
            current=idx,
            total=total,
            detail={"issue_kind": issue.kind, "path": issue.path},
        )
        fixer = registry.get(issue.kind)
        if fixer is None:
            skipped.append(
                {
                    "issue_index": idx,
                    "issue_path": issue.path,
                    "issue_kind": issue.kind,
                    "reason": f"no fixer registered for kind {issue.kind!r}",
                }
            )
            continue
        try:
            proposal = await fixer.propose(issue, ctx, reporter)
        except Exception as e:
            await reporter.log(
                "WARN",
                f"fixer for {issue.path} ({issue.kind}) raised: {e}",
            )
            skipped.append(
                {
                    "issue_index": idx,
                    "issue_path": issue.path,
                    "issue_kind": issue.kind,
                    "reason": f"fixer raised: {e}",
                }
            )
            continue
        if proposal is None:
            skipped.append(
                {
                    "issue_index": idx,
                    "issue_path": issue.path,
                    "issue_kind": issue.kind,
                    "reason": "fixer returned None",
                }
            )
            continue
        proposals.append(proposal)

    # Final progress event — total/total — so subscribers see the loop
    # completed even when the last issue was skipped.
    if total:
        await reporter.progress(
            phase="lint_propose",
            current=total,
            total=total,
            detail={"done": True},
        )

    return FixProposalReport(proposals=proposals, skipped=skipped)


def _wiki_doc_id(path: str) -> str:
    """Mirror of ``api._doc_id_for(Layer.WIKI, path)`` without the cycle.

    The on-disk format uses ``"<layer>:<normalized_path>"`` as the
    canonical id; we re-implement here so :mod:`lint_fix` doesn't need
    to import from :mod:`api` (the dependency arrow points the other
    way: ``api`` may import knowledge, not the reverse).
    """
    from ..data.path_norm import normalize_path

    return f"{Layer.WIKI.value}:{normalize_path(path)}"


def _build_page_from_op(op: FixOperation) -> WikiPage:
    """Materialise a :class:`WikiPage` from an op's frontmatter + body.

    Missing required fields (``id`` / ``type`` / ``created`` / ``updated``)
    are filled in with sensible defaults so a fixer that doesn't carry
    e.g. a stable ``id`` (PR2's stub-page LLM fallback) can still be
    persisted. ``write_page`` uses these to assemble the YAML header.
    """
    if op.new_body is None:
        raise ValueError(f"op {op.kind} for {op.path} missing new_body")
    fm: dict[str, Any] = dict(op.new_frontmatter or {})
    title = str(fm.pop("title", Path(op.path).stem.replace("-", " ").title()))
    type_ = str(fm.pop("type", "note"))
    page_id = str(fm.pop("id", f"K-{abs(hash(op.path)) % 0xFFFFFFFFFFFF:012x}"))
    now = now_iso()
    created = str(fm.pop("created", now))
    updated = str(fm.pop("updated", now))
    tags = list(fm.pop("tags", []) or [])
    sources = list(fm.pop("sources", []) or [])
    return WikiPage(
        path=op.path,
        id=page_id,
        type=type_,
        title=title,
        body=op.new_body,
        tags=tags,
        sources=sources,
        created=created,
        updated=updated,
        extras=fm,
    )


async def run_lint_apply(
    *,
    proposal_report: FixProposalReport,
    storage: Storage,
    wiki_root: Path,
    pick: list[int] | None = None,
    skip: list[int] | None = None,
    reporter: Any,  # ProgressReporter
) -> ApplyReport:
    """Mutate ``wiki/`` per a previously-produced :class:`FixProposalReport`.

    PR1 scope: write/unlink files + reconcile outgoing wikilinks for
    updated/created pages + ``deactivate_document`` for deletes. We
    intentionally do not re-chunk / re-embed — that's a follow-up
    ``dikw ingest``'s job (it'll see ``doc.hash`` mismatch and
    re-index). Keeping apply provider-free means a heuristic-only
    proposal can land without an embedder configured.

    ``pick`` / ``skip`` filter the proposal list by index. Both may be
    set; pick is applied first, then skip removes from that subset.
    """
    proposals = list(proposal_report.proposals)
    if pick is not None:
        pick_set = set(pick)
        proposals = [p for i, p in enumerate(proposal_report.proposals) if i in pick_set]
    if skip is not None:
        skip_set = set(skip)
        if pick is not None:
            kept = [(i, p) for i, p in enumerate(proposal_report.proposals)
                    if i in set(pick) and i not in skip_set]
        else:
            kept = [(i, p) for i, p in enumerate(proposal_report.proposals)
                    if i not in skip_set]
        proposals = [p for _, p in kept]

    # Pre-load K-layer doc rows for path→doc_id and title→path resolution.
    docs = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    path_to_doc_id: dict[str, str] = {d.path: d.doc_id for d in docs}
    title_to_path: dict[str, str] = {}
    for d in docs:
        if d.title and d.title not in title_to_path:
            title_to_path[d.title] = d.path

    applied: list[FixOperation] = []
    skipped: list[dict[str, Any]] = []
    paths_changed: set[str] = set()
    deleted_paths: set[str] = set()

    total_ops = sum(len(p.operations) for p in proposals)
    op_counter = 0

    for proposal in proposals:
        for op in proposal.operations:
            op_counter += 1
            reporter.cancel_token().raise_if_cancelled()
            await reporter.progress(
                phase="lint_apply",
                current=op_counter,
                total=total_ops,
                detail={"op": op.kind, "path": op.path},
            )
            outcome = await _apply_one_op(
                op=op,
                storage=storage,
                wiki_root=wiki_root,
                proposal_id=proposal.proposal_id,
                path_to_doc_id=path_to_doc_id,
            )
            if outcome["ok"]:
                applied.append(op)
                if op.kind == "delete_page":
                    deleted_paths.add(op.path)
                else:
                    paths_changed.add(op.path)
            else:
                skipped.append(outcome["skip"])

    # Reconcile outgoing wikilinks for every still-extant changed page.
    for path in sorted(paths_changed):
        abs_path = (wiki_root / path).resolve()
        if not abs_path.is_file():
            continue
        body = abs_path.read_text(encoding="utf-8")
        # Strip a leading frontmatter block — link parsing operates on body only.
        body_only = _strip_frontmatter(body)
        doc_id = path_to_doc_id.get(path) or _wiki_doc_id(path)
        parsed = parse_links(body_only)
        resolved, _unresolved = resolve_links(
            doc_id, parsed, title_to_path=title_to_path
        )
        # Replace_links_from requires the source doc row to exist. For
        # a freshly-created page (not in path_to_doc_id) we skip — the
        # next ``dikw ingest`` will pick it up and reconcile.
        if path in path_to_doc_id:
            await storage.replace_links_from(doc_id, resolved)

    return ApplyReport(
        applied=applied,
        skipped=skipped,
        wiki_paths_changed=sorted(paths_changed | deleted_paths),
    )


async def _apply_one_op(
    *,
    op: FixOperation,
    storage: Storage,
    wiki_root: Path,
    proposal_id: str,
    path_to_doc_id: dict[str, str],
) -> dict[str, Any]:
    """Execute one op; return ``{"ok": True}`` or ``{"ok": False, "skip": {...}}``."""
    abs_path = (wiki_root / op.path).resolve()

    if op.kind in ("update_page", "delete_page"):
        if not abs_path.is_file():
            return _skip(proposal_id, op, "file not found on disk")
        if op.expected_hash:
            actual = file_sha256(abs_path)
            if actual != op.expected_hash:
                return _skip(
                    proposal_id, op,
                    f"hash mismatch — concurrent edit detected (expected {op.expected_hash[:8]}…, got {actual[:8]}…)",
                )

    if op.kind == "create_page":
        if abs_path.exists():
            return _skip(proposal_id, op, "file already exists at create_page path")
        try:
            page = _build_page_from_op(op)
            write_page(wiki_root, page)
        except (OSError, ValueError) as e:
            return _skip(proposal_id, op, f"write_page failed: {e}")
        return {"ok": True}

    if op.kind == "update_page":
        try:
            page = _build_page_from_op(op)
            write_page(wiki_root, page)
        except (OSError, ValueError) as e:
            return _skip(proposal_id, op, f"write_page failed: {e}")
        return {"ok": True}

    if op.kind == "delete_page":
        try:
            abs_path.unlink()
        except OSError as e:
            return _skip(proposal_id, op, f"unlink failed: {e}")
        doc_id = path_to_doc_id.get(op.path)
        if doc_id is not None:
            await storage.deactivate_document(doc_id)
        return {"ok": True}

    return _skip(proposal_id, op, f"unknown op kind {op.kind!r}")


def _skip(proposal_id: str, op: FixOperation, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "skip": {
            "proposal_id": proposal_id,
            "op": op.kind,
            "path": op.path,
            "reason": reason,
        },
    }


_FRONTMATTER_FENCE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block so link parsing operates
    only on the body. Mirrors ``frontmatter.load`` without re-parsing
    the YAML — apply has already validated the file."""
    return _FRONTMATTER_FENCE.sub("", text, count=1)


# Type-hint indirection so static checkers see ``LinkRecord`` is used.
_ = LinkRecord, LinkType
