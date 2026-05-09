"""Tests for ``run_lint_apply`` — the executor that mutates ``wiki/``
based on a :class:`FixProposalReport`.

PR1 design: apply only touches disk + outgoing-link reconciliation. It
intentionally skips re-chunking / re-embedding so apply stays
provider-free; a follow-up ``dikw ingest`` reconciles ``doc.hash``
and chunks/embeddings on the next pass.

Storage-touching tests are parametrised over the ``parametrized_storage``
fixture so they auto-run against Postgres when ``DIKW_TEST_POSTGRES_DSN``
is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint_fix import (
    ApplyReport,
    FixOperation,
    FixProposal,
    FixProposalReport,
    file_sha256,
    run_lint_apply,
)
from dikw_core.schemas import DocumentRecord, Layer, LinkType
from dikw_core.storage.base import Storage


@dataclass
class _NullReporter:
    token: Any = None

    async def progress(self, **_: Any) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        from dikw_core.progress import CancelToken
        if self.token is None:
            self.token = CancelToken()
        return self.token


def _wiki_doc_id(path: str) -> str:
    """Mirror of ``api._doc_id_for(Layer.WIKI, path)`` without the cycle."""
    from dikw_core.domains.data.path_norm import normalize_path

    return f"wiki:{normalize_path(path)}"


async def _seed_page(
    *, storage: Storage, wiki_root: Path,
    path: str, title: str, body: str,
) -> str:
    """Write a page on disk + register a DocumentRecord. Returns the doc_id."""
    abs_path = wiki_root / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = f"---\ntitle: {title}\n---\n\n{body}"
    abs_path.write_text(body_text, encoding="utf-8")
    doc_id = _wiki_doc_id(path)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id, path=path, title=title,
            hash=f"hash-{path}", mtime=0.0,
            layer=Layer.WIKI, active=True,
        )
    )
    return doc_id


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.asyncio
async def test_update_page_writes_new_body_and_reconciles_links(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    target_doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/foo-bar.md", title="Foo Bar",
        body="# Foo Bar\nbody\n",
    )
    src_doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/source.md", title="Source",
        body="# Source\n\nSee [[foo  bar]] for context.\n",
    )
    abs_src = wiki_root / "wiki/concepts/source.md"
    expected_hash = file_sha256(abs_src)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/concepts/source.md",
        issue_detail="[[foo  bar]] has no matching wiki page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="wiki/concepts/source.md",
                new_frontmatter={"title": "Source"},
                new_body="# Source\n\nSee [[Foo Bar]] for context.\n",
                expected_hash=expected_hash,
            )
        ],
        rationale="fuzzy match",
        source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert isinstance(report, ApplyReport)
    assert len(report.applied) == 1
    assert report.skipped == []
    assert "wiki/concepts/source.md" in report.wiki_paths_changed

    # File was rewritten with new body.
    rewritten = abs_src.read_text(encoding="utf-8")
    assert "[[Foo Bar]]" in rewritten
    assert "[[foo  bar]]" not in rewritten

    # Links table: outgoing edge from source → foo-bar should now exist.
    links = await storage.links_from(src_doc_id)
    wikilinks = [link_row for link_row in links if link_row.link_type is LinkType.WIKILINK]
    assert len(wikilinks) == 1
    assert wikilinks[0].dst_path == "wiki/concepts/foo-bar.md"
    _ = target_doc_id  # held alive for the upsert side-effect


@pytest.mark.asyncio
async def test_hash_mismatch_skips_op_and_does_not_write_file(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/source.md", title="Source",
        body="# Source\n\nSee [[foo]] here.\n",
    )
    abs_src = wiki_root / "wiki/source.md"
    body_before = abs_src.read_text(encoding="utf-8")

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
        issue_detail="[[foo]] has no matching wiki page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="wiki/source.md",
                new_frontmatter={"title": "Source"},
                new_body="# Source\n\nDifferent body\n",
                # Wrong expected_hash — simulates concurrent edit.
                expected_hash="0" * 64,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert report.applied == []
    assert len(report.skipped) == 1
    assert "hash" in report.skipped[0]["reason"].lower()
    # File untouched.
    assert abs_src.read_text(encoding="utf-8") == body_before


@pytest.mark.asyncio
async def test_create_page_writes_new_file(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    new_path = "wiki/concepts/brand-new.md"
    abs_new = wiki_root / new_path
    assert not abs_new.exists()

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
        issue_detail="stub",
        operations=[
            FixOperation(
                kind="create_page",
                path=new_path,
                new_frontmatter={
                    "id": "K-newpage",
                    "type": "concept",
                    "title": "Brand New",
                    "created": "2026-05-09T00:00:00+00:00",
                    "updated": "2026-05-09T00:00:00+00:00",
                },
                new_body="# Brand New\n\nStub page.\n",
                expected_hash=None,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert len(report.applied) == 1
    assert report.skipped == []
    assert abs_new.is_file()
    content = abs_new.read_text(encoding="utf-8")
    assert "Brand New" in content


@pytest.mark.asyncio
async def test_create_page_skipped_when_file_already_exists(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/exists.md", title="Exists",
        body="# Exists\n",
    )

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
        issue_detail="stub",
        operations=[
            FixOperation(
                kind="create_page",
                path="wiki/concepts/exists.md",
                new_frontmatter={"title": "Exists"},
                new_body="# Exists\nWill not overwrite\n",
                expected_hash=None,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "exist" in report.skipped[0]["reason"].lower()


@pytest.mark.asyncio
async def test_delete_page_unlinks_file_and_deactivates_doc(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/dead.md", title="Dead",
        body="# Dead\n",
    )
    abs_dead = wiki_root / "wiki/concepts/dead.md"
    expected_hash = file_sha256(abs_dead)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="duplicate_title",
        issue_path="wiki/concepts/dead.md",
        issue_detail="dup",
        operations=[
            FixOperation(
                kind="delete_page",
                path="wiki/concepts/dead.md",
                expected_hash=expected_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert not abs_dead.exists()

    # Doc should be deactivated.
    actives = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    assert all(d.doc_id != doc_id for d in actives)


@pytest.mark.asyncio
async def test_pick_filter_applies_only_picked_proposals(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/foo.md", title="Foo",
        body="# Foo\n",
    )
    abs_foo = wiki_root / "wiki/foo.md"
    foo_hash = file_sha256(abs_foo)

    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/bar.md", title="Bar",
        body="# Bar\n",
    )
    abs_bar = wiki_root / "wiki/bar.md"
    bar_hash = file_sha256(abs_bar)
    bar_body_before = abs_bar.read_text(encoding="utf-8")

    p_foo = FixProposal(
        proposal_id="pa",
        issue_kind="broken_wikilink",
        issue_path="wiki/foo.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="wiki/foo.md",
                new_frontmatter={"title": "Foo"},
                new_body="# Foo\nupdated\n",
                expected_hash=foo_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    p_bar = FixProposal(
        proposal_id="pb",
        issue_kind="broken_wikilink",
        issue_path="wiki/bar.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="wiki/bar.md",
                new_frontmatter={"title": "Bar"},
                new_body="# Bar\nupdated\n",
                expected_hash=bar_hash,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_foo, p_bar]),
        storage=storage, wiki_root=wiki_root,
        pick=[0],  # only the first
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "wiki/foo.md"
    # Bar untouched.
    assert abs_bar.read_text(encoding="utf-8") == bar_body_before


@pytest.mark.asyncio
async def test_skip_filter_drops_skipped_proposals(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/a.md", title="A", body="# A\n",
    )
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/b.md", title="B", body="# B\n",
    )
    a_hash = file_sha256(wiki_root / "wiki/a.md")
    b_hash = file_sha256(wiki_root / "wiki/b.md")

    p_a = FixProposal(
        proposal_id="pa", issue_kind="broken_wikilink",
        issue_path="wiki/a.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="wiki/a.md",
            new_frontmatter={"title": "A"},
            new_body="# A\nv2\n", expected_hash=a_hash,
        )],
        rationale="r", source="heuristic",
    )
    p_b = FixProposal(
        proposal_id="pb", issue_kind="broken_wikilink",
        issue_path="wiki/b.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="wiki/b.md",
            new_frontmatter={"title": "B"},
            new_body="# B\nv2\n", expected_hash=b_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_a, p_b]),
        storage=storage, wiki_root=wiki_root,
        skip=[1],  # drop p_b
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "wiki/a.md"
