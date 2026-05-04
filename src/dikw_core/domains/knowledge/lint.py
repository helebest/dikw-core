"""K-layer hygiene checker.

Reports three classes of issue that are safe to detect deterministically:

* ``broken_wikilink`` — wikilinks whose target title isn't a known K/W page.
* ``orphan_page`` — pages with no inbound wikilinks and no listing source.
* ``duplicate_title`` — more than one K-layer page with identical title.

Phases 3+ may add semantic checks (stale claims, missing evidence for
approved wisdom items, etc.); this module intentionally stays lexical.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from ...schemas import Layer, LinkType
from ...storage.base import Storage
from .links import parse_links


@dataclass(frozen=True)
class LintIssue:
    kind: str            # broken_wikilink | orphan_page | duplicate_title
    path: str            # the doc exhibiting the issue
    detail: str
    line: int | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in self.issues:
            counts[i.kind] = counts.get(i.kind, 0) + 1
        return counts


async def run_lint(storage: Storage, *, root: Path) -> LintReport:
    """Scan K-layer pages and return a structured report."""
    issues: list[LintIssue] = []

    wiki_docs = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    title_to_paths: dict[str, list[str]] = defaultdict(list)
    inbound: Counter[str] = Counter()
    paths: list[str] = []

    for doc in wiki_docs:
        title = doc.title or Path(doc.path).stem
        title_to_paths[title].append(doc.path)
        paths.append(doc.path)

    title_set = set(title_to_paths)

    # broken wikilinks — re-parse on-disk bodies; storage.links only records
    # resolved links, so we must look at the raw text to find unresolved ones.
    for doc in wiki_docs:
        abs_path = (root / doc.path).resolve()
        if not abs_path.is_file():
            continue
        try:
            post = frontmatter.load(str(abs_path))
        except Exception:
            continue
        for link in parse_links(post.content):
            if (
                link.kind is LinkType.WIKILINK
                and link.target not in title_set
                and not any(t.lower() == link.target.lower() for t in title_set)
            ):
                issues.append(
                    LintIssue(
                        kind="broken_wikilink",
                        path=doc.path,
                        detail=f"[[{link.target}]] has no matching wiki page",
                        line=link.line,
                    )
                )

        # accumulate inbound link counts (resolved links from storage)
        for stored in await storage.links_from(doc.doc_id):
            if stored.link_type is LinkType.WIKILINK:
                inbound[stored.dst_path] += 1

    # orphans — no inbound wikilinks AND not referenced from index.md/log.md
    orphan_exclusions = {"wiki/index.md", "wiki/log.md"}
    for doc in wiki_docs:
        if doc.path in orphan_exclusions:
            continue
        if inbound[doc.path] == 0:
            issues.append(
                LintIssue(
                    kind="orphan_page",
                    path=doc.path,
                    detail="no inbound wikilinks from other K-layer pages",
                )
            )

    # duplicate titles — reported per extra path beyond the first
    for title, dup_paths in title_to_paths.items():
        if len(dup_paths) > 1:
            primary = dup_paths[0]
            for extra in dup_paths[1:]:
                issues.append(
                    LintIssue(
                        kind="duplicate_title",
                        path=extra,
                        detail=f"title '{title}' also used by {primary}",
                    )
                )

    return LintReport(issues=issues)
