from __future__ import annotations

from dikw_core.domains.knowledge.links import parse_links, resolve_links
from dikw_core.schemas import LinkType


def test_parse_wikilink_with_anchor_and_alias() -> None:
    body = "See [[Karpathy Wiki#Operations|the ops section]] for details.\n"
    links = parse_links(body)
    assert len(links) == 1
    assert links[0].kind is LinkType.WIKILINK
    assert links[0].target == "Karpathy Wiki"
    assert links[0].anchor == "Operations"
    assert links[0].line == 1


def test_parse_plain_markdown_link_and_url() -> None:
    body = (
        "line one\n"
        "\n"
        "Visit [RRF paper](https://example.com/rrf) or just https://example.com/home.\n"
    )
    links = parse_links(body)
    # we expect two: one MARKDOWN (as URL because https), one URL for the bare link
    kinds = [link.kind for link in links]
    assert LinkType.URL in kinds
    assert len(links) >= 2


def test_parse_local_markdown_link_is_markdown_type() -> None:
    body = "See [related](../notes/other.md) for more.\n"
    links = parse_links(body)
    assert len(links) == 1
    assert links[0].kind is LinkType.MARKDOWN
    assert links[0].target == "../notes/other.md"


def test_resolve_links_flags_unresolved_wikilinks() -> None:
    body = "Both [[Known Page]] and [[Unknown Page]] appear here."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"Known Page": "wiki/concepts/known.md"},
    )
    assert len(resolved) == 1
    assert resolved[0].dst_path == "wiki/concepts/known.md"
    assert len(unresolved) == 1
    assert "Unknown Page" in unresolved[0].target_text
