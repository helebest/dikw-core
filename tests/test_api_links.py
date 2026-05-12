"""Engine-side unit tests for ``api.list_links``.

The HTTP-layer surface lives in ``tests/server/test_routes_page_links.py``;
this file exercises the pure helper that produces a ``PageLinksResult``
from ``(root, path, direction, limit)`` so the seam (path-not-registered
→ ``PageNotFound``, outgoing carries dst_path+link_type+line, incoming
carries src_doc_id+src_path+link_type+line, direction filter on each
list, limit cap on each list) stays guarded without booting a server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api import _doc_id_for, _with_storage
from dikw_core.schemas import DocumentRecord, Layer, LinkRecord, LinkType

from .fakes import init_test_wiki


def _doc(path: str, layer: Layer = Layer.WIKI) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_doc_id_for(layer, path),
        path=path,
        hash="0" * 64,
        mtime=0.0,
        layer=layer,
        active=True,
    )


async def _seed_triangle(root: Path) -> tuple[str, str, str]:
    """Seed three wiki docs ``a → b → c`` plus ``c → a`` so every doc has
    both an outgoing and an incoming edge. Returns ``(a_path, b_path,
    c_path)`` for assertions.
    """
    init_test_wiki(root)
    a_path = "wiki/a.md"
    b_path = "wiki/b.md"
    c_path = "wiki/c.md"

    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        for p in (a_path, b_path, c_path):
            await storage.upsert_document(_doc(p))
        # a -> b
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.WIKI, a_path),
                dst_path=b_path,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=5,
            )
        )
        # b -> c
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.WIKI, b_path),
                dst_path=c_path,
                link_type=LinkType.WIKILINK,
                anchor="Section",
                line=7,
            )
        )
        # c -> a
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.WIKI, c_path),
                dst_path=a_path,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=11,
            )
        )
    finally:
        await storage.close()
    return a_path, b_path, c_path


@pytest.mark.asyncio
async def test_list_links_both_returns_outgoing_and_incoming(
    tmp_path: Path,
) -> None:
    a_path, b_path, c_path = await _seed_triangle(tmp_path)

    result = await api.list_links(tmp_path, b_path, direction="both")
    assert result.path == b_path
    # b -> c is the only outgoing edge.
    assert len(result.outgoing) == 1
    out = result.outgoing[0]
    assert out.dst_path == c_path
    assert out.link_type == LinkType.WIKILINK
    assert out.anchor == "Section"
    assert out.line == 7
    # a -> b is the only incoming edge.
    assert len(result.incoming) == 1
    inb = result.incoming[0]
    assert inb.src_path == a_path
    assert inb.src_doc_id == _doc_id_for(Layer.WIKI, a_path)
    assert inb.link_type == LinkType.WIKILINK
    assert inb.line == 5


@pytest.mark.asyncio
async def test_list_links_out_direction_drops_incoming(
    tmp_path: Path,
) -> None:
    _, b_path, _ = await _seed_triangle(tmp_path)
    result = await api.list_links(tmp_path, b_path, direction="out")
    assert result.outgoing  # b -> c remains
    assert result.incoming == []


@pytest.mark.asyncio
async def test_list_links_in_direction_drops_outgoing(
    tmp_path: Path,
) -> None:
    _, b_path, _ = await _seed_triangle(tmp_path)
    result = await api.list_links(tmp_path, b_path, direction="in")
    assert result.incoming  # a -> b remains
    assert result.outgoing == []


@pytest.mark.asyncio
async def test_list_links_limit_caps_each_list_independently(
    tmp_path: Path,
) -> None:
    """``limit`` caps outgoing AND incoming independently — it is a
    per-direction cap, not a total. Without this an agent with
    ``limit=5`` could see 5 outgoing and 0 incoming on a popular hub
    page even though the hub has plenty of inbound edges."""
    init_test_wiki(tmp_path)
    hub_path = "wiki/hub.md"
    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(hub_path))
        # Three outgoing edges hub -> {x, y, z}.
        for i, dst in enumerate(("wiki/x.md", "wiki/y.md", "wiki/z.md"), start=1):
            await storage.upsert_document(_doc(dst))
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=_doc_id_for(Layer.WIKI, hub_path),
                    dst_path=dst,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=i,
                )
            )
        # Three incoming edges {p, q, r} -> hub.
        for i, src in enumerate(("wiki/p.md", "wiki/q.md", "wiki/r.md"), start=1):
            await storage.upsert_document(_doc(src))
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=_doc_id_for(Layer.WIKI, src),
                    dst_path=hub_path,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=10 + i,
                )
            )
    finally:
        await storage.close()

    result = await api.list_links(tmp_path, hub_path, direction="both", limit=2)
    assert len(result.outgoing) == 2
    assert len(result.incoming) == 2


@pytest.mark.asyncio
async def test_list_links_unknown_path_raises(tmp_path: Path) -> None:
    init_test_wiki(tmp_path)
    with pytest.raises(api.PageNotFound):
        await api.list_links(tmp_path, "wiki/does-not-exist.md")


@pytest.mark.asyncio
async def test_list_links_outgoing_filters_inactive_or_unindexed_dst(
    tmp_path: Path,
) -> None:
    """``outgoing`` must only surface edges whose ``dst_path`` resolves
    to an active document — without this, bare URLs / markdown links to
    non-indexed files / pages-deactivated-since-synth leak into
    ``outgoing[]`` and break the graph-hop contract: the caller cannot
    fetch ``dst_path`` back through ``GET /v1/base/pages/{path}`` (it
    would 404).
    """
    init_test_wiki(tmp_path)
    src_path = "wiki/src.md"
    live_dst = "wiki/live.md"
    dead_dst = "wiki/dead.md"
    bare_url = "https://example.com/external"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path))
        await storage.upsert_document(_doc(live_dst))
        await storage.upsert_document(_doc(dead_dst))
        await storage.deactivate_document(_doc_id_for(Layer.WIKI, dead_dst))

        for i, dst in enumerate((live_dst, dead_dst, bare_url), start=1):
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=_doc_id_for(Layer.WIKI, src_path),
                    dst_path=dst,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=i,
                )
            )
    finally:
        await storage.close()

    result = await api.list_links(tmp_path, src_path, direction="out")
    assert [e.dst_path for e in result.outgoing] == [live_dst]


@pytest.mark.asyncio
async def test_list_links_limit_zero_returns_empty_lists(
    tmp_path: Path,
) -> None:
    """``limit=0`` is valid per the route's ``Query(ge=0)`` and must be
    honored symmetrically. Outgoing slices on the materialised list so
    ``[:0]`` is empty; incoming used to append the first active edge
    before checking the cap, returning one entry while outgoing returned
    none — fix slices both halves post-filter.
    """
    _, b_path, _ = await _seed_triangle(tmp_path)
    result = await api.list_links(tmp_path, b_path, direction="both", limit=0)
    assert result.outgoing == []
    assert result.incoming == []


@pytest.mark.asyncio
async def test_list_links_incoming_resolves_src_path_via_documents(
    tmp_path: Path,
) -> None:
    """Storage exposes inbound edges keyed on ``src_doc_id``; the engine
    helper must translate that to ``src_path`` by joining against the
    documents table so the response is path-readable without a second
    round trip."""
    init_test_wiki(tmp_path)
    target_path = "wiki/target.md"
    src_path = "wiki/origin.md"
    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(target_path))
        await storage.upsert_document(_doc(src_path))
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.WIKI, src_path),
                dst_path=target_path,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )
    finally:
        await storage.close()

    result = await api.list_links(tmp_path, target_path, direction="in")
    assert len(result.incoming) == 1
    assert result.incoming[0].src_path == src_path
    assert result.incoming[0].src_doc_id == _doc_id_for(Layer.WIKI, src_path)
