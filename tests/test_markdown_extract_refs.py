"""Tests for markdown image-reference extraction.

Pins the spec: every ``![alt](path)`` and ``![[file|alias]]`` in the body
becomes one ``AssetRef`` with ``start`` / ``end`` byte offsets that exactly
cover the literal reference syntax. ``parse_text`` populates
``ParsedDocument.asset_refs`` automatically.
"""

from __future__ import annotations

from dikw_core.data.backends.markdown import extract_image_refs, parse_text


def test_extract_markdown_image() -> None:
    body = "Hello ![alt text](./img.png) world"
    refs = extract_image_refs(body)
    assert len(refs) == 1
    r = refs[0]
    assert r.original_path == "./img.png"
    assert r.alt == "alt text"
    assert r.syntax == "markdown"
    assert body[r.start : r.end] == "![alt text](./img.png)"


def test_extract_markdown_image_with_title() -> None:
    body = '![arch](./diagrams/x.png "Architecture diagram")'
    refs = extract_image_refs(body)
    assert len(refs) == 1
    assert refs[0].original_path == "./diagrams/x.png"
    assert refs[0].alt == "arch"
    # Span covers the entire reference including the title attribute.
    assert body[refs[0].start : refs[0].end] == body


def test_extract_markdown_image_empty_alt() -> None:
    refs = extract_image_refs("![](standalone.png)")
    assert len(refs) == 1
    assert refs[0].alt == ""
    assert refs[0].original_path == "standalone.png"


def test_extract_obsidian_wikilink() -> None:
    body = "Pre ![[image.png]] post"
    refs = extract_image_refs(body)
    assert len(refs) == 1
    assert refs[0].original_path == "image.png"
    assert refs[0].alt == ""
    assert refs[0].syntax == "wikilink"
    assert body[refs[0].start : refs[0].end] == "![[image.png]]"


def test_extract_obsidian_wikilink_with_dimension_alias() -> None:
    """Obsidian's ``![[file|150]]`` alias carries display dimension."""
    refs = extract_image_refs("![[arch.png|150]]")
    assert len(refs) == 1
    assert refs[0].original_path == "arch.png"
    assert refs[0].alt == "150"


def test_extract_obsidian_wikilink_with_caption_alias() -> None:
    refs = extract_image_refs("![[arch.png|System architecture]]")
    assert refs[0].alt == "System architecture"
    assert refs[0].original_path == "arch.png"


def test_extract_no_refs() -> None:
    assert extract_image_refs("") == []
    assert extract_image_refs("Just text without images.") == []
    # A plain link (no leading !) is not an image and must not be captured.
    assert extract_image_refs("[link](http://x.com)") == []
    # A wikilink without a leading ! is also a plain link, not an embed.
    assert extract_image_refs("[[note.md]]") == []


def test_extract_returns_sorted_by_position() -> None:
    body = "![[b.png]] middle ![alt](a.png)"
    refs = extract_image_refs(body)
    assert len(refs) == 2
    assert refs[0].start < refs[1].start
    assert refs[0].syntax == "wikilink"
    assert refs[1].syntax == "markdown"


def test_extract_multiple_markdown_refs() -> None:
    body = "![one](a.png) and ![two](b.jpg) and ![three](c.gif)"
    refs = extract_image_refs(body)
    assert [r.original_path for r in refs] == ["a.png", "b.jpg", "c.gif"]
    assert [r.alt for r in refs] == ["one", "two", "three"]


def test_extract_remote_url_still_captured() -> None:
    """v1 captures all syntactically-valid references; the materialize layer
    decides what to do with remote URLs (skips them, logs)."""
    body = "![remote](https://example.com/img.png)"
    refs = extract_image_refs(body)
    assert len(refs) == 1
    assert refs[0].original_path == "https://example.com/img.png"


def test_extract_path_with_spaces() -> None:
    """Obsidian vaults commonly have filenames with spaces — the regex
    must capture them as a single path."""
    refs = extract_image_refs("![diagram](My Diagram.png)")
    assert len(refs) == 1
    assert refs[0].original_path == "My Diagram.png"


def test_extract_path_with_spaces_and_title() -> None:
    body = '![arch](My Big Diagram.png "Optional title")'
    refs = extract_image_refs(body)
    assert len(refs) == 1
    assert refs[0].original_path == "My Big Diagram.png"
    assert refs[0].alt == "arch"


def test_extract_cjk_in_path_and_alt() -> None:
    body = "![架构图](./图表/系统架构.png)"
    refs = extract_image_refs(body)
    assert len(refs) == 1
    assert refs[0].alt == "架构图"
    assert refs[0].original_path == "./图表/系统架构.png"


def test_extract_normalizes_windows_backslash() -> None:
    refs = extract_image_refs(r"![](images\images\00001.jpeg)")
    assert len(refs) == 1
    assert refs[0].original_path == "images/images/00001.jpeg"
    assert refs[0].syntax == "markdown"


def test_extract_normalizes_backslash_in_wikilink() -> None:
    refs = extract_image_refs(r"![[images\sub\arch.png]]")
    assert len(refs) == 1
    assert refs[0].original_path == "images/sub/arch.png"
    assert refs[0].syntax == "wikilink"


def test_parse_text_populates_asset_refs() -> None:
    body = "# Title\n\n![arch](arch.png)\n\nMore text with ![[diagram.svg|400]] inline."
    parsed = parse_text(path="x.md", text=body, mtime=1234.5)
    assert len(parsed.asset_refs) == 2
    assert parsed.asset_refs[0].original_path == "arch.png"
    assert parsed.asset_refs[0].syntax == "markdown"
    assert parsed.asset_refs[1].original_path == "diagram.svg"
    assert parsed.asset_refs[1].alt == "400"
    assert parsed.asset_refs[1].syntax == "wikilink"
    # Body must remain unmodified — parser only adds metadata.
    assert "![arch](arch.png)" in parsed.body
    assert "![[diagram.svg|400]]" in parsed.body


def test_parse_text_no_refs_yields_empty_list() -> None:
    parsed = parse_text(path="x.md", text="No images here.", mtime=0.0)
    assert parsed.asset_refs == []
