from __future__ import annotations

from pathlib import Path

from dikw_core.domains.data.backends.markdown import content_hash, parse_file, parse_text


def test_parse_text_uses_frontmatter_title() -> None:
    text = "---\ntitle: Custom\n---\n\n# Actual heading\n\nBody."
    parsed = parse_text(path="notes/a.md", text=text, mtime=1.0)
    assert parsed.title == "Custom"
    assert parsed.body.startswith("# Actual heading")
    assert parsed.hash == content_hash(parsed.body)


def test_parse_text_falls_back_to_first_heading() -> None:
    parsed = parse_text(path="notes/b.md", text="# First heading\n\nBody.", mtime=1.0)
    assert parsed.title == "First heading"


def test_parse_text_falls_back_to_stem() -> None:
    parsed = parse_text(path="notes/stem-name.md", text="no heading here", mtime=1.0)
    assert parsed.title == "stem-name"


def test_parse_file_reads_from_disk(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("# Hello\n\nBody", encoding="utf-8")
    parsed = parse_file(f, rel_path="x.md")
    assert parsed.path == "x.md"
    assert parsed.title == "Hello"
    assert parsed.mtime == f.stat().st_mtime
