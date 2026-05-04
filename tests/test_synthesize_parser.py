from __future__ import annotations

import pytest

from dikw_core.domains.knowledge.synthesize import SynthesisError, parse_synthesis_response

_RESPONSE = """
Here's the page:

<page path="wiki/concepts/dikw-pyramid.md" type="concept">
---
tags: [dikw, model]
---

# DIKW pyramid

The DIKW pyramid organises data into four layers.

See also [[Karpathy LLM Wiki]].
</page>
"""


def test_parse_response_extracts_page() -> None:
    page = parse_synthesis_response(_RESPONSE, source_path="sources/notes/dikw.md")
    assert page.path == "wiki/concepts/dikw-pyramid.md"
    assert page.type == "concept"
    assert page.title == "DIKW pyramid"
    assert "Karpathy LLM Wiki" in page.body
    assert page.tags == ["dikw", "model"]
    assert page.sources == ["sources/notes/dikw.md"]


def test_parse_missing_block_raises() -> None:
    with pytest.raises(SynthesisError):
        parse_synthesis_response("no page here", source_path="x")


def test_parse_bad_type_falls_back_to_note() -> None:
    raw = (
        "<page path=\"wiki/notes/x.md\" type=\"random\">\n"
        "---\ntags: []\n---\n\n"
        "# Random\n\nbody\n"
        "</page>"
    )
    page = parse_synthesis_response(raw, source_path="sources/x.md")
    assert page.type == "note"
