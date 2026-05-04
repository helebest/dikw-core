from __future__ import annotations

from dikw_core.domains.wisdom.distill import (
    MIN_EVIDENCE,
    make_wisdom_id,
    parse_distill_response,
)
from dikw_core.schemas import WisdomKind

_GOOD = """
<wisdom kind="principle">
---
confidence: 0.8
evidence:
  - doc: wiki/concepts/karpathy-llm-wiki.md
    line: 5
    excerpt: "Scoping should be deterministic."
  - doc: wiki/concepts/hybrid-retrieval.md
    excerpt: "RRF ignores raw scores."
---

# Prefer deterministic scoping over probabilistic retrieval

Body text citing [#1] and [#2].
</wisdom>
"""

_ONE_EVIDENCE = """
<wisdom kind="principle">
---
confidence: 0.9
evidence:
  - doc: wiki/x.md
    excerpt: "one only"
---

# Sole source claim

Body.
</wisdom>
"""

_BAD_KIND = """
<wisdom kind="rumour">
---
confidence: 0.5
evidence:
  - doc: a.md
    excerpt: "x"
  - doc: b.md
    excerpt: "y"
---

# Nope

body
</wisdom>
"""


def test_parses_valid_block() -> None:
    result = parse_distill_response(_GOOD)
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.kind is WisdomKind.PRINCIPLE
    assert cand.title.startswith("Prefer deterministic")
    assert len(cand.evidence) >= MIN_EVIDENCE
    assert cand.confidence == 0.8
    assert cand.item_id == make_wisdom_id("principle", cand.title)


def test_rejects_fewer_than_two_evidence() -> None:
    result = parse_distill_response(_ONE_EVIDENCE)
    assert result.candidates == []
    assert any("evidence" in reason for _, reason in result.rejected)


def test_rejects_invalid_kind() -> None:
    result = parse_distill_response(_BAD_KIND)
    assert result.candidates == []
    assert any("invalid kind" in reason for _, reason in result.rejected)


def test_empty_response_yields_no_candidates() -> None:
    assert parse_distill_response("no wisdom here").candidates == []
