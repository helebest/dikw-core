"""Surface approved wisdom at query time.

The scoring is deliberately cheap for Phase 3: Jaccard overlap between the
question's lowercased word set and the concatenation of each item's title
and body. Top-K items above a minimum overlap are returned. When no items
clear the bar we return an empty list — the query prompt then simply omits
the wisdom section rather than padding it with weakly-related items.

A later phase may upgrade this to cosine-over-embeddings; the Protocol
below (``pick_applicable``) shouldn't need to change.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ...schemas import WisdomItem

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9']+")

# Suffix stems ordered longest-first so multi-letter suffixes match before their
# prefixes (e.g. ``ically`` before ``ly``). This is a deliberately cheap
# substitute for a real stemmer — it catches common English morphological
# variants so "scoping" / "scope" and "deterministic" / "deterministically"
# collide in the overlap score.
_SUFFIXES: tuple[str, ...] = (
    "ications",
    "ically",
    "ations",
    "ingly",
    "ation",
    "ings",
    "tion",
    "ical",
    "ing",
    "ies",
    "ied",
    "ly",
    "ic",
    "ed",
    "er",
    "est",
    "es",
    "s",
)

MIN_OVERLAP = 0.08  # require at least 8% Jaccard to count as "applicable"


@dataclass(frozen=True)
class ApplicableWisdom:
    item: WisdomItem
    score: float


def _stem(token: str) -> str:
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    if len(token) > 4 and token.endswith("e"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(w.lower()) for w in _WORD.findall(text)}


def pick_applicable(
    question: str,
    items: Sequence[WisdomItem],
    *,
    limit: int = 3,
    min_overlap: float = MIN_OVERLAP,
) -> list[ApplicableWisdom]:
    """Return up to ``limit`` wisdom items whose token overlap exceeds ``min_overlap``."""
    q_tokens = _tokens(question)
    if not q_tokens or not items:
        return []

    scored: list[ApplicableWisdom] = []
    for item in items:
        item_tokens = _tokens(item.title) | _tokens(item.body)
        if not item_tokens:
            continue
        intersection = q_tokens & item_tokens
        union = q_tokens | item_tokens
        if not union:
            continue
        jaccard = len(intersection) / len(union)
        if jaccard < min_overlap:
            continue
        scored.append(ApplicableWisdom(item=item, score=jaccard))

    scored.sort(key=lambda a: (a.score, a.item.confidence), reverse=True)
    return scored[:limit]
