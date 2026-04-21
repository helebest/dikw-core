"""Test doubles for LLM + embedding providers.

The embedder produces deterministic vectors so retrieval tests are stable:
it tokenises each text, hashes each token into a fixed-size bucket, and
returns the bucket counts as an L2-normalised vector. Semantically similar
inputs (overlapping bag-of-words) cluster as expected.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from dikw_core.providers import LLMResponse, ToolSpec

EMBED_DIM = 64
_WORD = re.compile(r"[A-Za-z]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _bucket(tok: str) -> int:
    h = hashlib.sha1(tok.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % EMBED_DIM


class FakeEmbeddings:
    """Deterministic bag-of-words embeddings over a fixed ``EMBED_DIM`` space."""

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = model
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * EMBED_DIM
            for tok in _tokens(text):
                vec[_bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@dataclass
class FakeLLM:
    """Captures the last call and returns a scripted or default response."""

    response_text: str = "STUB: wired up."
    last_system: str | None = field(default=None, init=False)
    last_user: str | None = field(default=None, init=False)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        _ = (model, max_tokens, temperature, tools)
        self.last_system = system
        self.last_user = user
        return LLMResponse(text=self.response_text, finish_reason="end_turn")
