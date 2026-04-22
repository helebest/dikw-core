"""Test doubles for LLM + embedding providers.

``FakeEmbeddings`` lives in ``src/dikw_core/eval/fake_embedder.py`` so it
ships with the wheel for ``dikw eval``. This module re-exports it for
backward compatibility with existing ``from tests.fakes import FakeEmbeddings``
imports; new code should reach into ``dikw_core.eval.fake_embedder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dikw_core.eval.fake_embedder import EMBED_DIM, FakeEmbeddings
from dikw_core.providers import LLMResponse, ToolSpec

__all__ = ["EMBED_DIM", "FakeEmbeddings", "FakeLLM"]


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
