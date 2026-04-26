"""Test doubles for LLM + embedding providers.

``FakeEmbeddings`` lives in ``src/dikw_core/eval/fake_embedder.py`` so it
ships with the wheel for ``dikw eval``. This module re-exports it for
backward compatibility with existing ``from tests.fakes import FakeEmbeddings``
imports; new code should reach into ``dikw_core.eval.fake_embedder``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from dikw_core.eval.fake_embedder import EMBED_DIM, FakeEmbeddings
from dikw_core.providers import LLMResponse, ToolSpec
from dikw_core.schemas import MultimodalInput

__all__ = [
    "EMBED_DIM",
    "CountingEmbedder",
    "FakeEmbeddings",
    "FakeLLM",
    "FakeMultimodalEmbedding",
]


@dataclass
class CountingEmbedder:
    """Counts ``embed`` calls + total texts; delegates to ``FakeEmbeddings``.

    Used by streaming + perf tests to assert "cache hit means zero
    provider calls" (``embed_calls == 0``) or to enforce per-batch
    streaming visibility (call count > 1).

    ``fail_after`` lets a test simulate a mid-flight crash: the
    ``(fail_after+1)``-th call raises ``RuntimeError``. Default
    ``None`` = never fail.
    """

    inner: FakeEmbeddings = field(default_factory=FakeEmbeddings)
    embed_calls: int = field(default=0, init=False)
    total_texts: int = field(default=0, init=False)
    fail_after: int | None = None

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.embed_calls += 1
        self.total_texts += len(texts)
        if self.fail_after is not None and self.embed_calls > self.fail_after:
            raise RuntimeError(
                f"CountingEmbedder simulated failure on call {self.embed_calls}"
            )
        return await self.inner.embed(texts, model=model)

    def reset(self) -> None:
        self.embed_calls = 0
        self.total_texts = 0


@dataclass
class FakeLLM:
    """Captures the last call and returns a scripted or default response."""

    response_text: str = "STUB: wired up."
    last_system: str | None = field(default=None, init=False)
    last_user: str | None = field(default=None, init=False)
    last_max_tokens: int | None = field(default=None, init=False)

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
        _ = (model, temperature, tools)
        self.last_system = system
        self.last_user = user
        self.last_max_tokens = max_tokens
        return LLMResponse(text=self.response_text, finish_reason="end_turn")


@dataclass
class FakeMultimodalEmbedding:
    """Deterministic ``MultimodalEmbeddingProvider`` for tests.

    Each input is hashed into a vector of the configured ``dim``. Same
    input → same vector; text and image inputs are tagged with distinct
    namespaces so a text payload never collides with an image of the
    same bytes.
    """

    dim: int = 4
    last_inputs: list[MultimodalInput] = field(default_factory=list, init=False)
    last_model: str | None = field(default=None, init=False)

    async def embed(
        self, inputs: list[MultimodalInput], *, model: str
    ) -> list[list[float]]:
        self.last_inputs = list(inputs)
        self.last_model = model
        return [self._vector_for(inp) for inp in inputs]

    def _vector_for(self, inp: MultimodalInput) -> list[float]:
        """Deterministic projection of an input into a ``dim``-vector.

        Hash text + image bytes (each in its own namespace prefix) and
        sample ``dim`` floats out of the resulting digest. Stable across
        runs, distinct across inputs of different modality even when
        their content is byte-identical."""
        h = hashlib.sha256()
        if inp.text is not None:
            h.update(b"text:")
            h.update(inp.text.encode("utf-8"))
        for img in inp.images:
            h.update(b"image:")
            h.update(img.mime.encode("utf-8"))
            h.update(b"|")
            h.update(img.bytes)
        digest = h.digest()
        # Tile the 32-byte digest to fill ``dim`` floats; map each byte to
        # a value in [-1, 1] so vectors are dense and non-zero.
        out: list[float] = []
        for i in range(self.dim):
            b = digest[i % len(digest)]
            out.append((b / 127.5) - 1.0)
        return out
