"""Anthropic LLM provider.

Phase 0 ships a thin stub that lets the factory resolve without making real
API calls. Phase 1 will flesh this out with actual completions + prompt
caching. Anthropic has no embeddings endpoint, so embeddings must use an
OpenAI-compatible provider.
"""

from __future__ import annotations

import os

from .base import LLMResponse, ProviderError, ToolSpec


class AnthropicLLM:
    """Wrap the official ``anthropic`` SDK. Phase 0 = stub."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

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
        _ = (system, user, model, max_tokens, temperature, tools)
        raise NotImplementedError(
            "AnthropicLLM.complete lands in Phase 1; "
            "Phase 0 only exercises the factory wiring"
        )


def require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ProviderError("ANTHROPIC_API_KEY is not set")
    return key
