"""OpenAI-compatible LLM + Embedding provider.

Covers OpenAI, Azure OpenAI, Ollama, vLLM, TEI, DeepSeek, Gemini-compat, and
anything else that speaks the OpenAI HTTP surface via ``base_url`` + ``api_key``.
Phase 0 = stub; real calls land in Phase 1.
"""

from __future__ import annotations

import os

from .base import LLMResponse, ProviderError, ToolSpec


class OpenAICompatLLM:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

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
            "OpenAICompatLLM.complete lands in Phase 1; "
            "Phase 0 only exercises the factory wiring"
        )


class OpenAICompatEmbeddings:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = (texts, model)
        raise NotImplementedError(
            "OpenAICompatEmbeddings.embed lands in Phase 1; "
            "Phase 0 only exercises the factory wiring"
        )


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set")
    return key
