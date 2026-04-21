"""OpenAI-compatible LLM + Embedding provider.

Covers OpenAI, Azure OpenAI, Ollama, vLLM, TEI, DeepSeek, Gemini-compat, and
anything else that speaks the OpenAI HTTP surface via ``base_url`` + ``api_key``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import LLMResponse, ProviderError, ToolSpec

if TYPE_CHECKING:  # avoid importing openai at module load for envs without it
    from openai import AsyncOpenAI

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError(
            "OPENAI_API_KEY is not set. Export it or pass `api_key` explicitly."
        )
    return key


def _client(base_url: str, api_key: str) -> AsyncOpenAI:
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


class OpenAICompatLLM:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key_explicit = api_key
        self._client_cache: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client_cache is None:
            self._client_cache = _client(self._base_url, _resolve_api_key(self._api_key_explicit))
        return self._client_cache

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
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Tools are not translated into OpenAI-compat tool_call format yet; Phase 1
        # only needs plain text completion for query answering.
        _ = tools
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = {}
        if resp.usage is not None:
            usage = {
                "input_tokens": int(resp.usage.prompt_tokens or 0),
                "output_tokens": int(resp.usage.completion_tokens or 0),
            }
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason,
            usage=usage,
        )


class OpenAICompatEmbeddings:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key_explicit = api_key
        self._client_cache: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client_cache is None:
            self._client_cache = _client(self._base_url, _resolve_api_key(self._api_key_explicit))
        return self._client_cache

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = await client.embeddings.create(model=model, input=texts)
        return [list(r.embedding) for r in resp.data]
