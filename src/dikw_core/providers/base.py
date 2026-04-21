"""LLM + Embedding provider abstractions.

Engine code talks only to these Protocols; concrete adapters in sibling files
wrap the official SDKs. Swapping providers is a config-only change at the
``providers/__init__.py`` factory level.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    """Base class for provider errors (auth, network, invalid model, etc.)."""


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "ToolSpec",
]
