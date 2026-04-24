"""LLM + Embedding provider abstractions.

Engine code talks only to these Protocols; concrete adapters in sibling files
wrap the official SDKs. Swapping providers is a config-only change at the
``providers/__init__.py`` factory level.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..schemas import MultimodalInput


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


@runtime_checkable
class MultimodalEmbeddingProvider(Protocol):
    """Embedding provider that can encode text, images, or any combination
    into a single shared vector space.

    v1 callers use either text-only (chunks) or image-only (assets) inputs;
    the schema's ``MultimodalInput`` permits combined inputs for v1.5
    chunk-with-images joint encoding without breaking the wire contract.

    Output ordering must match input ordering so callers can pair vectors
    with their source rows. All vectors must have the same dimension —
    ``EmbeddingVersion.dim`` records that dim and the storage layer
    validates each row against it.
    """

    async def embed(
        self,
        inputs: list[MultimodalInput],
        *,
        model: str,
    ) -> list[list[float]]: ...


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "MultimodalEmbeddingProvider",
    "ProviderError",
    "ToolSpec",
]
