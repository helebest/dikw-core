"""Provider factory: resolves LLM + Embedding instances from ``ProviderConfig``."""

from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicLLM
from .base import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    MultimodalEmbeddingProvider,
    ProviderError,
    ToolSpec,
)
from .gitee_multimodal import GiteeMultimodalEmbedding
from .openai_compat import OpenAICompatEmbeddings, OpenAICompatLLM


def build_llm(config: ProviderConfig) -> LLMProvider:
    if config.llm == "anthropic":
        return AnthropicLLM(
            base_url=config.llm_base_url,
            max_retries=config.llm_max_retries,
            timeout_seconds=config.llm_timeout_seconds,
        )
    if config.llm == "openai_compat":
        return OpenAICompatLLM(
            base_url=config.llm_base_url,
            max_retries=config.llm_max_retries,
            timeout_seconds=config.llm_timeout_seconds,
        )
    raise ProviderError(f"unknown LLM provider: {config.llm!r}")


def build_embedder(config: ProviderConfig) -> EmbeddingProvider:
    # Anthropic has no embeddings API; both paths route through OpenAI-compat
    # using ``embedding_base_url`` so users configure one endpoint explicitly.
    if config.embedding == "openai_compat":
        return OpenAICompatEmbeddings(
            base_url=config.embedding_base_url,
            default_dimensions=config.embedding_dimensions,
            max_retries=config.embedding_max_retries,
            timeout_seconds=config.embedding_timeout_seconds,
        )
    raise ProviderError(f"unknown embedding provider: {config.embedding!r}")


def build_multimodal_embedder(
    provider: str, *, base_url: str | None = None, batch: int = 16
) -> MultimodalEmbeddingProvider:
    """Build a multimodal embedder by name.

    One provider ships today: ``gitee_multimodal`` covers every multimodal
    model Gitee AI serves (Qwen3-VL-Embedding-8B, jina-clip-v2, …) — they
    share one wire shape, the model name in ``assets.multimodal.model``
    discriminates which one runs server-side. Additional vendors (Voyage,
    Cohere, Jina-direct) are easy follow-ons: drop a new file under
    ``providers/`` and add a branch here.
    """
    if provider == "gitee_multimodal":
        return GiteeMultimodalEmbedding(base_url=base_url, batch=batch)
    raise ProviderError(f"unknown multimodal embedding provider: {provider!r}")


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "MultimodalEmbeddingProvider",
    "ProviderError",
    "ToolSpec",
    "build_embedder",
    "build_llm",
    "build_multimodal_embedder",
]
