"""Provider factory: resolves LLM + Embedding instances from ``ProviderConfig``."""

from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicLLM
from .base import EmbeddingProvider, LLMProvider, LLMResponse, ProviderError, ToolSpec
from .openai_compat import OpenAICompatEmbeddings, OpenAICompatLLM


def build_llm(config: ProviderConfig) -> LLMProvider:
    if config.llm == "anthropic":
        return AnthropicLLM(base_url=config.llm_base_url)
    if config.llm == "openai_compat":
        return OpenAICompatLLM(base_url=config.llm_base_url)
    raise ProviderError(f"unknown LLM provider: {config.llm!r}")


def build_embedder(config: ProviderConfig) -> EmbeddingProvider:
    # Anthropic has no embeddings API; both paths route through OpenAI-compat
    # using ``embedding_base_url`` so users configure one endpoint explicitly.
    if config.embedding == "openai_compat":
        return OpenAICompatEmbeddings(
            base_url=config.embedding_base_url,
            default_dimensions=config.embedding_dimensions,
        )
    raise ProviderError(f"unknown embedding provider: {config.embedding!r}")


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "ToolSpec",
    "build_embedder",
    "build_llm",
]
