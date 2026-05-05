"""``build_llm`` resolves ``llm: openai_codex`` to ``OpenAICodexLLM``.

Mirrors ``test_provider_openai_compat_retries.py``'s factory-wiring
checks: assert that the cfg fields the factory reads land on the
provider instance via the path the engine actually traverses.
"""

from __future__ import annotations

from dikw_core.providers import build_llm
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM

from .fakes import make_provider_cfg


def test_build_llm_returns_openai_codex_instance() -> None:
    cfg = make_provider_cfg(
        llm="openai_codex", llm_base_url=DEFAULT_CODEX_BASE_URL
    )
    provider = build_llm(cfg)
    assert isinstance(provider, OpenAICodexLLM)


def test_build_llm_threads_base_url() -> None:
    cfg = make_provider_cfg(
        llm="openai_codex", llm_base_url=DEFAULT_CODEX_BASE_URL
    )
    provider = build_llm(cfg)
    assert isinstance(provider, OpenAICodexLLM)
    # Read through the documented test seam — _base_url is the attribute the
    # provider hands to AsyncOpenAI, asserting that link is enough.
    assert provider._base_url == DEFAULT_CODEX_BASE_URL


def test_build_llm_threads_max_retries_and_timeout() -> None:
    cfg = make_provider_cfg(
        llm="openai_codex",
        llm_base_url=DEFAULT_CODEX_BASE_URL,
        llm_max_retries=7,
        llm_timeout_seconds=42.0,
    )
    provider = build_llm(cfg)
    assert isinstance(provider, OpenAICodexLLM)
    assert provider._max_retries == 7
    assert provider._timeout_seconds == 42.0
