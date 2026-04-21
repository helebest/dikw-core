"""Verify the Anthropic provider threads ``base_url`` to the SDK.

The Anthropic SDK accepts a ``base_url`` kwarg that retargets it at any
Anthropic-protocol-compatible endpoint (e.g., MiniMax). We expose this
through ``AnthropicLLM(base_url=...)`` and plumb it from
``ProviderConfig.llm_base_url`` in the factory so users don't have to
switch provider types.
"""

from __future__ import annotations

import pytest

from dikw_core.config import ProviderConfig
from dikw_core.providers import build_llm
from dikw_core.providers.anthropic import AnthropicLLM


@pytest.fixture()
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``anthropic.AsyncAnthropic`` with a stub that records its kwargs."""
    captured: dict[str, object] = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAsyncAnthropic)
    return captured


def test_anthropic_client_uses_base_url_when_provided(
    captured_kwargs: dict[str, object],
) -> None:
    llm = AnthropicLLM(api_key="sk-explicit", base_url="http://fake.example/v1")
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("base_url") == "http://fake.example/v1"
    assert kwargs.get("api_key") == "sk-explicit"


def test_anthropic_client_omits_base_url_by_default(
    captured_kwargs: dict[str, object],
) -> None:
    llm = AnthropicLLM(api_key="sk-explicit")
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    # When base_url is unset, don't pass the kwarg at all — let the SDK use its default.
    assert "base_url" not in kwargs


def test_build_llm_wires_base_url_from_config(
    captured_kwargs: dict[str, object],
) -> None:
    cfg = ProviderConfig(llm="anthropic", llm_base_url="http://minimax.example/anthropic")
    llm = build_llm(cfg)
    assert isinstance(llm, AnthropicLLM)
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("base_url") == "http://minimax.example/anthropic"
