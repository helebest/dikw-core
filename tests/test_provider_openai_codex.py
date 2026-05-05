"""``OpenAICodexLLM.complete()`` and the SDK plumbing around it.

Mirrors the shape of ``test_provider_openai_compat_retries.py``:
monkeypatch ``openai.AsyncOpenAI`` with a stub that captures init
kwargs + ``responses.create`` calls so we never touch the network.
``access_token_override`` is the production-time test seam — passing it
bypasses ``codex_auth.resolve_access_token`` entirely so these tests
don't need a fake ``~/.codex/auth.json``.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dikw_core.providers.base import LLMResponse
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(claims: dict[str, Any]) -> str:
    header = _b64url(json.dumps({"alg": "none"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    return f"{header}.{payload}.sig"


def _make_response(
    *,
    text: str = "hello",
    status: str = "completed",
    input_tokens: int = 5,
    output_tokens: int = 7,
    output: list[Any] | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like ``openai.types.responses.Response``."""
    if output is None:
        output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
    return SimpleNamespace(
        output=output,
        status=status,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "init_kwargs": None,
        "create_kwargs": None,
        "next_response": _make_response(),
        "close_calls": 0,
    }

    class FakeResponses:
        async def create(self, **kwargs: Any) -> Any:
            rec["create_kwargs"] = kwargs
            return rec["next_response"]

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            rec["close_calls"] += 1

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    return rec


# --------------------------------------------------------------------------- #
# Construction + auth header injection
# --------------------------------------------------------------------------- #


async def test_complete_passes_explicit_base_url(captured: dict[str, Any]) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="test-token"
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["base_url"] == DEFAULT_CODEX_BASE_URL


async def test_complete_passes_access_token_as_api_key(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="my-secret-token"
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["api_key"] == "my-secret-token"


async def test_complete_passes_codex_cloudflare_headers(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="plain-token"
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")


async def test_complete_includes_account_id_when_token_is_jwt(
    captured: dict[str, Any],
) -> None:
    token = _make_jwt({"chatgpt_account_id": "acc-42", "exp": 9_999_999_999})
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override=token
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["ChatGPT-Account-ID"] == "acc-42"


async def test_complete_omits_account_id_for_non_jwt_token(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="plain-not-jwt"
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert "ChatGPT-Account-ID" not in headers


# --------------------------------------------------------------------------- #
# responses.create kwargs shape
# --------------------------------------------------------------------------- #


async def test_complete_calls_responses_create_with_responses_api_shape(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    await provider.complete(
        system="be helpful",
        user="hello world",
        model="gpt-5.5",
        max_tokens=512,
        temperature=0.4,
    )
    kwargs = captured["create_kwargs"]
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "be helpful"
    assert kwargs["store"] is False
    assert kwargs["max_output_tokens"] == 512
    assert kwargs["temperature"] == 0.4
    # Input is the Responses API shape — list of items with content parts.
    assert kwargs["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello world"}],
        }
    ]


async def test_complete_does_not_pass_messages_kwarg(
    captured: dict[str, Any],
) -> None:
    """Regression: Responses API uses `instructions` + `input`, NOT `messages`."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert "messages" not in captured["create_kwargs"]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


async def test_complete_returns_text_from_output_messages(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(text="hello world")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"


async def test_complete_concatenates_multiple_output_text_parts(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="hello "),
                    SimpleNamespace(type="output_text", text="world"),
                ],
            )
        ],
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.text == "hello world"


async def test_complete_skips_non_message_output_items(
    captured: dict[str, Any],
) -> None:
    """reasoning items and tool_call items must not bleed into ``text``."""
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", summary="thought"),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="answer")],
            ),
        ],
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.text == "answer"


async def test_complete_maps_status_completed_to_stop(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(status="completed")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "stop"


async def test_complete_maps_status_incomplete_to_length(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(status="incomplete")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "length"


async def test_complete_extracts_usage_input_output_tokens(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(input_tokens=42, output_tokens=99)
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.usage == {"input_tokens": 42, "output_tokens": 99}


async def test_complete_handles_response_without_usage(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="x")],
            )
        ],
        status="completed",
        usage=None,
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, access_token_override="t"
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.usage == {}
