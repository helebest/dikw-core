"""OpenAI Codex provider — ChatGPT-backend Responses API with auto-refreshing OAuth.

Distinct from ``openai_compat.py`` despite sharing the openai SDK: Codex
speaks the **Responses API** (``client.responses.create``), authenticates
with a ChatGPT-issued OAuth ``access_token`` (resolved + refreshed via
``codex_auth``, not ``OPENAI_API_KEY``), and requires Cloudflare-mitigation
headers (``originator``, ``ChatGPT-Account-ID`` from the JWT). ``gpt-5.5``
and the rest of the codex model family live exclusively on
``https://chatgpt.com/backend-api/codex``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .base import LLMResponse, LLMStreamEvent, ToolSpec
from .codex_auth import account_id_from_jwt, resolve_access_token

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Cloudflare requires these headers on every chatgpt.com/backend-api/codex
# request — without them the gateway returns 403 before the request hits
# the upstream model. The originator string is the literal codex CLI
# reports; matching it is the only way to satisfy the gate today.
_CODEX_BASE_HEADERS: dict[str, str] = {
    "originator": "codex_cli_rs",
    "User-Agent": "codex_cli_rs/0.1 (dikw-core)",
}

_DEFAULT_TIMEOUT_SECONDS = 60.0

_FINISH_REASON_MAP: dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
    "cancelled": "stop",
}


def _build_async_client(
    *,
    base_url: str,
    access_token: str,
    max_retries: int | None,
    timeout_seconds: float | None,
) -> AsyncOpenAI:
    """Construct a fresh ``AsyncOpenAI`` for one request lifecycle.

    We rebuild per-call rather than caching: the OAuth access_token is
    short-lived and a stale client cached across token refreshes would
    silently 401. The ``_is_expiring`` check is nanosecond-cheap; the
    rebuild costs are dominated by httpx connection setup, comparable to
    the SDK's own behaviour on cache miss.
    """
    import httpx
    from openai import AsyncOpenAI

    headers: dict[str, str] = dict(_CODEX_BASE_HEADERS)
    account_id = account_id_from_jwt(access_token)
    if account_id is not None:
        headers["ChatGPT-Account-ID"] = account_id

    seconds = (
        timeout_seconds if timeout_seconds is not None else _DEFAULT_TIMEOUT_SECONDS
    )
    timeout = httpx.Timeout(connect=10.0, read=seconds, write=seconds, pool=5.0)

    kwargs: dict[str, Any] = {
        "api_key": access_token,
        "base_url": base_url,
        "default_headers": headers,
        "timeout": timeout,
        # Disable connection keepalive for the same reason openai_compat
        # does: idle keepalives behind aggressive proxies (Cloudflare in
        # this case) can drop silently and turn the SDK's retry path into
        # a multi-minute hang on the same dead socket.
        "http_client": httpx.AsyncClient(
            timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0)
        ),
    }
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return AsyncOpenAI(**kwargs)


def _extract_text_from_response(response: Any) -> str:
    """Walk ``response.output``, gather output_text from message items.

    Reasoning items, tool_call items, and any other type-tagged items are
    skipped — Codex's response.output is a heterogeneous list, only the
    ``message`` items carry user-facing text.
    """
    parts: list[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) == "output_text":
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


class OpenAICodexLLM:
    def __init__(
        self,
        *,
        base_url: str,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
        # Test seam: pre-resolved access_token bypasses ~/.codex/auth.json
        # I/O entirely. Production callers leave this None and
        # resolve_access_token() walks the standard codex_home flow on
        # every request.
        access_token_override: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._access_token_override = access_token_override

    def _resolve_token(self) -> str:
        if self._access_token_override is not None:
            return self._access_token_override
        return resolve_access_token()

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
        # Tools aren't translated into Responses API function-call format
        # yet — synth/distill/query are plain-text completions today, same
        # as the other two providers.
        _ = tools
        token = self._resolve_token()
        client = _build_async_client(
            base_url=self._base_url,
            access_token=token,
            max_retries=self._max_retries,
            timeout_seconds=self._timeout_seconds,
        )
        try:
            response = await client.responses.create(
                model=model,
                instructions=system,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user}],
                    }
                ],
                store=False,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

        text = _extract_text_from_response(response)
        status = str(getattr(response, "status", "") or "")
        finish_reason = _FINISH_REASON_MAP.get(status, "stop")
        usage = _extract_usage(response)
        return LLMResponse(text=text, finish_reason=finish_reason, usage=usage)

    def complete_stream(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        # Streaming lands in the next commit.
        _ = (system, user, model, max_tokens, temperature, tools)
        raise NotImplementedError(
            "OpenAICodexLLM.complete_stream is not yet implemented"
        )
