"""Gitee AI multimodal embedding provider — v1 default.

Wraps Gitee AI's hosted multimodal embedding endpoints (e.g. ``jina-clip-v2``)
behind the ``MultimodalEmbeddingProvider`` Protocol so chunk text and image
bytes share one vector space.

**Wire-format note (verify against current Gitee AI docs at deployment time):**

This implementation targets the OpenAI-compatible content-block surface that
Gitee AI exposes for multimodal models:

    POST {base_url}/v1/embeddings
    {
      "model": "<model name>",
      "input": [
        "plain text input",
        {"image_url": {"url": "data:image/png;base64,...."}},
        ...
      ]
    }

Different multimodal providers (Voyage, Cohere v4, native Jina) use slightly
different request shapes. If Gitee AI's endpoint diverges, override
``_serialize_input`` only — the Protocol surface and batching logic stay
the same.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from ..schemas import MultimodalInput
from .base import ProviderError

_DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
_DEFAULT_TIMEOUT = 60.0


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("DIKW_EMBEDDING_API_KEY")
    if not key:
        raise ProviderError(
            "DIKW_EMBEDDING_API_KEY is not set. Export it or pass `api_key` "
            "explicitly. The embedding leg never falls back to "
            "OPENAI_API_KEY (so LLM and embedding keys can differ)."
        )
    return key


def _serialize_input(inp: MultimodalInput) -> Any:
    """Convert one ``MultimodalInput`` into the wire-shape Gitee AI expects.

    Text-only → bare string (the simpler form most providers accept).
    With images → OpenAI-style content-block list with data-URL images.
    """
    if not inp.images:
        return inp.text or ""
    blocks: list[Any] = []
    if inp.text is not None:
        blocks.append({"type": "text", "text": inp.text})
    for img in inp.images:
        b64 = base64.b64encode(img.bytes).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img.mime};base64,{b64}"},
            }
        )
    return blocks


class GiteeMultimodalEmbedding:
    """``MultimodalEmbeddingProvider`` impl over Gitee AI's HTTP surface."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        batch: int = 16,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("DIKW_EMBEDDING_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key_explicit = api_key
        self._batch = batch
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def embed(
        self, inputs: list[MultimodalInput], *, model: str
    ) -> list[list[float]]:
        if not inputs:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        for start in range(0, len(inputs), self._batch):
            batch = inputs[start : start + self._batch]
            payload = {
                "model": model,
                "input": [_serialize_input(i) for i in batch],
            }
            resp = await client.post("/v1/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compat response: {"data": [{"index": int, "embedding": [...]}]}
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            if len(rows) != len(batch):
                raise ProviderError(
                    f"Gitee AI returned {len(rows)} vectors for "
                    f"{len(batch)} inputs"
                )
            out.extend([list(r["embedding"]) for r in rows])
        return out

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {_resolve_api_key(self._api_key_explicit)}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["GiteeMultimodalEmbedding"]
