"""Gitee AI multimodal embedding provider — v1 default.

Wraps Gitee AI's hosted multimodal embedding endpoints (e.g. ``jina-clip-v2``)
behind the ``MultimodalEmbeddingProvider`` Protocol so chunk text and image
bytes share one vector space.

**Wire-format note (verify against current Gitee AI docs at deployment time):**

This implementation targets the OpenAI-compatible content-block surface that
Gitee AI exposes for multimodal models:

    POST {base_url}/embeddings        # base_url already ends in /v1
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
from .openai_compat import _resolve_embedding_api_key

_DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
_DEFAULT_TIMEOUT = 60.0


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
            resp = await client.post(f"{self._base_url}/embeddings", json=payload)
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
                    "Authorization": f"Bearer {_resolve_embedding_api_key(self._api_key_explicit)}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _serialize_input_qwen_vl(inp: MultimodalInput) -> Any:
    """Convert one ``MultimodalInput`` into the wire-shape Qwen3-VL expects.

    Qwen3-VL-Embedding-8B on Gitee AI rejects the OpenAI-vision content-block
    shape that ``_serialize_input`` uses for jina-clip-v2 (``[{"type": "text"
    ...}, {"type": "image_url" ...}]`` returns 400 ``No schema matches``).
    Its accepted form is per-modality input dicts:

        [{"text": "..."}, {"image": "data:..."}]

    Each ``MultimodalInput`` becomes one wire input. The schema permits
    text+image combinations on a single ``MultimodalInput`` for hypothetical
    joint encoding, but Qwen3-VL has no joint-encode mode — it embeds
    per-modality independently. The v1 pipeline never produces combined
    inputs in practice (chunks are text-only, assets are image-only), so
    we reject the combined case explicitly to surface the mismatch loudly
    instead of silently dropping a modality.
    """
    has_text = inp.text is not None and inp.text != ""
    has_images = bool(inp.images)
    if has_text and has_images:
        raise ProviderError(
            "Qwen3-VL embeds per-modality; got a MultimodalInput with both "
            "text and images. Split into separate inputs."
        )
    if has_text:
        return {"text": inp.text}
    if has_images:
        if len(inp.images) > 1:
            raise ProviderError(
                "Qwen3-VL accepts one image per input; got "
                f"{len(inp.images)}. Split across multiple inputs."
            )
        img = inp.images[0]
        b64 = base64.b64encode(img.bytes).decode("ascii")
        return {"image": f"data:{img.mime};base64,{b64}"}
    # Empty input: send empty text to keep the 1:1 input/output shape.
    # The provider returns a usable (if uninformative) zero-ish vector;
    # Storage's vec_search filters NULL distances so degenerate hits drop.
    return {"text": ""}


class QwenVLMultimodalEmbedding:
    """``MultimodalEmbeddingProvider`` impl for Qwen3-VL on Gitee AI.

    Same HTTP surface as ``GiteeMultimodalEmbedding`` but with a different
    per-input wire shape (see ``_serialize_input_qwen_vl``). Output dim is
    1024 for ``Qwen3-VL-Embedding-8B``.
    """

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
                "input": [_serialize_input_qwen_vl(i) for i in batch],
            }
            resp = await client.post(f"{self._base_url}/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            if len(rows) != len(batch):
                raise ProviderError(
                    f"Qwen3-VL endpoint returned {len(rows)} vectors for "
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
                    "Authorization": f"Bearer {_resolve_embedding_api_key(self._api_key_explicit)}",
                    "Content-Type": "application/json",
                },
                limits=httpx.Limits(max_keepalive_connections=0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["GiteeMultimodalEmbedding", "QwenVLMultimodalEmbedding"]
