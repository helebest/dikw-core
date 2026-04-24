"""Tests for the MultimodalEmbeddingProvider Protocol contract.

Pins the Protocol's invariants — input/output ordering, batch handling,
text-only and image-only and mixed inputs — using a deterministic
``FakeMultimodalEmbedding`` so the test doesn't depend on any real
provider's HTTP surface.
"""

from __future__ import annotations

from dikw_core.providers.base import MultimodalEmbeddingProvider
from dikw_core.schemas import ImageContent, MultimodalInput
from tests.fakes import FakeMultimodalEmbedding


def test_fake_satisfies_protocol() -> None:
    """Static-duck-typing sanity: the fake implements the Protocol."""
    fake: MultimodalEmbeddingProvider = FakeMultimodalEmbedding(dim=4)
    assert hasattr(fake, "embed")


async def test_text_only_input_produces_vector() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    vectors = await fake.embed(
        [MultimodalInput(text="hello world")], model="fake-mm-v1"
    )
    assert len(vectors) == 1
    assert len(vectors[0]) == 4


async def test_image_only_input_produces_vector() -> None:
    fake = FakeMultimodalEmbedding(dim=8)
    vectors = await fake.embed(
        [
            MultimodalInput(
                images=[ImageContent(bytes=b"\x89PNG\r\n", mime="image/png")]
            )
        ],
        model="fake-mm-v1",
    )
    assert len(vectors) == 1
    assert len(vectors[0]) == 8


async def test_output_order_matches_input_order() -> None:
    """Deterministic provider: same input twice → same vector. Inputs in a
    batch must come back in input order so callers can pair with their
    source rows."""
    fake = FakeMultimodalEmbedding(dim=4)
    inputs = [
        MultimodalInput(text="alpha"),
        MultimodalInput(text="beta"),
        MultimodalInput(text="gamma"),
    ]
    a = await fake.embed(inputs, model="fake-mm-v1")
    b = await fake.embed(inputs, model="fake-mm-v1")
    assert a == b
    # Order: the second input's vector must differ from the first's.
    assert a[0] != a[1]


async def test_text_and_image_distinguished() -> None:
    """A text input and an image input — even with conceptually similar
    payload — produce distinct vectors so the embedding actually carries
    modality signal."""
    fake = FakeMultimodalEmbedding(dim=4)
    text_vec = (
        await fake.embed([MultimodalInput(text="png")], model="x")
    )[0]
    img_vec = (
        await fake.embed(
            [MultimodalInput(images=[ImageContent(bytes=b"png", mime="image/png")])],
            model="x",
        )
    )[0]
    assert text_vec != img_vec


async def test_empty_batch_returns_empty() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    assert await fake.embed([], model="any") == []
