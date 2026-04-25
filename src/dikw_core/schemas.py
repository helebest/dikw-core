"""Cross-layer record types shared by the engine and storage adapters.

These DTOs cross the Storage Protocol boundary, so they must stay backend-agnostic:
no SQL types, no ORM handles, no cursors.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Layer(StrEnum):
    """Which DIKW layer a record belongs to."""

    SOURCE = "source"
    WIKI = "wiki"
    WISDOM = "wisdom"


class LinkType(StrEnum):
    WIKILINK = "wikilink"
    MARKDOWN = "markdown"
    URL = "url"


class WisdomKind(StrEnum):
    PRINCIPLE = "principle"
    LESSON = "lesson"
    PATTERN = "pattern"


class WisdomStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    ARCHIVED = "archived"


class AssetKind(StrEnum):
    """Multimedia asset modality. v1 ships ``IMAGE`` only; ``AUDIO`` /
    ``VIDEO`` are reserved for v2 transcription support."""

    IMAGE = "image"


class DocumentRecord(BaseModel):
    doc_id: str
    path: str
    title: str | None = None
    hash: str
    mtime: float
    layer: Layer
    active: bool = True


class ChunkRecord(BaseModel):
    chunk_id: int | None = None  # assigned by storage on insert
    doc_id: str
    seq: int
    start: int
    end: int
    text: str


class EmbeddingRow(BaseModel):
    chunk_id: int
    model: str
    embedding: list[float]


class FTSHit(BaseModel):
    doc_id: str
    chunk_id: int | None = None
    score: float
    snippet: str | None = None


class VecHit(BaseModel):
    doc_id: str
    chunk_id: int
    distance: float  # smaller = more similar (cosine distance)


class LinkRecord(BaseModel):
    src_doc_id: str
    dst_path: str
    link_type: LinkType
    anchor: str | None = None
    line: int


class WikiLogEntry(BaseModel):
    ts: float
    action: Literal["ingest", "synth", "distill", "review", "lint", "delete"]
    src: str | None = None
    dst: str | None = None
    note: str | None = None


class WisdomEvidence(BaseModel):
    doc_id: str
    excerpt: str
    line: int | None = None


class WisdomItem(BaseModel):
    item_id: str
    kind: WisdomKind
    status: WisdomStatus = WisdomStatus.CANDIDATE
    path: str | None = None
    title: str
    body: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    created_ts: float
    approved_ts: float | None = None


class StorageCounts(BaseModel):
    """Diagnostic counts surfaced by `Storage.counts()`."""

    documents_by_layer: dict[str, int] = Field(default_factory=dict)
    chunks: int = 0
    embeddings: int = 0
    links: int = 0
    wisdom_by_status: dict[str, int] = Field(default_factory=dict)
    last_wiki_log_ts: float | None = None
    assets: int = 0
    asset_embeddings: int = 0


# ---- Media / Assets ------------------------------------------------------


class AssetRef(BaseModel):
    """One image reference encountered while parsing a markdown body.

    ``start`` / ``end`` are character offsets into the *original* document
    body (the same coordinate space ``ChunkRecord`` uses), so the chunker
    can keep the reference intact and ``chunk_asset_refs`` can be populated
    after chunking with chunk-relative offsets.

    ``original_path`` is canonicalised to forward slashes at construction
    so Windows-pasted backslash paths resolve on POSIX CI.
    """

    original_path: str
    alt: str = ""
    start: int
    end: int
    syntax: Literal["markdown", "wikilink"]

    @field_validator("original_path")
    @classmethod
    def _forward_slashes(cls, v: str) -> str:
        return v.replace("\\", "/")


class ImageContent(BaseModel):
    """Raw image payload destined for a multimodal embedding provider."""

    bytes: bytes
    mime: str  # "image/png" | "image/jpeg" | "image/webp" | "image/gif"


class MultimodalInput(BaseModel):
    """One input unit for a ``MultimodalEmbeddingProvider``.

    v1 uses either ``text`` or ``images`` (not both); the schema permits
    arbitrary combinations so v1.5 chunk-with-images joint encoding can
    land without breaking the wire format.
    """

    text: str | None = None
    images: list[ImageContent] = Field(default_factory=list)


class AssetRecord(BaseModel):
    """A multimedia asset materialized into the engine-managed vault path.

    ``asset_id`` is the sha256 of the bytes and is the single source of
    truth for identity; ``hash`` is kept as a separate column so it can
    be queried/indexed without parsing the asset_id.
    """

    asset_id: str
    hash: str
    kind: AssetKind
    mime: str
    stored_path: str  # relative to project_root, e.g. "assets/ab/ab3f12ef-foo.png"
    original_paths: list[str] = Field(default_factory=list)
    bytes: int
    width: int | None = None
    height: int | None = None
    caption: str | None = None  # v1 stays None; v1.5 backfills via VisionProvider
    caption_model: str | None = None
    created_ts: float


class ChunkAssetRef(BaseModel):
    """Many-to-many bridge: which assets a text chunk references, and where
    inside the chunk text each reference lives."""

    chunk_id: int
    asset_id: str
    ord: int  # ordinal within the chunk (0, 1, 2, …)
    alt: str = ""
    start_in_chunk: int
    end_in_chunk: int


class AssetEmbeddingRow(BaseModel):
    """One asset-level embedding, written to ``vec_assets_v<version_id>``.

    The vector dimension is fixed by ``EmbeddingVersion.dim``; mismatched
    rows are rejected at the storage boundary."""

    asset_id: str
    version_id: int
    embedding: list[float]


class AssetVecHit(BaseModel):
    """One result row from ``Storage.vec_search_assets``."""

    asset_id: str
    distance: float  # smaller = more similar (cosine distance, by default)


# ---- Embedding versioning ------------------------------------------------


class EmbeddingVersion(BaseModel):
    """Composite identity for an embedding generation.

    Any field different = different version = different vector table. The
    storage layer enforces ``UNIQUE (provider, model, revision, dim,
    normalize, distance)`` and assigns ``version_id``.

    ``revision`` is the user-facing escape hatch when a provider silently
    refreshes weights behind a stable model name (e.g. OpenAI's monthly
    "text-embedding-3-small" reroll); bumping it triggers a new version
    row and a fresh ``vec_*_v<id>`` table without touching any other knob.
    """

    version_id: int | None = None  # assigned by storage on insert
    provider: str
    model: str
    revision: str = ""
    dim: int
    normalize: bool
    distance: Literal["cosine", "l2", "dot"]
    modality: Literal["text", "multimodal"]
    created_ts: float | None = None
    is_active: bool = True
