"""Cross-layer record types shared by the engine and storage adapters.

These DTOs cross the Storage Protocol boundary, so they must stay backend-agnostic:
no SQL types, no ORM handles, no cursors.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


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
