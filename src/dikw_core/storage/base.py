"""Storage Protocol: the only seam between the engine and any backend.

The engine depends solely on this module; backend implementations live in
sibling files and are resolved through ``storage/__init__.py``.

Design invariants:
  * Every argument and return value is a plain Pydantic DTO from ``schemas.py``.
  * No SQL, cursor objects, or ORM handles cross this boundary.
  * Hybrid search (RRF fusion, reranking) is built on top of ``fts_search`` +
    ``vec_search`` in ``info/search.py`` — NOT inside an adapter.
  * Each engine-level operation is one transactional unit of work on the adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ..schemas import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    FTSHit,
    Layer,
    LinkRecord,
    StorageCounts,
    VecHit,
    WikiLogEntry,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)


class StorageError(RuntimeError):
    """Base class for storage-adapter errors."""


class NotSupported(StorageError):
    """Raised by an adapter when an operation isn't supported in its mode.

    For example, the filesystem backend with ``embed=false`` raises this from
    ``vec_search`` so ``info/search.py`` can fall back to LLM-navigation mode.
    """


@runtime_checkable
class Storage(Protocol):
    """Abstract storage backend. Implementations: SQLite (MVP), Postgres, Filesystem."""

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def migrate(self) -> None:
        """Apply schema migrations idempotently. Safe to call on every startup."""
        ...

    # ---- D layer ---------------------------------------------------------

    async def put_content(self, hash_: str, body: str) -> None: ...
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]: ...
    async def deactivate_document(self, doc_id: str) -> None: ...

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(self, doc_id: str, chunks: Sequence[ChunkRecord]) -> None: ...
    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None: ...
    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None: ...
    async def fts_search(
        self,
        q: str,
        *,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[FTSHit]: ...
    async def vec_search(
        self,
        embedding: list[float],
        *,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[VecHit]: ...

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None: ...
    async def links_from(self, src_doc_id: str) -> list[LinkRecord]: ...
    async def links_to(self, dst_path: str) -> list[LinkRecord]: ...
    async def append_wiki_log(self, entry: WikiLogEntry) -> None: ...

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None: ...
    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]: ...
    async def set_wisdom_status(self, item_id: str, status: WisdomStatus) -> None: ...

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts: ...


__all__ = ["NotSupported", "Storage", "StorageError"]
