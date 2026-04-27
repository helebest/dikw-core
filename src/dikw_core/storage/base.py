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
from typing import Literal, Protocol, runtime_checkable

from ..schemas import (
    AssetEmbeddingRow,
    AssetRecord,
    AssetVecHit,
    CachedEmbeddingRow,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
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

    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def get_documents(
        self, doc_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        """Batch-fetch documents by id. Missing ids are dropped silently —
        the caller key-by-id when they need a hit/miss distinction.

        Single-query equivalent of looping ``get_document``; chunk-level
        retrieval calls this on every search to avoid N+1 over repeating
        ``doc_id``s in the hit list.
        """
        ...
    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]: ...
    async def deactivate_document(self, doc_id: str) -> None: ...

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> list[int]:
        """Replace all chunks for ``doc_id``. Return the assigned ``chunk_id``s
        in the same ``seq`` order as the input so the caller can pair
        embeddings with persisted rows."""
        ...
    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None: ...

    # Content-hash embed cache. Decouples vector reuse from chunks.chunk_id
    # so re-ingest under replace_chunks's delete-and-reinsert semantics
    # doesn't lose API spend on byte-identical chunk text. Adapters that
    # don't implement the cache raise ``NotSupported``.
    async def get_cached_embeddings(
        self, content_hashes: Sequence[str], *, version_id: int
    ) -> dict[str, list[float]]:
        """Batch lookup keyed by ``sha256(chunk.text)`` for a given version.

        Returns a dict mapping content_hash -> vector for HITS only;
        missing hashes are misses (absent from the dict). Empty input
        is a no-op returning an empty dict.
        """
        ...

    async def cache_embeddings(self, rows: Sequence[CachedEmbeddingRow]) -> None:
        """Idempotent batch insert.

        ``(content_hash, version_id)`` is the primary key; collisions are
        no-ops (do NOT overwrite — vectors for the same content under
        the same version identity must be deterministic). Empty input
        is a no-op.
        """
        ...

    async def list_chunks_missing_embedding(
        self, *, version_id: int
    ) -> list[ChunkRecord]:
        """Chunks present in storage with no ``chunk_embed_meta`` row for ``version_id``.

        Used by the resume-scan path in ``api.ingest``: after a mid-flight
        crash the doc-level shortcut on retry skips docs whose hash already
        landed, but their chunks may have only partially embedded. This
        method surfaces the missing tail so the caller can re-run them
        (cache hits make most of those free; only true misses re-pay
        the provider).
        """
        ...

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None: ...
    async def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        """Batch-fetch chunks by id. Missing ids are dropped silently.

        Single-query equivalent of looping ``get_chunk``; chunk-level
        retrieval needs every retrieved chunk's body + seq, and going
        through ``get_chunk`` per hit would N+1 the connection.
        """
        ...
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
        version_id: int | None = None,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[VecHit]:
        """ANN search over chunk embeddings for ``version_id``.

        ``version_id=None`` means "the active text version" — adapters
        resolve via ``get_active_embed_version(modality="text")`` and
        raise ``NotSupported`` if no text embeddings have been indexed
        yet. Pass an explicit ``version_id`` to search a non-active
        version (e.g., during eval ablations or post-swap migrations).
        """
        ...

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None: ...
    async def links_from(self, src_doc_id: str) -> list[LinkRecord]: ...
    async def links_to(self, dst_path: str) -> list[LinkRecord]: ...
    async def append_wiki_log(self, entry: WikiLogEntry) -> None: ...
    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        """Return wiki-log entries in chronological order."""
        ...

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
    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        """Update a wisdom item's status. Pass ``approved_ts`` to stamp approvals."""
        ...

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        """Return evidence rows attached to ``item_id`` in insert order."""
        ...

    async def get_wisdom(self, item_id: str) -> WisdomItem | None: ...

    # ---- D layer: multimedia assets --------------------------------------

    async def upsert_asset(self, asset: AssetRecord) -> None:
        """Insert or replace an ``AssetRecord``.

        Idempotent by ``asset_id`` (= sha256). Adapters that already have
        a row at this id should preserve ``original_paths`` semantics
        themselves only if explicitly told to merge — the materialize
        layer in ``data/assets.py`` is what dedup-merges entries; this
        method is a plain replace.
        """
        ...

    async def get_asset(self, asset_id: str) -> AssetRecord | None: ...

    # ---- I layer: chunk ↔ asset bridge -----------------------------------

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        """Replace all ``chunk_asset_refs`` rows for ``chunk_id`` in one shot."""
        ...

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        """Return refs grouped by chunk_id, each list sorted by ``ord``.
        Missing chunk_ids map to empty lists."""
        ...

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        """Reverse-lookup: for each asset_id, the chunk_ids that reference it.
        Used by hybrid search to promote asset-vec hits to their parent
        chunks via the ``chunk_asset_refs`` bridge."""
        ...

    # ---- I layer: asset embeddings (multimodal) --------------------------

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        """Persist asset-level embedding vectors. The vector dimension
        must match the dim of the row's ``version_id`` in
        ``embed_versions``; otherwise raise ``StorageError``."""
        ...

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        """ANN search against the asset vector table for ``version_id``."""
        ...

    # ---- Embedding versioning --------------------------------------------

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        """Idempotent upsert of an embedding version identity.

        Match key is ``(provider, model, revision, dim, normalize, distance, modality)``.
        On a hit, returns the existing ``version_id`` and leaves
        ``is_active`` untouched. On a miss, inserts a fresh row and
        marks every other version of the same ``modality`` as
        ``is_active = 0`` so the new one becomes the sole active version.
        """
        ...

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None: ...

    async def list_embed_versions(
        self, *, modality: Literal["text", "multimodal"] | None = None
    ) -> list[EmbeddingVersion]:
        """Return embedding versions in registration order. ``modality=None``
        returns every version; pass ``"text"`` or ``"multimodal"`` to
        filter."""
        ...

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts: ...


__all__ = ["NotSupported", "Storage", "StorageError"]
