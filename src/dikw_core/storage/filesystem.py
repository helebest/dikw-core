"""Filesystem storage adapter — DB-less, Obsidian-vault-native.

The entire index lives as JSONL sidecars under ``.dikw/fs/`` so a wiki is
a single portable directory you can zip, `git clone`, or open in Obsidian.
Scale is intentionally bounded (≤ a few hundred pages); beyond that the
plan expects users to migrate to the SQLite backend.

Design:

* **Source of truth** = the sidecars. The engine still writes human-readable
  markdown files through ``wisdom/io.py`` and ``knowledge/wiki.py`` — those
  are views, not the authority.
* **Append-only** = ``wiki_log.jsonl`` and ``wisdom_evidence.jsonl``.
  Everything else is fully rewritten on mutation. At this scale
  rewrite-on-every-write is simpler than reconciling append logs.
* **In-memory** = on ``connect()`` the adapter loads all sidecars into
  dicts and keeps them there; subsequent reads are O(N) over small N,
  writes flush the relevant sidecar to disk.
* **FTS** = naive inverted index built on demand; scored by
  query-term-presence weighted by IDF. Good enough at small scale.
* **Vectors** = per-chunk JSON sidecar under ``.dikw/fs/vecs/<chunk_id>.json``
  written only when embeddings are enabled. Search is pure-Python cosine.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

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
    LinkType,
    StorageCounts,
    VecHit,
    WikiLogEntry,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from .base import NotSupported, StorageError

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


class FilesystemStorage:
    """Storage Protocol impl backed by JSONL files under ``.dikw/fs/``."""

    DOC_FILE = "documents.jsonl"
    CONTENT_DIR = "content"
    CHUNKS_FILE = "chunks.jsonl"
    EMBED_META_FILE = "embed_meta.jsonl"
    EMBED_DIM_FILE = "embed_dim.txt"
    VECS_DIR = "vecs"
    LINKS_FILE = "links.jsonl"
    WIKI_LOG_FILE = "wiki_log.jsonl"
    WISDOM_ITEMS_FILE = "wisdom_items.jsonl"
    WISDOM_EVIDENCE_FILE = "wisdom_evidence.jsonl"
    CHUNK_COUNTER_FILE = "chunk_counter.txt"

    def __init__(self, root: str | Path, *, embed: bool = False) -> None:
        self._root = Path(root)
        self._embed_enabled = embed
        self._lock = asyncio.Lock()

        # In-memory caches populated by ``connect()``.
        self._docs: dict[str, DocumentRecord] = {}
        self._chunks: dict[int, ChunkRecord] = {}
        self._by_doc_chunks: dict[str, list[int]] = defaultdict(list)
        self._embed_meta: dict[int, str] = {}
        self._embeddings: dict[int, list[float]] = {}
        self._embedding_dim: int | None = None
        self._links: list[LinkRecord] = []
        self._wiki_log: list[WikiLogEntry] = []
        self._wisdom_items: dict[str, WisdomItem] = {}
        self._wisdom_evidence: dict[str, list[WisdomEvidence]] = defaultdict(list)
        self._chunk_counter = 0

    # ---- filesystem paths ------------------------------------------------

    def _p(self, name: str) -> Path:
        return self._root / name

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        def _load() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            (self._p(self.CONTENT_DIR)).mkdir(exist_ok=True)
            (self._p(self.VECS_DIR)).mkdir(exist_ok=True)

            # documents
            for obj in _read_jsonl(self._p(self.DOC_FILE)):
                doc = DocumentRecord(**obj)
                self._docs[doc.doc_id] = doc
            # chunks
            for obj in _read_jsonl(self._p(self.CHUNKS_FILE)):
                chunk = ChunkRecord(**obj)
                cid = chunk.chunk_id
                if cid is None:
                    continue
                self._chunks[cid] = chunk
                self._by_doc_chunks[chunk.doc_id].append(cid)
            # embed meta + vectors (only if on-disk)
            for obj in _read_jsonl(self._p(self.EMBED_META_FILE)):
                self._embed_meta[int(obj["chunk_id"])] = obj["model"]
            dim_file = self._p(self.EMBED_DIM_FILE)
            if dim_file.is_file():
                try:
                    self._embedding_dim = int(dim_file.read_text().strip())
                except ValueError:
                    self._embedding_dim = None
            if self._embedding_dim is not None and self._embed_enabled:
                for vec_file in (self._p(self.VECS_DIR)).iterdir():
                    if vec_file.suffix == ".json":
                        chunk_id = int(vec_file.stem)
                        data = json.loads(vec_file.read_text())
                        self._embeddings[chunk_id] = list(data["embedding"])
            # links
            for obj in _read_jsonl(self._p(self.LINKS_FILE)):
                self._links.append(
                    LinkRecord(
                        src_doc_id=obj["src_doc_id"],
                        dst_path=obj["dst_path"],
                        link_type=LinkType(obj["link_type"]),
                        anchor=obj.get("anchor"),
                        line=int(obj["line"]),
                    )
                )
            # wiki log
            for obj in _read_jsonl(self._p(self.WIKI_LOG_FILE)):
                self._wiki_log.append(WikiLogEntry(**obj))
            # wisdom
            for obj in _read_jsonl(self._p(self.WISDOM_ITEMS_FILE)):
                item = WisdomItem(**obj)
                self._wisdom_items[item.item_id] = item
            for obj in _read_jsonl(self._p(self.WISDOM_EVIDENCE_FILE)):
                ev = WisdomEvidence(
                    doc_id=obj["doc_id"],
                    excerpt=obj["excerpt"],
                    line=obj.get("line"),
                )
                self._wisdom_evidence[obj["item_id"]].append(ev)
            # chunk counter
            cf = self._p(self.CHUNK_COUNTER_FILE)
            if cf.is_file():
                try:
                    self._chunk_counter = int(cf.read_text().strip() or "0")
                except ValueError:
                    self._chunk_counter = 0

        await asyncio.to_thread(_load)

    async def close(self) -> None:
        # Nothing to release — flushes happen on every mutation.
        return None

    async def migrate(self) -> None:
        # No schema to apply; just ensure directories exist (idempotent).
        def _run() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            (self._p(self.CONTENT_DIR)).mkdir(exist_ok=True)
            (self._p(self.VECS_DIR)).mkdir(exist_ok=True)

        await asyncio.to_thread(_run)

    # ---- D layer ---------------------------------------------------------

    async def put_content(self, hash_: str, body: str) -> None:
        async with self._lock:
            path = self._p(self.CONTENT_DIR) / f"{hash_}.txt"
            if path.exists():
                return

            def _write() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")

            await asyncio.to_thread(_write)

    async def upsert_document(self, doc: DocumentRecord) -> None:
        async with self._lock:
            self._docs[doc.doc_id] = doc
            await self._flush_docs()

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    async def get_documents(
        self, doc_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        return [self._docs[did] for did in doc_ids if did in self._docs]

    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]:
        out: list[DocumentRecord] = []
        for doc in self._docs.values():
            if layer is not None and doc.layer != layer:
                continue
            if active is not None and doc.active != bool(active):
                continue
            if since_ts is not None and doc.mtime < since_ts:
                continue
            out.append(doc)
        return out

    async def deactivate_document(self, doc_id: str) -> None:
        async with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return
            self._docs[doc_id] = doc.model_copy(update={"active": False})
            await self._flush_docs()

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> list[int]:
        async with self._lock:
            # Drop existing chunks for this doc.
            for cid in list(self._by_doc_chunks.get(doc_id, [])):
                self._chunks.pop(cid, None)
                self._embed_meta.pop(cid, None)
                self._embeddings.pop(cid, None)
                vec_path = self._p(self.VECS_DIR) / f"{cid}.json"
                if vec_path.exists():
                    await asyncio.to_thread(vec_path.unlink)
            self._by_doc_chunks[doc_id] = []

            # Insert new ones with freshly-minted ids.
            assigned: list[int] = []
            for chunk in chunks:
                self._chunk_counter += 1
                cid = self._chunk_counter
                stored = chunk.model_copy(update={"chunk_id": cid, "doc_id": doc_id})
                self._chunks[cid] = stored
                self._by_doc_chunks[doc_id].append(cid)
                assigned.append(cid)

            await self._flush_chunks()
            await self._flush_counter()
            # embed_meta must drop stale rows
            await self._flush_embed_meta()
            return assigned

    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        if not rows:
            return
        if not self._embed_enabled:
            raise NotSupported(
                "embedding disabled — set storage.embed=true to enable vector search"
            )
        async with self._lock:
            dim = len(rows[0].embedding)
            if self._embedding_dim is None:
                self._embedding_dim = dim
                await asyncio.to_thread(
                    self._p(self.EMBED_DIM_FILE).write_text, str(dim), "utf-8"
                )
            elif self._embedding_dim != dim:
                raise StorageError(
                    f"embedding dim mismatch: index uses {self._embedding_dim}, got {dim}"
                )
            for row in rows:
                if len(row.embedding) != dim:
                    raise StorageError("inconsistent embedding dimensions in batch")
                self._embed_meta[row.chunk_id] = row.model
                self._embeddings[row.chunk_id] = list(row.embedding)
                vec_path = self._p(self.VECS_DIR) / f"{row.chunk_id}.json"
                await asyncio.to_thread(
                    vec_path.write_text,
                    json.dumps({"model": row.model, "embedding": row.embedding}),
                    "utf-8",
                )
            await self._flush_embed_meta()

    async def get_cached_embeddings(
        self, content_hashes: Sequence[str], *, model: str
    ) -> dict[str, list[float]]:
        # Pre-alpha: filesystem backend doesn't carry an embed cache;
        # re-ingest re-pays the provider. Add when needed.
        del content_hashes, model
        raise NotSupported("filesystem backend doesn't implement embed_cache")

    async def cache_embeddings(self, rows: Sequence[CachedEmbeddingRow]) -> None:
        del rows
        raise NotSupported("filesystem backend doesn't implement embed_cache")

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        return self._chunks.get(chunk_id)

    async def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    async def fts_search(
        self, q: str, *, limit: int = 20, layer: Layer | None = None
    ) -> list[FTSHit]:
        q_tokens = set(_tokenise(q))
        if not q_tokens:
            return []

        # Build (or rebuild) a cheap IDF table from the current chunk corpus.
        chunk_tokens: dict[int, set[str]] = {}
        df: dict[str, int] = defaultdict(int)
        for cid, chunk in self._chunks.items():
            doc = self._docs.get(chunk.doc_id)
            if doc is None or not doc.active:
                continue
            if layer is not None and doc.layer != layer:
                continue
            toks = set(_tokenise(chunk.text))
            chunk_tokens[cid] = toks
            for t in toks:
                df[t] += 1

        n_docs = max(len(chunk_tokens), 1)
        scored: list[tuple[float, int]] = []
        for cid, toks in chunk_tokens.items():
            hits = q_tokens & toks
            if not hits:
                continue
            score = sum(math.log(1 + n_docs / (df[t] + 1)) for t in hits)
            scored.append((score, cid))

        scored.sort(key=lambda s: s[0], reverse=True)

        # Fold multiple chunks of the same doc into one hit — the engine scores
        # over doc_ids, so we keep the best chunk per doc.
        best_by_doc: dict[str, tuple[float, int]] = {}
        for score, cid in scored:
            chunk = self._chunks[cid]
            if chunk.doc_id not in best_by_doc:
                best_by_doc[chunk.doc_id] = (score, cid)

        out: list[FTSHit] = []
        for doc_id, (score, cid) in sorted(
            best_by_doc.items(), key=lambda kv: kv[1][0], reverse=True
        ):
            chunk = self._chunks[cid]
            snippet = chunk.text.strip().replace("\n", " ")[:240]
            out.append(FTSHit(doc_id=doc_id, chunk_id=cid, score=score, snippet=snippet))
            if len(out) >= limit:
                break
        return out

    async def vec_search(
        self, embedding: list[float], *, limit: int = 20, layer: Layer | None = None
    ) -> list[VecHit]:
        if not self._embed_enabled or not self._embeddings:
            raise NotSupported(
                "no embeddings available — enable storage.embed and re-ingest"
            )
        if self._embedding_dim is not None and len(embedding) != self._embedding_dim:
            raise StorageError(
                f"query embedding dim {len(embedding)} != index dim {self._embedding_dim}"
            )

        q_norm = _norm(embedding)
        if q_norm == 0.0:
            return []

        hits: list[VecHit] = []
        for cid, vec in self._embeddings.items():
            chunk = self._chunks.get(cid)
            if chunk is None:
                continue
            doc = self._docs.get(chunk.doc_id)
            if doc is None or not doc.active:
                continue
            if layer is not None and doc.layer != layer:
                continue
            d_norm = _norm(vec)
            if d_norm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(embedding, vec, strict=True))
            cos = dot / (q_norm * d_norm)
            hits.append(VecHit(doc_id=chunk.doc_id, chunk_id=cid, distance=max(0.0, 1 - cos)))

        hits.sort(key=lambda h: h.distance)
        return hits[:limit]

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None:
        async with self._lock:
            # Dedup on (src, dst, line) triple.
            self._links = [
                existing
                for existing in self._links
                if not (
                    existing.src_doc_id == link.src_doc_id
                    and existing.dst_path == link.dst_path
                    and existing.line == link.line
                )
            ]
            self._links.append(link)
            await self._flush_links()

    async def links_from(self, src_doc_id: str) -> list[LinkRecord]:
        return [link for link in self._links if link.src_doc_id == src_doc_id]

    async def links_to(self, dst_path: str) -> list[LinkRecord]:
        return [link for link in self._links if link.dst_path == dst_path]

    async def append_wiki_log(self, entry: WikiLogEntry) -> None:
        async with self._lock:
            self._wiki_log.append(entry)
            await asyncio.to_thread(
                _append_jsonl, self._p(self.WIKI_LOG_FILE), entry.model_dump(mode="json")
            )

    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        rows = sorted(self._wiki_log, key=lambda e: e.ts)
        if since_ts is not None:
            rows = [r for r in rows if r.ts >= since_ts]
        if limit is not None:
            rows = rows[:limit]
        return rows

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None:
        async with self._lock:
            self._wisdom_items[item.item_id] = item
            self._wisdom_evidence[item.item_id] = list(evidence)
            await self._flush_wisdom()

    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]:
        items = list(self._wisdom_items.values())
        if status is not None:
            items = [i for i in items if i.status == status]
        if kind is not None:
            items = [i for i in items if i.kind == kind]
        items.sort(key=lambda i: i.created_ts, reverse=True)
        return items

    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        async with self._lock:
            item = self._wisdom_items.get(item_id)
            if item is None:
                return
            updates: dict[str, Any] = {"status": status}
            if approved_ts is not None:
                updates["approved_ts"] = approved_ts
            self._wisdom_items[item_id] = item.model_copy(update=updates)
            await self._flush_wisdom()

    async def get_wisdom(self, item_id: str) -> WisdomItem | None:
        return self._wisdom_items.get(item_id)

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        return list(self._wisdom_evidence.get(item_id, []))

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts:
        by_layer: dict[str, int] = defaultdict(int)
        for doc in self._docs.values():
            if doc.active:
                by_layer[doc.layer.value] += 1
        by_status: dict[str, int] = defaultdict(int)
        for item in self._wisdom_items.values():
            by_status[item.status.value] += 1
        last_ts = max((e.ts for e in self._wiki_log), default=None)
        return StorageCounts(
            documents_by_layer=dict(by_layer),
            chunks=len(self._chunks),
            embeddings=len(self._embed_meta),
            links=len(self._links),
            wisdom_by_status=dict(by_status),
            last_wiki_log_ts=last_ts,
        )

    # ---- multimedia assets (Phase 5: not yet implemented) ----------------
    #
    # The filesystem adapter's asset / version-aware embedding support lands
    # in a follow-up phase alongside the Postgres impl. Until then every new
    # method raises NotSupported so callers (and contract tests) can detect
    # and skip cleanly without a misleading "method missing" error.

    async def upsert_asset(self, asset: AssetRecord) -> None:
        raise NotSupported("filesystem adapter: assets not implemented yet")

    async def get_asset(self, asset_id: str) -> AssetRecord | None:
        raise NotSupported("filesystem adapter: assets not implemented yet")

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        raise NotSupported("filesystem adapter: chunk_asset_refs not implemented yet")

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        raise NotSupported("filesystem adapter: chunk_asset_refs not implemented yet")

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        raise NotSupported("filesystem adapter: chunk_asset_refs not implemented yet")

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        raise NotSupported("filesystem adapter: asset embeddings not implemented yet")

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        raise NotSupported("filesystem adapter: asset embeddings not implemented yet")

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        raise NotSupported("filesystem adapter: embed versioning not implemented yet")

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        raise NotSupported("filesystem adapter: embed versioning not implemented yet")

    async def list_embed_versions(self) -> list[EmbeddingVersion]:
        raise NotSupported("filesystem adapter: embed versioning not implemented yet")

    # ---- flushers --------------------------------------------------------

    async def _flush_docs(self) -> None:
        payload = [doc.model_dump(mode="json") for doc in self._docs.values()]
        await asyncio.to_thread(_write_jsonl, self._p(self.DOC_FILE), payload)

    async def _flush_chunks(self) -> None:
        payload = [chunk.model_dump(mode="json") for chunk in self._chunks.values()]
        await asyncio.to_thread(_write_jsonl, self._p(self.CHUNKS_FILE), payload)

    async def _flush_counter(self) -> None:
        await asyncio.to_thread(
            self._p(self.CHUNK_COUNTER_FILE).write_text, str(self._chunk_counter), "utf-8"
        )

    async def _flush_embed_meta(self) -> None:
        payload = [
            {"chunk_id": cid, "model": model}
            for cid, model in self._embed_meta.items()
        ]
        await asyncio.to_thread(_write_jsonl, self._p(self.EMBED_META_FILE), payload)

    async def _flush_links(self) -> None:
        payload = [
            {
                "src_doc_id": link.src_doc_id,
                "dst_path": link.dst_path,
                "link_type": link.link_type.value,
                "anchor": link.anchor,
                "line": link.line,
            }
            for link in self._links
        ]
        await asyncio.to_thread(_write_jsonl, self._p(self.LINKS_FILE), payload)

    async def _flush_wisdom(self) -> None:
        items_payload = [item.model_dump(mode="json") for item in self._wisdom_items.values()]
        evidence_payload: list[dict[str, Any]] = []
        for item_id, rows in self._wisdom_evidence.items():
            for ev in rows:
                evidence_payload.append(
                    {
                        "item_id": item_id,
                        "doc_id": ev.doc_id,
                        "excerpt": ev.excerpt,
                        "line": ev.line,
                    }
                )
        await asyncio.to_thread(
            _write_jsonl, self._p(self.WISDOM_ITEMS_FILE), items_payload
        )
        await asyncio.to_thread(
            _write_jsonl, self._p(self.WISDOM_EVIDENCE_FILE), evidence_payload
        )


# ---- helpers -------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))
