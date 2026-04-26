-- Content-addressed embedding cache.
--
-- Decoupled from chunks.chunk_id so re-ingest under replace_chunks's
-- delete-and-reinsert semantics (FK ON DELETE CASCADE wipes embed_meta)
-- doesn't lose API spend on byte-identical chunk text.
--
-- Lookup by (sha256(chunk.text), model). The dim column duplicates
-- len(embedding) but is stored explicitly so a future cleanup query can
-- filter cheaply without unpacking blobs. created_ts pins insertion
-- time for a future LRU/TTL eviction (out of scope for the initial PR).
--
-- Vector blob format: float32 little-endian, same packing as
-- storage/sqlite.py::_serialize_vec, so the cache and the live
-- chunks_vec virtual table share serialization.

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    BLOB NOT NULL,
    created_ts   REAL NOT NULL,
    PRIMARY KEY (content_hash, model)
);
