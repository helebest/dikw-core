-- Content-addressed embedding cache.
--
-- Decoupled from chunks.chunk_id so re-ingest under replace_chunks's
-- delete-and-reinsert semantics doesn't lose API spend on byte-identical
-- chunk text.
--
-- Lookup by (sha256(chunk.text), version_id). Keying on version_id
-- (rather than the raw model name) makes the cache strictly version-
-- isolated: a normalize/distance flip without a model name change is
-- now a different version_id and therefore a different cache key, so
-- there is no way for stale vectors to leak across version boundaries.
--
-- ``dim`` duplicates len(embedding) but is stored explicitly so a
-- future cleanup query can filter cheaply without unpacking blobs.
-- ``created_ts`` pins insertion time for a future LRU/TTL eviction.
--
-- Vector blob format: float32 little-endian, same packing as
-- storage/sqlite.py::_serialize_vec.

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT    NOT NULL,
    version_id   INTEGER NOT NULL REFERENCES embed_versions(version_id),
    dim          INTEGER NOT NULL,
    embedding    BLOB    NOT NULL,
    created_ts   REAL    NOT NULL,
    PRIMARY KEY (content_hash, version_id)
);
