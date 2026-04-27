-- Content-addressed embedding cache (Postgres).
--
-- Mirrors the SQLite schema (storage/migrations/sqlite/003_embed_cache.sql).
-- BYTEA stores the same struct-packed float32 little-endian bytes as the
-- SQLite BLOB so the cross-adapter cache contract is byte-exact. Keying
-- on ``version_id`` keeps the cache strictly version-isolated; the FK
-- to ``embed_versions`` (defined in 003_assets.sql) guarantees the cache
-- never holds vectors for an unregistered identity.

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT             NOT NULL,
    version_id   BIGINT           NOT NULL REFERENCES embed_versions(version_id),
    dim          INTEGER          NOT NULL,
    embedding    BYTEA            NOT NULL,
    created_ts   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (content_hash, version_id)
);
