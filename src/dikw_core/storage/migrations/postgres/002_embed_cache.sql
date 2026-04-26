-- Content-addressed embedding cache (Postgres).
--
-- Mirrors the SQLite schema (storage/migrations/sqlite/003_embed_cache.sql).
-- BYTEA stores the same struct-packed float32 little-endian bytes as the
-- SQLite BLOB so the cross-adapter cache contract is byte-exact.

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT     NOT NULL,
    model        TEXT     NOT NULL,
    dim          INTEGER  NOT NULL,
    embedding    BYTEA    NOT NULL,
    created_ts   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (content_hash, model)
);
