-- dikw-core Postgres schema v3: multimedia assets (mirror of SQLite 002).
--
-- v1 ships images only; the chunk_asset_refs bridge, embed_versions, and
-- asset_embed_meta tables (and the per-version pgvector tables they imply)
-- stay SQLite-only until Phase 5. The Python adapter therefore implements
-- only ``upsert_asset`` / ``get_asset`` against this table and keeps every
-- other asset method as ``NotSupported`` until that phase lands.
--
-- ``bytes`` widens to BIGINT here because PG INTEGER is 32-bit while the
-- SQLite ``INTEGER`` column is variable-width (effectively 64-bit) — this
-- is the conservative side of the cross-backend size contract.

CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,
    hash           TEXT UNIQUE NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('image')),
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    original_paths TEXT NOT NULL,           -- JSON list (parity with SQLite)
    bytes          BIGINT NOT NULL,
    media_meta     TEXT,                    -- per-kind JSON (image: {"kind","width","height"})
    created_ts     DOUBLE PRECISION NOT NULL
);
