-- dikw-core Postgres schema v2: embedding versioning + multimedia assets.
-- Mirror of SQLite 002_assets.sql.
--
-- ``embed_versions`` is the registry of every embedding generation,
-- text + multimodal alike. Each row produces its own per-version vector
-- table (``vec_chunks_v<id>`` for text, ``vec_assets_v<id>`` for
-- multimodal), created at runtime in ``storage/postgres.py``: pgvector
-- needs the dim parameterised into the ``vector(<dim>)`` column type.
--
-- ``bytes`` on assets widens to BIGINT because PG INTEGER is 32-bit;
-- SQLite ``INTEGER`` is variable-width (effectively 64-bit). Conservative
-- side of the cross-backend size contract.

CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,        -- sha256 hex of the bytes; content-addressed identity
    kind           TEXT NOT NULL CHECK (kind IN ('image')),
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    original_paths TEXT NOT NULL,           -- JSON list (parity with SQLite)
    bytes          BIGINT NOT NULL,
    media_meta     TEXT,                    -- per-kind JSON (image: {"kind","width","height"})
    created_ts     DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_asset_refs (
    chunk_id        BIGINT  NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    asset_id        TEXT    NOT NULL REFERENCES assets(asset_id),
    ord             INTEGER NOT NULL,        -- 0-based ordinal within chunk
    alt             TEXT    NOT NULL DEFAULT '',
    start_in_chunk  INTEGER NOT NULL,
    end_in_chunk    INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, ord),
    -- See migrations/sqlite/002_assets.sql for the rationale.
    CHECK (start_in_chunk < end_in_chunk),
    UNIQUE (chunk_id, start_in_chunk, end_in_chunk)
);

CREATE INDEX IF NOT EXISTS chunk_asset_refs_asset
    ON chunk_asset_refs(asset_id);

-- Composite identity: any field different = different version = different
-- vector table. ``revision`` lets users force a new version when a
-- provider silently refreshes weights behind a stable model name.
-- ``modality`` is part of the UNIQUE so a single CLIP-style model can
-- register both a text and a multimodal version without collision.
CREATE TABLE IF NOT EXISTS embed_versions (
    version_id  BIGSERIAL PRIMARY KEY,
    provider    TEXT             NOT NULL,
    model       TEXT             NOT NULL,
    revision    TEXT             NOT NULL DEFAULT '',
    dim         INTEGER          NOT NULL,
    normalize   BOOLEAN          NOT NULL,
    distance    TEXT             NOT NULL CHECK (distance IN ('cosine','l2','dot')),
    modality    TEXT             NOT NULL CHECK (modality IN ('text','multimodal')),
    created_ts  DOUBLE PRECISION NOT NULL,
    is_active   BOOLEAN          NOT NULL DEFAULT TRUE,
    UNIQUE (provider, model, revision, dim, normalize, distance, modality)
);

CREATE INDEX IF NOT EXISTS embed_versions_active
    ON embed_versions(modality, is_active);

-- Per-chunk embedding metadata. The vector itself lives in the
-- vec_chunks_v<version_id> regular table created at runtime.
CREATE TABLE IF NOT EXISTS chunk_embed_meta (
    chunk_id   BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (chunk_id, version_id)
);

CREATE INDEX IF NOT EXISTS chunk_embed_meta_version
    ON chunk_embed_meta(version_id);

-- Per-asset embedding metadata. Mirror of chunk_embed_meta.
CREATE TABLE IF NOT EXISTS asset_embed_meta (
    asset_id   TEXT   NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (asset_id, version_id)
);

CREATE INDEX IF NOT EXISTS asset_embed_meta_version
    ON asset_embed_meta(version_id);
