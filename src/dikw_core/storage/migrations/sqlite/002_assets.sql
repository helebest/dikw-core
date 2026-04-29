-- dikw-core SQLite schema v2: embedding versioning + multimedia assets.
--
-- The ``embed_versions`` table is the single registry of every embedding
-- generation (text + multimodal alike). Each row produces its own per-
-- version vector table (``vec_chunks_v<id>`` for text, ``vec_assets_v<id>``
-- for multimodal), created at runtime in ``storage/sqlite.py`` because
-- sqlite-vec needs the embedding dim parameterized into the CREATE.
-- Switching a model = new ``embed_versions`` row = new vec table; prior
-- vectors survive in-place.

-- ---- Multimedia assets ---------------------------------------------------

CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,         -- sha256 hex of the bytes; content-addressed identity
    kind           TEXT NOT NULL CHECK (kind IN ('image')),
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,            -- relative to project_root
    original_paths TEXT NOT NULL,            -- JSON list of source-side names
    bytes          INTEGER NOT NULL,
    media_meta     TEXT,                     -- per-kind JSON; image: {"kind":"image","width":...,"height":...}
    created_ts     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_asset_refs (
    chunk_id        INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    asset_id        TEXT    NOT NULL REFERENCES assets(asset_id),
    ord             INTEGER NOT NULL,        -- 0-based ordinal within chunk
    alt             TEXT    NOT NULL DEFAULT '',
    start_in_chunk  INTEGER NOT NULL,
    end_in_chunk    INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, ord),
    -- CHECK guarantees a non-degenerate span; the markdown image regex
    -- already enforces this, so the constraint is the schema-level
    -- safety net. UNIQUE forbids two refs landing in the same byte range
    -- within one chunk — duplicates would indicate a chunker bug.
    CHECK (start_in_chunk < end_in_chunk),
    UNIQUE (chunk_id, start_in_chunk, end_in_chunk)
);

CREATE INDEX IF NOT EXISTS chunk_asset_refs_asset
    ON chunk_asset_refs(asset_id);

-- ---- Embedding versioning ------------------------------------------------

-- Composite identity: any field different = different version = different
-- vector table. ``revision`` lets users force a new version when a
-- provider silently refreshes weights behind a stable model name.
-- ``modality`` is part of the UNIQUE so a single CLIP-style model can
-- register both a text and a multimodal version without collision.
CREATE TABLE IF NOT EXISTS embed_versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    revision    TEXT    NOT NULL DEFAULT '',
    dim         INTEGER NOT NULL,
    normalize   INTEGER NOT NULL,            -- 0/1
    distance    TEXT    NOT NULL CHECK (distance IN ('cosine','l2','dot')),
    modality    TEXT    NOT NULL CHECK (modality IN ('text','multimodal')),
    created_ts  REAL    NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    UNIQUE (provider, model, revision, dim, normalize, distance, modality)
);

CREATE INDEX IF NOT EXISTS embed_versions_active
    ON embed_versions(modality, is_active);

-- Per-chunk embedding metadata. The vector itself lives in the
-- vec_chunks_v<version_id> virtual table created at runtime in Python
-- (sqlite-vec needs the embedding dimension parameterized at CREATE
-- time). See storage/sqlite.py: _ensure_chunk_vec_table().
CREATE TABLE IF NOT EXISTS chunk_embed_meta (
    chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (chunk_id, version_id)
);

CREATE INDEX IF NOT EXISTS chunk_embed_meta_version
    ON chunk_embed_meta(version_id);

-- Per-asset embedding metadata. Mirror of chunk_embed_meta. The vector
-- lives in vec_assets_v<version_id> (also runtime-created).
CREATE TABLE IF NOT EXISTS asset_embed_meta (
    asset_id   TEXT    NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (asset_id, version_id)
);

CREATE INDEX IF NOT EXISTS asset_embed_meta_version
    ON asset_embed_meta(version_id);
