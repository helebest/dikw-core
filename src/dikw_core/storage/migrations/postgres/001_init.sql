-- dikw-core Postgres schema v1
-- The vector extension is enabled at connect-time in Python (not here)
-- so the SQL below can use ``::vector`` casts on the first connection.
-- Per-version vec_chunks_v<id> / vec_assets_v<id> tables are created
-- lazily in Python so the engine can parameterise the embedding
-- dimension at first insert. See storage/postgres.py.

CREATE TABLE IF NOT EXISTS meta_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---- D layer -------------------------------------------------------------

-- See migrations/sqlite/001_init.sql for the path / path_key split
-- rationale.
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    path_key TEXT NOT NULL UNIQUE,
    title    TEXT,
    hash     TEXT NOT NULL,
    mtime    DOUBLE PRECISION,
    layer    TEXT NOT NULL CHECK (layer IN ('source','wiki','wisdom')),
    active   BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS documents_layer_active ON documents(layer, active);

-- Reverse lookup by content hash.
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(hash);

-- ---- I layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    start_off   INTEGER NOT NULL,
    end_off     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    fts         tsvector GENERATED ALWAYS AS
                (to_tsvector('simple', coalesce(text,''))) STORED,
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS chunks_doc_seq ON chunks(doc_id, seq);
CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING GIN (fts);

-- ``chunk_embed_meta`` is defined in 003_assets.sql alongside the
-- ``embed_versions`` registry it references.

-- ---- K layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS links (
    src_doc_id TEXT NOT NULL,
    dst_path   TEXT NOT NULL,
    link_type  TEXT NOT NULL CHECK (link_type IN ('wikilink','markdown','url')),
    anchor     TEXT,
    line       INTEGER NOT NULL,
    PRIMARY KEY (src_doc_id, dst_path, line)
);

CREATE INDEX IF NOT EXISTS links_dst ON links(dst_path);

CREATE TABLE IF NOT EXISTS wiki_log (
    id     BIGSERIAL PRIMARY KEY,
    ts     DOUBLE PRECISION NOT NULL,
    action TEXT NOT NULL,
    src    TEXT,
    dst    TEXT,
    note   TEXT
);

CREATE INDEX IF NOT EXISTS wiki_log_ts ON wiki_log(ts);

-- ---- W layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS wisdom_items (
    item_id     TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('principle','lesson','pattern')),
    status      TEXT NOT NULL DEFAULT 'candidate'
                CHECK (status IN ('candidate','approved','archived')),
    path        TEXT,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    confidence  DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_ts  DOUBLE PRECISION NOT NULL,
    approved_ts DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS wisdom_status ON wisdom_items(status);
CREATE INDEX IF NOT EXISTS wisdom_kind   ON wisdom_items(kind);

CREATE TABLE IF NOT EXISTS wisdom_evidence (
    id      BIGSERIAL PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id),
    excerpt TEXT NOT NULL,
    line    INTEGER
);

CREATE INDEX IF NOT EXISTS wisdom_evidence_item ON wisdom_evidence(item_id);
