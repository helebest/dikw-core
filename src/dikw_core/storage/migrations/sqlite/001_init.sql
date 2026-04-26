-- dikw-core SQLite schema v1
-- sqlite-vec must be loaded into the connection before running this migration.

-- ---- D layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path   TEXT UNIQUE NOT NULL,
    title  TEXT,
    hash   TEXT NOT NULL,
    mtime  REAL,
    layer  TEXT NOT NULL CHECK (layer IN ('source','wiki','wisdom')),
    active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS documents_layer_active
    ON documents(layer, active);

-- Reverse lookup by content hash.
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(hash);

-- ---- I layer -------------------------------------------------------------

-- FTS5 over chunk body; chunks.chunk_id aligns with fts rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path UNINDEXED,
    title,
    body,
    layer UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,
    start_off INTEGER NOT NULL,
    end_off   INTEGER NOT NULL,
    text      TEXT NOT NULL,
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS chunks_doc_seq ON chunks(doc_id, seq);

-- Note: the embedding virtual table is created in Python because sqlite-vec
-- needs the embedding dimension parameterized at table-creation time.
-- See storage/sqlite.py: SQLiteStorage._ensure_vec_table().

CREATE TABLE IF NOT EXISTS embed_meta (
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

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
    ts     REAL NOT NULL,
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
    confidence  REAL NOT NULL DEFAULT 0.5,
    created_ts  REAL NOT NULL,
    approved_ts REAL
);

CREATE INDEX IF NOT EXISTS wisdom_status ON wisdom_items(status);
CREATE INDEX IF NOT EXISTS wisdom_kind ON wisdom_items(kind);

CREATE TABLE IF NOT EXISTS wisdom_evidence (
    item_id TEXT NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id),
    excerpt TEXT NOT NULL,
    line    INTEGER
);

CREATE INDEX IF NOT EXISTS wisdom_evidence_item ON wisdom_evidence(item_id);
