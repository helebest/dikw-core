-- dikw-core SQLite schema v1
-- sqlite-vec must be loaded into the connection before running this migration.

-- ``meta_kv`` is also created inline in ``SQLiteStorage.migrate()`` before
-- this file runs, because ``_read_schema_version_sqlite`` must read its
-- row to decide whether to skip already-applied migration files. Declaring
-- it here too keeps the SQLite schema's source-of-truth visible to anyone
-- diffing the two adapters' ``migrations/`` trees; ``IF NOT EXISTS`` makes
-- the second create a no-op.
CREATE TABLE IF NOT EXISTS meta_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---- D layer -------------------------------------------------------------

-- ``path`` carries the user's spelling (display path); ``path_key`` is
-- the engine's NFC + casefold lookup key. Splitting the two lets the
-- same logical file under different macOS NFD / NTFS-case spellings
-- resolve to a single row while ``dikw status`` still shows whichever
-- spelling is on disk. Uniqueness moves to ``path_key`` accordingly;
-- ``path`` stays NOT NULL but plain so a rename-with-case-change can
-- update the display value in place. See data/path_norm.py.
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    path_key TEXT NOT NULL UNIQUE,
    title    TEXT,
    hash     TEXT NOT NULL,
    mtime    REAL,
    layer    TEXT NOT NULL CHECK (layer IN ('source','wiki','wisdom')),
    active   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS documents_layer_active
    ON documents(layer, active);

-- Reverse lookup by content hash.
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(hash);

-- ---- I layer -------------------------------------------------------------

-- FTS5 over chunk body; chunks.chunk_id aligns with fts rowid.
-- Scope is intentionally narrow:
--   * Only ``body`` is indexed — ``path`` / ``title`` / ``layer`` come
--     back via a JOIN onto ``chunks`` + ``documents`` at search time,
--     mirroring the PG side which indexes only ``chunks.text``.
--   * ``unicode61 remove_diacritics 0`` so that ``café`` and ``cafe``
--     are different tokens — same byte-level behavior as PG's
--     ``to_tsvector('simple', ...)``. Aligning here avoids a silent
--     cross-backend recall divergence on accented text. The ``0`` is
--     intentional and explicit: ``unicode61`` defaults to
--     ``remove_diacritics 1`` which still strips most diacritics.
-- Legacy DBs built before this scoping ship through
-- ``SQLiteStorage._migrate_legacy_documents_fts``.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    body,
    tokenize = "unicode61 remove_diacritics 0"
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

-- Note: the per-version chunk-vector tables (``vec_chunks_v<version_id>``)
-- are created in Python at first ingest because sqlite-vec needs the
-- embedding dimension parameterized at table-creation time. See
-- storage/sqlite.py: SQLiteStorage._ensure_chunk_vec_table(). Each
-- ``embed_versions`` row produces its own table; switching text models
-- creates a new version + new table, leaving prior data intact.
--
-- ``chunk_embed_meta`` is defined in 002_assets.sql alongside the
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

-- ``id`` is an explicit AUTOINCREMENT column so monotonic ordering is
-- preserved when multiple events share the same ``ts`` (``time.time()``
-- is float-second resolution; an ingest batch can append multiple rows
-- in the same second). Mirrors the Postgres ``BIGSERIAL`` column.
CREATE TABLE IF NOT EXISTS wiki_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
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

-- ``id`` mirrors the Postgres ``BIGSERIAL PRIMARY KEY`` so the two
-- adapters return evidence rows in the same insertion order via the
-- same ``ORDER BY id`` clause. Without it SQLite has to lean on the
-- implicit rowid, which is correct in practice but invisible to
-- contract tests / external readers.
CREATE TABLE IF NOT EXISTS wisdom_evidence (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id),
    excerpt TEXT NOT NULL,
    line    INTEGER
);

CREATE INDEX IF NOT EXISTS wisdom_evidence_item ON wisdom_evidence(item_id);
