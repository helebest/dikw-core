-- dikw-core SQLite schema v4: wisdom-item embedding metadata.
--
-- Mirror of chunk_embed_meta and asset_embed_meta. The per-version
-- vector itself lives in vec_wisdom_v<version_id>, created at runtime
-- in storage/sqlite.py because sqlite-vec needs the embedding dim
-- parameterized into the virtual-table CREATE.
--
-- Wisdom rides on the text modality so a single active text
-- embed_versions row covers both chunks and wisdom — apply-at-query
-- compares a question's text embedding against wisdom embeddings in
-- the same cosine space. The per-version vec table stays separate
-- (vec_wisdom_v<id> keyed by item_id TEXT vs vec_chunks_v<id> keyed by
-- chunk_id INTEGER) because the identity columns can't share a table.

CREATE TABLE IF NOT EXISTS wisdom_embed_meta (
    item_id    TEXT    NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (item_id, version_id)
);

CREATE INDEX IF NOT EXISTS wisdom_embed_meta_version
    ON wisdom_embed_meta(version_id);
