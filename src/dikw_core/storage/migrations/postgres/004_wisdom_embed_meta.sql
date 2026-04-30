-- dikw-core Postgres schema v4: wisdom-item embedding metadata.
-- Mirror of SQLite 004_wisdom_embed_meta.sql.
--
-- The per-version vector lives in vec_wisdom_v<version_id> (created at
-- runtime in storage/postgres.py because pgvector needs the dim
-- parameterised into the vector(<dim>) column type).
--
-- Wisdom reuses the text modality on embed_versions; see the SQLite
-- mirror for the rationale.

CREATE TABLE IF NOT EXISTS wisdom_embed_meta (
    item_id    TEXT   NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (item_id, version_id)
);

CREATE INDEX IF NOT EXISTS wisdom_embed_meta_version
    ON wisdom_embed_meta(version_id);
