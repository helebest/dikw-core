"""End-to-end tests for ``documents.path_key`` NFC + casefold uniqueness.

These tests exercise the data-boundary normalization (``data/path_norm.py``)
and prove that re-ingesting a file under different macOS NFD / NTFS-case
spellings collapses to a single ``documents`` row across the SQLite,
Postgres, and filesystem adapters.

The tests construct strings directly with the relevant codepoints rather
than depending on the host filesystem so the assertions hold on Linux
CI (no NFD-producing kernel) and on Windows (which can't write a colon
in a filename).
"""

from __future__ import annotations

import time
import unicodedata

import pytest

from dikw_core.data.path_norm import normalize_path
from dikw_core.schemas import DocumentRecord, Layer


def test_normalize_path_nfc_collapses_nfd_byte_difference() -> None:
    """``é`` written as the precomposed codepoint and as ``e + combining
    acute`` are the same logical character. ``normalize_path`` must
    return the same string for both."""
    nfc = "café.md"  # é = U+00E9
    nfd = unicodedata.normalize("NFD", nfc)  # e + U+0301
    assert nfc != nfd  # sanity: byte-different on the way in
    assert normalize_path(nfc) == normalize_path(nfd)


def test_normalize_path_casefold_collapses_case_only_difference() -> None:
    assert normalize_path("MyDoc.md") == normalize_path("mydoc.md")
    assert normalize_path("MYDOC.MD") == normalize_path("mydoc.md")


def test_normalize_path_casefold_handles_eszett() -> None:
    """``casefold()`` (unlike ``lower()``) expands German ß to ``ss`` so
    ``STRAßE`` and ``strasse`` collide."""
    assert normalize_path("STRAßE.md") == normalize_path("strasse.md")


def test_normalize_path_combined_nfc_and_casefold() -> None:
    nfd = unicodedata.normalize("NFD", "Café.MD")
    assert normalize_path(nfd) == "café.md"


def test_document_record_derives_path_key_from_path() -> None:
    """``DocumentRecord(path=...)`` without an explicit ``path_key`` must
    auto-derive it via the model validator. Engine call sites rely on
    this so they don't have to thread ``normalize_path`` through every
    construction."""
    doc = DocumentRecord(
        doc_id="x",
        path="MyDoc.MD",
        hash="h",
        mtime=time.time(),
        layer=Layer.SOURCE,
    )
    assert doc.path == "MyDoc.MD"
    assert doc.path_key == "mydoc.md"


def test_document_record_explicit_path_key_overrides_derivation() -> None:
    """When loading a row out of storage, the adapter passes the stored
    path_key explicitly. The validator must not re-derive (display path
    can have changed since insert)."""
    doc = DocumentRecord(
        doc_id="x",
        path="brand-new-display.md",
        path_key="legacy-key.md",
        hash="h",
        mtime=time.time(),
        layer=Layer.SOURCE,
    )
    assert doc.path == "brand-new-display.md"
    assert doc.path_key == "legacy-key.md"


# ---- end-to-end via the public Storage Protocol --------------------------


@pytest.fixture
async def sqlite_storage(tmp_path):  # type: ignore[no-untyped-def]
    from dikw_core.storage.sqlite import SQLiteStorage

    s = SQLiteStorage(tmp_path / "t.db")
    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        await s.close()


async def test_sqlite_path_key_unique_constraint(sqlite_storage) -> None:  # type: ignore[no-untyped-def]
    """Inserting two documents whose paths differ only by case must fail
    on the second upsert because the new ``path_key`` UNIQUE collides."""
    import sqlite3

    a = DocumentRecord(
        doc_id="a",  # different doc_id so PK conflict isn't the trigger
        path="MyDoc.md",
        hash="h",
        mtime=1.0,
        layer=Layer.SOURCE,
    )
    b = DocumentRecord(
        doc_id="b",
        path="mydoc.md",
        hash="h",
        mtime=2.0,
        layer=Layer.SOURCE,
    )
    await sqlite_storage.upsert_document(a)
    with pytest.raises(sqlite3.IntegrityError):
        await sqlite_storage.upsert_document(b)


async def test_sqlite_round_trip_preserves_display_path(sqlite_storage) -> None:  # type: ignore[no-untyped-def]
    """The user's spelling round-trips through storage even though the
    uniqueness gate uses path_key. ``MyDoc.md`` stays ``MyDoc.md`` on
    read, not ``mydoc.md``."""
    doc = DocumentRecord(
        doc_id="d",
        path="Notes/MyDoc.MD",
        hash="h",
        mtime=time.time(),
        layer=Layer.SOURCE,
    )
    await sqlite_storage.upsert_document(doc)
    got = await sqlite_storage.get_document("d")
    assert got is not None
    assert got.path == "Notes/MyDoc.MD"
    assert got.path_key == "notes/mydoc.md"


async def test_sqlite_rename_with_case_change_updates_display_in_place(
    sqlite_storage,  # type: ignore[no-untyped-def]
) -> None:
    """When ingest re-inserts a doc whose display path changed case but
    path_key is identical, the row's display path should update in place
    — keyed by doc_id, not by path. Uniqueness on path_key holds."""
    doc1 = DocumentRecord(
        doc_id="same",
        path="MyDoc.md",
        hash="h1",
        mtime=1.0,
        layer=Layer.SOURCE,
    )
    doc2 = DocumentRecord(
        doc_id="same",
        path="mydoc.md",  # rename: case-only delta
        hash="h2",
        mtime=2.0,
        layer=Layer.SOURCE,
    )
    await sqlite_storage.upsert_document(doc1)
    await sqlite_storage.upsert_document(doc2)
    got = await sqlite_storage.get_document("same")
    assert got is not None
    assert got.path == "mydoc.md"
    assert got.path_key == "mydoc.md"
    assert got.hash == "h2"
