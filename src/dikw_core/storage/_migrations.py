"""Migration-file discovery + schema-version constants shared by adapters.

The SQLite and Postgres adapters both apply numbered ``NNN_*.sql`` files
under ``storage/migrations/{sqlite,postgres}/`` and audit progress via
``meta_kv['schema_version']``. The eval-snapshot cache key in
``eval/runner.py`` also derives from the migration-file count so a
schema bump invalidates stale snapshots.

Centralizing the parsing here keeps those three call sites in lockstep
— if the file-naming convention ever shifts (e.g. ``NNNN_*``), one
edit covers every consumer.
"""

from __future__ import annotations

from importlib import resources

SCHEMA_VERSION_KEY = "schema_version"


def migration_number(filename: str) -> int | None:
    """Return the integer prefix of ``NNN_*.sql`` or None if not numeric."""
    head = filename.split("_", 1)[0]
    return int(head) if head.isdigit() else None


def ordered_migrations(pkg_name: str) -> list[tuple[int, str]]:
    """Return ``(number, filename)`` pairs in ascending numeric order.

    Files outside the ``NNN_*.sql`` shape are skipped silently so a
    stray README or `__init__.py` in the migrations directory doesn't
    break the loader.
    """
    out: list[tuple[int, str]] = []
    for r in resources.files(pkg_name).iterdir():
        if not (r.is_file() and r.name.endswith(".sql")):
            continue
        n = migration_number(r.name)
        if n is not None:
            out.append((n, r.name))
    out.sort(key=lambda t: t[0])
    return out


__all__ = ["SCHEMA_VERSION_KEY", "migration_number", "ordered_migrations"]
