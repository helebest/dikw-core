"""Local sources/+assets/ → tar.gz + manifest packing.

The server's ``POST /v1/upload/sources`` contract requires:

* a tar.gz whose top-level directories are exactly ``sources/`` and/or
  ``assets/``;
* a JSON manifest with one entry per file in the archive
  (``{path, size, sha256}``); paths must match in-archive paths verbatim.

This module turns a local directory that already contains those top-level
subtrees into the (tarball, manifest) pair the transport ships. Two
shapes are accepted:

1. ``<src>/sources/...`` and optionally ``<src>/assets/...`` already in
   place — most common: the user dropped markdown into a fresh wiki's
   ``sources/`` and wants to upload it.
2. ``<src>/`` with markdown files at the top — we re-root them under
   ``sources/`` automatically so the user doesn't have to mkdir before
   their first upload.

We **never** silently include files outside the allowed top-level dirs:
that's the client-side mirror of the server's ``tar_unexpected_path``
guard, surfaced as a clear local error before the upload starts.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import IO

# Spooled buffer threshold — 16 MiB matches the plan's recommendation.
# Below it, the tarball stays in RAM (zero disk I/O); above it, the
# spooled file rolls to a real tempfile so we don't OOM on big asset
# bundles.
_SPOOL_MAX_SIZE = 16 * 1024 * 1024
_ALLOWED_TOP_DIRS = ("sources", "assets")


@dataclass(frozen=True)
class ManifestEntry:
    """One file's metadata. Mirrors the server's ManifestEntry shape."""

    path: str  # POSIX, relative to the tarball root, e.g. ``sources/foo.md``
    size: int
    sha256: str  # lowercase hex


@dataclass
class UploadBundle:
    """A ready-to-send tar.gz + its manifest.

    ``payload`` is a file-like positioned at byte 0; the caller hands it
    to the transport. ``payload`` is owned by this dataclass — close it
    via :meth:`close` (or use as a context manager) when done.
    """

    payload: IO[bytes]
    manifest_json: str
    files_count: int
    bytes: int

    def close(self) -> None:
        # Tempfile cleanup races with delete-on-close on Windows;
        # don't let a swallowed cleanup error mask the real upload
        # error from the surrounding traceback.
        with contextlib.suppress(Exception):
            self.payload.close()

    def __enter__(self) -> UploadBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_upload(
    src: Path,
    *,
    extra_extensions: Iterable[str] = (),
) -> UploadBundle:
    """Pack ``src`` into a tar.gz + manifest the server will accept.

    ``extra_extensions`` lets callers admit additional file types beyond
    the conservative default (``.md``, image extensions, ``.pdf``). The
    extension check is the only file-type filter — we don't infer mime
    types or peek at content, so a ``.md`` file that happens to be
    binary still ships through.

    Raises :class:`UploadError` for empty inputs or unrecognised file
    types so the user sees the problem before the upload begins.
    """
    src = src.resolve()
    if not src.is_dir():
        raise UploadError(f"upload source is not a directory: {src}")

    allowed_ext = _allowed_extensions(extra_extensions)
    files = list(_discover(src, allowed_ext=allowed_ext))
    if not files:
        raise UploadError(
            f"no files found under {src}; expected sources/**.md or assets/**"
        )

    # SpooledTemporaryFile is the payload we hand back to the caller;
    # it must outlive this function. SIM115 wants a ``with`` block, but
    # closing here would invalidate the bundle — suppress at this site.
    payload = SpooledTemporaryFile(  # noqa: SIM115
        max_size=_SPOOL_MAX_SIZE, mode="w+b"
    )
    manifest_entries: list[ManifestEntry] = []
    total_bytes = 0
    try:
        # Build the gzip stream around the spooled file directly so
        # large bundles never materialise in RAM. ``mode='w'`` on the
        # tarfile + GzipFile pair is the documented streaming write
        # idiom and matches what httpx will send on the wire.
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz, tarfile.TarFile(
            fileobj=gz, mode="w"
        ) as tf:
            for entry in files:
                tarinfo = tarfile.TarInfo(name=entry.archive_path)
                tarinfo.size = entry.size
                tarinfo.mtime = int(entry.mtime)
                tarinfo.mode = 0o644
                # Strip uid/gid/uname/gname so the archive is byte-stable
                # across users running the same upload — useful when a
                # CI job and a developer both upload the same tree.
                tarinfo.uid = 0
                tarinfo.gid = 0
                tarinfo.uname = ""
                tarinfo.gname = ""
                with entry.path.open("rb") as fh:
                    tf.addfile(tarinfo, fh)
                manifest_entries.append(
                    ManifestEntry(
                        path=entry.archive_path,
                        size=entry.size,
                        sha256=entry.sha256,
                    )
                )
                total_bytes += entry.size
    except Exception:
        payload.close()
        raise

    payload.seek(0)
    manifest_json = json.dumps(
        {
            "files": [e.__dict__ for e in manifest_entries],
            "total_bytes": total_bytes,
        }
    )
    return UploadBundle(
        payload=payload,
        manifest_json=manifest_json,
        files_count=len(manifest_entries),
        bytes=total_bytes,
    )


# ---- internals ----------------------------------------------------------


_DEFAULT_EXTENSIONS = frozenset(
    {".md", ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}
)


def _allowed_extensions(extra: Iterable[str]) -> frozenset[str]:
    extras = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extra}
    return _DEFAULT_EXTENSIONS | frozenset(extras)


@dataclass(frozen=True)
class _DiscoveredFile:
    """One on-disk file plus its computed archive path + hash + size."""

    path: Path
    archive_path: str
    size: int
    sha256: str
    mtime: float


class UploadError(Exception):
    """Surface client-side rejection (empty input, bad extension, …)
    before any bytes leave the machine."""


def _discover(
    src: Path, *, allowed_ext: frozenset[str]
) -> Iterator[_DiscoveredFile]:
    """Yield every uploadable file under ``src``, with archive path resolved.

    Two top-level shapes are recognised:

    * ``src/sources/**``, ``src/assets/**`` already laid out — files keep
      their relative path inside the tarball.
    * ``src/**.md`` at the top — files get re-rooted under ``sources/``.
      Any other extension at the top is rejected so the user notices
      they forgot the ``sources/`` directory rather than silently
      missing assets.

    Mixed shapes (top-level md AND a sources/ subdir) are rejected — too
    ambiguous to guess the right merge.
    """
    sources_dir = src / "sources"
    assets_dir = src / "assets"
    has_sources_subtree = sources_dir.is_dir()
    has_assets_subtree = assets_dir.is_dir()
    has_subtree = has_sources_subtree or has_assets_subtree

    # Inventory every visible top-level entry so we can loud-fail on
    # anything that wouldn't ship. Hidden entries (``.dikw/``, ``.git/``)
    # are excluded since ``_walk_files`` skips them anyway.
    top_entries = [p for p in sorted(src.iterdir()) if not p.name.startswith(".")]

    if has_subtree:
        # Subtree-shape: every top-level entry must be either ``sources/``
        # or ``assets/``. Loose files (even allowed-ext ones) are
        # rejected as ambiguous — the user clearly meant the subtree
        # layout, so a stray ``stray.md`` is more likely a mistake than
        # a deliberate addition. Other top-level dirs (``drafts/``,
        # ``docs/``) would otherwise be silently dropped.
        stray = [p for p in top_entries if p.name not in _ALLOWED_TOP_DIRS]
        if stray:
            raise UploadError(
                f"{src} contains stray top-level entries alongside "
                f"sources/ or assets/: {[p.name for p in stray]}; "
                "move them under sources/ or assets/ to disambiguate — "
                "silently dropping them would ship an incomplete tree."
            )
    else:
        # Flat-folder shape: every top-level entry must be a
        # whitelisted file. A top-level dir is a hint the user
        # forgot to wrap things in ``sources/``; surface it instead
        # of silently skipping its contents.
        bad: list[Path] = [
            p
            for p in top_entries
            if p.is_dir()
            or (p.is_file() and p.suffix.lower() not in allowed_ext)
        ]
        if bad:
            raise UploadError(
                f"{src} contains entries that aren't recognised as "
                f"sources or assets: {[p.name for p in bad]}"
            )

    if has_subtree:
        for top in _ALLOWED_TOP_DIRS:
            top_path = src / top
            if not top_path.is_dir():
                continue
            for path in sorted(_walk_files(top_path)):
                if path.suffix.lower() not in allowed_ext:
                    raise UploadError(
                        f"unsupported file type for upload: {path} "
                        f"(allowed: {sorted(allowed_ext)})"
                    )
                rel = path.relative_to(src).as_posix()
                yield _make_entry(path, archive_path=rel)
        return

    # No sources/ or assets/ subtree → re-root: markdown lands under
    # ``sources/``, everything else under ``assets/``. The default
    # whitelist includes image / pdf extensions for the assets path,
    # so a flat dir with ``note.md + diagram.png`` gets the png mapped
    # to ``assets/diagram.png`` (where commit_staging + ingest expect
    # asset files) rather than ``sources/diagram.png`` (where ingest
    # would skip or misclassify it).
    _MD_LIKE = {".md", ".html", ".htm"}
    for path in sorted(_walk_files(src)):
        if path.name == "_payload.tar.gz":
            continue
        if path.suffix.lower() not in allowed_ext:
            raise UploadError(
                f"unsupported file type for upload: {path} "
                f"(allowed: {sorted(allowed_ext)})"
            )
        rel = path.relative_to(src).as_posix()
        bucket = "sources" if path.suffix.lower() in _MD_LIKE else "assets"
        yield _make_entry(path, archive_path=f"{bucket}/{rel}")


def _walk_files(root: Path) -> Iterator[Path]:
    for parent, dirs, files in os.walk(root):
        # Drop hidden dirs in-place so os.walk skips them entirely;
        # ``.dikw/`` (engine state, may contain stale embeddings) is the
        # main one we don't want to upload by accident.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            yield Path(parent) / name


def _make_entry(path: Path, *, archive_path: str) -> _DiscoveredFile:
    # Reject symlinks before we open() — otherwise the open follows the
    # link and archives whatever it points at as a regular file (e.g.
    # ``sources/note.md -> /etc/passwd``). The server-side tar check
    # blocks symlink *members* but can't tell that the bytes inside a
    # regular member came from a followed symlink, so the gate has to
    # live here.
    if path.is_symlink():
        raise UploadError(
            f"refusing to upload symlink: {path} "
            f"(target: {os.readlink(path)!r}); "
            "copy the file in place if you really mean to include it"
        )
    size = path.stat().st_size
    sha = _sha256_file(path)
    return _DiscoveredFile(
        path=path,
        archive_path=archive_path,
        size=size,
        sha256=sha,
        mtime=path.stat().st_mtime,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Quiet ``io`` import: we expose ``IO[bytes]`` as the file-like type.
_ = io


__all__ = [
    "ManifestEntry",
    "UploadBundle",
    "UploadError",
    "build_upload",
]
