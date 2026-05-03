"""Multipart sources upload — Phase 3 of the migration.

Clients turn an arbitrary input directory into the dikw input contract
(``sources/**`` markdown + ``assets/**`` binaries) locally, then ship a
single tar.gz here. The server unpacks into a per-upload staging
directory under ``<wiki>/.dikw/upload-staging/<upload_id>/``, verifies
every file's claimed sha256 against the manifest, and returns the
``upload_id``. Phase-3 ingest tasks then reference that ``upload_id``
to commit the staged tree into ``<wiki>/sources/`` + ``<wiki>/assets/``
*atomically* before running ``api.ingest``.

Two-step (upload → ingest) instead of one POST is deliberate:
  * Upload validation (tar shape, sha256, path-traversal) finishes
    before the server does any wiki-mutating work.
  * Long ingest tasks survive client disconnect because the source
    bytes are already on the server.
  * Failed uploads leave a ``staging`` tree (not the wiki tree) for
    operator inspection.

Wire format (``POST /v1/upload/sources``)::

    Content-Type: multipart/form-data
        payload  — tar.gz, paths must start with ``sources/`` or ``assets/``
        manifest — JSON string with ``{ files: [{path, size, sha256}], total_bytes }``

The ``manifest.files[*].path`` MUST match the in-archive path verbatim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from .errors import BadRequest
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


# ``DIKW_SERVER_MAX_UPLOAD_BYTES`` overrides the default at process start.
# 1 GiB is generous for an md+assets bundle but avoids the worst-case
# wedge on a runaway tarball.
_DEFAULT_MAX_UPLOAD_BYTES = 1 * 1024 * 1024 * 1024
_ALLOWED_TOP_DIRS = ("sources", "assets")

STAGING_DIRNAME = ".dikw/upload-staging"


# ---- request / response models ------------------------------------------


class ManifestEntry(BaseModel):
    path: str
    size: int = Field(ge=0)
    sha256: str  # lowercase hex


class Manifest(BaseModel):
    files: list[ManifestEntry]
    total_bytes: int = Field(ge=0)


class UploadResponse(BaseModel):
    upload_id: str
    files_count: int
    bytes: int
    applied_at: str  # ISO8601 UTC; matches the staging mtime
    staging_path: str  # wiki-root-relative path of the staging tree


# ---- router -------------------------------------------------------------


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.post("/upload/sources", response_model=UploadResponse)
    async def upload_sources(
        request: Request,
        payload: UploadFile = File(..., description="tar.gz of sources/+assets/"),
        manifest: str = Form(..., description="JSON Manifest body."),
    ) -> UploadResponse:
        rt: ServerRuntime = get_runtime(request.app)
        max_bytes = _resolve_max_bytes()

        # Cheap content-length pre-check; the per-write counter inside
        # ``_save_payload`` is the authoritative limit.
        cl = request.headers.get("content-length")
        if cl is not None and int(cl) > max_bytes:
            raise BadRequest(
                f"upload exceeds {max_bytes} bytes (content-length={cl})",
                code="upload_too_large",
                detail={"max_bytes": max_bytes},
            )

        manifest_obj = _parse_manifest(manifest)

        upload_id = uuid.uuid4().hex[:12]
        staging_root = rt.root / STAGING_DIRNAME / upload_id
        staging_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=False)

        try:
            tarball_path = staging_root / "_payload.tar.gz"
            written = await _save_payload(payload, tarball_path, max_bytes)
            _extract_safely(tarball_path, staging_root)
            os.unlink(tarball_path)
            _verify_manifest(staging_root, manifest_obj)
        except Exception:
            # Anything goes wrong → staging dir is junk; remove it so a
            # botched upload doesn't squat on the wiki forever.
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

        return UploadResponse(
            upload_id=upload_id,
            files_count=len(manifest_obj.files),
            bytes=written,
            applied_at=_isoformat(),
            staging_path=str(
                staging_root.relative_to(rt.root).as_posix()
            ),
        )

    return router


# ---- helpers ------------------------------------------------------------


def _resolve_max_bytes() -> int:
    raw = os.environ.get("DIKW_SERVER_MAX_UPLOAD_BYTES")
    if not raw:
        return _DEFAULT_MAX_UPLOAD_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_UPLOAD_BYTES


def _parse_manifest(raw: str) -> Manifest:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BadRequest(
            f"manifest is not valid JSON: {e}", code="manifest_malformed"
        ) from e
    try:
        return Manifest.model_validate(data)
    except Exception as e:
        raise BadRequest(
            f"manifest schema mismatch: {e}", code="manifest_invalid"
        ) from e


async def _save_payload(
    payload: UploadFile, dest: Path, max_bytes: int
) -> int:
    """Stream ``payload`` to ``dest`` while enforcing ``max_bytes``."""
    total = 0
    chunk_size = 1024 * 1024  # 1 MiB
    with dest.open("wb") as fh:
        while True:
            chunk = await payload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BadRequest(
                    f"upload exceeds {max_bytes} bytes",
                    code="upload_too_large",
                    detail={"max_bytes": max_bytes},
                )
            fh.write(chunk)
    return total


def _extract_safely(tarball: Path, target: Path) -> None:
    """Extract a tar.gz, rejecting anything that escapes ``target``.

    Two layers of defence:
      * member.name path-traversal check (``..`` segments, absolute
        paths) — common ``tarbomb`` defence.
      * post-resolve ``commonpath`` check — guards against symlink
        members and platform-specific edge cases (``\\`` on Windows).
    """
    with tarfile.open(tarball, mode="r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            _check_member(m, target)
        # Python 3.12+ ships a ``filter='data'`` helper; we use it as a
        # belt to ``_check_member``'s suspenders.
        tf.extractall(target, filter="data")


def _check_member(member: tarfile.TarInfo, target: Path) -> None:
    name = member.name
    if not name:
        raise BadRequest(
            "tarball contains an empty path", code="tar_invalid"
        )
    if member.issym() or member.islnk():
        raise BadRequest(
            f"tarball symlinks/hardlinks are not allowed: {name!r}",
            code="tar_link_forbidden",
        )
    # Reject Windows-style separators so a Windows-built tar can't ship
    # ``..\..\etc\foo`` and slip past POSIX path-resolution.
    if "\\" in name:
        raise BadRequest(
            f"tarball path uses backslashes: {name!r}",
            code="tar_invalid",
        )
    parts = Path(name).parts
    if any(p == ".." for p in parts) or Path(name).is_absolute():
        raise BadRequest(
            f"tarball path escapes root: {name!r}",
            code="tar_path_traversal",
        )
    top = parts[0] if parts else ""
    if member.isdir():
        return
    if top not in _ALLOWED_TOP_DIRS:
        raise BadRequest(
            f"tarball file outside sources/+assets/: {name!r}",
            code="tar_unexpected_path",
            detail={"allowed": list(_ALLOWED_TOP_DIRS)},
        )
    # Final resolved-path check.
    resolved = (target / name).resolve()
    base = target.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as e:
        raise BadRequest(
            f"tarball path escapes root after resolve: {name!r}",
            code="tar_path_traversal",
        ) from e


def _verify_manifest(staging_root: Path, manifest: Manifest) -> None:
    """Hash every staged file and compare to the manifest's claim.

    Mismatches surface a clear ``code`` so the client can distinguish
    transport corruption (sha256 wrong) from path-shape errors
    (missing / extra files) without parsing the message.
    """
    declared = {entry.path: entry for entry in manifest.files}
    on_disk: dict[str, Path] = {}
    for path in staging_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "_payload.tar.gz":
            continue
        rel = path.relative_to(staging_root).as_posix()
        on_disk[rel] = path

    missing = sorted(declared.keys() - on_disk.keys())
    if missing:
        raise BadRequest(
            f"manifest declares files not present in tarball: {missing}",
            code="manifest_missing_files",
            detail={"missing": missing[:10]},
        )
    extra = sorted(on_disk.keys() - declared.keys())
    if extra:
        raise BadRequest(
            f"tarball contains files not in manifest: {extra}",
            code="manifest_extra_files",
            detail={"extra": extra[:10]},
        )
    bad: list[str] = []
    for rel, abs_path in on_disk.items():
        entry = declared[rel]
        if abs_path.stat().st_size != entry.size:
            bad.append(f"{rel}: size mismatch")
            continue
        h = _sha256_file(abs_path)
        if h != entry.sha256.lower():
            bad.append(f"{rel}: sha256 mismatch")
    if bad:
        raise BadRequest(
            f"manifest verification failed: {bad[:5]}",
            code="manifest_sha256_mismatch",
            detail={"errors": bad[:10]},
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _isoformat(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return (
        datetime.fromtimestamp(ts, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "STAGING_DIRNAME",
    "Manifest",
    "ManifestEntry",
    "UploadResponse",
    "make_router",
]
