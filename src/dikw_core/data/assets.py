"""Multimedia asset materialization + path composition.

v1 scope: images only. Audio/video extension points are reserved on the
schemas (`AssetKind`, `MultimodalInput`) but not exercised here.

This module owns the contract between Markdown source references like
``![alt](./diagrams/foo.png)`` and the engine-managed on-disk layout at
``<project_root>/assets/<h2>/<h8>-<sanitized-name>.<ext>``. The path scheme
combines a 256-way sha256 prefix shard (FS-friendly under heavy ingest)
with the original file name preserved for human/Obsidian readability.
"""

from __future__ import annotations

import os
import re
import unicodedata

# Windows reserved device names. Compared case-insensitively against the
# sanitized stem so a literal `con.png` in markdown doesn't break a
# Windows-mounted vault.
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Sanitized stem's max UTF-8 byte length. Combined with the ``<h8>-`` prefix
# (9 bytes), shard dir (3 bytes) and a few-byte extension this stays well
# under the 255-byte filename cap of every major filesystem (APFS, ext4,
# NTFS) — leaving headroom for cross-platform sync clients with their own
# stricter caps.
_STEM_BYTE_CAP = 150

_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)$")
_HYPHEN_RUN = re.compile(r"-+")


def _split_basename_and_ext(name: str) -> tuple[str, str]:
    """Strip directories, then return (stem, ext) where ext is lower-case
    without the leading dot. Empty ext = no extension detected."""
    # Treat both POSIX and Windows separators robustly even though markdown
    # references should be POSIX-only in practice.
    base = os.path.basename(name.replace("\\", "/"))
    m = _EXT_RE.search(base)
    if m is None:
        return base, ""
    return base[: m.start()], m.group(1).lower()


def _is_allowed_char(ch: str) -> bool:
    """The sanitize whitelist. Anything not in here is replaced with ``-``.

    Allowed:
      - ``-`` and ``_`` (literal)
      - Any Unicode "Letter" (Lu/Ll/Lt/Lm/Lo) — covers ASCII letters, Latin
        Extended (é, ñ, ü), CJK, Hiragana, Katakana, Hangul, Cyrillic,
        Greek, Arabic, Hebrew, Devanagari, etc.
      - Any Unicode "Number" (Nd/Nl/No) — covers ASCII digits, Arabic-
        Indic digits, full-width digits, etc.

    Excluded by virtue of category mismatch: punctuation (P*), symbols
    (S*, including emoji), separators (Z*), marks (M*, includes leftover
    NFD combining accents that NFC didn't fold), and control codes (C*).
    """
    if ch in ("-", "_"):
        return True
    # Letters and Numbers cover all writing systems' name characters with
    # one rule, instead of an ever-growing range list. Mn/Mc (combining
    # marks) deliberately excluded so any NFD residue gets sanitized away.
    return unicodedata.category(ch)[0] in ("L", "N")


def _truncate_to_utf8_bytes(s: str, byte_cap: int) -> str:
    """Truncate ``s`` to at most ``byte_cap`` UTF-8 bytes without ever
    splitting a multi-byte character."""
    encoded = s.encode("utf-8")
    if len(encoded) <= byte_cap:
        return s
    encoded = encoded[:byte_cap]
    while encoded:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def _sanitize_stem(stem: str) -> str:
    """The 10-step sanitize pipeline from the design doc.

    NFC-normalize, whitelist-filter, hyphen-collapse, byte-truncate,
    Windows-reserved guard, leading/trailing hyphen+dot strip.
    """
    # 3. NFC normalize. APFS NFD → unified NFC so the same logical filename
    # produces the same sanitize output across platforms.
    stem = unicodedata.normalize("NFC", stem)
    # 4. Character whitelist.
    out = "".join(ch if _is_allowed_char(ch) else "-" for ch in stem)
    # 5. Compress hyphen runs.
    out = _HYPHEN_RUN.sub("-", out)
    # 6. Strip leading/trailing hyphens and dots (dots may slip through
    #    when stem ends with `.foo.bar` and we split off `.bar` as ext).
    out = out.strip("-.")
    # 7. Byte-length cap, UTF-8 boundary safe.
    out = _truncate_to_utf8_bytes(out, _STEM_BYTE_CAP)
    # Truncation may have left a dangling hyphen at the new tail; re-strip.
    out = out.strip("-.")
    # 8. Windows reserved name → underscore prefix (case-insensitive match).
    if out.upper() in _WIN_RESERVED:
        out = "_" + out
    return out


def assets_relative_path(
    *, hash_: str, original_path: str, dir_: str = "assets"
) -> str:
    """Compose the engine-managed relative storage path for an asset.

    Format::

        <dir_>/<h2>/<h8>[-<sanitized-stem>][.<ext>]

    where ``h2 = hash_[:2]`` is the 256-way shard and ``h8 = hash_[:8]`` is
    the per-file disambiguation prefix that lets two assets with identical
    sanitized names but different content live side by side without
    collision. The returned path is POSIX-style (``/`` separators) so it
    serializes uniformly into databases and cross-platform vaults.

    ``original_path`` may be an arbitrary reference string (relative path,
    bare filename, or even a path-with-subdirs); only the basename feeds
    the sanitized stem. ``dir_`` defaults to ``assets`` to match the
    convention written into ``dikw.yml`` under ``assets.dir``.
    """
    if len(hash_) < 8:
        raise ValueError(f"hash too short for sharding: {hash_!r}")
    h2 = hash_[:2]
    h8 = hash_[:8]
    stem, ext = _split_basename_and_ext(original_path)
    sanitized = _sanitize_stem(stem)
    name = h8
    if sanitized:
        name = f"{name}-{sanitized}"
    if ext:
        name = f"{name}.{ext}"
    return f"{dir_}/{h2}/{name}"


__all__ = ["assets_relative_path"]
