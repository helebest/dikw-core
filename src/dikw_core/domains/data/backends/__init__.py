"""Source-backend registry.

Importing this module registers the built-in backends as a side effect; new
backends call ``register`` during their own import. The engine talks to
``parse_any`` / ``get_backend``; it never imports a concrete backend.
"""

from __future__ import annotations

from .base import (
    ParsedDocument,
    SourceBackend,
    UnsupportedFormat,
    get_backend,
    parse_any,
    register,
    supported_extensions,
)
from .markdown import MarkdownBackend

# Built-in registrations happen exactly once at import.
register(MarkdownBackend())


__all__ = [
    "MarkdownBackend",
    "ParsedDocument",
    "SourceBackend",
    "UnsupportedFormat",
    "get_backend",
    "parse_any",
    "register",
    "supported_extensions",
]
