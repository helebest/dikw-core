"""Client-side converter plugin contract + discovery.

Converters turn non-markdown inputs (``.pdf``, ``.epub``, …) into
md+assets on disk so that :func:`dikw_core.client.importer.build_import`
can package them like any other markdown source. They run **in-process**
inside the ``dikw client``, never on the server, and are shipped as
separate pypi packages (see the ``dikw-plugins`` sibling repo).

The contract is intentionally minimal:

* A :class:`Converter` is anything that has a ``name`` (engine label like
  ``"marker"``), an ``extensions`` tuple (which file suffixes it claims),
  and a ``convert(input_path, output_dir)`` method that writes md+assets
  into ``output_dir``.
* Plugins register one entry-point per Converter class in the
  :data:`ENTRY_POINT_GROUP` group; :func:`discover` instantiates each and
  indexes them by extension.
* :func:`pick` resolves which converter to invoke for a given extension,
  following the priority CLI flag → ``client.toml`` config → unique
  installed → :class:`ConverterError`.

Heavy imports (PyTorch, OCR models, …) belong to plugin packages — this
module touches none of them and stays in the ``stdlib + httpx + typer +
rich`` client weight class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "dikw.client.converters"

_DIKW_PLUGINS_URL = "https://github.com/opendikw/dikw-plugins"


@runtime_checkable
class Converter(Protocol):
    """A client-side converter that turns one non-markdown input into
    md+assets on disk.

    Plugin authors implement this Protocol as a class registered via
    entry-points. ``convert`` is expected to be deterministic for the
    same input bytes — dikw-core's ingest pipeline relies on md hashes
    being stable across imports to skip unchanged sources.
    """

    name: str  # engine label, e.g. "marker", "mineru"
    extensions: tuple[str, ...]  # claimed file suffixes, e.g. (".pdf",)

    def convert(self, input_path: Path, output_dir: Path) -> None: ...


class ConverterError(Exception):
    """Raised when discovery or dispatch fails — bad plugin metadata,
    ambiguous extension routing, or a missing/unknown engine selection."""


Registry = dict[str, list[Converter]]


def _index(converters: Iterable[Converter]) -> Registry:
    """Build a ``{ext: [converter, ...]}`` index from converter instances."""
    out: Registry = {}
    for c in converters:
        for ext in c.extensions:
            out.setdefault(ext.lower(), []).append(c)
    return out


def _validate(klass: type, instance: object) -> None:
    """Surface plugin-author mistakes as ``ConverterError`` at discover
    time rather than letting an ``AttributeError`` leak out of
    :func:`pick`.

    ``runtime_checkable`` Protocols only verify methods, not attribute
    presence — so a class that defines ``convert`` but forgets ``name``
    or ``extensions`` would slip past ``isinstance``.
    """
    name = getattr(instance, "name", None)
    if not isinstance(name, str) or not name:
        raise ConverterError(
            f"converter plugin {klass.__name__!r} missing a non-empty "
            "``name`` attribute"
        )
    extensions = getattr(instance, "extensions", None)
    if not isinstance(extensions, tuple) or not extensions:
        raise ConverterError(
            f"converter plugin {klass.__name__!r} missing a non-empty "
            "``extensions`` tuple"
        )
    if not all(isinstance(e, str) and e.startswith(".") for e in extensions):
        raise ConverterError(
            f"converter plugin {klass.__name__!r} extensions must be "
            f"strings starting with '.', got {extensions!r}"
        )


def _load_entry_points() -> list[Converter]:
    """Load + instantiate every converter registered under
    :data:`ENTRY_POINT_GROUP`.

    Called only when dispatch on a non-md extension is needed — keeps
    ML-heavy plugin imports off the common CLI startup path.
    """
    out: list[Converter] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        klass = ep.load()
        instance = klass()
        _validate(klass, instance)
        out.append(instance)
    return out


def discover() -> Registry:
    """Return the extension → ``[converter, ...]`` registry from
    installed plugins. May raise :class:`ConverterError` if any
    registered plugin has malformed metadata."""
    return _index(_load_entry_points())


def pick(
    ext: str,
    registry: Registry,
    *,
    converter: str | None = None,
    config: Mapping[str, str] | None = None,
) -> Converter:
    """Resolve which converter handles ``ext``.

    Priority:

    1. ``converter`` — explicit ``--converter=<name>`` CLI flag.
    2. ``config[ext]`` — ``client.toml`` ``[default.converters]`` entry
       (which the caller has already merged with any
       ``DIKW_CLIENT_CONVERTER_<EXT>`` env override).
    3. Exactly one converter registered for ``ext`` → that one.
    4. Otherwise raise :class:`ConverterError` listing options and
       remediation.
    """
    ext = ext.lower()
    candidates = registry.get(ext, [])
    if not candidates:
        raise ConverterError(
            f"no converter installed for {ext!r}. "
            f"See {_DIKW_PLUGINS_URL} for available plugins."
        )

    if converter:
        for c in candidates:
            if c.name == converter:
                return c
        installed = ", ".join(sorted(c.name for c in candidates))
        raise ConverterError(
            f"converter {converter!r} not registered for {ext!r}; "
            f"installed: {installed}"
        )

    if config and ext in config:
        desired = config[ext]
        for c in candidates:
            if c.name == desired:
                return c
        installed = ", ".join(sorted(c.name for c in candidates))
        raise ConverterError(
            f"client.toml [default.converters] specifies {desired!r} for "
            f"{ext!r}, but only these are installed: {installed}"
        )

    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(sorted(c.name for c in candidates))
    raise ConverterError(
        f"multiple converters registered for {ext!r}: {names}. "
        f"Choose one via --converter=<name> or set client.toml "
        f"[default.converters] entry {ext!r} = '<name>'."
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "Converter",
    "ConverterError",
    "Registry",
    "discover",
    "pick",
]
