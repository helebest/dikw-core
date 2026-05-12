"""Tests for ``dikw_core.client.converters`` — the client-side converter
plugin contract.

The contract lives entirely in the client (``dikw client``) and is
discovered via ``importlib.metadata`` entry-points so plugins can be
shipped as separate pypi packages without modifying dikw-core. These
tests cover ``discover()``'s indexing of entry-points and ``pick()``'s
selection priority (CLI flag > client.toml config > unique installed →
error on conflict / missing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core.client.converters import (
    ENTRY_POINT_GROUP,
    Converter,
    ConverterError,
    Registry,
    _index,
    discover,
    pick,
)


class _FakeConverter:
    def __init__(self, name: str, extensions: tuple[str, ...]) -> None:
        self.name = name
        self.extensions = extensions

    def convert(self, input_path: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / (input_path.stem + ".md")).write_text(
            f"# {input_path.stem}\n", encoding="utf-8"
        )


def _registry(*converters: _FakeConverter) -> Registry:
    return _index(converters)


def test_protocol_runtime_check() -> None:
    assert isinstance(_FakeConverter("fake", (".fake",)), Converter)


# ---- pick(): priority order ----------------------------------------------


def test_pick_unique_returns_it() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    assert pick(".pdf", _registry(marker)) is marker


def test_pick_extension_case_insensitive() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    assert pick(".PDF", _registry(marker)) is marker


def test_pick_missing_extension_raises_with_plugin_link() -> None:
    with pytest.raises(ConverterError) as exc:
        pick(".pdf", _registry())
    msg = str(exc.value)
    assert ".pdf" in msg
    assert "dikw-plugins" in msg


def test_pick_cli_flag_wins() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    mineru = _FakeConverter("mineru", (".pdf",))
    registry = _registry(marker, mineru)
    assert pick(".pdf", registry, converter="mineru") is mineru
    assert pick(".pdf", registry, converter="marker") is marker


def test_pick_cli_flag_unknown_raises() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    with pytest.raises(ConverterError) as exc:
        pick(".pdf", _registry(marker), converter="mineru")
    msg = str(exc.value)
    assert "mineru" in msg
    assert "marker" in msg


def test_pick_config_used_when_no_cli() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    mineru = _FakeConverter("mineru", (".pdf",))
    registry = _registry(marker, mineru)
    assert pick(".pdf", registry, config={".pdf": "mineru"}) is mineru


def test_pick_cli_overrides_config() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    mineru = _FakeConverter("mineru", (".pdf",))
    registry = _registry(marker, mineru)
    chosen = pick(".pdf", registry, converter="marker", config={".pdf": "mineru"})
    assert chosen is marker


def test_pick_config_unknown_raises() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    with pytest.raises(ConverterError) as exc:
        pick(".pdf", _registry(marker), config={".pdf": "ghostscript"})
    msg = str(exc.value)
    assert "ghostscript" in msg
    assert "marker" in msg


def test_pick_conflict_lists_options_and_remediation() -> None:
    marker = _FakeConverter("marker", (".pdf",))
    mineru = _FakeConverter("mineru", (".pdf",))
    with pytest.raises(ConverterError) as exc:
        pick(".pdf", _registry(marker, mineru))
    msg = str(exc.value)
    assert "marker" in msg
    assert "mineru" in msg
    assert "--converter" in msg
    assert "client.toml" in msg


# ---- discover(): entry-points integration --------------------------------


class _FakeEP:
    """Mimic just enough of importlib.metadata.EntryPoint for tests."""

    def __init__(self, klass: type) -> None:
        self.name = klass.__name__
        self._klass = klass

    def load(self) -> type:
        return self._klass


def _make_converter_class(name: str, extensions: tuple[str, ...]) -> type:
    """Build an ad-hoc Converter class — entry-points register classes,
    not instances, so discover() must do the construction."""
    return type(
        f"{name.capitalize()}Converter",
        (),
        {
            "name": name,
            "extensions": extensions,
            "convert": lambda self, ip, od: None,
        },
    )


def test_discover_calls_entry_points_with_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_entry_points(*, group: str) -> list[Any]:
        captured.append(group)
        return []

    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points", fake_entry_points
    )
    discover()
    assert captured == [ENTRY_POINT_GROUP]


def test_discover_loads_and_indexes_by_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_cls = _make_converter_class("marker", (".pdf",))
    mineru_cls = _make_converter_class("mineru", (".pdf",))
    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points",
        lambda *, group: [_FakeEP(marker_cls), _FakeEP(mineru_cls)],
    )
    registry = discover()
    assert set(registry.keys()) == {".pdf"}
    assert sorted(c.name for c in registry[".pdf"]) == ["marker", "mineru"]


def test_discover_multi_extension_plugin_appears_under_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combo_cls = _make_converter_class("combo", (".pdf", ".epub"))
    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points",
        lambda *, group: [_FakeEP(combo_cls)],
    )
    registry = discover()
    assert set(registry.keys()) == {".pdf", ".epub"}
    assert registry[".pdf"][0] is registry[".epub"][0]


def test_discover_rejects_plugin_missing_protocol_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin author who forgets ``name`` or ``extensions`` should get
    a clear error pointing at the offending entry-point, not an
    AttributeError surfacing later from ``pick()``."""
    broken_cls = type(
        "BrokenConverter",
        (),
        {"convert": lambda self, ip, od: None},
    )
    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points",
        lambda *, group: [_FakeEP(broken_cls)],
    )
    with pytest.raises(ConverterError) as exc:
        discover()
    assert "BrokenConverter" in str(exc.value)


def test_discover_skips_plugin_that_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One broken plugin (missing optional dep → ImportError, bad
    constructor → TypeError, …) must not block dispatch for every
    other extension. ``discover()`` logs a warning and continues."""

    class _ExplodingEP:
        name = "broken"

        def load(self) -> type:
            raise ImportError("optional dep 'torch' not installed")

    working_cls = _make_converter_class("marker", (".pdf",))

    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points",
        lambda *, group: [_ExplodingEP(), _FakeEP(working_cls)],
    )
    with caplog.at_level("WARNING", logger="dikw_core.client.converters"):
        registry = discover()

    # The working plugin survived; the broken one was skipped + logged.
    assert sorted(c.name for c in registry[".pdf"]) == ["marker"]
    assert any("broken" in rec.message and "ImportError" in rec.message for rec in caplog.records)


def test_discover_skips_plugin_with_broken_constructor(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plugin whose ``__init__`` raises (e.g. requires args we don't
    pass) is skipped the same way as a load failure."""

    def _bad_init(self: object) -> None:
        raise TypeError("MarkerConverter.__init__ requires model_path")

    bad_cls = type(
        "BadInitConverter",
        (),
        {
            "name": "bad",
            "extensions": (".pdf",),
            "convert": lambda self, ip, od: None,
            "__init__": _bad_init,
        },
    )
    monkeypatch.setattr(
        "dikw_core.client.converters.entry_points",
        lambda *, group: [_FakeEP(bad_cls)],
    )
    with caplog.at_level("WARNING", logger="dikw_core.client.converters"):
        registry = discover()

    assert registry == {}
    assert any("TypeError" in rec.message for rec in caplog.records)
