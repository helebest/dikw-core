# ADR 0001: Client-side converter plugins for non-md inputs

- Status: accepted
- Date: 2026-05-12

## Context

`dikw client import` ships accepting `.md` only — every non-markdown file
(`paper.pdf`, `book.epub`, lecture audio, …) has to be converted to
md+assets externally before it can enter `<base>/sources/`. The
hard-coded `_DEFAULT_MD_EXTENSIONS = {".md"}` in `importer.py` and the
single `MarkdownBackend` in `domains/data/backends/` lock that boundary.

We want users to be able to run `dikw client import paper.pdf` and have
it work. The conversion logic itself (parsing PDFs with marker / MinerU
/ docling, EPUBs with ebook2md, …) is a heavyweight, fast-changing
ecosystem that doesn't belong inside dikw-core — both because of
dependency weight (PyTorch, OCR models, GB of weights) and because
each tool has its own release cadence we don't want to inherit.

## Decision

Add a **client-side converter plugin** mechanism. Plugins are separate
pypi packages discovered by `dikw client` via
`importlib.metadata.entry_points(group="dikw.client.converters")`. They
run in the client process, write md+assets to a temp staging directory,
and let the normal import flow package that directory like any other
md tree. The server never sees the plugin or its dependencies.

The contract is intentionally minimal:

```python
class Converter(Protocol):
    name: str                          # e.g. "marker", "mineru"
    extensions: tuple[str, ...]        # e.g. (".pdf",)
    def convert(self, input_path: Path, output_dir: Path) -> None: ...
```

Selection priority when an extension has multiple registered plugins:

1. `--converter=<name>` CLI flag (one-shot override).
2. `client.toml` `[default.converters]` entry for the extension
   (with per-extension `DIKW_CLIENT_CONVERTER_<EXT>` env override).
3. The single registered plugin if exactly one exists.
4. Otherwise raise `ConverterError` listing options + remediation.

The reference implementation lives in
[`src/dikw_core/client/converters.py`](../../src/dikw_core/client/converters.py);
plugins live in the sibling [`dikw-plugins`](https://github.com/opendikw/dikw-plugins) monorepo.

## Alternatives considered

### Reject: extend `SourceBackend` to handle `.pdf` / `.epub`

`domains/data/backends/` has a registry built exactly for "add a new
format = one class + register()" extensions. Why not put PDF support
there?

Because that registry is **engine-side** — `parse_any()` is called from
`api.ingest`, inside the server process. Wiring PDF support through it
means the server has to import PyTorch / Surya / whatever the
chosen converter pulls in. That breaks the `server/*` dependency
whitelist (CLAUDE.md: server depends only on `api/schemas/storage/providers`)
and means a remote `dikw serve` process needs the user's GPU to
process PDFs the user uploaded — reverse flow, doubly slow.

### Reject: server-side plugin (same entry-points contract, but on server)

Same dependency-injection objection as the SourceBackend route plus an
extra one: it requires the user to upload the raw PDF bytes to the
server first (the import endpoint validates md only today). For remote
deployments, PDFs would travel client → server → conversion →
storage; for local-dev the round trip is wasted. And the server can't
benefit from GPUs the client has access to.

### Reject: pure upstream tool, no integration

The cleanest possible decoupling: a separate CLI (`dikw-loaders pdf
paper.pdf -o /tmp/paper/`) that produces md+assets, then `dikw client
import /tmp/paper/` ingests them. dikw-core doesn't change at all.

This was actually the position our stored conventions documented
("preprocessing is upstream"), and it remains a valid escape hatch.
The single reason we picked client-side plugins over it: the user
explicitly wants the one-step UX `dikw client import paper.pdf`.
Two-step UX makes every PDF import require remembering two commands
and a temp-dir convention.

### Reject: optional extra `pip install dikw-core[pdf]`

We already have `pip install dikw-core[postgres]` for the Postgres
storage adapter, so the pattern exists. But postgres is genuinely part
of core (it's a storage backend selected by `storage.kind`); PDF
conversion isn't — we explicitly want the converter ecosystem off the
critical path of dikw-core's release cycle. Bundling them as an extra
forces every dikw-core release to coordinate with every plugin's deps,
inverting the decoupling we want.

## Consequences

**Positive**

- dikw-core's release cadence stays independent of any plugin's. Marker
  ships a breaking change → only `dikw-converter-pdf` cares.
- The dispatch path is small (~150 LOC in `client/converters.py` plus a
  short staging context manager in `importer.py`) and entirely client-
  scoped — `server/`, `storage/`, `providers/`, and ingest are untouched.
- Plugins can be authored by anyone; the contract is published, stable,
  and minimal. Once an external author starts shipping
  `dikw-converter-foo`, breaking the entry-points group name or the
  `Converter` Protocol signature requires a version-bump conversation.

**Negative**

- The client `pyproject.toml` weight class promise ("stdlib + httpx +
  typer + rich") now reads "core client is light; installing a plugin
  pulls its own deps into the client's venv". Users with heavy plugins
  + a lean baseline may want a separate venv per plugin.
- Plugin authors must image-ref every asset they emit
  (`![original](assets/paper.pdf)`) so `md_inspect` picks them up;
  failure to do so silently drops the asset (or trips the orphan check
  if the extension is in `_DEFAULT_ASSET_EXTENSIONS`). This is the
  same rule user-authored md+assets follow, but worth flagging in
  plugin-author docs.
- Multiple plugins claiming the same extension produces an error at
  dispatch time. v1 has no fallback / "auto-pick by quality" — the
  user must disambiguate explicitly. Acceptable cost for predictability.

## Out of scope (deliberately)

- Directory-mode dispatch (mixed md + non-md folders). v1 single-file
  only; users with a folder of PDFs run one import per file or convert
  externally. Revisit once usage shows the friction is real.
- Conversion result caching (`<base>/.dikw/converter-cache/`). Plugins
  re-convert on every import. Add a cache layer when re-import latency
  starts mattering.
- Progress + cancellation events from inside the plugin. v1 shows a
  spinner during conversion; if a plugin hangs the user can Ctrl+C.
- Server-side conversion of any kind. The architectural answer is
  "no" — see "Alternatives considered" above.
