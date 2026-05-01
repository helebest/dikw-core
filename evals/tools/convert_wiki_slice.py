"""Build a small public-licensed multimodal eval slice from Wikipedia.

Fetches a hardcoded list of Wikipedia article summaries via the REST API,
downloads each article's lead image from Wikimedia Commons, and writes
them into the dikw-core 4-file dataset format
(``dataset.yaml`` + ``corpus/`` + ``targets.yaml`` + ``queries.yaml``).

License surface:

* Wikipedia summary text -> CC-BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>)
* Image bytes -> vary per file; see ``ATTRIBUTION.md`` for per-file source URL.
  Most lead images on en.wikipedia are CC-BY-SA or PD; the converter
  records the Commons page URL so a downstream user can verify.

The script is **manually run** (one-shot) and the generated output is
committed to ``evals/datasets/wiki-mini-mm/``. Re-running fetches fresh
content from Wikipedia, which may drift; the committed snapshot is
the source of truth for eval reproducibility.

Usage::

    set HTTPS_PROXY=http://localhost:1235  # if needed for network access
    uv run python -m evals.tools.convert_wiki_slice \\
        --out evals/datasets/wiki-mini-mm
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from ._common import dump_dataset_yaml, dump_queries_yaml, dump_targets_yaml

IMAGE_MAX_EDGE = 1024  # downscale target so committed images stay repo-friendly


@dataclass(frozen=True)
class ArticleSpec:
    slug: str            # stable id base (e.g., "eiffel_tower")
    wiki_title: str      # Wikipedia URL slug (e.g., "Eiffel_Tower")


# 6 articles spanning landmarks / animals / food / art so the corpus
# covers diverse visual content. Anchors are ASCII-slug stable identifiers
# usable as named ids in targets.yaml.
ARTICLES: list[ArticleSpec] = [
    ArticleSpec(slug="eiffel_tower", wiki_title="Eiffel_Tower"),
    ArticleSpec(slug="great_wall_of_china", wiki_title="Great_Wall_of_China"),
    ArticleSpec(slug="mount_fuji", wiki_title="Mount_Fuji"),
    ArticleSpec(slug="lion", wiki_title="Lion"),
    ArticleSpec(slug="sushi", wiki_title="Sushi"),
    ArticleSpec(slug="mona_lisa", wiki_title="Mona_Lisa"),
]


@dataclass
class FetchedArticle:
    spec: ArticleSpec
    title: str
    extract: str
    image_source_url: str
    image_bytes: bytes
    page_url: str
    description: str = ""


def _user_agent() -> str:
    """Wikipedia REST API requests require a UA per WMF policy."""
    return "dikw-core-eval-slice/0.1 (https://github.com/helebest/dikw-core)"


def _http_get(url: str, *, accept: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _resize_jpeg(data: bytes, *, max_edge: int) -> bytes:
    """Re-encode + downscale so the longer edge <= max_edge."""
    from PIL import Image  # Pillow is not a project dep; user installs once.

    img = Image.open(BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    out = BytesIO()
    img.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()


def fetch_article(spec: ArticleSpec) -> FetchedArticle:
    """Two API calls per article: REST ``/page/summary`` for the lead
    image + short description, MediaWiki ``api.php?prop=extracts`` for
    the full lead-section plain text. Summary alone (~150-500 chars) is
    too short for the chunker to split a doc into Description + Image
    sections; the extracts endpoint reliably yields 2k-4k chars so the
    heading-aware chunker breaks at each H2.
    """
    summary_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{spec.wiki_title}"
    )
    summary: dict[str, Any] = json.loads(
        _http_get(summary_url, accept="application/json")
    )

    title = str(summary.get("title") or spec.wiki_title.replace("_", " "))
    description = str(summary.get("description") or "")
    page_url = (
        summary.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{spec.wiki_title}"
    )
    img_block = summary.get("originalimage") or summary.get("thumbnail") or {}
    image_source_url = str(img_block.get("source") or "")
    if not image_source_url:
        raise RuntimeError(
            f"article {spec.wiki_title!r}: no image in REST summary response"
        )

    extracts_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "format": "json",
            "titles": spec.wiki_title,
            "exintro": "true",
            "explaintext": "true",
        }
    )
    extracts_data: dict[str, Any] = json.loads(
        _http_get(extracts_url, accept="application/json")
    )
    pages = extracts_data.get("query", {}).get("pages", {})
    full_extract = next(
        (str(p.get("extract") or "") for p in pages.values()), ""
    ).strip()
    if not full_extract:
        # Fall back to the summary's short extract if extracts API is empty.
        full_extract = str(summary.get("extract") or "").strip()

    print(f"  fetching image: {image_source_url}", flush=True)
    image_bytes = _resize_jpeg(_http_get(image_source_url), max_edge=IMAGE_MAX_EDGE)

    return FetchedArticle(
        spec=spec,
        title=title,
        extract=full_extract,
        image_source_url=image_source_url,
        image_bytes=image_bytes,
        page_url=page_url,
        description=description,
    )


def render_markdown(article: FetchedArticle) -> str:
    """Compose a 2-section markdown body — Description and Image — with
    explicit ``Target: <anchor>`` lines so the eval loader can resolve
    each chunk by anchor regardless of chunker boundaries.
    """
    slug = article.spec.slug
    return (
        f"---\n"
        f"title: {article.title}\n"
        f"source: {article.page_url}\n"
        f"text_license: CC-BY-SA-4.0\n"
        f"image_license: see ATTRIBUTION.md\n"
        f"---\n\n"
        f"# {article.title}\n\n"
        f"## Description\n\n"
        f"Target: {slug}.description\n\n"
        f"{article.extract}\n\n"
        f"## Image\n\n"
        f"Target: {slug}.image\n\n"
        f"![{article.title}](images/{slug}.jpg)\n\n"
        f"The image above shows {article.title}"
        f"{(' — ' + article.description) if article.description else ''}.\n"
    )


def build_targets(
    articles: list[FetchedArticle],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (assets, chunks) lists for ``targets.yaml``."""
    assets: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for a in articles:
        slug = a.spec.slug
        assets.append(
            {
                "id": f"{slug}.image",
                "doc": slug,
                "path": f"images/{slug}.jpg",
                "heading": "Image",
                "anchor": f"{slug}.image",
            }
        )
        chunks.append(
            {
                "id": f"{slug}.description",
                "doc": slug,
                "heading": "Description",
                "anchor": f"{slug}.description",
            }
        )
        chunks.append(
            {
                "id": f"{slug}.image_chunk",
                "doc": slug,
                "heading": "Image",
                "anchor": f"{slug}.image",
                "asset_id": f"{slug}.image",
            }
        )
    return assets, chunks


# Hand-authored queries per article: 1 doc, 1 description-chunk,
# 1 image-asset, 1 image-chunk = 4 queries x 6 articles = 24 queries.
# Each query's wording avoids verbatim copy from the wiki extract so
# retrieval has to actually generalize. Doc-level queries are the
# loosest; chunk and asset queries should narrow.
HAND_QUERIES: list[dict[str, Any]] = [
    {"id": "eiffel_doc", "query_type": "doc", "q": "What is the Eiffel Tower?",
     "expect_any": ["eiffel_tower"]},
    {"id": "eiffel_desc", "query_type": "text_chunk",
     "q": "Who built the Eiffel Tower and when?",
     "expect_any": ["eiffel_tower"], "expect_chunk_any": ["eiffel_tower.description"]},
    {"id": "eiffel_img_asset", "query_type": "asset",
     "q": "Find the photograph of the Eiffel Tower",
     "expect_any": ["eiffel_tower"], "expect_asset_any": ["eiffel_tower.image"]},
    {"id": "eiffel_img_chunk", "query_type": "text_chunk",
     "q": "section that shows the Eiffel Tower image",
     "expect_any": ["eiffel_tower"], "expect_chunk_any": ["eiffel_tower.image_chunk"]},

    {"id": "great_wall_doc", "query_type": "doc",
     "q": "ancient defensive wall in northern China",
     "expect_any": ["great_wall_of_china"]},
    {"id": "great_wall_desc", "query_type": "text_chunk",
     "q": "Why was the Great Wall of China built?",
     "expect_any": ["great_wall_of_china"],
     "expect_chunk_any": ["great_wall_of_china.description"]},
    {"id": "great_wall_img_asset", "query_type": "asset",
     "q": "photograph of the Great Wall winding over hills",
     "expect_any": ["great_wall_of_china"],
     "expect_asset_any": ["great_wall_of_china.image"]},
    {"id": "great_wall_img_chunk", "query_type": "text_chunk",
     "q": "section with the Great Wall photograph",
     "expect_any": ["great_wall_of_china"],
     "expect_chunk_any": ["great_wall_of_china.image_chunk"]},

    {"id": "fuji_doc", "query_type": "doc", "q": "tallest mountain in Japan",
     "expect_any": ["mount_fuji"]},
    {"id": "fuji_desc", "query_type": "text_chunk",
     "q": "Where is Mount Fuji located and how tall is it?",
     "expect_any": ["mount_fuji"], "expect_chunk_any": ["mount_fuji.description"]},
    {"id": "fuji_img_asset", "query_type": "asset",
     "q": "snow-capped Mount Fuji photo",
     "expect_any": ["mount_fuji"], "expect_asset_any": ["mount_fuji.image"]},
    {"id": "fuji_img_chunk", "query_type": "text_chunk",
     "q": "section showing the Fuji image",
     "expect_any": ["mount_fuji"], "expect_chunk_any": ["mount_fuji.image_chunk"]},

    {"id": "lion_doc", "query_type": "doc",
     "q": "large African big cat species",
     "expect_any": ["lion"]},
    {"id": "lion_desc", "query_type": "text_chunk",
     "q": "What does the lion look like and where does it live?",
     "expect_any": ["lion"], "expect_chunk_any": ["lion.description"]},
    {"id": "lion_img_asset", "query_type": "asset",
     "q": "photograph of a lion with a mane",
     "expect_any": ["lion"], "expect_asset_any": ["lion.image"]},
    {"id": "lion_img_chunk", "query_type": "text_chunk",
     "q": "section that displays the lion picture",
     "expect_any": ["lion"], "expect_chunk_any": ["lion.image_chunk"]},

    {"id": "sushi_doc", "query_type": "doc",
     "q": "Japanese rice and seafood dish",
     "expect_any": ["sushi"]},
    {"id": "sushi_desc", "query_type": "text_chunk",
     "q": "What ingredients does sushi traditionally use?",
     "expect_any": ["sushi"], "expect_chunk_any": ["sushi.description"]},
    {"id": "sushi_img_asset", "query_type": "asset",
     "q": "photo of sushi pieces on a plate",
     "expect_any": ["sushi"], "expect_asset_any": ["sushi.image"]},
    {"id": "sushi_img_chunk", "query_type": "text_chunk",
     "q": "section presenting the sushi photograph",
     "expect_any": ["sushi"], "expect_chunk_any": ["sushi.image_chunk"]},

    {"id": "mona_lisa_doc", "query_type": "doc",
     "q": "famous Renaissance portrait painted by Leonardo da Vinci",
     "expect_any": ["mona_lisa"]},
    {"id": "mona_lisa_desc", "query_type": "text_chunk",
     "q": "Who painted the Mona Lisa and where is it displayed?",
     "expect_any": ["mona_lisa"], "expect_chunk_any": ["mona_lisa.description"]},
    {"id": "mona_lisa_img_asset", "query_type": "asset",
     "q": "image of the Mona Lisa painting",
     "expect_any": ["mona_lisa"], "expect_asset_any": ["mona_lisa.image"]},
    {"id": "mona_lisa_img_chunk", "query_type": "text_chunk",
     "q": "section that contains the Mona Lisa image",
     "expect_any": ["mona_lisa"], "expect_chunk_any": ["mona_lisa.image_chunk"]},

    # Out-of-domain queries — surface unrelated retrieval behavior.
    {"id": "neg_quantum", "q": "How does quantum entanglement work in superconductors?",
     "expect_none": True},
    {"id": "neg_python_async", "q": "Python asyncio event loop best practices",
     "expect_none": True},
]


def render_attribution_md(articles: list[FetchedArticle]) -> str:
    lines = [
        "# Attribution",
        "",
        "Text content for each article is excerpted from English Wikipedia",
        "under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).",
        "",
        "Per-image attribution (image bytes were re-encoded as JPEG and",
        "downscaled; the source URL points at the upstream Commons file",
        "where the original license terms apply):",
        "",
    ]
    for a in articles:
        commons_filename = urllib.parse.unquote(
            posixpath.basename(urllib.parse.urlparse(a.image_source_url).path)
        )
        commons_page = (
            "https://commons.wikimedia.org/wiki/File:"
            + urllib.parse.quote(commons_filename, safe="")
        )
        lines.extend(
            [
                f"## {a.title}",
                "",
                f"- Article: <{a.page_url}>",
                f"- Image source: <{a.image_source_url}>",
                f"- Commons page: <{commons_page}>",
                f"- Stored at: corpus/images/{a.spec.slug}.jpg "
                f"(re-encoded JPEG, max edge {IMAGE_MAX_EDGE}px)",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target dataset directory (will be created/overwritten).",
    )
    args = parser.parse_args()
    out_dir: Path = args.out

    for var in ("HTTPS_PROXY", "HTTP_PROXY"):
        if os.environ.get(var):
            print(f"Using {var}={os.environ[var]}", file=sys.stderr, flush=True)

    print(f"Fetching {len(ARTICLES)} articles...", flush=True)
    fetched: list[FetchedArticle] = []
    for spec in ARTICLES:
        print(f"[{spec.slug}] {spec.wiki_title}", flush=True)
        fetched.append(fetch_article(spec))
        time.sleep(0.5)  # Be polite to Wikipedia REST.

    print(f"\nWriting to {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus"
    images = corpus / "images"
    images.mkdir(parents=True, exist_ok=True)

    for a in fetched:
        md_path = corpus / f"{a.spec.slug}.md"
        md_path.write_text(render_markdown(a), encoding="utf-8")
        img_path = images / f"{a.spec.slug}.jpg"
        img_path.write_bytes(a.image_bytes)
        print(
            f"  wrote {md_path.relative_to(out_dir)} "
            f"({len(a.extract)} chars) + {img_path.relative_to(out_dir)} "
            f"({len(a.image_bytes) // 1024} KB)"
        )

    assets, chunks = build_targets(fetched)
    dump_dataset_yaml(
        out_dir,
        name="wiki-mini-mm",
        description=(
            "Tiny public-licensed multimodal IR slice — six well-known "
            "Wikipedia articles (landmarks, animals, food, art) each rendered "
            "as a 2-section markdown (Description + Image) with the article's "
            "lead image from Wikimedia Commons. "
            "Text under CC-BY-SA-4.0; image licenses vary per file (see "
            "ATTRIBUTION.md). Generated via evals/tools/convert_wiki_slice.py."
        ),
        thresholds={},
    )
    dump_targets_yaml(out_dir, assets=assets, chunks=chunks)
    dump_queries_yaml(out_dir, HAND_QUERIES)
    (out_dir / "ATTRIBUTION.md").write_text(
        render_attribution_md(fetched), encoding="utf-8"
    )

    total_kb = sum(len(a.image_bytes) for a in fetched) // 1024
    print(
        f"\nDone. {len(fetched)} articles, "
        f"{len(HAND_QUERIES)} queries, total image ~{total_kb} KB."
    )


if __name__ == "__main__":
    main()
