from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List

from .cookies import CookieStore
from .dedup import dedupe
from .fetch import fetch_full_text
from .models import Paper
from .search.arxiv import search_arxiv
from .search.core import search_core
from .search.crossref import search_crossref
from .search.openalex import search_openalex
from .search.semantic_scholar import search_semantic_scholar


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id, "title": p.title, "authors": p.authors, "year": p.year,
        "venue": p.venue, "abstract": p.abstract, "doi": p.doi,
        "arxiv_id": p.arxiv_id, "source": p.source,
        "oa_pdf_url": p.oa_pdf_url, "landing_url": p.landing_url,
    }


def _run_source(name: str, call: Callable[[], List[Paper]]) -> List[Paper]:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - any source failure must not abort the run
        print(f"warning: {name} search failed, skipping: {exc}", file=sys.stderr)
        return []


def run_search(queries: List[str], limit: int, out_path: Path) -> None:
    papers: List[Paper] = []
    for query in queries:
        papers.extend(_run_source("arxiv", lambda: search_arxiv(query, limit=limit)))
        papers.extend(
            _run_source(
                "semantic_scholar",
                lambda: search_semantic_scholar(
                    query, api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"), limit=limit
                ),
            )
        )
        if os.environ.get("CROSSREF_MAILTO"):
            papers.extend(
                _run_source(
                    "crossref",
                    lambda: search_crossref(
                        query, mailto=os.environ["CROSSREF_MAILTO"], limit=limit
                    ),
                )
            )
        if os.environ.get("OPENALEX_API_KEY"):
            papers.extend(
                _run_source(
                    "openalex",
                    lambda: search_openalex(
                        query, api_key=os.environ["OPENALEX_API_KEY"], limit=limit
                    ),
                )
            )
        if os.environ.get("CORE_API_KEY"):
            papers.extend(
                _run_source(
                    "core",
                    lambda: search_core(query, api_key=os.environ["CORE_API_KEY"], limit=limit),
                )
            )

    deduped = dedupe(papers)
    out_path.write_text(json.dumps([_paper_to_dict(p) for p in deduped], indent=2))


def run_fetch(
    candidates_path: Path, ids: List[str], out_dir: Path, cookies_path: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(candidates_path.read_text())
    id_set = set(ids)
    cookie_store = CookieStore(cookies_path) if cookies_path.exists() else None

    manifest = {}
    for item in candidates:
        if item["id"] not in id_set:
            continue
        paper = Paper(**item)
        try:
            result = fetch_full_text(paper, cookie_store)
        except Exception as exc:  # noqa: BLE001 - a single paper's fetch failure must not abort the batch
            print(
                f"warning: fetch failed for {paper.id}, marking unavailable: {exc}",
                file=sys.stderr,
            )
            manifest[paper.id] = "unavailable"
            continue
        manifest[paper.id] = result.status
        if result.text:
            safe_name = paper.id.replace(":", "_").replace("/", "_")
            (out_dir / f"{safe_name}.txt").write_text(result.text)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto_researcher")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search")
    search_p.add_argument("--query", action="append", required=True, dest="queries")
    search_p.add_argument("--limit", type=int, default=25)
    search_p.add_argument("--out", type=Path, required=True)

    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--in", type=Path, required=True, dest="candidates_path")
    fetch_p.add_argument("--ids", required=True)
    fetch_p.add_argument("--out-dir", type=Path, required=True)
    fetch_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"))

    args = parser.parse_args()
    if args.command == "search":
        run_search(args.queries, args.limit, args.out)
    elif args.command == "fetch":
        run_fetch(args.candidates_path, args.ids.split(","), args.out_dir, args.cookies)


if __name__ == "__main__":
    main()
