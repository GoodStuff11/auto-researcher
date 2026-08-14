from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable, List

from . import store
from .cookies import CookieStore
from .dedup import dedupe
from .fetch import fetch_full_text
from .models import Paper
from .search.arxiv import search_arxiv
from .search.core import search_core
from .search.crossref import search_crossref
from .search.openalex import search_openalex
from .search.semantic_scholar import search_semantic_scholar

_PAPER_FIELDS = {f.name for f in fields(Paper)}


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


def run_search(
    queries: List[str],
    limit: int,
    out_path: Path,
    topic: str | None = None,
    question: str | None = None,
    store_root: Path = Path("store"),
) -> None:
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

    if topic:
        for p in deduped:
            store.upsert_paper(store_root, p)
        store.record_query(store_root, topic, question or "", deduped)


def run_fetch(
    candidates_path: Path,
    ids: List[str],
    out_dir: Path,
    cookies_path: Path,
    store_root: Path = Path("store"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(candidates_path.read_text())
    id_set = set(ids)
    cookie_store = CookieStore(cookies_path) if cookies_path.exists() else None
    email = os.environ.get("CROSSREF_MAILTO")

    manifest = {}
    for item in candidates:
        if item["id"] not in id_set:
            continue
        paper = Paper(**{k: v for k, v in item.items() if k in _PAPER_FIELDS})
        store.upsert_paper(store_root, paper)

        if store.has_fulltext(store_root, paper.id):
            cached = store.load_paper(store_root, paper.id)
            manifest[paper.id] = cached["fetch_status"]["status"]
            if cached.get("fulltext"):
                safe_name = store.safe_paper_id(paper.id)
                (out_dir / f"{safe_name}.txt").write_text(cached["fulltext"])
            continue

        try:
            result = fetch_full_text(paper, cookie_store, email=email)
        except Exception as exc:  # noqa: BLE001 - a single paper's fetch failure must not abort the batch
            print(
                f"warning: fetch failed for {paper.id}, marking unavailable: {exc}",
                file=sys.stderr,
            )
            manifest[paper.id] = "unavailable"
            store.record_fulltext(store_root, paper.id, text=None, status="unavailable")
            continue

        manifest[paper.id] = result.status
        if result.text:
            safe_name = store.safe_paper_id(paper.id)
            (out_dir / f"{safe_name}.txt").write_text(result.text)
        store.record_fulltext(
            store_root, paper.id, text=result.text, status=result.status,
            source_url=paper.oa_pdf_url or paper.landing_url, pdf_bytes=result.pdf_bytes,
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_store_record_scores(store_root: Path, topic: str, scores: List[dict]) -> None:
    store.record_scores(store_root, topic, scores)


def run_store_record_synthesis(
    store_root: Path, topic: str, relevant_ids: List[str], synthesis_md: str
) -> None:
    store.record_synthesis(store_root, topic, relevant_ids, synthesis_md)


def run_store_show(store_root: Path, topic: str) -> None:
    print(json.dumps(store.load_query(store_root, topic), indent=2))


def run_store_show_paper(store_root: Path, paper_id: str) -> None:
    print(json.dumps(store.load_paper(store_root, paper_id), indent=2))


def run_store_list(store_root: Path) -> None:
    print(json.dumps(store.list_queries(store_root), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto_researcher")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search")
    search_p.add_argument("--query", action="append", required=True, dest="queries")
    search_p.add_argument("--limit", type=int, default=25)
    search_p.add_argument("--out", type=Path, required=True)
    search_p.add_argument("--topic", required=True)
    search_p.add_argument("--question", required=True)
    search_p.add_argument("--store-root", type=Path, default=Path("store"))

    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--in", type=Path, required=True, dest="candidates_path")
    fetch_p.add_argument("--ids", required=True)
    fetch_p.add_argument("--out-dir", type=Path, required=True)
    fetch_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"))
    fetch_p.add_argument("--store-root", type=Path, default=Path("store"))

    store_p = sub.add_parser("store")
    store_sub = store_p.add_subparsers(dest="store_command", required=True)

    rs_p = store_sub.add_parser("record-scores")
    rs_p.add_argument("--topic", required=True)
    rs_p.add_argument("--scores", type=Path, required=True)
    rs_p.add_argument("--store-root", type=Path, default=Path("store"))

    rsyn_p = store_sub.add_parser("record-synthesis")
    rsyn_p.add_argument("--topic", required=True)
    rsyn_p.add_argument("--relevant-ids", required=True)
    rsyn_p.add_argument("--synthesis", type=Path, required=True)
    rsyn_p.add_argument("--store-root", type=Path, default=Path("store"))

    show_p = store_sub.add_parser("show")
    show_p.add_argument("--topic", required=True)
    show_p.add_argument("--store-root", type=Path, default=Path("store"))

    show_paper_p = store_sub.add_parser("show-paper")
    show_paper_p.add_argument("--id", required=True, dest="paper_id")
    show_paper_p.add_argument("--store-root", type=Path, default=Path("store"))

    list_p = store_sub.add_parser("list")
    list_p.add_argument("--store-root", type=Path, default=Path("store"))

    args = parser.parse_args()
    if args.command == "search":
        run_search(
            args.queries, args.limit, args.out,
            topic=args.topic, question=args.question, store_root=args.store_root,
        )
    elif args.command == "fetch":
        run_fetch(
            args.candidates_path, args.ids.split(","), args.out_dir, args.cookies,
            store_root=args.store_root,
        )
    elif args.command == "store":
        if args.store_command == "record-scores":
            scores = json.loads(args.scores.read_text())
            run_store_record_scores(args.store_root, args.topic, scores)
        elif args.store_command == "record-synthesis":
            run_store_record_synthesis(
                args.store_root, args.topic, args.relevant_ids.split(","), args.synthesis.read_text()
            )
        elif args.store_command == "show":
            run_store_show(args.store_root, args.topic)
        elif args.store_command == "show-paper":
            run_store_show_paper(args.store_root, args.paper_id)
        elif args.store_command == "list":
            run_store_list(args.store_root)


if __name__ == "__main__":
    main()
