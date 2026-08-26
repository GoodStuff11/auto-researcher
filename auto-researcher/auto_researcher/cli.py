from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable, List, Optional

from urllib.parse import urlsplit

from . import store
from .browser_login import is_local_browser_available, print_manual_instructions, refresh_cookies_interactive
from .cookies import CookieStore
from .dedup import dedupe
from .fetch import fetch_full_text, to_proxy_url
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
    candidates_path: Optional[Path] = None,
    ids: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    cookies_path: Path = Path(".cookies.txt"),
    topic: Optional[str] = None,
    min_relevance: Optional[float] = None,
    min_read_priority: Optional[float] = None,
    store_root: Path = Path("store"),
) -> None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if candidates_path is not None and candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
    elif topic is not None:
        candidates = store.get_candidates(store_root, topic)
    else:
        raise ValueError("Either candidates_path or topic must be provided to run_fetch")

    if ids is not None:
        id_set = set(ids)
    elif topic is not None and (min_relevance is not None or min_read_priority is not None):
        scores = store.list_scores(
            store_root, topic, min_relevance=min_relevance, min_read_priority=min_read_priority
        )
        id_set = {s["id"] for s in scores}
    else:
        id_set = {c["id"] for c in candidates}

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
            if cached.get("fulltext") and out_dir is not None:
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
        if result.text and out_dir is not None:
            safe_name = store.safe_paper_id(paper.id)
            (out_dir / f"{safe_name}.txt").write_text(result.text)
        store.record_fulltext(
            store_root, paper.id, text=result.text, status=result.status,
            source_url=paper.oa_pdf_url or paper.landing_url, pdf_bytes=result.pdf_bytes,
        )
    if out_dir is not None:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_cookies_status(domains: List[str], cookies_path: Path) -> dict:
    cookie_store = CookieStore(cookies_path) if cookies_path.exists() else None
    result = {}
    for domain in domains:
        proxy_domain = urlsplit(to_proxy_url(f"https://{domain}/")).netloc
        result[domain] = bool(cookie_store and cookie_store.is_fresh(proxy_domain))
    return result


def run_cookies_refresh(domains: List[str], cookies_path: Path) -> None:
    if not is_local_browser_available():
        print_manual_instructions(domains, cookies_path)
        return
    refresh_cookies_interactive(domains, cookies_path)


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


def run_store_list_ids(store_root: Path, topic: str) -> None:
    print(json.dumps(store.list_candidate_ids(store_root, topic), indent=2))


def run_store_get_candidates(store_root: Path, topic: str, ids: List[str]) -> None:
    print(json.dumps(store.get_candidates_by_ids(store_root, topic, ids), indent=2))


def run_store_unscored(store_root: Path, topic: str, limit: Optional[int]) -> None:
    ids = store.get_unscored_ids(store_root, topic, limit=limit)
    print(json.dumps(store.get_candidates_by_ids(store_root, topic, ids), indent=2))


def run_store_append_score(
    store_root: Path, topic: str, paper_id: str, relevance: float,
    read_priority: float, reason: str,
) -> None:
    store.append_score(store_root, topic, paper_id, relevance, read_priority, reason)


def _score_table(scores: List[dict]) -> str:
    lines = [f"{'id':<40} {'relevance':>9} {'read_priority':>13}  reason"]
    for s in scores:
        reason = (s.get("reason") or "")[:80]
        lines.append(
            f"{s['id'][:40]:<40} {s.get('relevance', 0):>9} {s.get('read_priority', 0):>13}  {reason}"
        )
    return "\n".join(lines)


def run_store_scores(
    store_root: Path, topic: str, min_relevance: Optional[float],
    min_read_priority: Optional[float], sort_by: str, top: Optional[int], fmt: str,
) -> None:
    scores = store.list_scores(
        store_root, topic, min_relevance=min_relevance,
        min_read_priority=min_read_priority, sort_by=sort_by, top=top,
    )
    if fmt == "json":
        print(json.dumps(scores, indent=2))
    else:
        print(_score_table(scores))


def run_store_chunk_paper(store_root: Path, paper_id: str, chunk_words: int) -> None:
    print(json.dumps(store.chunk_paper(store_root, paper_id, chunk_words=chunk_words), indent=2))


def run_store_paper_chunks(store_root: Path, paper_id: str) -> None:
    print(json.dumps(store.list_chunks(store_root, paper_id), indent=2))


def run_store_append_note(store_root: Path, paper_id: str, chunk_n: int, note: str) -> None:
    store.append_paper_note(store_root, paper_id, chunk_n, note)


def run_store_paper_notes(store_root: Path, paper_id: str) -> None:
    notes = store.get_paper_notes(store_root, paper_id)
    print(notes if notes is not None else "(no notes recorded for this paper yet)")


def run_store_summary(
    store_root: Path,
    topic: str,
    min_relevance: float = 5.0,
    min_read_priority: float = 7.0,
    fmt: str = "json",
) -> None:
    summary = store.get_topic_summary(
        store_root, topic, min_relevance=min_relevance, min_read_priority=min_read_priority
    )
    if fmt == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(f"Topic: {summary['topic_slug']}")
        print(f"Question: {summary['question']}")
        print(f"Total Candidates: {summary['total_candidates']}")
        print(f"Total Scored: {summary['total_scored']}")
        print(f"Unscored: {summary['unscored_count']}")
        print(f"Qualified (relevance >= {min_relevance}): {summary['qualified_count']}")
        print(f"Deep Read (read_priority >= {min_read_priority}): {summary['deep_read_count']}")
        print(f"Fulltext Cached: {summary['fetched_fulltext_count']}")
        print(f"Notes Recorded: {summary['notes_recorded_count']}")


def run_store_chunk_papers(
    store_root: Path,
    topic: str,
    ids: Optional[List[str]] = None,
    min_relevance: Optional[float] = None,
    min_read_priority: Optional[float] = None,
    chunk_words: int = 1200,
) -> None:
    chunks = store.chunk_papers_for_topic(
        store_root, topic, ids=ids, min_relevance=min_relevance,
        min_read_priority=min_read_priority, chunk_words=chunk_words,
    )
    print(json.dumps({pid: len(chk_list) for pid, chk_list in chunks.items()}, indent=2))


def run_store_bibliography(
    store_root: Path,
    topic: str,
    max_read: int = 50,
    min_relevance: float = 5.0,
    read_priority_threshold: float = 7.0,
    fmt: str = "json",
) -> None:
    bib = store.build_bibliography(
        store_root, topic, max_read=max_read,
        min_relevance=min_relevance, read_priority_threshold=read_priority_threshold,
    )
    if fmt == "json":
        print(json.dumps(bib, indent=2))
    else:
        for i, b in enumerate(bib, 1):
            star = "★ " if b["read_status"] == "full_text_read" else ""
            authors_str = ", ".join(b.get("authors") or [])
            year_str = f" ({b['year']})" if b.get("year") else ""
            print(f"{i}. {star}{authors_str}{year_str} — {b['title']} [{b['read_status']}]")


def run_store_generate_report(
    store_root: Path,
    topic: str,
    out_path: Path,
    direct_answer_path: Optional[Path] = None,
    synthesis_path: Optional[Path] = None,
    gaps_path: Optional[Path] = None,
    caveats: Optional[str] = None,
    max_read: int = 50,
    min_relevance: float = 5.0,
    read_priority_threshold: float = 7.0,
) -> None:
    query = store.load_query(store_root, topic)
    if not query:
        raise ValueError(f"Topic {topic} not found in store {store_root}")

    direct_answer = direct_answer_path.read_text() if direct_answer_path and direct_answer_path.exists() else ""
    synthesis_body = synthesis_path.read_text() if synthesis_path and synthesis_path.exists() else (query.get("synthesis") or "")
    gaps_body = gaps_path.read_text() if gaps_path and gaps_path.exists() else ""

    report_md = store.generate_report_markdown(
        root=store_root,
        topic_slug=topic,
        direct_answer=direct_answer,
        synthesis_body=synthesis_body,
        references=None,
        gaps_body=gaps_body,
        caveats=caveats,
        max_read=max_read,
        min_relevance=min_relevance,
        read_priority_threshold=read_priority_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)
    print(f"Report generated and written to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the comprehensive command-line argument parser.

    Supported commands:
      - search: Query academic databases (arXiv, Semantic Scholar, CrossRef, OpenAlex, CORE)
      - fetch: Retrieve full text for papers (via open-access or university proxy)
      - cookies: Manage session cookies for subscription-gated publisher domains (status, refresh)
      - store: Interact with persistent queries, candidates, scores, notes, and report generation
    """
    parser = argparse.ArgumentParser(
        prog="auto_researcher",
        description="Autonomous literature discovery, scoring, full-text reading, and synthesis pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Search
    search_p = sub.add_parser("search", help="Search academic APIs for candidates.")
    search_p.add_argument("--query", action="append", required=True, dest="queries", help="Search query string (repeatable for multiple queries).")
    search_p.add_argument("--limit", type=int, default=25, help="Maximum number of candidates to fetch per query per source (default: 25).")
    search_p.add_argument("--out", type=Path, required=True, help="Destination path for writing raw candidate JSON.")
    search_p.add_argument("--topic", required=True, help="Unique topic slug for persistent storage in store/queries/<topic>/.")
    search_p.add_argument("--question", required=True, help="Original natural language question.")
    search_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root directory of the persistent paper store (default: store).")

    # 2. Fetch
    fetch_p = sub.add_parser("fetch", help="Fetch full-text PDFs/text for candidate papers.")
    fetch_p.add_argument("--in", type=Path, default=None, dest="candidates_path", help="Path to input candidates.json file.")
    fetch_p.add_argument("--topic", default=None, help="Topic slug in store to fetch candidates from directly.")
    fetch_p.add_argument("--ids", default=None, help="Comma-separated list of candidate IDs to fetch.")
    fetch_p.add_argument("--min-relevance", type=float, default=None, help="Minimum relevance score filter when selecting candidate IDs from topic.")
    fetch_p.add_argument("--min-read-priority", type=float, default=None, help="Minimum read priority score filter when selecting candidate IDs from topic.")
    fetch_p.add_argument("--out-dir", type=Path, default=None, help="Optional output directory to dump extracted text files.")
    fetch_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"), help="Path to Netscape-format cookie file for proxy authentication.")
    fetch_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root directory of the persistent paper store (default: store).")

    # 3. Cookies
    cookies_p = sub.add_parser("cookies", help="Inspect and refresh proxy cookies.")
    cookies_sub = cookies_p.add_subparsers(dest="cookies_command", required=True)

    cookies_status_p = cookies_sub.add_parser("status", help="Check if fresh proxy cookies are present for given domains.")
    cookies_status_p.add_argument("--domains", required=True, help="Comma-separated publisher domains (e.g. ieeexplore.ieee.org,link.springer.com).")
    cookies_status_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"), help="Path to cookies file.")

    cookies_refresh_p = cookies_sub.add_parser("refresh", help="Interactively refresh proxy cookies in browser or print manual export steps.")
    cookies_refresh_p.add_argument("--domains", required=True, help="Comma-separated publisher domains.")
    cookies_refresh_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"), help="Path to cookies file.")

    # 4. Store
    store_p = sub.add_parser("store", help="Access and manage persistent research store records.")
    store_sub = store_p.add_subparsers(dest="store_command", required=True)

    rs_p = store_sub.add_parser("record-scores", help="Persist a batch of scores into candidates.json.")
    rs_p.add_argument("--topic", required=True, help="Topic slug.")
    rs_p.add_argument("--scores", type=Path, required=True, help="JSON file containing list of {id, relevance, reason}.")
    rs_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    rsyn_p = store_sub.add_parser("record-synthesis", help="Persist final synthesized markdown and relevant candidate IDs.")
    rsyn_p.add_argument("--topic", required=True, help="Topic slug.")
    rsyn_p.add_argument("--relevant-ids", required=True, help="Comma-separated candidate IDs cited in synthesis.")
    rsyn_p.add_argument("--synthesis", type=Path, required=True, help="Path to synthesis markdown file.")
    rsyn_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    show_p = store_sub.add_parser("show", help="Display query record JSON.")
    show_p.add_argument("--topic", required=True, help="Topic slug.")
    show_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    show_paper_p = store_sub.add_parser("show-paper", help="Display paper metadata and cache status.")
    show_paper_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID (e.g. arxiv:2112.05731v1).")
    show_paper_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    list_p = store_sub.add_parser("list", help="List all stored research queries.")
    list_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    list_ids_p = store_sub.add_parser("list-ids", help="List all candidate IDs for a topic.")
    list_ids_p.add_argument("--topic", required=True, help="Topic slug.")
    list_ids_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    get_candidates_p = store_sub.add_parser("get-candidates", help="Retrieve full candidate metadata for specified IDs.")
    get_candidates_p.add_argument("--topic", required=True, help="Topic slug.")
    get_candidates_p.add_argument("--ids", required=True, help="Comma-separated paper IDs.")
    get_candidates_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    unscored_p = store_sub.add_parser("unscored", help="Retrieve candidate objects that have not yet been scored.")
    unscored_p.add_argument("--topic", required=True, help="Topic slug.")
    unscored_p.add_argument("--limit", type=int, default=None, help="Maximum number of unscored candidates to return.")
    unscored_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    append_score_p = store_sub.add_parser("append-score", help="Append a relevance & read priority score for a paper to scores.jsonl.")
    append_score_p.add_argument("--topic", required=True, help="Topic slug.")
    append_score_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID.")
    append_score_p.add_argument("--relevance", type=float, required=True, help="Relevance score (0.0 - 10.0).")
    append_score_p.add_argument("--read-priority", type=float, default=0, help="Read priority score (0.0 - 10.0).")
    append_score_p.add_argument("--reason", default="", help="One-sentence rationale for assigned score.")
    append_score_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    scores_p = store_sub.add_parser("scores", help="List and filter scores for candidates.")
    scores_p.add_argument("--topic", required=True, help="Topic slug.")
    scores_p.add_argument("--min-relevance", type=float, default=None, help="Minimum relevance threshold.")
    scores_p.add_argument("--min-read-priority", type=float, default=None, help="Minimum read priority threshold.")
    scores_p.add_argument("--sort-by", choices=["relevance", "read_priority"], default="relevance", help="Field to sort by.")
    scores_p.add_argument("--top", type=int, default=None, help="Maximum number of records to return.")
    scores_p.add_argument("--format", choices=["table", "json"], default="table", dest="fmt", help="Output format: 'table' or 'json'.")
    scores_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    summary_p = store_sub.add_parser("summary", help="Show topic summary metrics and pipeline progress.")
    summary_p.add_argument("--topic", required=True, help="Topic slug.")
    summary_p.add_argument("--min-relevance", type=float, default=5.0, help="Cutoff for qualified candidate count (default: 5.0).")
    summary_p.add_argument("--min-read-priority", type=float, default=7.0, help="Cutoff for deep-read candidate count (default: 7.0).")
    summary_p.add_argument("--format", choices=["table", "json"], default="json", dest="fmt", help="Output format: 'table' or 'json'.")
    summary_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    chunk_p = store_sub.add_parser("chunk-paper", help="Split single stored paper full text into chunk files.")
    chunk_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID.")
    chunk_p.add_argument("--chunk-words", type=int, default=1200, help="Word count per chunk (default: 1200).")
    chunk_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    chunks_p = store_sub.add_parser("chunk-papers", help="Split multiple stored papers into chunk files for a topic.")
    chunks_p.add_argument("--topic", required=True, help="Topic slug.")
    chunks_p.add_argument("--ids", default=None, help="Optional comma-separated list of paper IDs to chunk.")
    chunks_p.add_argument("--min-relevance", type=float, default=None, help="Minimum relevance filter for papers to chunk.")
    chunks_p.add_argument("--min-read-priority", type=float, default=None, help="Minimum read priority filter for papers to chunk.")
    chunks_p.add_argument("--chunk-words", type=int, default=1200, help="Word count per chunk (default: 1200).")
    chunks_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    paper_chunks_p = store_sub.add_parser("paper-chunks", help="List existing chunks for a paper.")
    paper_chunks_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID.")
    paper_chunks_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    append_note_p = store_sub.add_parser("append-note", help="Append a distilled insight note for a chunk to notes.md.")
    append_note_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID.")
    append_note_p.add_argument("--chunk", type=int, required=True, dest="chunk_n", help="Chunk sequence number.")
    append_note_p.add_argument("--note", required=True, help="Extracted insight note text.")
    append_note_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    paper_notes_p = store_sub.add_parser("paper-notes", help="View all accumulated notes for a paper.")
    paper_notes_p.add_argument("--id", required=True, dest="paper_id", help="Paper ID.")
    paper_notes_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    bib_p = store_sub.add_parser("bibliography", help="Generate full bibliography for a topic.")
    bib_p.add_argument("--topic", required=True, help="Topic slug.")
    bib_p.add_argument("--max-read", type=int, default=50, help="Maximum number of papers included in ranked set.")
    bib_p.add_argument("--min-relevance", type=float, default=5.0, help="Minimum relevance threshold.")
    bib_p.add_argument("--read-priority-threshold", type=float, default=7.0, help="Read priority threshold.")
    bib_p.add_argument("--format", choices=["table", "json"], default="json", dest="fmt", help="Output format: 'table' or 'json'.")
    bib_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    report_p = store_sub.add_parser("generate-report", help="Assemble and write a markdown report for a topic.")
    report_p.add_argument("--topic", required=True, help="Topic slug.")
    report_p.add_argument("--out", type=Path, required=True, help="Destination report path.")
    report_p.add_argument("--direct-answer", type=Path, default=None, help="Optional text file containing direct answer.")
    report_p.add_argument("--synthesis", type=Path, default=None, help="Optional text file containing synthesis body.")
    report_p.add_argument("--gaps", type=Path, default=None, help="Optional text file containing explicit gaps.")
    report_p.add_argument("--caveats", default=None, help="Optional caveat string for header.")
    report_p.add_argument("--max-read", type=int, default=50, help="Maximum papers in read set.")
    report_p.add_argument("--min-relevance", type=float, default=5.0, help="Minimum relevance threshold.")
    report_p.add_argument("--read-priority-threshold", type=float, default=7.0, help="Read priority threshold.")
    report_p.add_argument("--store-root", type=Path, default=Path("store"), help="Root store directory.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "search":
        run_search(
            args.queries, args.limit, args.out,
            topic=args.topic, question=args.question, store_root=args.store_root,
        )
    elif args.command == "fetch":
        ids_list = args.ids.split(",") if args.ids else None
        run_fetch(
            candidates_path=args.candidates_path,
            ids=ids_list,
            out_dir=args.out_dir,
            cookies_path=args.cookies,
            topic=args.topic,
            min_relevance=args.min_relevance,
            min_read_priority=args.min_read_priority,
            store_root=args.store_root,
        )
    elif args.command == "cookies":
        if args.cookies_command == "status":
            print(json.dumps(run_cookies_status(args.domains.split(","), args.cookies), indent=2))
        elif args.cookies_command == "refresh":
            run_cookies_refresh(args.domains.split(","), args.cookies)
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
        elif args.store_command == "list-ids":
            run_store_list_ids(args.store_root, args.topic)
        elif args.store_command == "get-candidates":
            run_store_get_candidates(args.store_root, args.topic, args.ids.split(","))
        elif args.store_command == "unscored":
            run_store_unscored(args.store_root, args.topic, args.limit)
        elif args.store_command == "append-score":
            run_store_append_score(
                args.store_root, args.topic, args.paper_id, args.relevance,
                args.read_priority, args.reason,
            )
        elif args.store_command == "scores":
            run_store_scores(
                args.store_root, args.topic, args.min_relevance, args.min_read_priority,
                args.sort_by, args.top, args.fmt,
            )
        elif args.store_command == "summary":
            run_store_summary(
                args.store_root, args.topic, args.min_relevance, args.min_read_priority, args.fmt
            )
        elif args.store_command == "chunk-paper":
            run_store_chunk_paper(args.store_root, args.paper_id, args.chunk_words)
        elif args.store_command == "chunk-papers":
            ids_list = args.ids.split(",") if args.ids else None
            run_store_chunk_papers(
                args.store_root, args.topic, ids=ids_list, min_relevance=args.min_relevance,
                min_read_priority=args.min_read_priority, chunk_words=args.chunk_words,
            )
        elif args.store_command == "paper-chunks":
            run_store_paper_chunks(args.store_root, args.paper_id)
        elif args.store_command == "append-note":
            run_store_append_note(args.store_root, args.paper_id, args.chunk_n, args.note)
        elif args.store_command == "paper-notes":
            run_store_paper_notes(args.store_root, args.paper_id)
        elif args.store_command == "bibliography":
            run_store_bibliography(
                args.store_root, args.topic, args.max_read, args.min_relevance,
                args.read_priority_threshold, args.fmt,
            )
        elif args.store_command == "generate-report":
            run_store_generate_report(
                args.store_root, args.topic, args.out, args.direct_answer,
                args.synthesis, args.gaps, args.caveats, args.max_read,
                args.min_relevance, args.read_priority_threshold,
            )



if __name__ == "__main__":
    main()
