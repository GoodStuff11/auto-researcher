from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Paper


def safe_paper_id(paper_id: str) -> str:
    return paper_id.replace(":", "_").replace("/", "_")


def _paper_dir(root: Path, paper_id: str) -> Path:
    return Path(root) / "papers" / safe_paper_id(paper_id)


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id, "title": p.title, "authors": p.authors, "year": p.year,
        "venue": p.venue, "abstract": p.abstract, "doi": p.doi,
        "arxiv_id": p.arxiv_id, "source": p.source,
        "oa_pdf_url": p.oa_pdf_url, "landing_url": p.landing_url,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_paper(root: Path, paper: Paper) -> Path:
    paper_dir = _paper_dir(root, paper.id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    meta_path = paper_dir / "meta.json"

    new = _paper_to_dict(paper)
    if meta_path.exists():
        merged = json.loads(meta_path.read_text())
        for key, value in new.items():
            if not merged.get(key) and value:
                merged[key] = value
    else:
        merged = new
    meta_path.write_text(json.dumps(merged, indent=2))

    abstract_path = paper_dir / "abstract.txt"
    if paper.abstract and not abstract_path.exists():
        abstract_path.write_text(paper.abstract)

    return paper_dir


def has_fulltext(root: Path, paper_id: str) -> bool:
    return (_paper_dir(root, paper_id) / "fulltext.txt").exists()


def record_fulltext(
    root: Path,
    paper_id: str,
    text: Optional[str],
    status: str,
    source_url: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
) -> None:
    paper_dir = _paper_dir(root, paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    if pdf_bytes:
        (paper_dir / "fulltext.pdf").write_bytes(pdf_bytes)
    if text:
        (paper_dir / "fulltext.txt").write_text(text)
    (paper_dir / "fetch_status.json").write_text(
        json.dumps({"status": status, "source_url": source_url, "fetched_at": _now()}, indent=2)
    )


def load_paper(root: Path, paper_id: str) -> Optional[dict]:
    paper_dir = _paper_dir(root, paper_id)
    meta_path = paper_dir / "meta.json"
    if not meta_path.exists():
        return None

    result: dict = json.loads(meta_path.read_text())

    abstract_path = paper_dir / "abstract.txt"
    if abstract_path.exists():
        result["abstract"] = abstract_path.read_text()

    fulltext_path = paper_dir / "fulltext.txt"
    if fulltext_path.exists():
        result["fulltext"] = fulltext_path.read_text()

    status_path = paper_dir / "fetch_status.json"
    if status_path.exists():
        result["fetch_status"] = json.loads(status_path.read_text())

    return result


def _query_dir(root: Path, topic_slug: str) -> Path:
    return Path(root) / "queries" / topic_slug


def record_query(root: Path, topic_slug: str, question: str, candidates: List[Paper]) -> None:
    query_dir = _query_dir(root, topic_slug)
    query_dir.mkdir(parents=True, exist_ok=True)

    (query_dir / "question.txt").write_text(question)
    (query_dir / "candidates.json").write_text(
        json.dumps([_paper_to_dict(p) for p in candidates], indent=2)
    )

    created_path = query_dir / "created_at.txt"
    if not created_path.exists():
        created_path.write_text(_now())
    (query_dir / "updated_at.txt").write_text(_now())


def get_candidates(root: Path, topic_slug: str) -> List[dict]:
    """The full candidate list for a topic, as last written by record_query."""
    candidates_path = _query_dir(root, topic_slug) / "candidates.json"
    return json.loads(candidates_path.read_text()) if candidates_path.exists() else []


def list_candidate_ids(root: Path, topic_slug: str) -> List[str]:
    """Just the ids — cheap to hand to an orchestrator that shouldn't have to
    hold every candidate's title/abstract in context at once."""
    return [c["id"] for c in get_candidates(root, topic_slug)]


def get_candidates_by_ids(root: Path, topic_slug: str, ids: List[str]) -> List[dict]:
    id_set = set(ids)
    return [c for c in get_candidates(root, topic_slug) if c["id"] in id_set]


def _scores_path(root: Path, topic_slug: str) -> Path:
    return _query_dir(root, topic_slug) / "scores.jsonl"


def append_score(
    root: Path,
    topic_slug: str,
    paper_id: str,
    relevance: float,
    read_priority: float = 0,
    reason: str = "",
) -> None:
    """Append one candidate's score as a single JSON line. This is the
    incremental counterpart to record_scores: an agent that has just scored
    ONE paper (or a small batch, one call per paper) calls this immediately,
    so scoring never requires holding every candidate — or every score so
    far — in context at once. Re-scoring a paper is just appending again;
    load_scores/list_scores use the LAST line for a given id."""
    query_dir = _query_dir(root, topic_slug)
    query_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": paper_id,
        "relevance": relevance,
        "read_priority": read_priority,
        "reason": reason,
    }
    with open(_scores_path(root, topic_slug), "a") as f:
        f.write(json.dumps(entry) + "\n")
    (query_dir / "updated_at.txt").write_text(_now())


def load_scores(root: Path, topic_slug: str) -> Dict[str, dict]:
    """Every scored candidate, keyed by id, using the last-appended entry
    for ids scored more than once, with fallback to candidates.json."""
    result: Dict[str, dict] = {}
    path = _scores_path(root, topic_slug)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            result[entry["id"]] = entry
    else:
        for c in get_candidates(root, topic_slug):
            if c.get("relevance") is not None:
                result[c["id"]] = {
                    "id": c["id"],
                    "relevance": c.get("relevance"),
                    "read_priority": c.get("read_priority", 0),
                    "reason": c.get("reason", ""),
                }
    return result


def get_unscored_ids(root: Path, topic_slug: str, limit: Optional[int] = None) -> List[str]:
    """Candidate ids with no score yet, in the order they were found — the
    next batch for a scoring agent to pull via get_candidates_by_ids."""
    scored = set(load_scores(root, topic_slug).keys())
    ids = [cid for cid in list_candidate_ids(root, topic_slug) if cid not in scored]
    return ids[:limit] if limit is not None else ids


def list_scores(
    root: Path,
    topic_slug: str,
    min_relevance: Optional[float] = None,
    min_read_priority: Optional[float] = None,
    sort_by: str = "relevance",
    top: Optional[int] = None,
) -> List[dict]:
    """Navigate the scored set without re-reading every candidate: filter by
    a relevance/read_priority floor, sort by either field, and cap the
    count — the way an agent pages through hundreds of scored papers to
    decide what to read next."""
    scores = list(load_scores(root, topic_slug).values())
    if min_relevance is not None:
        scores = [s for s in scores if s.get("relevance", 0) >= min_relevance]
    if min_read_priority is not None:
        scores = [s for s in scores if s.get("read_priority", 0) >= min_read_priority]
    scores.sort(key=lambda s: s.get(sort_by, 0) or 0, reverse=True)
    return scores[:top] if top is not None else scores


def _chunks_dir(root: Path, paper_id: str) -> Path:
    return _paper_dir(root, paper_id) / "chunks"


def chunk_paper(root: Path, paper_id: str, chunk_words: int = 1200) -> List[dict]:
    """Split a paper's stored full text into word-bounded chunk files so an
    agent reads one chunk at a time instead of loading the whole paper into
    context. Returns the chunk index: [{n, path, word_count}, ...]."""
    fulltext_path = _paper_dir(root, paper_id) / "fulltext.txt"
    if not fulltext_path.exists():
        raise FileNotFoundError(f"no fulltext stored for {paper_id}; fetch it first")

    words = fulltext_path.read_text().split()
    chunks_dir = _chunks_dir(root, paper_id)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob("chunk_*.txt"):
        old.unlink()

    index: List[dict] = []
    n = 0
    for start in range(0, max(len(words), 1), chunk_words):
        n += 1
        piece = words[start:start + chunk_words]
        if not piece:
            continue
        chunk_path = chunks_dir / f"chunk_{n:04d}.txt"
        chunk_path.write_text(" ".join(piece))
        index.append({"n": n, "path": str(chunk_path), "word_count": len(piece)})

    (chunks_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def list_chunks(root: Path, paper_id: str) -> List[dict]:
    index_path = _chunks_dir(root, paper_id) / "index.json"
    return json.loads(index_path.read_text()) if index_path.exists() else []


def append_paper_note(root: Path, paper_id: str, chunk_n: int, note: str) -> None:
    """Append one chunk's distilled insight to the paper's running notes
    file. Notes accumulate on disk as a paper is read chunk-by-chunk, so the
    agent only ever needs the current chunk plus this small notes file in
    context — never the whole paper at once — and the notes (not the raw
    text) are what a later synthesis step reads."""
    paper_dir = _paper_dir(root, paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_dir / "notes.md", "a") as f:
        f.write(f"## Chunk {chunk_n}\n\n{note.strip()}\n\n")


def get_paper_notes(root: Path, paper_id: str) -> Optional[str]:
    notes_path = _paper_dir(root, paper_id) / "notes.md"
    return notes_path.read_text() if notes_path.exists() else None


def record_scores(root: Path, topic_slug: str, scores: List[dict]) -> None:
    query_dir = _query_dir(root, topic_slug)
    candidates_path = query_dir / "candidates.json"
    candidates = json.loads(candidates_path.read_text())

    score_by_id = {s["id"]: s for s in scores}
    for c in candidates:
        s = score_by_id.get(c["id"])
        if s:
            c["relevance"] = s.get("relevance")
            if "read_priority" in s:
                c["read_priority"] = s.get("read_priority")
            c["reason"] = s.get("reason")

    candidates_path.write_text(json.dumps(candidates, indent=2))
    (query_dir / "updated_at.txt").write_text(_now())


def record_synthesis(root: Path, topic_slug: str, relevant_ids: List[str], synthesis_md: str) -> None:
    query_dir = _query_dir(root, topic_slug)
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / "relevant_ids.json").write_text(json.dumps(relevant_ids, indent=2))
    (query_dir / "synthesis.md").write_text(synthesis_md)
    (query_dir / "updated_at.txt").write_text(_now())


def load_query(root: Path, topic_slug: str) -> Optional[dict]:
    query_dir = _query_dir(root, topic_slug)
    question_path = query_dir / "question.txt"
    if not question_path.exists():
        return None

    result: dict = {"topic_slug": topic_slug, "question": question_path.read_text()}

    candidates_path = query_dir / "candidates.json"
    result["candidates"] = json.loads(candidates_path.read_text()) if candidates_path.exists() else []

    # scores.jsonl (the incremental append-score path) is the primary source
    # going forward, but record_scores' candidates.json-embedded relevance
    # is honored too if that's the only thing present — merge scores.jsonl
    # in on top so callers see relevance/read_priority/reason either way.
    scores_by_id = load_scores(root, topic_slug)
    if scores_by_id:
        for c in result["candidates"]:
            s = scores_by_id.get(c["id"])
            if s:
                c["relevance"] = s.get("relevance")
                c["read_priority"] = s.get("read_priority")
                c["reason"] = s.get("reason")

    relevant_path = query_dir / "relevant_ids.json"
    result["relevant_ids"] = json.loads(relevant_path.read_text()) if relevant_path.exists() else []

    synthesis_path = query_dir / "synthesis.md"
    result["synthesis"] = synthesis_path.read_text() if synthesis_path.exists() else None

    return result


def list_queries(root: Path) -> List[dict]:
    queries_dir = Path(root) / "queries"
    if not queries_dir.exists():
        return []

    result = []
    for slug_dir in sorted(queries_dir.iterdir()):
        question_path = slug_dir / "question.txt"
        if not question_path.exists():
            continue
        created_path = slug_dir / "created_at.txt"
        result.append({
            "topic_slug": slug_dir.name,
            "question": question_path.read_text(),
            "created_at": created_path.read_text() if created_path.exists() else None,
        })
    return result


def get_topic_summary(
    root: Path,
    topic_slug: str,
    min_relevance: float = 5.0,
    min_read_priority: float = 7.0,
) -> dict:
    """Return summary statistics for a topic query in the store.

    Parameters:
        root: Base store directory path.
        topic_slug: Unique identifier for the research query topic.
        min_relevance: Cutoff relevance score for qualified candidates.
        min_read_priority: Cutoff read priority for deep read candidates.

    Returns:
        Dictionary containing counts and ID lists for candidates, scored,
        unscored, qualified, deep-read, fetched full-text, and noted papers.
    """
    query_dir = _query_dir(root, topic_slug)
    question_path = query_dir / "question.txt"
    question = question_path.read_text().strip() if question_path.exists() else ""

    candidate_ids = list_candidate_ids(root, topic_slug)
    scores = load_scores(root, topic_slug)
    unscored_ids = [cid for cid in candidate_ids if cid not in scores]

    qualified = [s for s in scores.values() if s.get("relevance", 0) >= min_relevance]
    qualified.sort(key=lambda s: s.get("relevance", 0), reverse=True)

    deep_read = [s for s in qualified if s.get("read_priority", 0) >= min_read_priority]
    deep_read.sort(key=lambda s: (s.get("read_priority", 0), s.get("relevance", 0)), reverse=True)

    fetched_count = sum(1 for cid in candidate_ids if has_fulltext(root, cid))
    notes_count = sum(1 for cid in candidate_ids if get_paper_notes(root, cid) is not None)

    return {
        "topic_slug": topic_slug,
        "question": question,
        "total_candidates": len(candidate_ids),
        "total_scored": len(scores),
        "unscored_count": len(unscored_ids),
        "unscored_ids": unscored_ids,
        "qualified_count": len(qualified),
        "qualified_ids": [s["id"] for s in qualified],
        "deep_read_count": len(deep_read),
        "deep_read_ids": [s["id"] for s in deep_read],
        "fetched_fulltext_count": fetched_count,
        "notes_recorded_count": notes_count,
    }


def chunk_papers_for_topic(
    root: Path,
    topic_slug: str,
    ids: Optional[List[str]] = None,
    min_relevance: Optional[float] = None,
    min_read_priority: Optional[float] = None,
    chunk_words: int = 1200,
) -> Dict[str, List[dict]]:
    """Chunk all available fulltext papers for a topic matching given IDs or thresholds.

    Parameters:
        root: Base store directory path.
        topic_slug: Unique identifier for the research query topic.
        ids: Optional explicit list of paper IDs to chunk.
        min_relevance: Cutoff relevance score for selecting papers to chunk.
        min_read_priority: Cutoff read priority for selecting papers to chunk.
        chunk_words: Maximum words per chunk.

    Returns:
        Mapping of paper_id to list of chunk descriptors.
    """
    if ids is not None:
        target_ids = ids
    else:
        scores = list_scores(root, topic_slug, min_relevance=min_relevance, min_read_priority=min_read_priority)
        target_ids = [s["id"] for s in scores]

    result: Dict[str, List[dict]] = {}
    for pid in target_ids:
        if has_fulltext(root, pid):
            chunks = chunk_paper(root, pid, chunk_words=chunk_words)
            result[pid] = chunks
    return result


def paper_link(paper: dict | Paper) -> Optional[str]:
    """Extract primary URL link for a paper record.

    Parameters:
        paper: Paper object or dictionary containing paper metadata.

    Returns:
        Best available URL link or None.
    """
    if isinstance(paper, Paper):
        p = _paper_to_dict(paper)
    else:
        p = paper
    if p.get("oa_pdf_url"):
        return p["oa_pdf_url"]
    if p.get("doi"):
        return f"https://doi.org/{p['doi']}"
    if p.get("landing_url"):
        return p["landing_url"]
    if p.get("arxiv_id"):
        return f"https://arxiv.org/abs/{p['arxiv_id']}"
    return None


def build_bibliography(
    root: Path,
    topic_slug: str,
    max_read: int = 50,
    min_relevance: float = 5.0,
    read_priority_threshold: float = 7.0,
) -> List[dict]:
    """Build structured bibliography of qualified candidates for a topic with read status.

    Parameters:
        root: Base store directory path.
        topic_slug: Unique identifier for the research query topic.
        max_read: Maximum number of papers to include in the ranked read set.
        min_relevance: Cutoff relevance score.
        read_priority_threshold: Threshold to consider paper for deep read.

    Returns:
        List of candidate dictionaries enriched with metadata, score info,
        and read_status ('full_text_read', 'abstract_only', or 'not_read').
    """
    scores = list_scores(root, topic_slug, min_relevance=min_relevance, sort_by="relevance")
    ranked_scores = scores[:max_read]
    ranked_ids = {s["id"] for s in ranked_scores}

    candidates = get_candidates(root, topic_slug)
    cand_by_id = {c["id"]: c for c in candidates}

    entries = []
    for s in scores:
        cid = s["id"]
        c = cand_by_id.get(cid, {})
        has_ft = has_fulltext(root, cid)
        has_notes = get_paper_notes(root, cid) is not None

        if cid in ranked_ids:
            if has_notes or (has_ft and s.get("read_priority", 0) >= read_priority_threshold):
                status = "full_text_read"
            else:
                status = "abstract_only"
        else:
            status = "not_read"

        entries.append({
            "id": cid,
            "title": c.get("title") or cid,
            "authors": c.get("authors") or [],
            "year": c.get("year"),
            "venue": c.get("venue"),
            "link": paper_link(c),
            "relevance": s.get("relevance", 0),
            "read_priority": s.get("read_priority", 0),
            "reason": s.get("reason", ""),
            "read_status": status,
            "has_fulltext": has_ft,
            "has_notes": has_notes,
        })
    return entries


def generate_report_markdown(
    root: Path,
    topic_slug: str,
    direct_answer: str,
    synthesis_body: str,
    references: Optional[List[dict]] = None,
    gaps_body: Optional[str] = None,
    caveats: Optional[str] = None,
    max_read: int = 50,
    min_relevance: float = 5.0,
    read_priority_threshold: float = 7.0,
) -> str:
    """Generate standardized report markdown adhering to repository standards.

    Parameters:
        root: Base store directory path.
        topic_slug: Unique identifier for the research query topic.
        direct_answer: Direct answer text.
        synthesis_body: Detailed synthesis body text.
        references: Optional list of cited reference dicts.
        gaps_body: Optional text detailing explicit gaps.
        caveats: Optional caveat string for header.
        max_read: Maximum papers included in read set.
        min_relevance: Minimum relevance for inclusion in bibliography.
        read_priority_threshold: Threshold for full text deep reads.

    Returns:
        Complete formatted Markdown string for the report.
    """
    query = load_query(root, topic_slug)
    if not query:
        raise ValueError(f"Topic {topic_slug} not found in store {root}")
    question = query["question"].strip()

    summary = get_topic_summary(root, topic_slug, min_relevance=min_relevance, min_read_priority=read_priority_threshold)
    bib = build_bibliography(root, topic_slug, max_read=max_read, min_relevance=min_relevance, read_priority_threshold=read_priority_threshold)

    deep_read_count = sum(1 for b in bib if b["read_status"] == "full_text_read")

    lines = [
        f"# {question}",
        "",
        f"**Question:** {question}",
        "",
    ]
    if caveats:
        lines.extend([f"*{caveats}*", ""])
    else:
        lines.extend([f"*Deepened pass: full text was fetched and read for {deep_read_count} of the papers below (marked with ★ in the bibliography); the rest remain abstract-only.*", ""])

    lines.extend([
        "## Direct answer",
        "",
        direct_answer.strip(),
        "",
        "## Synthesis",
        "",
        synthesis_body.strip(),
        "",
    ])

    if references:
        lines.extend([
            "## References",
            "",
        ])
        for r in references:
            n = r.get("n", "")
            authors = ", ".join(r.get("authors") or []) or "Unknown authors"
            year = r.get("year") or "n.d."
            title = r.get("title") or "Untitled"
            venue = f", *{r['venue']}*" if r.get("venue") else ""
            link = r.get("link") or ""
            link_str = f" {link}" if link else ""
            deep_tag = " [full text read]" if r.get("deepRead") or r.get("read_status") == "full_text_read" else ""
            lines.append(f"[{n}] {authors} ({year}) — \"{title}\"{venue}.{link_str}{deep_tag}")
        lines.append("")

    if gaps_body:
        lines.extend([
            "## Explicit gaps (not resolved by this search)",
            "",
            gaps_body.strip(),
            "",
        ])

    lines.extend([
        "## Bibliography",
        "",
        "*(★ = full text fetched and read for this pass; others remain abstract-only)*",
        "",
    ])

    for i, b in enumerate(bib, 1):
        star = "★ " if b["read_status"] == "full_text_read" else ""
        authors = ", ".join(b.get("authors") or []) or "Unknown authors"
        year = f" ({b['year']})" if b.get("year") else ""
        title = f"\"{b['title']}\""
        venue = f", *{b['venue']}*" if b.get("venue") else ""
        link = f" {b['link']}" if b.get("link") else ""

        if b["read_status"] == "full_text_read":
            status_tag = ""
        elif b["read_status"] == "abstract_only":
            status_tag = " — *abstract-only*"
        else:
            status_tag = " — *not read*"

        lines.append(f"{i}. {star}{authors}{year} — {title}{venue}.{link}{status_tag}")

    lines.extend([
        "",
        "---",
        f"*Full record — all {summary['total_candidates']} raw candidates, scores, extracts, and synthesis — is durably stored under `auto-researcher/store/queries/{topic_slug}/`, and cached full texts are preserved under `auto-researcher/store/papers/` for reuse by any future question. Revisit with the `research-followup` skill.*",
        "",
    ])

    return "\n".join(lines)

