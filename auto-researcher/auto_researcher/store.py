from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
