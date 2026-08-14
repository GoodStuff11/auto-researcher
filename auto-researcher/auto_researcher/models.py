from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    source: str
    oa_pdf_url: Optional[str] = None
    landing_url: Optional[str] = None


def make_id(
    doi: Optional[str],
    arxiv_id: Optional[str],
    title: str,
    year: Optional[int],
) -> str:
    if doi:
        return f"doi:{doi.strip().lower()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip().lower()}"
    basis = f"{title.strip().lower()}|{year or ''}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"
