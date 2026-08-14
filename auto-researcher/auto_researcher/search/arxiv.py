from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, limit: int = 25) -> List[Paper]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 100)}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers: List[Paper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
        if not title:
            continue
        summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip()
        id_url = entry.findtext(f"{ATOM_NS}id") or ""
        arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url.rsplit("/", 1)[-1]
        published = entry.findtext(f"{ATOM_NS}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        papers.append(
            Paper(
                id=make_id(None, arxiv_id, title, year),
                title=title,
                authors=[a for a in authors if a],
                year=year,
                venue="arXiv",
                abstract=summary or None,
                doi=None,
                arxiv_id=arxiv_id,
                source="arxiv",
                oa_pdf_url=pdf_url,
                landing_url=id_url or None,
            )
        )
    return papers
