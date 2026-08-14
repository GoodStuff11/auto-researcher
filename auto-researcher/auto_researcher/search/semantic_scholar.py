from __future__ import annotations

from typing import List, Optional

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,externalIds,venue,openAccessPdf"


def search_semantic_scholar(
    query: str, api_key: Optional[str] = None, limit: int = 25
) -> List[Paper]:
    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}
    resp = request_with_retry("GET", BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("data", []):
        title = item.get("title") or ""
        if not title:
            continue
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        oa_pdf = (item.get("openAccessPdf") or {}).get("url")
        paper_id = item.get("paperId")

        papers.append(
            Paper(
                id=make_id(doi, arxiv_id, title, item.get("year")),
                title=title,
                authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
                year=item.get("year"),
                venue=item.get("venue") or None,
                abstract=item.get("abstract"),
                doi=doi,
                arxiv_id=arxiv_id,
                source="semantic_scholar",
                oa_pdf_url=oa_pdf,
                landing_url=(
                    f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None
                ),
            )
        )
    return papers
