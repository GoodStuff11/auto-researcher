from __future__ import annotations

from typing import List

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.core.ac.uk/v3/search/works/"


def search_core(query: str, api_key: str, limit: int = 25) -> List[Paper]:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "limit": min(limit, 100)}
    resp = request_with_retry("GET", BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("results", []):
        title = item.get("title") or ""
        if not title:
            continue
        authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

        papers.append(
            Paper(
                id=make_id(item.get("doi"), None, title, item.get("yearPublished")),
                title=title,
                authors=authors,
                year=item.get("yearPublished"),
                venue=item.get("publisher"),
                abstract=item.get("abstract"),
                doi=item.get("doi"),
                arxiv_id=None,
                source="core",
                oa_pdf_url=item.get("downloadUrl"),
                landing_url=item.get("downloadUrl"),
            )
        )
    return papers
