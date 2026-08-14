from __future__ import annotations

from typing import List, Optional

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, api_key: str, limit: int = 25) -> List[Paper]:
    params = {"search": query, "per-page": min(limit, 200), "api_key": api_key}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("results", []):
        title = item.get("title") or ""
        if not title:
            continue
        authors = [
            a["author"]["display_name"]
            for a in item.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ]
        doi = item.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        oa = item.get("open_access") or {}
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}

        papers.append(
            Paper(
                id=make_id(doi, None, title, item.get("publication_year")),
                title=title,
                authors=authors,
                year=item.get("publication_year"),
                venue=source.get("display_name"),
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                doi=doi,
                arxiv_id=None,
                source="openalex",
                oa_pdf_url=oa.get("oa_url"),
                landing_url=item.get("id"),
            )
        )
    return papers
