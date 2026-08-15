from __future__ import annotations

import re
from typing import List, Optional

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.crossref.org/works"

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_abstract(raw: Optional[str]) -> Optional[str]:
    """CrossRef abstracts, when present, are JATS XML (e.g. "<jats:p>...</jats:p>")."""
    if not raw:
        return None
    text = _TAG_RE.sub("", raw).strip()
    return text or None


def search_crossref(query: str, mailto: str, limit: int = 25) -> List[Paper]:
    params = {"query": query, "rows": min(limit, 100), "mailto": mailto}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        if not title:
            continue

        authors = []
        for a in item.get("author", []) or []:
            name = " ".join(part for part in [a.get("given"), a.get("family")] if part)
            if name:
                authors.append(name)

        year = None
        date_parts = (item.get("published") or {}).get("date-parts")
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else None
        doi = item.get("DOI")

        papers.append(
            Paper(
                id=make_id(doi, None, title, year),
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                abstract=_clean_abstract(item.get("abstract")),
                doi=doi,
                arxiv_id=None,
                source="crossref",
                oa_pdf_url=None,
                landing_url=item.get("URL"),
            )
        )
    return papers
