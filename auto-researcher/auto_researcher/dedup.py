from __future__ import annotations

import difflib
from typing import List

from .models import Paper


def _score(p: Paper) -> tuple:
    return (p.abstract is not None, p.doi is not None, p.oa_pdf_url is not None)


def _merge(a: Paper, b: Paper) -> Paper:
    winner = a if _score(a) >= _score(b) else b
    loser = b if winner is a else a
    return Paper(
        id=winner.id,
        title=winner.title,
        authors=winner.authors or loser.authors,
        year=winner.year or loser.year,
        venue=winner.venue or loser.venue,
        abstract=winner.abstract or loser.abstract,
        doi=winner.doi or loser.doi,
        arxiv_id=winner.arxiv_id or loser.arxiv_id,
        source=f"{winner.source}+{loser.source}" if winner.source != loser.source else winner.source,
        oa_pdf_url=winner.oa_pdf_url or loser.oa_pdf_url,
        landing_url=winner.landing_url or loser.landing_url,
    )


def dedupe(papers: List[Paper], title_similarity_threshold: float = 0.92) -> List[Paper]:
    by_id: dict[str, Paper] = {}
    for p in papers:
        by_id[p.id] = _merge(by_id[p.id], p) if p.id in by_id else p

    result: List[Paper] = []
    for p in by_id.values():
        match_idx = None
        for i, existing in enumerate(result):
            if existing.year != p.year:
                continue
            ratio = difflib.SequenceMatcher(
                None, existing.title.lower(), p.title.lower()
            ).ratio()
            if ratio >= title_similarity_threshold:
                match_idx = i
                break
        if match_idx is None:
            result.append(p)
        else:
            result[match_idx] = _merge(result[match_idx], p)
    return result
