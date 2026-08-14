from auto_researcher.dedup import dedupe
from auto_researcher.models import Paper


def _paper(**kwargs):
    base = dict(
        id="x", title="T", authors=[], year=2020, venue=None,
        abstract=None, doi=None, arxiv_id=None, source="s",
    )
    base.update(kwargs)
    return Paper(**base)


def test_dedupe_merges_exact_id_matches():
    a = _paper(id="doi:10.1/x", title="A Paper", abstract=None, source="crossref")
    b = _paper(id="doi:10.1/x", title="A Paper", abstract="An abstract", source="semantic_scholar")
    result = dedupe([a, b])
    assert len(result) == 1
    assert result[0].abstract == "An abstract"


def test_dedupe_merges_fuzzy_title_matches_same_year():
    a = _paper(id="title:aaa", title="Neural Quantum States for Fermions", year=2022, source="arxiv")
    b = _paper(
        id="title:bbb", title="Neural Quantum States for Fermions.", year=2022, source="openalex"
    )
    result = dedupe([a, b])
    assert len(result) == 1


def test_dedupe_keeps_distinct_papers():
    a = _paper(id="title:aaa", title="Neural Quantum States for Fermions", year=2022)
    b = _paper(id="title:bbb", title="Coupled Cluster Methods for Hubbard Models", year=2022)
    result = dedupe([a, b])
    assert len(result) == 2
