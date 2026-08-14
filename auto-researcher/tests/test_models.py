from auto_researcher.models import Paper, make_id


def test_make_id_prefers_doi():
    assert make_id("10.1000/xyz", "2101.00001", "Title", 2021) == "doi:10.1000/xyz"


def test_make_id_falls_back_to_arxiv():
    assert make_id(None, "2101.00001", "Title", 2021) == "arxiv:2101.00001"


def test_make_id_falls_back_to_title_hash():
    id1 = make_id(None, None, "Some Title", 2021)
    id2 = make_id(None, None, "Some Title", 2021)
    id3 = make_id(None, None, "Different Title", 2021)
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("title:")


def test_paper_dataclass_defaults():
    p = Paper(
        id="x", title="T", authors=[], year=2020, venue=None,
        abstract=None, doi=None, arxiv_id=None, source="test",
    )
    assert p.oa_pdf_url is None
    assert p.landing_url is None
