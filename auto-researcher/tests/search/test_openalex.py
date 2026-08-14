from unittest.mock import MagicMock, patch

from auto_researcher.search.openalex import _reconstruct_abstract, search_openalex

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Sample OpenAlex Paper",
            "publication_year": 2022,
            "doi": "https://doi.org/10.1000/sample",
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {"source": {"display_name": "Journal of Sample Studies"}},
            "open_access": {"oa_url": "https://example.com/sample.pdf"},
            "abstract_inverted_index": {"This": [0], "is": [1], "an": [2], "abstract": [3]},
            "id": "https://openalex.org/W123",
        }
    ]
}


def test_reconstruct_abstract_orders_words():
    assert (
        _reconstruct_abstract({"This": [0], "is": [1], "an": [2], "abstract": [3]})
        == "This is an abstract"
    )


def test_reconstruct_abstract_handles_none():
    assert _reconstruct_abstract(None) is None


def test_search_openalex_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.openalex.request_with_retry", return_value=fake_resp):
        papers = search_openalex("sample query", api_key="fake-key", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample OpenAlex Paper"
    assert p.doi == "10.1000/sample"
    assert p.authors == ["Ada Lovelace"]
    assert p.venue == "Journal of Sample Studies"
    assert p.abstract == "This is an abstract"
    assert p.oa_pdf_url == "https://example.com/sample.pdf"
    assert p.source == "openalex"
