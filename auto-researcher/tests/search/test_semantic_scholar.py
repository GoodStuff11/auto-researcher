from unittest.mock import MagicMock, patch

from auto_researcher.search.semantic_scholar import search_semantic_scholar

SAMPLE_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Sample S2 Paper",
            "abstract": "An S2 abstract.",
            "year": 2020,
            "venue": "S2 Conference",
            "authors": [{"name": "Grace Hopper"}],
            "externalIds": {"DOI": "10.1000/s2", "ArXiv": "2001.00001"},
            "openAccessPdf": {"url": "https://example.com/s2.pdf"},
        }
    ]
}


def test_search_semantic_scholar_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch(
        "auto_researcher.search.semantic_scholar.request_with_retry", return_value=fake_resp
    ):
        papers = search_semantic_scholar("sample query", api_key=None, limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample S2 Paper"
    assert p.doi == "10.1000/s2"
    assert p.arxiv_id == "2001.00001"
    assert p.authors == ["Grace Hopper"]
    assert p.oa_pdf_url == "https://example.com/s2.pdf"
    assert p.landing_url == "https://www.semanticscholar.org/paper/abc123"
