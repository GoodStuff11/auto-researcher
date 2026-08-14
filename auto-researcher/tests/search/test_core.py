from unittest.mock import MagicMock, patch

from auto_researcher.search.core import search_core

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Sample CORE Paper",
            "abstract": "A CORE abstract.",
            "authors": [{"name": "Marie Curie"}],
            "yearPublished": 2018,
            "doi": "10.1000/core-sample",
            "downloadUrl": "https://core.ac.uk/download/sample.pdf",
            "publisher": "Sample Press",
        }
    ]
}


def test_search_core_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.core.request_with_retry", return_value=fake_resp):
        papers = search_core("sample query", api_key="fake-key", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample CORE Paper"
    assert p.abstract == "A CORE abstract."
    assert p.authors == ["Marie Curie"]
    assert p.doi == "10.1000/core-sample"
    assert p.oa_pdf_url == "https://core.ac.uk/download/sample.pdf"
