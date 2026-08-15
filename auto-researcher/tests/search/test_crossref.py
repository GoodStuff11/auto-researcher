from unittest.mock import MagicMock, patch

from auto_researcher.search.crossref import search_crossref

SAMPLE_RESPONSE = {
    "message": {
        "items": [
            {
                "title": ["Sample Crossref Paper"],
                "author": [{"given": "Alan", "family": "Turing"}],
                "published": {"date-parts": [[2019, 5]]},
                "container-title": ["Journal of Computation"],
                "DOI": "10.1000/crossref-sample",
                "URL": "https://doi.org/10.1000/crossref-sample",
            }
        ]
    }
}


def test_search_crossref_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.crossref.request_with_retry", return_value=fake_resp):
        papers = search_crossref("sample query", mailto="you@example.com", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample Crossref Paper"
    assert p.authors == ["Alan Turing"]
    assert p.year == 2019
    assert p.venue == "Journal of Computation"
    assert p.doi == "10.1000/crossref-sample"
    assert p.landing_url == "https://doi.org/10.1000/crossref-sample"
    assert p.abstract is None


def test_search_crossref_extracts_jats_abstract():
    response = {
        "message": {
            "items": [
                {
                    "title": ["Paper With Abstract"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "published": {"date-parts": [[2020]]},
                    "container-title": ["Journal of Computation"],
                    "DOI": "10.1000/crossref-abstract",
                    "URL": "https://doi.org/10.1000/crossref-abstract",
                    "abstract": "<jats:p>This paper studies <jats:italic>things</jats:italic>.</jats:p>",
                }
            ]
        }
    }
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = response
    with patch("auto_researcher.search.crossref.request_with_retry", return_value=fake_resp):
        papers = search_crossref("sample query", mailto="you@example.com", limit=5)
    assert papers[0].abstract == "This paper studies things."
