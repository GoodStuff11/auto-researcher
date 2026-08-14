from unittest.mock import MagicMock, patch

from auto_researcher.search.unpaywall import find_oa_location


def test_find_oa_location_returns_pdf_url():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "best_oa_location": {
            "url_for_pdf": "https://example.com/oa.pdf",
            "url": "https://example.com/oa",
        }
    }
    with patch("auto_researcher.search.unpaywall.request_with_retry", return_value=fake_resp):
        url = find_oa_location("10.1000/sample", email="you@example.com")
    assert url == "https://example.com/oa.pdf"


def test_find_oa_location_returns_none_on_404():
    fake_resp = MagicMock(status_code=404)
    with patch("auto_researcher.search.unpaywall.request_with_retry", return_value=fake_resp):
        url = find_oa_location("10.1000/missing", email="you@example.com")
    assert url is None
