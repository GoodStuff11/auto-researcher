import io
import time
from unittest.mock import MagicMock, patch

import pypdf

from auto_researcher.cookies import CookieStore
from auto_researcher.fetch import fetch_full_text, to_proxy_url
from auto_researcher.models import Paper


def _minimal_pdf_bytes() -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _paper(**kwargs):
    base = dict(
        id="x", title="T", authors=[], year=2020, venue=None, abstract=None,
        doi=None, arxiv_id=None, source="s", oa_pdf_url=None, landing_url=None,
    )
    base.update(kwargs)
    return Paper(**base)


def test_to_proxy_url_rewrites_host():
    assert to_proxy_url("https://ieeexplore.ieee.org/document/123") == (
        "https://ieeexplore-ieee-org.proxy.library.cornell.edu/document/123"
    )


def test_fetch_full_text_prefers_open_access():
    paper = _paper(oa_pdf_url="https://example.com/oa.pdf")
    fake_resp = MagicMock(status_code=200, text="full text here")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.status == "open_access"
    assert result.text == "full text here"


def test_fetch_full_text_unavailable_without_oa_or_cookies():
    paper = _paper(landing_url="https://ieeexplore.ieee.org/document/123")
    result = fetch_full_text(paper, cookie_store=None)
    assert result.status == "unavailable"
    assert result.text is None


def test_fetch_full_text_uses_proxy_when_cookies_fresh(tmp_path):
    # Realistic EZproxy cookie: scoped to the shared parent proxy domain,
    # not to the exact per-paper rewritten host.
    cookie_path = tmp_path / "cookies.txt"
    cookie_path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".proxy.library.cornell.edu\tTRUE\t/\tTRUE\t"
        f"{int(time.time()) + 3600}\tsession\tabc\n"
    )
    store = CookieStore(cookie_path)
    paper = _paper(landing_url="https://ieeexplore.ieee.org/document/123")
    fake_resp = MagicMock(
        status_code=200, text="proxied full text", headers={"content-type": "text/html"}
    )
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper, cookie_store=store)
    assert result.status == "proxy"
    assert result.text == "proxied full text"


def test_fetch_full_text_extracts_text_from_pdf_response():
    paper = _paper(oa_pdf_url="https://arxiv.org/pdf/1234.5678")
    pdf_bytes = _minimal_pdf_bytes()
    fake_resp = MagicMock(
        status_code=200,
        content=pdf_bytes,
        headers={"content-type": "application/pdf"},
    )
    # .text is intentionally left as a MagicMock/garbage value to prove the
    # PDF branch is used instead of falling back to raw response text.
    fake_resp.text = pdf_bytes.decode("latin-1")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.status == "open_access"
    assert isinstance(result.text, str)
    assert not result.text.startswith("%PDF")


def test_fetch_full_text_captures_raw_pdf_bytes():
    paper = _paper(oa_pdf_url="https://arxiv.org/pdf/1234.5678")
    pdf_bytes = _minimal_pdf_bytes()
    fake_resp = MagicMock(
        status_code=200,
        content=pdf_bytes,
        headers={"content-type": "application/pdf"},
    )
    fake_resp.text = pdf_bytes.decode("latin-1")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.pdf_bytes == pdf_bytes


def test_fetch_full_text_html_response_has_no_pdf_bytes():
    paper = _paper(oa_pdf_url="https://example.com/oa-landing")
    fake_resp = MagicMock(
        status_code=200,
        content=b"<html>full text here</html>",
        text="full text here",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.pdf_bytes is None


def test_fetch_full_text_html_open_access_still_uses_response_text():
    paper = _paper(oa_pdf_url="https://example.com/oa-landing")
    fake_resp = MagicMock(
        status_code=200,
        content=b"<html>full text here</html>",
        text="full text here",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.status == "open_access"
    assert result.text == "full text here"


def test_fetch_full_text_corrupt_pdf_falls_through_to_unavailable():
    paper = _paper(oa_pdf_url="https://example.com/broken.pdf")
    fake_resp = MagicMock(
        status_code=200,
        content=b"%PDF-not-actually-valid-content",
        headers={"content-type": "application/pdf"},
    )
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.status == "unavailable"
    assert result.text is None


def test_fetch_full_text_uses_unpaywall_when_no_direct_oa_url():
    paper = _paper(doi="10.1234/example")
    fake_resp = MagicMock(
        status_code=200,
        content=b"<html>unpaywall article</html>",
        text="unpaywall article",
        headers={"content-type": "text/html"},
    )
    with patch(
        "auto_researcher.fetch.find_oa_location", return_value="https://example.com/unpaywall.html"
    ) as mock_find, patch(
        "auto_researcher.fetch.request_with_retry", return_value=fake_resp
    ):
        result = fetch_full_text(paper, email="me@example.com")
    mock_find.assert_called_once_with("10.1234/example", email="me@example.com")
    assert result.status == "open_access"
    assert result.text == "unpaywall article"


def test_fetch_full_text_unpaywall_none_falls_through_to_unavailable():
    paper = _paper(doi="10.1234/example")
    with patch("auto_researcher.fetch.find_oa_location", return_value=None):
        result = fetch_full_text(paper, email="me@example.com")
    assert result.status == "unavailable"
    assert result.text is None


def test_fetch_full_text_skips_unpaywall_without_email():
    paper = _paper(doi="10.1234/example")
    with patch("auto_researcher.fetch.find_oa_location") as mock_find:
        result = fetch_full_text(paper, email=None)
    mock_find.assert_not_called()
    assert result.status == "unavailable"
