import time
from unittest.mock import MagicMock, patch

from auto_researcher.cookies import CookieStore
from auto_researcher.fetch import fetch_full_text, to_proxy_url
from auto_researcher.models import Paper


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
    cookie_path = tmp_path / "cookies.txt"
    cookie_path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".ieeexplore-ieee-org.proxy.library.cornell.edu\tTRUE\t/\tTRUE\t"
        f"{int(time.time()) + 3600}\tsession\tabc\n"
    )
    store = CookieStore(cookie_path)
    paper = _paper(landing_url="https://ieeexplore.ieee.org/document/123")
    fake_resp = MagicMock(status_code=200, text="proxied full text")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper, cookie_store=store)
    assert result.status == "proxy"
    assert result.text == "proxied full text"
