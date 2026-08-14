from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from .cookies import CookieStore
from .http_utils import request_with_retry
from .models import Paper

CORNELL_PROXY_SUFFIX = ".proxy.library.cornell.edu"


@dataclass
class FullTextResult:
    paper_id: str
    status: str  # "open_access", "proxy", "unavailable"
    text: Optional[str] = None


def to_proxy_url(url: str) -> str:
    parts = urlsplit(url)
    proxied_host = parts.netloc.replace(".", "-") + CORNELL_PROXY_SUFFIX
    return urlunsplit((parts.scheme, proxied_host, parts.path, parts.query, parts.fragment))


def fetch_full_text(
    paper: Paper, cookie_store: Optional[CookieStore] = None
) -> FullTextResult:
    if paper.oa_pdf_url:
        resp = request_with_retry("GET", paper.oa_pdf_url)
        if resp.status_code == 200:
            return FullTextResult(paper.id, "open_access", resp.text)

    if paper.landing_url and cookie_store is not None:
        proxy_url = to_proxy_url(paper.landing_url)
        proxy_domain = urlsplit(proxy_url).netloc
        if cookie_store.is_fresh(proxy_domain):
            resp = request_with_retry(
                "GET", proxy_url, cookies=cookie_store.as_requests_cookies()
            )
            if resp.status_code == 200:
                return FullTextResult(paper.id, "proxy", resp.text)

    return FullTextResult(paper.id, "unavailable", None)
