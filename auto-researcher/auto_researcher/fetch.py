from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import pypdf

from .cookies import CookieStore
from .http_utils import request_with_retry
from .models import Paper
from .search.unpaywall import find_oa_location

CORNELL_PROXY_SUFFIX = ".proxy.library.cornell.edu"


@dataclass
class FullTextResult:
    paper_id: str
    status: str  # "open_access", "proxy", "unavailable"
    text: Optional[str] = None
    pdf_bytes: Optional[bytes] = None


def to_proxy_url(url: str) -> str:
    parts = urlsplit(url)
    proxied_host = parts.netloc.replace(".", "-") + CORNELL_PROXY_SUFFIX
    return urlunsplit((parts.scheme, proxied_host, parts.path, parts.query, parts.fragment))


def _is_pdf_response(resp) -> bool:
    content_type = resp.headers.get("content-type", "")
    return "pdf" in content_type.lower() or resp.content[:4] == b"%PDF"


def _extract_text(resp) -> Optional[str]:
    if not _is_pdf_response(resp):
        return resp.text
    try:
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:  # noqa: BLE001 - a corrupt/encrypted PDF must not crash the run
        return None


def fetch_full_text(
    paper: Paper,
    cookie_store: Optional[CookieStore] = None,
    email: Optional[str] = None,
) -> FullTextResult:
    oa_url = paper.oa_pdf_url
    if not oa_url and paper.doi and email:
        oa_url = find_oa_location(paper.doi, email=email)

    if oa_url:
        resp = request_with_retry("GET", oa_url)
        if resp.status_code == 200:
            text = _extract_text(resp)
            if text is not None:
                pdf_bytes = resp.content if _is_pdf_response(resp) else None
                return FullTextResult(paper.id, "open_access", text, pdf_bytes)

    if paper.landing_url and cookie_store is not None:
        proxy_url = to_proxy_url(paper.landing_url)
        proxy_domain = urlsplit(proxy_url).netloc
        if cookie_store.is_fresh(proxy_domain):
            resp = request_with_retry(
                "GET", proxy_url, cookies=cookie_store.as_requests_cookies()
            )
            if resp.status_code == 200:
                text = _extract_text(resp)
                if text is not None:
                    pdf_bytes = resp.content if _is_pdf_response(resp) else None
                    return FullTextResult(paper.id, "proxy", text, pdf_bytes)

    return FullTextResult(paper.id, "unavailable", None)
