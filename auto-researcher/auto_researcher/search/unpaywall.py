from __future__ import annotations

from typing import Optional

from ..http_utils import request_with_retry

BASE_URL = "https://api.unpaywall.org/v2/{doi}"


def find_oa_location(doi: str, email: str) -> Optional[str]:
    url = BASE_URL.format(doi=doi)
    resp = request_with_retry("GET", url, params={"email": email})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")
