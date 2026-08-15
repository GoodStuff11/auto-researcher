from __future__ import annotations

import time

import requests

# A default requests.request() call identifies itself as "python-requests/x.x",
# which publisher sites behind Cloudflare (e.g. journals.aps.org) block outright
# with a 403 challenge page before even checking cookies/auth. A browser-like
# User-Agent avoids that false-negative; callers can still override it via headers=.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    **kwargs,
) -> requests.Response:
    timeout = kwargs.pop("timeout", 20)
    headers = {"User-Agent": DEFAULT_USER_AGENT, **kwargs.pop("headers", {})}
    last_exc: Exception | None = None
    resp: requests.Response | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, headers=headers, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff_seconds * (2**attempt))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(backoff_seconds * (2**attempt))
            continue
        return resp
    if resp is not None:
        return resp
    assert last_exc is not None
    raise last_exc
