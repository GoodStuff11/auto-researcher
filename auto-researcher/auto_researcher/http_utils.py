from __future__ import annotations

import time

import requests


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    **kwargs,
) -> requests.Response:
    timeout = kwargs.pop("timeout", 20)
    last_exc: Exception | None = None
    resp: requests.Response | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
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
