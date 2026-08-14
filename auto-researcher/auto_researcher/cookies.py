from __future__ import annotations

import http.cookiejar
import time
from pathlib import Path
from typing import Dict


class CookieStore:
    def __init__(self, cookies_path: Path):
        self.cookies_path = Path(cookies_path)
        self.jar = http.cookiejar.MozillaCookieJar()
        if self.cookies_path.exists():
            self.jar.load(str(self.cookies_path), ignore_discard=True, ignore_expires=True)

    def is_fresh(self, domain: str) -> bool:
        now = time.time()
        for cookie in self.jar:
            if domain in cookie.domain and cookie.expires and cookie.expires > now:
                return True
        return False

    def as_requests_cookies(self) -> Dict[str, str]:
        return {c.name: c.value for c in self.jar if c.value is not None}
