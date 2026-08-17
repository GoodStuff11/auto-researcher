from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List

from .fetch import to_proxy_url

NETSCAPE_HEADER = "# Netscape HTTP Cookie File"


def is_local_browser_available() -> bool:
    """True if this process can plausibly pop open a real GUI browser window.

    False over SSH (no one is at the keyboard to see the window or tap Duo)
    or on Linux with no display server. Cookies always expire in hours, so
    this is checked fresh every time a refresh is requested, not cached.
    """
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def proxy_login_url(domain: str) -> str:
    return to_proxy_url(f"https://{domain}/")


def _bool_flag(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def cookies_to_netscape(cookies: List[dict]) -> str:
    """Serialize Playwright-style cookie dicts to the Netscape cookie file
    format `CookieStore`/`http.cookiejar.MozillaCookieJar` already reads."""
    lines = [NETSCAPE_HEADER]
    for c in cookies:
        domain = c.get("domain", "")
        expires = c.get("expires")
        expires_str = "0" if expires is None or expires < 0 else str(int(expires))
        lines.append(
            "\t".join(
                [
                    domain,
                    _bool_flag(domain.startswith(".")),
                    c.get("path", "/"),
                    _bool_flag(bool(c.get("secure"))),
                    expires_str,
                    c.get("name", ""),
                    c.get("value", ""),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def refresh_cookies_interactive(
    domains: List[str],
    cookies_path: Path,
    announce: Callable[[str], None] = print,
    poll_interval_s: float = 3.0,
    timeout_s: float = 600.0,
) -> None:
    """Open a real, visible browser for each proxied domain and wait for the
    user to complete Cornell NetID + Duo login themselves — Duo cannot be
    scripted past by design, so this automates everything *except* that tap.
    Polls for the resulting session cookies rather than blocking on stdin,
    since the caller may not have an interactive terminal to hand back to.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed. Run: pip install playwright "
            "&& playwright install chromium"
        ) from exc

    target_hosts = [proxy_login_url(d).split("//", 1)[1].rstrip("/") for d in domains]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        for domain in domains:
            announce(
                f"Opening {domain} through the Cornell proxy in a new tab — "
                "log in with your NetID and approve Duo if prompted."
            )
            context.new_page().goto(proxy_login_url(domain))

        announce(f"Waiting up to {int(timeout_s // 60)} minutes for login on all sites...")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            seen_hosts = {c["domain"].lstrip(".") for c in context.cookies()}
            if all(
                any(host == h or host.endswith("." + h) for host in seen_hosts)
                for h in target_hosts
            ):
                break
            time.sleep(poll_interval_s)
        else:
            announce("Timed out waiting for login; saving whatever cookies were captured so far.")

        cookies = context.cookies()
        browser.close()

    cookies_path.write_text(cookies_to_netscape(cookies))
    announce(f"Saved {len(cookies)} cookies to {cookies_path}")


MANUAL_INSTRUCTIONS = """No local browser is available here (SSH session or headless), so the
automated login can't run. Do this on your own desktop instead:

1. Log into the Cornell library proxy in a normal browser — visit any
   publisher page through the proxy (e.g. one of the domains below with
   `.proxy.library.cornell.edu` appended) and complete NetID + Duo login.
2. Visit at least one page on each domain you need, through the proxy:
{domains}
3. Export cookies with a browser extension that writes the Netscape
   format, e.g. "Get cookies.txt LOCALLY" — export the whole cookie jar,
   not just the current site.
4. Copy the file to this machine as `{cookies_path}`:
   scp cookies.txt <user>@<host>:{cookies_path}

Cookies expire in a few hours — repeat this each time a fetch reports
stale/missing proxy access, not just once."""


def print_manual_instructions(domains: List[str], cookies_path: Path, announce: Callable[[str], None] = print) -> None:
    domain_lines = "\n".join(f"   - {proxy_login_url(d).split('//', 1)[1].rstrip('/')}" for d in domains)
    announce(MANUAL_INSTRUCTIONS.format(domains=domain_lines, cookies_path=cookies_path))
