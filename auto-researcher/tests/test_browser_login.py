import time

from auto_researcher.browser_login import (
    cookies_to_netscape,
    is_local_browser_available,
    proxy_login_url,
)
from auto_researcher.cookies import CookieStore


def test_proxy_login_url_rewrites_host():
    assert proxy_login_url("ieeexplore.ieee.org") == (
        "https://ieeexplore-ieee-org.proxy.library.cornell.edu/"
    )


def test_is_local_browser_available_false_over_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 1 5.6.7.8 22")
    monkeypatch.delenv("SSH_TTY", raising=False)
    assert is_local_browser_available() is False


def test_is_local_browser_available_false_on_headless_linux(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert is_local_browser_available() is False


def test_is_local_browser_available_true_on_linux_with_display(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    assert is_local_browser_available() is True


def test_cookies_to_netscape_round_trips_through_cookie_store(tmp_path):
    cookies = [
        {
            "name": "session",
            "value": "abc123",
            "domain": ".proxy.library.cornell.edu",
            "path": "/",
            "secure": True,
            "expires": time.time() + 3600,
        }
    ]
    cookies_path = tmp_path / "cookies.txt"
    cookies_path.write_text(cookies_to_netscape(cookies))

    store = CookieStore(cookies_path)
    assert store.is_fresh("ieeexplore-ieee-org.proxy.library.cornell.edu") is True
    assert store.as_requests_cookies() == {"session": "abc123"}


def test_cookies_to_netscape_treats_negative_expiry_as_session_cookie(tmp_path):
    cookies = [
        {
            "name": "session",
            "value": "abc123",
            "domain": ".proxy.library.cornell.edu",
            "path": "/",
            "secure": True,
            "expires": -1,
        }
    ]
    cookies_path = tmp_path / "cookies.txt"
    cookies_path.write_text(cookies_to_netscape(cookies))

    store = CookieStore(cookies_path)
    assert store.is_fresh("proxy.library.cornell.edu") is True
