import time
from pathlib import Path

from auto_researcher.cookies import CookieStore

NETSCAPE_HEADER = "# Netscape HTTP Cookie File\n"


def _write_cookie_file(path: Path, domain: str, expires: int) -> None:
    line = f"{domain}\tTRUE\t/\tTRUE\t{expires}\tsession\tabc123\n"
    path.write_text(NETSCAPE_HEADER + line)


def test_is_fresh_true_for_unexpired_cookie(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", int(time.time()) + 3600)
    store = CookieStore(cookie_path)
    assert store.is_fresh("proxy.library.cornell.edu") is True


def test_is_fresh_false_for_expired_cookie(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", int(time.time()) - 3600)
    store = CookieStore(cookie_path)
    assert store.is_fresh("proxy.library.cornell.edu") is False


def test_missing_cookie_file_has_no_fresh_cookies(tmp_path):
    store = CookieStore(tmp_path / "missing.txt")
    assert store.is_fresh("anything.com") is False


def test_is_fresh_true_for_rewritten_host_under_parent_proxy_domain(tmp_path):
    # Cornell EZproxy issues a session cookie scoped to the shared parent
    # domain, while fetch.py queries per-paper rewritten hosts. A stored
    # cookie for the parent domain must still match the rewritten host.
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", int(time.time()) + 3600)
    store = CookieStore(cookie_path)
    assert store.is_fresh("ieeexplore-ieee-org.proxy.library.cornell.edu") is True


def test_is_fresh_true_for_session_cookie_with_no_expiry(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", 0)
    store = CookieStore(cookie_path)
    assert store.is_fresh("ieeexplore-ieee-org.proxy.library.cornell.edu") is True
