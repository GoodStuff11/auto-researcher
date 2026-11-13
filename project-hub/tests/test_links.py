import http.client
import textwrap
import urllib.error
from pathlib import Path

from hub.config import load_config
from hub.links import check_links


def _vault(tmp_path):
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text("name: demo\npeople: {alice: {name: Alice}}\nseries: {}\nroots: {CODE: /x}\n")
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {code}\n")
    (vault / "Locator.md").write_text(textwrap.dedent("""\
        # Locator
        - Solver: `$CODE/src`
        - Old script: `$CODE/old.py`
        - Elsewhere: $NOPE/thing
        - Energy $E_0$ and $x$ are maths, not paths.
        - Repo: https://github.com/example/demo
        """))
    return load_config(vault)


def test_offline(tmp_path):
    rows = check_links(_vault(tmp_path))
    by_ref = {r["ref"]: r for r in rows}
    assert set(by_ref) == {"$CODE/src", "$CODE/old.py", "$NOPE/thing", "https://github.com/example/demo"}
    assert by_ref["$CODE/src"]["ok"] is True
    assert by_ref["$CODE/old.py"]["ok"] is False and "does not exist" in by_ref["$CODE/old.py"]["problem"]
    assert by_ref["$NOPE/thing"]["ok"] is False and "unknown root" in by_ref["$NOPE/thing"]["problem"]
    assert by_ref["https://github.com/example/demo"]["ok"] is None


def test_online_uses_opener(tmp_path):
    calls = []

    class Resp:
        status = 404

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        calls.append(req.full_url)
        return Resp()

    rows = check_links(_vault(tmp_path), online=True, opener=opener)
    url = next(r for r in rows if r["kind"] == "url")
    assert calls == ["https://github.com/example/demo"]
    assert url["ok"] is False and "404" in url["problem"]


def test_figure_meta_paths(tmp_path):
    cfg = _vault(tmp_path)
    d = cfg.root / "Figures" / "f1"
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text("id: f1\nsource: $CODE/fig.png\nscript: ''\ndata: [$CODE/src]\n")
    refs = {r["ref"]: r for r in check_links(cfg) if r["file"] == "Figures/f1/meta.yaml"}
    assert refs["$CODE/fig.png"]["ok"] is False and refs["$CODE/src"]["ok"] is True


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _with_locator(cfg, text):
    (cfg.root / "Locator.md").write_text(text, encoding="utf-8")
    return cfg


def test_binary_locator_does_not_crash(tmp_path):
    cfg = _vault(tmp_path)
    (cfg.root / "Locator.md").write_bytes(b"\xff\xfe\x00\x80 binary \x00 $CODE/src \x81")
    rows = check_links(cfg)
    assert any(r["ref"] == "$CODE/src" and r["ok"] is True for r in rows)


def test_unreadable_locator_row_and_figures_still_checked(tmp_path):
    cfg = _vault(tmp_path)
    (cfg.root / "Locator.md").unlink()
    (cfg.root / "Locator.md").mkdir()
    d = cfg.root / "Figures" / "f1"
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text("source: $CODE/src\n")
    rows = check_links(cfg)
    loc = [r for r in rows if r["file"] == "Locator.md"]
    assert len(loc) == 1
    assert loc[0]["ref"] == "Locator.md" and loc[0]["kind"] == "path" and loc[0]["ok"] is False
    assert loc[0]["problem"].startswith("unreadable: ")
    assert any(r["file"] == "Figures/f1/meta.yaml" and r["ok"] is True for r in rows)


def test_transport_errors_do_not_abort(tmp_path):
    cfg = _with_locator(_vault(tmp_path), "- A https://a.example/incomplete\n- B https://b.example/boom\n- C https://c.example/ok\n")

    def opener(req, timeout):
        if "incomplete" in req.full_url:
            raise http.client.IncompleteRead(b"")
        if "boom" in req.full_url:
            raise RuntimeError("boom")
        return _Resp(200)

    by_ref = {r["ref"]: r for r in check_links(cfg, online=True, opener=opener)}
    assert "IncompleteRead" in by_ref["https://a.example/incomplete"]["problem"]
    assert by_ref["https://a.example/incomplete"]["ok"] is False
    assert by_ref["https://b.example/boom"]["problem"] == "unreachable: RuntimeError: boom"
    assert by_ref["https://c.example/ok"]["ok"] is True


def test_head_405_then_get_200(tmp_path):
    cfg = _with_locator(_vault(tmp_path), "see https://h.example/x\n")
    calls = []

    def opener(req, timeout):
        calls.append(req.get_method())
        if req.get_method() == "HEAD":
            raise urllib.error.HTTPError(req.full_url, 405, "no", {}, None)
        return _Resp(200)

    url = next(r for r in check_links(cfg, online=True, opener=opener) if r["kind"] == "url")
    assert url["ok"] is True
    assert calls == ["HEAD", "GET"]


def test_url_parentheses(tmp_path):
    cfg = _with_locator(_vault(tmp_path),
                        "- https://en.wikipedia.org/wiki/Foo_(bar)\n"
                        "- [x](https://ex.com/a)\n"
                        "- (see https://ex.com/b)\n"
                        "- end of sentence https://ex.com/c.\n")
    urls = {r["ref"] for r in check_links(cfg) if r["kind"] == "url"}
    assert urls == {"https://en.wikipedia.org/wiki/Foo_(bar)", "https://ex.com/a",
                    "https://ex.com/b", "https://ex.com/c"}


def test_closing_maths_dollar_is_not_a_path(tmp_path):
    cfg = _with_locator(_vault(tmp_path), "Sum $x/y$ is maths and $CODE/src is a path.\n")
    refs = {r["ref"] for r in check_links(cfg) if r["file"] == "Locator.md"}
    assert refs == {"$CODE/src"}


def test_spaced_backtick_path(tmp_path):
    cfg = _with_locator(_vault(tmp_path), "- Data: `$CODE/dir with spaces/file`\n")
    spaced = Path(cfg.roots["CODE"]) / "dir with spaces"
    spaced.mkdir()
    (spaced / "file").write_text("x")
    rows = [r for r in check_links(cfg) if r["file"] == "Locator.md"]
    assert len(rows) == 1
    assert rows[0]["ref"] == "$CODE/dir with spaces/file" and rows[0]["ok"] is True


def test_bad_refs_do_not_crash(tmp_path):
    long_ref = "$CODE/" + "a" * 5000
    nul_ref = "$CODE/bad\x00name"
    cfg = _with_locator(_vault(tmp_path), f"- {long_ref}\n- {nul_ref}\n")
    by_ref = {r["ref"]: r for r in check_links(cfg)}
    assert by_ref[long_ref]["ok"] is False
    assert by_ref[nul_ref]["ok"] is False
