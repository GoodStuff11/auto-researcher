import textwrap

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
