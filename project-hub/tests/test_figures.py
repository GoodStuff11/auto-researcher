import textwrap
from datetime import date

import pytest

from hub.config import load_config
from hub.figures import FigureError, add_figure, render_gallery, stale_figures, sync_figures, write_gallery
from hub.paths import PathError, resolve
from hub.snapshot import render_snapshot
from hub.store import Store

D = date(2026, 11, 13)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    (src / "fig.png").write_bytes(b"png-1")
    (src / "data" / "a.csv").write_text("1\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice}
        series: {}
        roots:
          SRC: /nowhere
        """))
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  SRC: {src}\n")
    return load_config(vault), vault, src


def test_resolve_forms(env):
    cfg, vault, src = env
    assert resolve(cfg, "$SRC/fig.png") == src / "fig.png"
    assert resolve(cfg, "$SRC") == src
    assert resolve(cfg, "Figures/x") == vault / "Figures" / "x"
    assert resolve(cfg, str(src / "fig.png")) == src / "fig.png"
    with pytest.raises(PathError, match="unknown root"):
        resolve(cfg, "$NOPE/x")


def test_add_sync_stale_gallery(env):
    cfg, vault, src = env
    d = add_figure(cfg, "energy-vs-u", "$SRC/fig.png", title="Energy vs U", data=["$SRC/data"],
                   licenses="Energy error falls with U.", caption="Energy error against U.")
    assert (d / "figure.png").read_bytes() == b"png-1"
    assert (d / "caption.md").read_text() == "Energy error against U.\n"
    assert sync_figures(cfg, today=D) == [{"id": "energy-vs-u", "copied": False, "stale": False, "problem": None}]
    (d / "caption.md").write_text("Edited by a human.\n")
    (src / "data" / "a.csv").write_text("2\n")                 # data changed, figure not regenerated
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    assert stale_figures(vault) == ["figure energy-vs-u: its data changed after the figure was made — "
                                    "regenerate it, then run `hub figures sync`"]
    (src / "fig.png").write_bytes(b"png-2")                   # figure regenerated
    row = sync_figures(cfg, today=D)[0]
    assert row["copied"] is True and row["stale"] is False
    assert (d / "figure.png").read_bytes() == b"png-2"
    assert (d / "caption.md").read_text() == "Edited by a human.\n"
    assert stale_figures(vault) == []
    text = render_gallery(vault)
    assert "![[Figures/energy-vs-u/figure.png]]" in text and "![[Figures/energy-vs-u/caption]]" in text
    assert "Energy error falls with U." in text
    assert write_gallery(vault) == vault / "Figures" / "Gallery.md"


def test_missing_source_is_a_problem_and_a_flag(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    (src / "fig.png").unlink()
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"].startswith("source missing")
    assert any("f1" in f for f in stale_figures(vault))
    assert "figure f1" in render_snapshot(Store(vault), cfg, D)


@pytest.mark.parametrize("fig_id,source,msg", [
    ("Bad Id", "$SRC/fig.png", "figure id"),
    ("f2", "$SRC/none.png", "does not exist"),
    ("f3", "$SRC/data", "not a file"),
])
def test_add_errors(env, fig_id, source, msg):
    cfg, vault, src = env
    with pytest.raises(FigureError, match=msg):
        add_figure(cfg, fig_id, source)


def test_duplicate_and_bad_data(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    with pytest.raises(FigureError, match="already registered"):
        add_figure(cfg, "f1", "$SRC/fig.png")
    with pytest.raises(FigureError, match="does not exist"):
        add_figure(cfg, "f2", "$SRC/fig.png", data=["$SRC/missing.csv"])
