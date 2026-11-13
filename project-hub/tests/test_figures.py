import textwrap
from datetime import date

import pytest
import yaml

import hub.figures as figures_mod
from hub.config import load_config
from hub.figures import FigureError, add_figure, render_gallery, stale_figures, sync_figures, write_gallery
from hub.paths import ROOT_REF_RE, PathError, resolve
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
    d = add_figure(cfg, "results-overview", "$SRC/fig.png", title="Results overview", data=["$SRC/data"],
                   licenses="Accuracy rises with sample size.", caption="Accuracy against sample size.")
    assert (d / "figure.png").read_bytes() == b"png-1"
    assert (d / "caption.md").read_text() == "Accuracy against sample size.\n"
    assert sync_figures(cfg, today=D) == [{"id": "results-overview", "copied": False, "stale": False,
                                           "problem": None, "skipped": None}]
    (d / "caption.md").write_text("Edited by a human.\n")
    (src / "data" / "a.csv").write_text("2\n")                 # data changed, figure not regenerated
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    assert stale_figures(vault) == ["figure results-overview: its data changed after the figure was made — "
                                    "regenerate it, then run `hub figures sync`"]
    (src / "fig.png").write_bytes(b"png-2")                   # figure regenerated
    row = sync_figures(cfg, today=D)[0]
    assert row["copied"] is True and row["stale"] is False
    assert (d / "figure.png").read_bytes() == b"png-2"
    assert (d / "caption.md").read_text() == "Edited by a human.\n"
    assert stale_figures(vault) == []
    text = render_gallery(vault)
    assert "![[Figures/results-overview/figure.png]]" in text and "![[Figures/results-overview/caption]]" in text
    assert "Accuracy rises with sample size." in text
    assert write_gallery(vault) == vault / "Figures" / "Gallery.md"


def test_missing_source_is_skipped_not_a_flag(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    (src / "fig.png").unlink()
    row = sync_figures(cfg, today=D)[0]
    assert row["skipped"] == "$SRC/fig.png" and row["problem"] is None
    assert stale_figures(vault) == []
    assert "figure f1" not in render_snapshot(Store(vault), cfg, D)


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


def _edit_meta(vault, fig_id, fn):
    p = vault / "Figures" / fig_id / "meta.yaml"
    meta = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(meta)
    p.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_sync_never_deletes_outside_the_figure(env):
    cfg, vault, src = env
    d = add_figure(cfg, "f1", "$SRC/fig.png")
    (d / "caption.md").write_text("keep caption\n")
    for bad in ["caption.md", "../../project.yaml"]:
        _edit_meta(vault, "f1", lambda m, bad=bad: m.update(figure_file=bad))
        (src / "fig.png").write_bytes(b"png-" + bad.encode())
        assert sync_figures(cfg, today=D)[0]["copied"] is True
        assert (d / "caption.md").read_text() == "keep caption\n"
        assert (vault / "project.yaml").exists()


def test_one_bad_meta_does_not_break_the_rest(env):
    cfg, vault, src = env
    add_figure(cfg, "bad", "$SRC/fig.png")
    add_figure(cfg, "good", "$SRC/fig.png")
    bad_meta = vault / "Figures" / "bad" / "meta.yaml"
    for content in ["id: bad\nsource: [unclosed\n", b"\xff\xfe\x00junk"]:
        if isinstance(content, bytes):
            bad_meta.write_bytes(content)
        else:
            bad_meta.write_text(content)
        rows = {r["id"]: r for r in sync_figures(cfg, today=D)}
        assert rows["good"] == {"id": "good", "copied": False, "stale": False, "problem": None, "skipped": None}
        assert rows["bad"]["copied"] is False and rows["bad"]["stale"] is False and rows["bad"]["problem"]
        text = render_gallery(vault)
        assert "## bad" in text and "**Problem:**" in text and "## good" in text
        assert any("figure bad" in f for f in stale_figures(vault))
        assert "figure bad" in render_snapshot(Store(vault), cfg, D)
        assert bad_meta.read_bytes() == (content if isinstance(content, bytes) else content.encode())


def test_stale_is_recomputed_not_sticky(env):
    cfg, vault, src = env
    add_figure(cfg, "s", "$SRC/fig.png", data=["$SRC/data"])
    (src / "data" / "a.csv").write_text("2\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    (src / "data" / "a.csv").write_text("1\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is False
    assert stale_figures(vault) == []


def test_hand_edited_data_list_rebaselines(env):
    cfg, vault, src = env
    add_figure(cfg, "s", "$SRC/fig.png", data=["$SRC/data"])
    (src / "more.csv").write_text("x\n")
    _edit_meta(vault, "s", lambda m: m.update(data=m["data"] + ["$SRC/more.csv"]))
    row = sync_figures(cfg, today=D)[0]
    assert row["stale"] is False and row["problem"] is None
    assert stale_figures(vault) == []


def test_old_meta_without_baseline_is_not_stale(env):
    cfg, vault, src = env
    add_figure(cfg, "s", "$SRC/fig.png", data=["$SRC/data"])
    _edit_meta(vault, "s", lambda m: m.pop("data_baseline", None))
    row = sync_figures(cfg, today=D)[0]
    assert row["stale"] is False and row["problem"] is None


@pytest.mark.parametrize("key,value,msg", [
    ("source", "", "source must be"),
    ("source", None, "source must be"),
    ("data", "$SRC/data", "data must be"),
    ("data", [""], "data must be"),
    ("data", [3], "data must be"),
])
def test_sync_reports_bad_types_as_problems(env, key, value, msg):
    cfg, vault, src = env
    add_figure(cfg, "t", "$SRC/fig.png")
    _edit_meta(vault, "t", lambda m: m.update({key: value}))
    row = sync_figures(cfg, today=D)[0]
    assert msg in row["problem"] and row["stale"] is False and row["copied"] is False
    assert "## t" in render_gallery(vault)


@pytest.mark.parametrize("source,data,msg", [
    ("", [], "source must be"),
    (None, [], "source must be"),
    ("$SRC/fig.png", "$SRC/data", "data must be"),
    ("$SRC/fig.png", [""], "data must be"),
    ("$SRC/fig.png", [3], "data must be"),
])
def test_add_rejects_bad_types(env, source, data, msg):
    cfg, vault, src = env
    with pytest.raises(FigureError, match=msg):
        add_figure(cfg, "t1", source, data=data)
    assert not (vault / "Figures" / "t1").exists()


def test_add_keeps_orphan_caption(env):
    cfg, vault, src = env
    d = vault / "Figures" / "x"
    d.mkdir(parents=True)
    (d / "caption.md").write_text("keep me")
    add_figure(cfg, "x", "$SRC/fig.png")
    assert (d / "caption.md").read_text() == "keep me"


def test_failed_add_removes_only_what_it_created(env, monkeypatch):
    cfg, vault, src = env

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(figures_mod.shutil, "copyfileobj", boom)
    with pytest.raises(OSError):
        add_figure(cfg, "new", "$SRC/fig.png")
    assert not (vault / "Figures" / "new").exists()
    d = vault / "Figures" / "keep"
    d.mkdir(parents=True)
    (d / "caption.md").write_text("keep me")
    with pytest.raises(OSError):
        add_figure(cfg, "keep", "$SRC/fig.png")
    assert (d / "caption.md").read_text() == "keep me"


@pytest.mark.parametrize("text,expected", [
    ("see $SRC/a/b.png.", "$SRC/a/b.png"),
    ("$SRC/x:", "$SRC/x"),
])
def test_root_ref_drops_trailing_punctuation(text, expected):
    m = ROOT_REF_RE.search(text)
    assert m and m.group(0) == expected


def test_root_ref_needs_a_word_boundary_before_dollar():
    assert ROOT_REF_RE.search("a$SRC/x") is None


def test_change_then_list_edit_stays_stale(env):
    cfg, vault, src = env
    (src / "more.csv").write_text("x\n")
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data"])
    (src / "data" / "a.csv").write_text("2\n")
    _edit_meta(vault, "f", lambda m: m.update(data=m["data"] + ["$SRC/more.csv"]))
    row = sync_figures(cfg, today=D)[0]
    assert row["stale"] is True and row["problem"] is None


def test_stale_survives_adding_an_entry(env):
    cfg, vault, src = env
    (src / "more.csv").write_text("x\n")
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data"])
    (src / "data" / "a.csv").write_text("2\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    _edit_meta(vault, "f", lambda m: m.update(data=m["data"] + ["$SRC/more.csv"]))
    assert sync_figures(cfg, today=D)[0]["stale"] is True


def test_reordering_the_list_changes_nothing(env):
    cfg, vault, src = env
    (src / "more.csv").write_text("x\n")
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data", "$SRC/more.csv"])
    _edit_meta(vault, "f", lambda m: m.update(data=list(reversed(m["data"]))))
    assert sync_figures(cfg, today=D)[0]["stale"] is False
    (src / "data" / "a.csv").write_text("2\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    _edit_meta(vault, "f", lambda m: m.update(data=list(reversed(m["data"]))))
    assert sync_figures(cfg, today=D)[0]["stale"] is True


def test_revert_clears_stale_after_list_edit(env):
    cfg, vault, src = env
    (src / "more.csv").write_text("x\n")
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data"])
    (src / "data" / "a.csv").write_text("2\n")
    _edit_meta(vault, "f", lambda m: m.update(data=m["data"] + ["$SRC/more.csv"]))
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    (src / "data" / "a.csv").write_text("1\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is False
    assert stale_figures(vault) == []


def test_new_entry_missing_is_skipped_until_it_exists(env):
    cfg, vault, src = env
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data"])
    _edit_meta(vault, "f", lambda m: m.update(data=m["data"] + ["$SRC/nope.csv"]))
    row = sync_figures(cfg, today=D)[0]
    assert row["skipped"] == "$SRC/nope.csv" and row["problem"] is None and row["stale"] is False
    (src / "nope.csv").write_text("1\n")
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"] is None and row["stale"] is False
    (src / "data" / "a.csv").write_text("CHANGED\n")
    assert sync_figures(cfg, today=D)[0]["stale"] is True


@pytest.mark.parametrize("stored", [None, "junk", {"$SRC/data": 7}])
def test_old_or_malformed_meta_is_baselined_not_stale(env, stored):
    cfg, vault, src = env
    add_figure(cfg, "f", "$SRC/fig.png", data=["$SRC/data"])

    def old_format(m):
        m.pop("data_hashes", None)
        m["data_hash"] = "deadbeefdeadbeef"
        m["data_baseline"] = "oops"
        if stored is not None:
            m["data_hashes"] = stored

    _edit_meta(vault, "f", old_format)
    (src / "data" / "a.csv").write_text("2\n")
    row = sync_figures(cfg, today=D)[0]
    assert row["stale"] is False and row["problem"] is None
    meta = yaml.safe_load((vault / "Figures" / "f" / "meta.yaml").read_text(encoding="utf-8"))
    assert "data_hash" not in meta and "data_baseline" not in meta
    assert set(meta["data_hashes"]) == {"$SRC/data"}


def test_regeneration_over_a_symlink_never_writes_through(env):
    cfg, vault, src = env
    d = add_figure(cfg, "f", "$SRC/fig.png")
    outside = vault.parent / "outside.png"
    outside.write_bytes(b"precious")
    (d / "figure.png").unlink()
    (d / "figure.png").symlink_to(outside)
    (src / "fig.png").write_bytes(b"png-2")
    assert sync_figures(cfg, today=D)[0]["copied"] is True
    assert outside.read_bytes() == b"precious"
    assert (d / "figure.png").read_bytes() == b"png-2" and not (d / "figure.png").is_symlink()


def test_extension_change_copies_new_then_removes_old(env):
    cfg, vault, src = env
    d = add_figure(cfg, "f", "$SRC/fig.png")
    (src / "fig.pdf").write_bytes(b"pdf")
    _edit_meta(vault, "f", lambda m: m.update(source="$SRC/fig.pdf"))
    assert sync_figures(cfg, today=D)[0]["copied"] is True
    assert (d / "figure.pdf").read_bytes() == b"pdf"
    assert not (d / "figure.png").exists()
    assert _meta(vault, "f")["figure_file"] == "figure.pdf"


def test_failed_extension_change_keeps_old_figure(env):
    cfg, vault, src = env
    d = add_figure(cfg, "f", "$SRC/fig.png")
    (src / "fig.pdf").write_bytes(b"pdf")
    _edit_meta(vault, "f", lambda m: m.update(source="$SRC/fig.pdf"))
    (d / "figure.pdf").mkdir()
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"] and row["copied"] is False
    assert (d / "figure.png").read_bytes() == b"png-1"
    assert _meta(vault, "f")["figure_file"] == "figure.png"


def _meta(vault, fig_id):
    return yaml.safe_load((vault / "Figures" / fig_id / "meta.yaml").read_text(encoding="utf-8"))
