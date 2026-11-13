"""Figure sync never publishes one machine's local view as shared truth (final review I1, m3)."""
import subprocess
import textwrap
from datetime import date

import pytest
import yaml

from hub.cli import main
from hub.config import load_config
from hub.figures import add_figure, render_gallery, stale_figures, sync_figures
from hub.gitsync import GITATTRIBUTES_LINES
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


def _meta_bytes(vault, fig_id):
    return (vault / "Figures" / fig_id / "meta.yaml").read_bytes()


def test_missing_source_is_skipped_and_meta_untouched(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png", data=["$SRC/data"])
    before = _meta_bytes(vault, "f1")
    (src / "fig.png").unlink()
    row = sync_figures(cfg, today=D)[0]
    assert row["skipped"] == "$SRC/fig.png" and row["problem"] is None and row["stale"] is False
    assert _meta_bytes(vault, "f1") == before
    assert stale_figures(vault) == []
    assert "figure f1" not in render_snapshot(Store(vault), cfg, D)


def test_missing_data_entry_is_skipped_even_when_source_changed(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png", data=["$SRC/data"])
    before = _meta_bytes(vault, "f1")
    (src / "data" / "a.csv").unlink()
    (src / "data").rmdir()
    (src / "fig.png").write_bytes(b"png-2")
    row = sync_figures(cfg, today=D)[0]
    assert row["skipped"] == "$SRC/data" and row["copied"] is False
    assert _meta_bytes(vault, "f1") == before
    assert (vault / "Figures" / "f1" / "figure.png").read_bytes() == b"png-1"


def test_unknown_root_on_this_machine_is_skipped(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    p = vault / "Figures" / "f1" / "meta.yaml"
    p.write_text(p.read_text().replace("$SRC/fig.png", "$OTHER/fig.png"))
    before = p.read_bytes()
    row = sync_figures(cfg, today=D)[0]
    assert row["skipped"] == "$OTHER/fig.png" and row["problem"] is None
    assert p.read_bytes() == before


def test_old_problem_key_is_cleared_when_figure_syncs(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    p = vault / "Figures" / "f1" / "meta.yaml"
    meta = yaml.safe_load(p.read_text())
    meta["problem"] = "source missing: $SRC/fig.png (/some/local/path)"
    p.write_text(yaml.safe_dump(meta, sort_keys=False))
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"] is None
    assert not yaml.safe_load(p.read_text()).get("problem")
    assert "Problem" not in render_gallery(vault)


def test_copy_failure_writes_no_local_path(env):
    cfg, vault, src = env
    d = add_figure(cfg, "f1", "$SRC/fig.png")
    (src / "fig.pdf").write_bytes(b"pdf")
    p = d / "meta.yaml"
    p.write_text(p.read_text().replace("$SRC/fig.png", "$SRC/fig.pdf"))
    before = p.read_bytes()
    (d / "figure.pdf").mkdir()
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"] and row["copied"] is False
    assert p.read_bytes() == before
    assert str(vault) not in p.read_text() and str(src) not in p.read_text()


def test_bad_meta_types_are_still_problems(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    p = vault / "Figures" / "f1" / "meta.yaml"
    p.write_text(p.read_text().replace("source: $SRC/fig.png", "source: ''"))
    row = sync_figures(cfg, today=D)[0]
    assert "source must be" in row["problem"] and row["skipped"] is None
    assert str(src) not in p.read_text()


def test_symlinked_figure_folder_is_a_problem_and_never_written(env, tmp_path):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "meta.yaml").write_text((vault / "Figures" / "f1" / "meta.yaml").read_text().replace("f1", "evil"))
    before = sorted((p.name, p.read_bytes()) for p in outside.iterdir())
    (vault / "Figures" / "evil").symlink_to(outside)
    (src / "fig.png").write_bytes(b"png-2")
    rows = {r["id"]: r for r in sync_figures(cfg, today=D)}
    assert rows["evil"]["problem"] and "symbolic link" in rows["evil"]["problem"]
    assert rows["f1"]["copied"] is True
    assert sorted((p.name, p.read_bytes()) for p in outside.iterdir()) == before


def test_cli_prints_skipped_and_exits_zero(env, capsys, monkeypatch):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    (src / "fig.png").unlink()
    monkeypatch.chdir(vault)
    code = main(["figures", "sync"])
    out = capsys.readouterr().out
    assert code == 0
    assert "f1  SKIPPED: not available on this machine ($SRC/fig.png)" in out


def test_gallery_merges_ours_and_is_regenerated_by_commit(env, monkeypatch, capsys):
    cfg, vault, src = env
    assert "Figures/Gallery.md merge=ours" in GITATTRIBUTES_LINES
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=vault, check=True)
    for k, v in (("user.name", "a"), ("user.email", "a@x")):
        subprocess.run(["git", "config", k, v], cwd=vault, check=True)
    add_figure(cfg, "f1", "$SRC/fig.png", title="Results overview")
    monkeypatch.chdir(vault)
    assert main(["commit", "add figure"]) == 0
    assert "Results overview" in (vault / "Figures" / "Gallery.md").read_text()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=vault, capture_output=True, text=True).stdout
    assert "Gallery.md" not in status
