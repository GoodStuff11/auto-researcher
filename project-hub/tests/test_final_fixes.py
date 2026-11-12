"""Plan 2 final-review fix wave: one section per finding (I1-I6 and the minors)."""
import subprocess
from datetime import date

import pytest

from hub.config import load_config
from hub.store import Store

D = date(2026, 11, 12)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------- I1: same-day watermark merges

def _wm(value, updated=None):
    return f"value: '{value}'\n" + (f"updated: '{updated}'\n" if updated else "")


@pytest.mark.parametrize("a, b, winner", [
    (_wm("x1", "2026-11-12T10:00:00+00:00"), _wm("x0", "2026-11-12T11:00:00+00:00"), "b"),  # newer updated
    (_wm("x9", "2026-11-12T11:00:00+00:00"), _wm("x0", "2026-11-12T10:00:00+00:00"), "a"),
    (_wm("x9"), _wm("x0", "2026-11-12T10:00:00+00:00"), "b"),                             # missing = oldest
    (_wm("2026-11-12T10:00", "2026-11-12T10:00:00+00:00"),
     _wm("2026-11-12T10:05", "2026-11-12T10:00:00+00:00"), "b"),                           # tie: larger value
], ids=["b-newer", "a-newer", "a-missing-updated", "tie-larger-value"])
def test_merge_watermark_driver_picks_newer(tmp_path, a, b, winner):
    from hub.merge_watermark import main
    paths = []
    for name, text in (("O", _wm("old")), ("A", a), ("B", b)):
        (tmp_path / name).write_text(text)
        paths.append(str(tmp_path / name))
    assert main(paths) == 0
    assert (tmp_path / "A").read_text() == (a if winner == "a" else b)


def test_merge_watermark_driver_rejects_garbage(tmp_path):
    from hub.merge_watermark import main
    for name, text in (("O", ""), ("A", "- a list\n"), ("B", _wm("x"))):
        (tmp_path / name).write_text(text)
    assert main([str(tmp_path / n) for n in "OAB"]) == 1


def test_watermark_write_stamps_updated_and_skips_unchanged(hub_root, monkeypatch):
    import yaml
    from hub.sync_state import load_state, save_state, set_watermark
    monkeypatch.setattr("hub.sync_state._now", lambda: "2026-11-12T10:00:00+00:00")
    set_watermark(hub_root, "clickup", "2026-11-12T09:00")
    p = hub_root / "Status" / "sync-state" / "watermark.clickup.yaml"
    assert yaml.safe_load(p.read_text()) == {"value": "2026-11-12T09:00", "updated": "2026-11-12T10:00:00+00:00"}
    monkeypatch.setattr("hub.sync_state._now", lambda: "2026-11-12T12:00:00+00:00")
    state = load_state(hub_root)
    state["next_meeting"]["group"] = date(2026, 11, 19)
    save_state(hub_root, state)  # clickup unchanged: its file (and `updated`) stays as it was
    assert yaml.safe_load(p.read_text())["updated"] == "2026-11-12T10:00:00+00:00"
    assert load_state(hub_root)["watermarks"] == {"clickup": "2026-11-12T09:00"}
    nm = yaml.safe_load((hub_root / "Status" / "sync-state" / "next_meeting.group.yaml").read_text())
    assert nm == {"value": date(2026, 11, 19), "updated": "2026-11-12T12:00:00+00:00"}


def test_ensure_repo_files_registers_watermark_driver(tmp_path):
    from hub.gitsync import ensure_repo_files
    git(tmp_path, "init", "-b", "main")
    ensure_repo_files(tmp_path)
    attrs = (tmp_path / ".gitattributes").read_text().splitlines()
    assert "Status/sync-state/watermark.*.yaml merge=hub-watermark" in attrs
    assert "Status/sync-state/next_meeting.*.yaml merge=hub-watermark" in attrs
    assert "hub.merge_watermark" in git(tmp_path, "config", "merge.hub-watermark.driver")


@pytest.fixture
def two_clones(tmp_path, hub_root):
    import shutil
    from hub.gitsync import ensure_repo_files
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    shutil.copytree(hub_root, seed)
    (seed / "hub.local.yaml").unlink()
    git(seed, "init", "-b", "main")
    git(seed, "config", "user.email", "t@example.com")
    git(seed, "config", "user.name", "T")
    ensure_repo_files(seed)
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "seed")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "-u", "origin", "main")
    clones = []
    for name in ("a", "b"):
        c = tmp_path / name
        git(tmp_path, "clone", str(bare), str(c))
        git(c, "config", "user.email", f"{name}@example.com")
        git(c, "config", "user.name", name)
        (c / "hub.local.yaml").write_text("me: alice\n")
        clones.append(c)
    return clones


@pytest.mark.parametrize("newer_pushes_last", [True, False])
def test_same_day_watermark_change_in_two_clones_merges(two_clones, monkeypatch, newer_pushes_last):
    from hub.gitsync import commit
    from hub.sync_state import load_state, set_watermark
    a, b = two_clones
    set_watermark(a, "clickup", "2026-11-11T09:00")  # yesterday's run, shared by both
    set_watermark(a, "git.CODE", "sha-old")
    commit(a, "yesterday")
    git(b, "pull", "--rebase")
    stamps = {"older": "2026-11-12T10:00:00+00:00", "newer": "2026-11-12T10:05:00+00:00"}
    first, last = (a, b) if newer_pushes_last else (b, a)
    monkeypatch.setattr("hub.sync_state._now", lambda: stamps["older"])
    set_watermark(first, "clickup", "2026-11-12T10:00")
    set_watermark(first, "git.CODE", "sha-first")
    monkeypatch.setattr("hub.sync_state._now", lambda: stamps["newer"])
    set_watermark(last, "clickup", "2026-11-12T10:05")
    set_watermark(last, "git.CODE", "sha-last")
    # Each clone commits its own run; the one that pushes second rebases onto the other's change.
    pusher_order = (first, last) if newer_pushes_last else (last, first)
    commit(pusher_order[0], "run 1")
    commit(pusher_order[1], "run 2")  # no SyncConflict
    git(pusher_order[0], "pull", "--rebase")
    for clone in (a, b):
        assert load_state(clone)["watermarks"] == {"clickup": "2026-11-12T10:05", "git.CODE": "sha-last"}
