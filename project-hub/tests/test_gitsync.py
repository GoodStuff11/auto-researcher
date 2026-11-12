import shutil
import subprocess
from datetime import date
from pathlib import Path
import pytest
from hub.gitsync import SyncConflict, SyncError, commit, ensure_repo_files
from hub.pending import add_proposal, load_proposals
from hub.store import Store

D = date(2026, 11, 12)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def two_clones(tmp_path, hub_root):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    shutil.copytree(hub_root, seed)
    (seed / "hub.local.yaml").unlink()
    git(seed, "init", "-b", "main")
    for c in (seed,):
        git(c, "config", "user.email", "t@example.com")
        git(c, "config", "user.name", "T")
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


def test_ensure_repo_files_idempotent(tmp_path):
    git(tmp_path, "init", "-b", "main")
    ensure_repo_files(tmp_path)
    ensure_repo_files(tmp_path)
    attrs = (tmp_path / ".gitattributes").read_text().splitlines()
    assert attrs.count("Status/Log/*.md merge=union") == 1
    assert "hub.local.yaml" in (tmp_path / ".gitignore").read_text().splitlines()
    assert git(tmp_path, "config", "merge.ours.driver").strip() == "true"


def test_concurrent_items_and_logs_merge(two_clones):
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")
    Store(a).log("A ran.", author="alice", source="m", day=D)
    Store(b).create("question", "From B", owner="bob", author="bob", day=D, source="m")
    Store(b).log("B ran.", author="bob", source="m", day=D)
    commit(a, "a")
    commit(b, "b")  # must rebase over a's push, union-merge the log
    git(a, "pull", "--rebase")
    titles = {i.title for i in Store(a).all()}
    assert titles == {"From A", "From B"}
    log = (a / "Status" / "Log" / "2026-11.md").read_text()
    assert "A ran." in log and "B ran." in log and "<<<<<<<" not in log
    assert "hub.local.yaml" not in git(a, "ls-files")


def test_regenerate_runs_after_pull(two_clones):
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    Store(b).create("question", "From B", owner="bob", author="bob", day=D, source="m")
    seen = []

    def regen():
        seen.append(len(Store(b).all()))
        (b / "Status" / "Snapshot.md").write_text(f"{len(Store(b).all())} items\n")

    commit(b, "b", regenerate=regen)
    assert seen == [2]
    assert git(b, "status", "--porcelain").strip() == ""


def test_same_item_conflict_reported(two_clones):
    a, b = two_clones
    q = Store(a).create("question", "Shared", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    git(b, "pull", "--rebase")
    Store(a).set_fields(q.id, {"title": "From A"})
    Store(b).set_fields(q.id, {"title": "From B"})
    commit(a, "a2")
    with pytest.raises(SyncConflict) as exc:
        commit(b, "b2")
    assert f"Status/Items/{q.id}.md" in exc.value.files
    assert not (b / ".git" / "rebase-merge").exists()


def test_concurrent_proposals_do_not_conflict(two_clones):
    a, b = two_clones
    pa = add_proposal(a, "From A", [], proposer="alice", source="s", reason="r", day=D)
    commit(a, "a")
    pb = add_proposal(b, "From B", [], proposer="bob", source="s", reason="r", day=D)
    commit(b, "b")  # rebases over a's push; each proposal is its own file
    git(a, "pull", "--rebase")
    assert {p.pid for p in load_proposals(a)} == {pa, pb}


def test_no_upstream_is_sync_error_not_conflict(two_clones):
    a, _ = two_clones
    git(a, "checkout", "-b", "other")
    Store(a).create("question", "Local", owner="alice", author="alice", day=D, source="m")
    with pytest.raises(SyncError) as exc:
        commit(a, "a-other")
    assert not isinstance(exc.value, SyncConflict)
    assert "no upstream branch configured" in str(exc.value)


def test_push_to_missing_remote_reports_git_message(two_clones, tmp_path):
    a, _ = two_clones
    missing = tmp_path / "gone.git"
    git(a, "remote", "set-url", "origin", str(missing))
    Store(a).create("question", "Lost", owner="alice", author="alice", day=D, source="m")
    with pytest.raises(SyncError, match="does not appear to be a git repository"):
        commit(a, "a")
