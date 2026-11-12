import shutil
import subprocess
from datetime import date
from pathlib import Path
import pytest
from hub.gitsync import SyncConflict, SyncError, commit, ensure_repo_files
from hub.pending import add_proposal, load_proposals
from hub.store import Store

D = date(2026, 11, 12)
NOOP = [{"op": "new", "type": "question", "title": "Placeholder"}]  # proposals need at least one op


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
    msg = str(exc.value)
    assert f"hub-item conflict in {q.id}: title:" in msg and "From A" in msg and "From B" in msg
    assert "git rebase --continue" in msg and "hub commit" in msg


def test_concurrent_proposals_do_not_conflict(two_clones):
    a, b = two_clones
    pa = add_proposal(a, "From A", NOOP, proposer="alice", source="s", reason="r", day=D)
    commit(a, "a")
    pb = add_proposal(b, "From B", NOOP, proposer="bob", source="s", reason="r", day=D)
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


def test_ensure_repo_files_registers_item_driver_and_ignores(tmp_path):
    git(tmp_path, "init", "-b", "main")
    ensure_repo_files(tmp_path)
    attrs = (tmp_path / ".gitattributes").read_text().splitlines()
    for line in ("Status/Items/*.md merge=hub-item", "*.md text eol=lf", "*.yaml text eol=lf"):
        assert line in attrs
    assert "hub.merge_item %O %A %B" in git(tmp_path, "config", "merge.hub-item.driver")
    assert git(tmp_path, "config", "merge.hub-item.name").strip() == "hub item merge"
    ignore = (tmp_path / ".gitignore").read_text().splitlines()
    assert "~$*" in ignore and ".~lock.*#" in ignore


def test_office_lock_files_not_committed(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "T")
    (tmp_path / "Agenda").mkdir()
    (tmp_path / "Agenda" / "~$group.xlsx").write_text("lock")
    (tmp_path / "Agenda" / ".~lock.group.xlsx#").write_text("lock")
    (tmp_path / "Agenda" / "group.xlsx").write_text("x")
    commit(tmp_path, "c")
    files = git(tmp_path, "ls-files").split()
    assert "Agenda/group.xlsx" in files and not any("lock" in f or "~$" in f for f in files)


def test_concurrent_entries_on_same_item_merge(two_clones):
    a, b = two_clones
    q = Store(a).create("question", "Shared", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    git(b, "pull", "--rebase")
    Store(a).entry(q.id, "Entry from A.", author="alice", source="circleback:1", day=D)
    Store(b).entry(q.id, "Entry from B.", author="bob", source="clickup:2", day=D)
    commit(a, "a2")
    commit(b, "b2")  # the hub-item driver merges both entries
    git(a, "pull", "--rebase")
    for clone in (a, b):
        item = Store(clone).get(q.id)
        assert [e.text for e in item.entries] == ["Raised.", "Entry from A.", "Entry from B."]
        assert item.meta["sources"] == ["m", "circleback:1", "clickup:2"]


def test_regenerate_failure_is_warning_and_still_pushes(two_clones, capsys):
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")

    def regen():
        raise RuntimeError("snapshot broke")

    commit(a, "a", regenerate=regen)
    assert "warning: could not regenerate generated files: snapshot broke" in capsys.readouterr().err
    git(b, "pull", "--rebase")
    assert [i.title for i in Store(b).all()] == ["From A"]


def test_pull_if_possible(two_clones, tmp_path):
    from hub.gitsync import pull_if_possible
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    assert pull_if_possible(b) is None
    assert [i.title for i in Store(b).all()] == ["From A"]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert pull_if_possible(plain) is None  # not a git repo: nothing to do
    git(b, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    warning = pull_if_possible(b)
    assert warning and warning.startswith("warning: could not pull")


def test_concurrent_watermarks_and_flags_merge(two_clones):
    from hub.sync_state import load_state, save_state, set_watermark
    a, b = two_clones
    set_watermark(a, "clickup", "2026-11-12T10:00")
    state = load_state(a)
    state["flags"] = ["flag from A"]
    save_state(a, state)
    set_watermark(b, "circleback", "2026-11-12T11:00")
    state = load_state(b)
    state["flags"] = ["flag from B"]
    save_state(b, state)
    commit(a, "a")
    commit(b, "b")  # different watermark files; flags.md union-merges
    git(a, "pull", "--rebase")
    for clone in (a, b):
        st = load_state(clone)
        assert st["watermarks"] == {"circleback": "2026-11-12T11:00", "clickup": "2026-11-12T10:00"}
        assert sorted(st["flags"]) == ["flag from A", "flag from B"]


def test_same_proposal_applied_in_two_clones_creates_one_item(two_clones):
    from hub.pending import apply_proposals
    a, b = two_clones
    q = Store(a).create("question", "Q", owner="alice", author="alice", day=D, source="m")
    pid = add_proposal(a, "Decide", [{"op": "new", "type": "decision", "title": "Do X",
                                      "links": {"answers": [q.id]}}],
                       proposer="alice", source="s", reason="r", day=D)
    commit(a, "stage")
    git(b, "pull", "--rebase")
    made_a = apply_proposals(a, Store(a), [pid], author="alice", day=D)
    made_b = apply_proposals(b, Store(b), [pid], author="bob", day=D)
    assert made_a == made_b
    commit(a, "apply in a")
    commit(b, "apply in b")  # add/add of the same item file: merged by the hub-item driver
    git(a, "pull", "--rebase")
    for clone in (a, b):
        decisions = Store(clone).query(type="decision")
        assert [d.id for d in decisions] == made_a
        assert sorted(e.author for e in decisions[0].entries) == ["alice", "bob"]


def test_cli_commit_with_garbage_item_still_pushes(two_clones):
    from hub.cli import main
    a, b = two_clones
    Store(a).create("question", "Good", owner="alice", author="alice", day=D, source="m")
    (a / "Status" / "Items" / "Q-20261112-bad.md").write_text("garbage\n")
    assert main(["--root", str(a), "commit", "with garbage"]) == 0
    git(b, "pull", "--rebase")
    assert (b / "Status" / "Items" / "Q-20261112-bad.md").is_file()
    assert "Q-20261112-bad.md" in (b / "Status" / "Snapshot.md").read_text()
