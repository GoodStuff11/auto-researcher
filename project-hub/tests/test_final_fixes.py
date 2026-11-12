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


# ---------------------------------------------------------------- I2: concurrent runs must not duplicate

def _action(store, title, cid=None):
    a = store.create("action", title, owner="bob", author="alice", day=D, source="manual")
    if cid:
        store.set_fields(a.id, {"clickup_id": cid})
    return a.id


def test_plan_clickup_flags_shared_task_and_validate_reports_it(hub_root):
    from hub.clickup import plan_clickup
    from hub.validate import validate
    store, cfg = Store(hub_root), load_config(hub_root)
    ids = sorted(_action(store, t, "t2") for t in ("Make fig", "Make fig", "Make fig again"))
    _action(store, "Other", "t9")
    tasks = [{"id": t, "name": n, "status": "to do", "status_type": "", "assignees": [], "due": None}
             for t, n in (("t2", "Make fig"), ("t9", "Other"))]
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert f"clickup: actions {', '.join(ids)} share task t2" in plan["flags"]
    assert not any("t9" in f for f in plan["flags"])
    dup = [v for v in validate(store, cfg) if v.rule == "duplicate-clickup-id"]
    assert sorted(v.item_id for v in dup) == ids[1:]  # one violation per extra action
    assert all(f"also on {ids[0]}" in v.message for v in dup)


def test_check_ops_rejects_new_action_for_an_already_linked_task(hub_root):
    from hub.pending import PendingError, add_proposal, apply_proposals, check_ops
    store = Store(hub_root)
    linked = _action(store, "Make fig", "t2")
    op = {"op": "new", "type": "action", "title": "Make fig", "fields": {"clickup_id": "t2"}}
    assert check_ops(store, [op]) == [f"op 1 (new): ClickUp task t2 already linked to {linked}"]
    # Two proposals for one new task (two clones staged it the same day): the second fails at apply time.
    op3 = dict(op, fields={"clickup_id": "t3"})
    p1 = add_proposal(hub_root, "New ClickUp task: X", [op3], proposer="alice", source="clickup:t3", reason="r",
                      day=D)
    p2 = add_proposal(hub_root, "New ClickUp task: X (2)", [op3], proposer="bob", source="clickup:t3", reason="r",
                      day=D)
    with pytest.raises(PendingError, match=rf"{p2} cannot be applied:\n  op 1 \(new\): ClickUp task t3 already "
                                           rf"linked to A-\S+ \(already applied: {p1}\)"):
        apply_proposals(hub_root, store, [p1, p2], author="alice", day=D)
    assert len([a for a in store.query(type="action") if a.meta.get("clickup_id") == "t3"]) == 1


# ---------------------------------------------------------------- minors: check_ops dry-runs values

@pytest.mark.parametrize("ops, expected", [
    ([], ["proposal has no ops"]),
    ([{"op": "new", "type": "decision", "title": "D", "fields": {"status": "answered"}}],
     ["op 1 (new): status 'answered' invalid for decision; expected one of ('proposed', 'made', 'rejected', "
      "'deferred')"]),
    ([{"op": "new", "type": "action", "title": "A", "fields": {"due": "Friday"}}],
     ["op 1 (new): invalid value for 'due': 'Friday'"]),
    ([{"op": "new", "type": "action", "ref": "$a", "title": "A"},
      {"op": "set", "id": "$a", "fields": {"status": "answered", "priority": "high"}}],
     ["op 2 (set): status 'answered' invalid for action; expected one of ('open', 'done', 'dropped')",
      "op 2 (set): invalid value for 'priority': 'high'"]),
], ids=["no-ops", "new-bad-status", "new-bad-date", "set-on-ref"])
def test_check_ops_dry_runs_values(hub_root, ops, expected):
    from hub.pending import check_ops
    assert sorted(check_ops(Store(hub_root), ops)) == sorted(expected)


def test_check_ops_set_on_existing_item_checks_status_and_dates(hub_root):
    from hub.pending import check_ops
    store = Store(hub_root)
    a = _action(store, "Act")
    errs = check_ops(store, [{"op": "set", "id": a, "fields": {"status": "done", "closed": "Nov 12"}},
                             {"op": "set", "id": a, "fields": {"status": "answered"}},
                             {"op": "set", "id": a, "fields": {"status": "done", "closed": "2026-11-12"}}])
    assert errs == ["op 1 (set): invalid value for 'closed': 'Nov 12'",
                    "op 2 (set): status 'answered' invalid for action; expected one of ('open', 'done', 'dropped')"]


def test_batch_rejects_proposals_with_bad_values(hub_root):
    from hub.batch import BatchError, run_batch
    store, cfg = Store(hub_root), load_config(hub_root)
    batch = {"proposals": [{"title": "Bad", "reason": "r", "source": "x:1",
                            "ops": [{"op": "new", "type": "decision", "title": "D",
                                     "fields": {"status": "answered"}}]},
                           {"title": "Empty", "reason": "r", "source": "x:1", "ops": []}]}
    with pytest.raises(BatchError) as exc:
        run_batch(hub_root, store, batch, author="alice", day=D, people=cfg.people)
    assert "status 'answered' invalid for decision" in str(exc.value)
    assert "proposal has no ops" in str(exc.value)
    assert not (hub_root / "Status" / "Pending").exists()


def test_store_create_collapses_title_whitespace(hub_root):
    store = Store(hub_root)
    q = store.create("question", "  Line one\n  line two\t end ", owner="alice", author="alice", day=D, source="m")
    assert store.get(q.id).title == "Line one line two end"


# ---------------------------------------------------------------- I3: meeting decided says reached / staged

def test_meeting_decided_reports_not_reached_and_staged(hub_root, monkeypatch, capsys):
    import json
    from openpyxl import load_workbook
    from hub.agenda.build import SHEET_D
    from hub.agenda.draft import save_draft, scaffold
    from hub.agenda.lifecycle import build_agenda, read_agenda
    from hub.cli import main
    from hub.pending import load_proposals
    store, cfg = Store(hub_root), load_config(hub_root)
    q = store.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m",
                     series=["group"])
    d1, d2 = (store.create("decision", t, owner="alice", author="alice", day=date(2026, 11, 2), source="m",
                           series=["group"], links={"answers": [q.id]}).id for t in ("Decide one", "Decide two"))
    draft = scaffold(store, hub_root, "group", date(2026, 11, 19))
    draft["summary"] = "S."
    save_draft(hub_root, draft)
    path = build_agenda(hub_root, "group")
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    rows = {ws.cell(r, 1).value: r for r in range(1, 40)}
    ws[f"H{rows[d1]}"], ws[f"I{rows[d1]}"] = "approve", "Do it"
    new_row = next(r for v, r in rows.items() if str(v or "").startswith("NEW"))
    ws[f"B{new_row}"], ws[f"H{new_row}"] = "Drop UCJ", "approve"
    wb.save(path)
    res = read_agenda(store, cfg, "group", author="alice", today=date(2026, 11, 19))
    assert res["not_reached"] == [d2] and len(res["staged"]) == 1
    staged = load_proposals(hub_root)[0]
    assert staged.body["reason"] == "Decision raised in meeting; after applying, link it to the question(s) it answers."
    monkeypatch.chdir(hub_root)
    assert main(["meeting", "decided", "group", "2026-11-19", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["id"]: r["reached"] for r in out["decisions"]} == {d1: True, d2: False}
    assert out["staged"] == res["staged"]
    assert main(["meeting", "decided", "group", "2026-11-19"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(ln.startswith(d1) and "— Do it" in ln for ln in lines)
    assert any(ln.startswith(d2) and "— not reached" in ln and "as proposed" not in ln for ln in lines)
    assert lines[-1] == f"staged: {res['staged'][0]}"


# ---------------------------------------------------------------- I6 + `--to`: repository entries

@pytest.fixture
def repo_vault(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    repo = tmp_path / "code"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Tester")
    for i in range(3):
        (repo / f"f{i}.txt").write_text(str(i))
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", f"c{i}")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")

    def configure(entry: str):
        (vault / "project.yaml").write_text(
            "name: demo\npeople:\n  alice: {name: Alice}\nseries:\n  group: {attendees: [alice]}\n"
            f"roots:\n  CODE: /nowhere\nsources:\n  repos:\n    - {entry}\n")
        return load_config(vault)
    return configure, repo, vault


@pytest.mark.parametrize("entry, branch", [
    ("{root: CODE, branch: main}", "main"),
    ("{root: CODE, branches: [main, dev]}", "main"),
    ("{root: CODE, branch: main, remote: origin}", "main"),
    ("{root: CODE}", "HEAD"),
])
def test_repo_entry_branch_or_branches(repo_vault, entry, branch):
    from hub.ingest_git import ingest_git
    configure, _, _ = repo_vault
    res = ingest_git(configure(entry), "CODE", today=date(2026, 11, 12), since=date(2000, 1, 1))
    assert res["branch"] == branch and len(res["commits"]) == 3


@pytest.mark.parametrize("entry, message", [
    ("{root: CODE, branchs: main}", "unknown key(s) branchs"),
    ("{root: CODE, branches: []}", "`branches` must be a non-empty list"),
    ("{root: CODE, branches: main}", "`branches` must be a non-empty list"),
    ("{root: CODE, branch: main, branches: [main]}", "either `branch` or `branches`"),
])
def test_repo_entry_errors(repo_vault, entry, message):
    from hub.ingest_git import IngestError, ingest_git
    configure, _, _ = repo_vault
    with pytest.raises(IngestError, match=re_escape(message)):
        ingest_git(configure(entry), "CODE", today=date(2026, 11, 12))


def re_escape(text):
    import re
    return re.escape(text)


def test_advance_to_stops_at_the_recorded_commit(repo_vault):
    from hub.ingest_git import IngestError, ingest_git
    from hub.sync_state import get_watermark
    configure, repo, vault = repo_vault
    cfg = configure("{root: CODE, branch: main}")
    today = date(2026, 11, 12)
    listed = ingest_git(cfg, "CODE", today=today)["commits"]
    recorded = listed[-1]["sha"]
    (repo / "late.txt").write_text("late")  # a commit lands after the agent recorded its batch
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "late commit")
    with pytest.raises(IngestError, match="--to needs --advance"):
        ingest_git(cfg, "CODE", today=today, to=recorded)
    with pytest.raises(IngestError, match="not one of the 4 commit"):
        ingest_git(cfg, "CODE", today=today, advance=True, to="0000000")
    with pytest.raises(IngestError, match="not one of"):
        ingest_git(cfg, "CODE", today=today, advance=True, to=recorded[:6])  # too short
    res = ingest_git(cfg, "CODE", today=today, advance=True, to=recorded[:7])
    assert res["advanced"] and get_watermark(vault, "git.CODE") == recorded
    after = ingest_git(cfg, "CODE", today=today)["commits"]
    assert [c["subject"] for c in after] == ["late commit"]  # not skipped


def test_cli_advance_to(repo_vault, monkeypatch, capsys):
    import json
    from hub.cli import main
    configure, repo, vault = repo_vault
    configure("{root: CODE, branch: main}")
    monkeypatch.chdir(vault)
    monkeypatch.setenv("HUB_TODAY", "2026-11-12")
    assert main(["ingest", "git", "CODE"]) == 0
    first = json.loads(capsys.readouterr().out)["commits"][0]["sha"]
    assert main(["ingest", "git", "CODE", "--advance", "--to", first]) == 0
    assert json.loads(capsys.readouterr().out)["advanced"] is True
    assert main(["ingest", "git", "CODE"]) == 0
    assert len(json.loads(capsys.readouterr().out)["commits"]) == 2


# ---------------------------------------------------------------- minors: load_tasks, validate_draft

@pytest.mark.parametrize("assignees", ["222", 222, {"id": 222}])
def test_load_tasks_assignees_must_be_a_list(tmp_path, assignees):
    import json
    from hub.clickup import ClickUpError, load_tasks
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"id": "t1", "name": "Act", "status": "open", "assignees": assignees}]))
    with pytest.raises(ClickUpError, match="task t1: assignees must be a list"):
        load_tasks(p)
    p.write_text(json.dumps([{"id": "t1", "name": "Act", "status": "open", "assignees": None}]))
    assert load_tasks(p)[0]["assignees"] == []


def test_validate_draft_weigh_in_must_be_a_string():
    from hub.agenda.draft import validate_draft
    draft = {"series": "group", "meeting_date": date(2026, 11, 19), "previous_meeting": None, "summary": "S.",
             "since_last": [], "questions": [],
             "decisions": [{"decide": "X", "minutes": 10, "weigh_in": ["alice", "bob"]}]}
    assert validate_draft(draft) == ["decisions[0]: weigh_in must be one string of comma-separated people keys "
                                     "(e.g. \"alice, bob\"), got ['alice', 'bob']"]
    draft["decisions"][0]["weigh_in"] = "alice, bob"
    assert validate_draft(draft) == []
