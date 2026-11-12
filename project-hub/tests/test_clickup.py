# project-hub/tests/test_clickup.py
import json
import textwrap
from datetime import date, datetime

import pytest

from hub.clickup import ClickUpError, apply_clickup, load_tasks, plan_clickup
from hub.config import load_config
from hub.pending import load_proposals, reject_proposals
from hub.store import Store
from hub.sync_state import load_state, save_state
from hub.validate import validate

D = date(2026, 11, 19)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    (tmp_path / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice, clickup: "111", email: alice@x.edu}
          bob: {name: Bob, clickup: "222"}
        series:
          group: {attendees: [alice, bob]}
        sources:
          clickup: {list_id: "900", done_statuses: [complete, closed]}
        """))
    (tmp_path / "hub.local.yaml").write_text("me: alice\n")
    store, cfg = Store(tmp_path), load_config(tmp_path)

    def action(title, owner="alice", **fields):
        return store.create("action", title, owner=owner, author="alice", day=date(2026, 11, 1), source="m",
                            fields=fields)
    return store, cfg, tmp_path, action


def write_tasks(root, tasks):
    p = root / "tasks.json"
    p.write_text(json.dumps(tasks))
    return load_tasks(p)


def test_load_tasks_normalises(env):
    root = env[2]
    tasks = write_tasks(root, [{"id": "t1", "name": "Do", "status": {"status": "in progress"},
                                "assignees": [{"id": 111, "username": "Alice A", "email": "a@x"}, "bob"],
                                "due": "2026-11-30T00:00:00"}])
    assert tasks == [{"id": "t1", "name": "Do", "status": "in progress", "status_type": "",
                      "assignees": [["111", "Alice A", "a@x"], ["bob"]], "due": "2026-11-30"}]
    for bad in ({"name": "x"}, [{"id": "t", "name": "x", "status": "open", "due": "soon"}]):
        (root / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ClickUpError):
            load_tasks(root / "bad.json")


def test_load_tasks_rejects_missing_status_and_spaced_id(env):
    root = env[2]
    for bad in ([{"id": "t1", "name": "x"}],
                [{"id": "t1", "name": "x", "status": {"type": "closed"}}],
                [{"id": "t 1", "name": "x", "status": "open"}]):
        (root / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ClickUpError):
            load_tasks(root / "bad.json")
    (root / "bad.json").write_text(json.dumps([{"id": "t1", "name": "x"}]))
    with pytest.raises(ClickUpError, match="missing status"):
        load_tasks(root / "bad.json")


def test_mirror_status_owner_due(env):
    store, cfg, root, action = env
    a1 = action("Rerun", clickup_id="t1", due="2026-11-20")
    a2 = action("Plot", clickup_id="t2")
    tasks = write_tasks(root, [
        {"id": "t1", "name": "Rerun", "status": "complete", "assignees": [{"id": "222"}], "due": None},
        {"id": "t2", "name": "Plot", "status": "open", "assignees": [], "due": "2026-12-01"}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    # item ids carry a random suffix, so store order is arbitrary: compare order-independently
    assert sorted(plan["set"], key=lambda s: s["source"]) == [
        {"id": a1.id, "fields": {"status": "done", "closed": "2026-11-19", "owner": "bob", "due": ""},
         "entry": "Closed in ClickUp (status 'complete').", "source": "clickup:t1"},
        {"id": a2.id, "fields": {"due": "2026-12-01"}, "entry": None, "source": "clickup:t2"}]
    assert plan["flags"] == [] and plan["proposals"] == [] and plan["push"] == []
    rep = apply_clickup(root, store, plan, author="alice", day=D)
    assert sorted(rep["updated"]) == sorted([a1.id, a2.id])
    got = store.get(a1.id)
    assert got.status == "done" and got.owner == "bob" and got.meta["due"] is None
    assert store.get(a2.id).meta["due"] == date(2026, 12, 1)
    assert validate(store, cfg) == []


def test_flags_proposals_push(env):
    store, cfg, root, action = env
    gone = action("Gone", clickup_id="t9")
    done = action("Finished", clickup_id="t3")
    store.entry(done.id, "Done.", author="alice", source="m", day=D)
    store.set_fields(done.id, {"status": "done", "closed": "2026-11-19"})
    action("Mine", clickup_id="t4")
    local = action("Local only", owner="bob", due="2026-11-25")
    tasks = write_tasks(root, [
        {"id": "t3", "name": "Finished", "status": "open", "assignees": []},
        {"id": "t4", "name": "Mine", "status": "open", "assignees": [{"username": "zed"}]},
        {"id": "t5", "name": "New  from ClickUp", "status": "open", "assignees": [{"email": "ALICE@x.edu"}],
         "due": "2026-12-05"},
        {"id": "t6", "name": "Old closed", "status": "closed", "assignees": []}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert sorted(plan["flags"]) == sorted([
        f"clickup: task t9 for {gone.id} (Gone) is not in the export (deleted or moved?)",
        f"clickup: {done.id} is done in the hub but task t3 is 'open' in ClickUp",
        "clickup: assignee 'zed' of task t4 is not in project.yaml people (add their ClickUp id as `clickup:`)"])
    assert plan["proposals"] == [{
        "title": "New ClickUp task: New from ClickUp", "source": "clickup:t5",
        "reason": "Task created directly in ClickUp; link it (spawned_by) to the decision it serves, then apply.",
        "ops": [{"op": "new", "type": "action", "title": "New  from ClickUp", "owner": "alice",
                 "fields": {"clickup_id": "t5", "due": "2026-12-05"}, "source": "clickup:t5"}]}]
    assert plan["push"] == [{"id": local.id, "title": "Local only", "owner": "bob", "assignee": "222",
                             "due": "2026-11-25"}]
    state = load_state(root)
    state["flags"] = ["clickup: stale", "other: keep"]
    save_state(root, state)
    rep = apply_clickup(root, store, plan, author="alice", day=D)
    assert len(rep["staged"]) == 1 and rep["push"] == plan["push"]
    assert load_state(root)["flags"] == ["other: keep"] + plan["flags"]
    assert plan_clickup(store, cfg, tasks, today=D)["proposals"] == []   # already pending
    assert [p.title for p in load_proposals(root)] == ["New ClickUp task: New from ClickUp"]


def test_due_date_epoch_ms_sets_and_keeps(env):
    store, cfg, root, action = env
    ms = int(datetime(2026, 12, 5, 12).timestamp() * 1000)
    action("Keep", clickup_id="t1", due="2026-12-05")
    tasks = write_tasks(root, [{"id": "t1", "name": "Keep", "status": "open", "assignees": [], "due_date": ms}])
    assert tasks[0]["due"] == "2026-12-05"
    assert plan_clickup(store, cfg, tasks, today=D)["set"] == []   # same due: not cleared
    action("Move", clickup_id="t2")
    tasks = write_tasks(root, [{"id": "t2", "name": "Move", "status": "open", "assignees": [], "due_date": ms}])
    assert plan_clickup(store, cfg, tasks, today=D)["set"][0]["fields"] == {"due": "2026-12-05"}


def test_status_type_closed_counts_as_done(env):
    store, cfg, root, action = env
    action("Shipped", clickup_id="t1")
    tasks = write_tasks(root, [{"id": "t1", "name": "Shipped", "status": {"status": "shipped", "type": "closed"},
                                "assignees": []}])
    assert tasks[0]["status_type"] == "closed"
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert plan["set"][0]["fields"]["status"] == "done"
    assert plan["set"][0]["entry"] == "Closed in ClickUp (status 'shipped')."


def test_rejected_proposal_not_reproposed(env):
    store, cfg, root, action = env
    tasks = write_tasks(root, [{"id": "t5", "name": "Unwanted", "status": "open", "assignees": []}])
    rep = apply_clickup(root, store, plan_clickup(store, cfg, tasks, today=D), author="alice", day=D)
    assert len(rep["staged"]) == 1
    reject_proposals(root, store, rep["staged"], author="alice", day=D)
    assert load_proposals(root) == []
    assert plan_clickup(store, cfg, tasks, today=D)["proposals"] == []


def test_renamed_pending_task_not_restaged(env):
    store, cfg, root, action = env
    tasks = write_tasks(root, [{"id": "t5", "name": "First name", "status": "open", "assignees": []}])
    apply_clickup(root, store, plan_clickup(store, cfg, tasks, today=D), author="alice", day=D)
    tasks = write_tasks(root, [{"id": "t5", "name": "Renamed", "status": "open", "assignees": []}])
    assert plan_clickup(store, cfg, tasks, today=D)["proposals"] == []
    assert len(load_proposals(root)) == 1


def test_unmapped_new_task_assignee_flagged(env):
    store, cfg, root, action = env
    tasks = write_tasks(root, [{"id": "t7", "name": "Stranger task", "status": "open",
                                "assignees": [{"username": "zed"}]}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert plan["flags"] == ["clickup: assignee 'zed' of new task t7 is not in project.yaml people"]
    assert plan["proposals"][0]["ops"][0]["owner"] is None
    rep = apply_clickup(root, store, plan, author="alice", day=D)
    assert len(rep["staged"]) == 1


def test_second_run_after_apply_has_no_sets(env):
    store, cfg, root, action = env
    action("Rerun", clickup_id="t1", due="2026-11-20")
    action("Plot", clickup_id="t2")
    tasks = write_tasks(root, [
        {"id": "t1", "name": "Rerun", "status": "complete", "assignees": [{"id": "222"}], "due": None},
        {"id": "t2", "name": "Plot", "status": "open", "assignees": [], "due": "2026-12-01"}])
    apply_clickup(root, store, plan_clickup(store, cfg, tasks, today=D), author="alice", day=D)
    assert plan_clickup(store, cfg, tasks, today=D)["set"] == []


def test_apply_with_missing_target_writes_nothing(env):
    store, cfg, root, action = env
    real = action("Real", clickup_id="t1")
    tasks = write_tasks(root, [
        {"id": "t1", "name": "Real", "status": "complete", "assignees": []},
        {"id": "t8", "name": "Fresh", "status": "open", "assignees": []}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert plan["set"] and plan["proposals"]
    plan["set"].append({"id": "action-nope", "fields": {"due": ""}, "entry": None, "source": "clickup:x"})
    with pytest.raises(ClickUpError, match="does not exist"):
        apply_clickup(root, store, plan, author="alice", day=D)
    assert store.get(real.id).status == "open"
    assert load_proposals(root) == []


def test_retry_after_failed_set_fields_closes_once(env, monkeypatch):
    store, cfg, root, action = env
    a = action("Rerun", clickup_id="t1")
    tasks = write_tasks(root, [{"id": "t1", "name": "Rerun", "status": "complete", "assignees": []}])
    real = Store.set_fields
    calls = {"n": 0}

    def flaky(self, item_id, fields):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(self, item_id, fields)

    monkeypatch.setattr(Store, "set_fields", flaky)
    with pytest.raises(RuntimeError):
        apply_clickup(root, store, plan_clickup(store, cfg, tasks, today=D), author="alice", day=D)
    apply_clickup(root, store, plan_clickup(store, cfg, tasks, today=D), author="alice", day=D)
    got = store.get(a.id)
    assert got.status == "done"
    assert [e.text for e in got.entries].count("Closed in ClickUp (status 'complete').") == 1
