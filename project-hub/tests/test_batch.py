from datetime import date

import pytest

from hub.batch import BatchError, check_batch, run_batch
from hub.pending import apply_proposals, load_proposals
from hub.store import Store
from hub.sync_state import get_watermark, load_state

D = date(2026, 11, 19)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def base(store):
    q = store.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m")
    return q, {
        "watermark": {"name": "circleback", "value": "2026-11-19"},
        "safe": [
            {"op": "entry", "id": q.id, "text": "Discussed;  Method A plateaus at size 12.", "source": "circleback:m1"},
            {"op": "log", "text": "Group meeting: framing settled.", "source": "circleback:m1"},
            {"op": "new", "type": "action", "title": "Rerun baseline", "owner": "bob", "source": "circleback:m1",
             "series": ["group"]},
        ],
        "proposals": [{"title": "Close: Why?", "reason": "Answered in meeting.", "source": "circleback:m1",
                       "ops": [{"op": "set", "id": q.id, "fields": {"status": "answered", "closed": "2026-11-19"}},
                               {"op": "entry", "id": q.id, "text": "Answered.", "source": "circleback:m1"}]}],
        "flags": ["circleback: meeting m2 has no notes yet"],
    }


def test_run_batch_applies_safe_stages_proposals(hub_root, store):
    q, batch = base(store)
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["applied"] == 3 and rep["skipped"] == 0 and len(rep["staged"]) == 1
    assert rep["flags_added"] == 1 and rep["watermark"] == "circleback=2026-11-19"
    assert store.get(q.id).entries[-1].text == "Discussed; Method A plateaus at size 12."
    assert [a.title for a in store.query(type="action")] == ["Rerun baseline"]
    assert "Group meeting: framing settled." in (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert [p.title for p in load_proposals(hub_root)] == ["Close: Why?"]
    assert get_watermark(hub_root, "circleback") == "2026-11-19"
    assert load_state(hub_root)["flags"] == ["circleback: meeting m2 has no notes yet"]


def test_rerun_is_a_no_op(hub_root, store):
    q, batch = base(store)
    run_batch(hub_root, store, batch, author="alice", day=D)
    rep = run_batch(hub_root, store, batch, author="bob", day=D)
    assert rep["applied"] == 0 and rep["skipped"] == 3 and rep["staged"] == [] and rep["skipped_proposals"] == 1
    assert rep["flags_added"] == 0
    assert len(store.get(q.id).entries) == 2 and len(store.query(type="action")) == 1


def test_applied_proposal_is_not_staged_again(hub_root, store):
    q, batch = base(store)
    pid = run_batch(hub_root, store, batch, author="alice", day=D)["staged"][0]
    apply_proposals(hub_root, store, [pid], author="alice", day=D)
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["staged"] == [] and rep["skipped_proposals"] == 1


@pytest.mark.parametrize("mutate,needle", [
    # judgement calls and structure
    (lambda b, q: b["safe"].append({"op": "new", "type": "decision", "title": "X", "source": "s"}), "decisions are proposals"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"status": "done"}}), "fields may only set due"),
    (lambda b, q: b["safe"].append({"op": "entry", "id": "Q-20990101-zzz", "text": "x", "source": "s"}), "does not exist"),
    (lambda b, q: b["safe"].append({"op": "log", "text": "x", "source": "has space"}), "no whitespace"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "links": {"spawned_by": ["$d"]}}), "does not exist"),
    (lambda b, q: b["proposals"].append({"title": "T", "reason": "r", "source": "s",
                                         "ops": [{"op": "set", "id": "Q-20990101-zzz", "fields": {}}]}), "does not exist"),
    (lambda b, q: b["proposals"].append({"title": "T", "reason": "r", "source": "s",
                                         "ops": [{"op": ["set"], "id": q.id, "fields": {}}]}), "op name"),
    (lambda b, q: b.update(extra=1), "unknown top-level key"),
    (lambda b, q: b.update(safe={"op": "log"}), "safe must be a list"),
    (lambda b, q: b.update(watermark={"name": "circleback"}), "watermark must be"),
    (lambda b, q: b.update(watermark={"name": "../x", "value": "1"}), "watermark name"),
    # types
    (lambda b, q: b["safe"].append({"op": "entry", "id": q.id, "text": date(2026, 11, 1), "source": "s"}),
     "text must be a non-empty string"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": 42, "source": "s"}),
     "title must be a non-empty string"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s", "owner": ["alice"]}),
     "owner must be a string"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s", "priority": "high"}),
     "priority must be an integer"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s", "priority": 7}),
     "priority must be an integer"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s", "series": [1]}),
     "series must be a list of strings"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s", "description": 3}),
     "description must be a string"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"due": "tomorrow"}}), "fields may only set due"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"clickup_id": "x"}}), "fields may only set due"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"sources": ["a"]}}), "fields may only set due"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "links": {"answers": [q.id]}}), "only spawned_by links are safe"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "links": {"blocks": ["Q-20990101-zzz"]}}), "only spawned_by links are safe"),
    (lambda b, q: b["safe"].append({"op": ["entry"], "id": q.id, "text": "x", "source": "s"}), "op must be one of"),
    (lambda b, q: b["safe"].append({"op": "entry", "id": [q.id], "text": "x", "source": "s"}), "id must be a string"),
    (lambda b, q: b["safe"].append({"op": "entry", "id": q.id, "text": "x", "source": "s", "status": "answered"}),
     "unknown key(s) status"),
])
def test_invalid_batches_write_nothing(hub_root, store, mutate, needle):
    q, batch = base(store)
    mutate(batch, q)
    assert any(needle in e for e in check_batch(store, batch))
    with pytest.raises(BatchError, match="nothing written"):
        run_batch(hub_root, store, batch, author="alice", day=D)
    assert len(store.get(q.id).entries) == 1
    assert store.query(type="action") == []
    assert load_proposals(hub_root) == []
    assert not (hub_root / "Status" / "Log").exists()


def test_owner_must_be_a_known_person(hub_root, store):
    q, batch = base(store)
    batch["safe"].append({"op": "new", "type": "action", "title": "Y", "source": "s", "owner": "mallory"})
    people = {"alice": {}, "bob": {}}
    assert any("mallory" in e for e in check_batch(store, batch, people=people))
    with pytest.raises(BatchError, match="nothing written"):
        run_batch(hub_root, store, batch, author="alice", day=D, people=people)
    assert len(store.get(q.id).entries) == 1
    assert store.query(type="action") == []
    assert not (hub_root / "Status" / "Log").exists()


def test_log_with_same_source_but_different_text_is_not_a_duplicate(hub_root, store):
    store.log("framing settled and more", author="alice", source="circleback:m1", day=date(2026, 11, 1))
    batch = {"safe": [{"op": "log", "text": "framing settled", "source": "circleback:m1"}]}
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["applied"] == 1 and rep["skipped"] == 0


def test_new_item_title_duplicate_ignores_case_and_spacing(hub_root, store):
    store.create("action", "Rerun baseline", owner="bob", author="alice", day=date(2026, 11, 1), source="circleback:m1")
    batch = {"safe": [{"op": "new", "type": "action", "title": "rerun baseline ", "source": "circleback:m1"}]}
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["applied"] == 0 and rep["skipped"] == 1
    assert len(store.query(type="action")) == 1


def test_proposal_title_duplicate_ignores_case(hub_root, store):
    q, batch = base(store)
    run_batch(hub_root, store, batch, author="alice", day=D)
    dup = {"proposals": [{"title": "close: why?", "reason": "r", "source": "circleback:m1",
                          "ops": [{"op": "entry", "id": q.id, "text": "x", "source": "s"}]}]}
    rep = run_batch(hub_root, store, dup, author="alice", day=D)
    assert rep["staged"] == [] and rep["skipped_proposals"] == 1


def test_safe_new_may_set_due_and_spawned_by(hub_root, store):
    q, _ = base(store)
    batch = {"safe": [{"op": "new", "type": "action", "title": "Due thing", "source": "c:1",
                       "links": {"spawned_by": [q.id]}, "fields": {"due": "2026-11-30"}}]}
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["applied"] == 1
    action = store.query(type="action")[0]
    assert action.meta["due"] == date(2026, 11, 30) and action.meta["links"]["spawned_by"] == [q.id]


def test_proposal_new_owner_must_be_a_known_person(hub_root, store):
    q, batch = base(store)
    batch["proposals"].append({"title": "Act", "reason": "r", "source": "s", "ops": [
        {"op": "new", "type": "action", "title": "Z", "owner": "mallory", "source": "s"}]})
    people = {"alice": {}, "bob": {}}
    assert any("mallory" in e and "proposal 2" in e for e in check_batch(store, batch, people=people))
    with pytest.raises(BatchError, match="nothing written"):
        run_batch(hub_root, store, batch, author="alice", day=D, people=people)
    assert load_proposals(hub_root) == []
