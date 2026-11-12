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
            {"op": "entry", "id": q.id, "text": "Discussed;  HVA plateaus at L=12.", "source": "circleback:m1"},
            {"op": "log", "text": "Group meeting: framing settled.", "source": "circleback:m1"},
            {"op": "new", "type": "action", "title": "Rerun HVA", "owner": "bob", "source": "circleback:m1",
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
    assert store.get(q.id).entries[-1].text == "Discussed; HVA plateaus at L=12."
    assert [a.title for a in store.query(type="action")] == ["Rerun HVA"]
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
    (lambda b, q: b["safe"].append({"op": "new", "type": "decision", "title": "X", "source": "s"}), "decisions are proposals"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"status": "done"}}), "without status"),
    (lambda b, q: b["safe"].append({"op": "entry", "id": "Q-20990101-zzz", "text": "x", "source": "s"}), "does not exist"),
    (lambda b, q: b["safe"].append({"op": "log", "text": "x", "source": "has space"}), "no whitespace"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "links": {"spawned_by": ["$d"]}}), "does not exist"),
    (lambda b, q: b["proposals"].append({"title": "T", "reason": "r", "source": "s",
                                         "ops": [{"op": "set", "id": "Q-20990101-zzz", "fields": {}}]}), "does not exist"),
    (lambda b, q: b.update(extra=1), "unknown top-level key"),
    (lambda b, q: b.update(safe={"op": "log"}), "safe must be a list"),
    (lambda b, q: b.update(watermark={"name": "circleback"}), "watermark must be"),
])
def test_invalid_batches_write_nothing(hub_root, store, mutate, needle):
    q, batch = base(store)
    mutate(batch, q)
    assert any(needle in e for e in check_batch(store, batch))
    with pytest.raises(BatchError, match="nothing written"):
        run_batch(hub_root, store, batch, author="alice", day=D)
    assert len(store.get(q.id).entries) == 1 and load_proposals(hub_root) == []
    assert not (hub_root / "Status" / "Log").exists()
