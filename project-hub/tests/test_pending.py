# project-hub/tests/test_pending.py
from datetime import date
import pytest
from hub.pending import (PendingError, add_proposal, apply_proposals, load_proposals, reject_proposals)
from hub.store import Store

D = date(2026, 11, 12)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def test_add_and_load(hub_root):
    pid = add_proposal(hub_root, "Close Q", [{"op": "set", "id": "Q-1", "fields": {"status": "answered"}}],
                       proposer="alice", source="circleback:9", reason="Answered in meeting.", day=D)
    props = load_proposals(hub_root)
    assert [p.pid for p in props] == [pid]
    assert props[0].title == "Close Q" and props[0].body["reason"] == "Answered in meeting."
    text = (hub_root / "Status" / "Pending" / f"{pid}.md").read_text()
    assert text.startswith(f"## {pid} · Close Q\n" + "`" * 3 + "yaml\n")


def test_apply_with_refs(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    ops = [
        {"op": "new", "ref": "$d", "type": "decision", "title": "Do X", "owner": "alice",
         "links": {"answers": [q.id]}, "fields": {"status": "made"}, "source": "agenda:group/2026-11-12"},
        {"op": "new", "type": "action", "title": "Run X", "owner": "bob",
         "links": {"spawned_by": ["$d"]}, "source": "agenda:group/2026-11-12"},
        {"op": "entry", "id": q.id, "text": "Decision taken.", "source": "agenda:group/2026-11-12"},
        {"op": "link", "id": "$d", "rel": "answers", "target": q.id},
    ]
    pid = add_proposal(hub_root, "Record decision", ops, proposer="alice", source="agenda", reason="r", day=D)
    created = apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert len(created) == 2
    d = store.get(created[0])
    assert d.status == "made" and d.links("answers") == [q.id]
    assert store.get(created[1]).links("spawned_by") == [d.id]
    assert store.get(q.id).entries[-1].text == "Decision taken."
    assert load_proposals(hub_root) == []
    log = (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert f"Applied {pid}: Record decision" in log


def test_reject_logs_and_removes(hub_root, store):
    pid = add_proposal(hub_root, "X", [], proposer="alice", source="s", reason="r", day=D)
    reject_proposals(hub_root, store, [pid], author="bob", day=D)
    assert load_proposals(hub_root) == []
    assert f"Rejected {pid}: X" in (hub_root / "Status" / "Log" / "2026-11.md").read_text()


def test_unknown_pid(hub_root, store):
    with pytest.raises(PendingError, match="P-zzzz"):
        apply_proposals(hub_root, store, ["P-zzzz"], author="bob", day=D)


def test_failed_apply_keeps_proposal(hub_root, store):
    pid = add_proposal(hub_root, "Bad", [{"op": "set", "id": "Q-20990101-zzz", "fields": {}}],
                       proposer="alice", source="s", reason="r", day=D)
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_entry_with_whitespace_source_is_pending_error(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    pid = add_proposal(hub_root, "Bad source",
                       [{"op": "entry", "id": q.id, "text": "t", "source": "meeting notes"}],
                       proposer="alice", source="s", reason="r", day=D)
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_non_mapping_op_is_pending_error(hub_root, store):
    pid = add_proposal(hub_root, "Oops", ["oops"], proposer="alice", source="s", reason="r", day=D)
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_bad_author_rejected_before_any_write(hub_root, store):
    pid = add_proposal(hub_root, "Create", [{"op": "new", "type": "question", "title": "Made", "owner": "alice"}],
                       proposer="alice", source="s", reason="r", day=D)
    before = store.ids()
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bad author", day=D)
    assert store.ids() == before
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_title_is_single_line(hub_root):
    pid = add_proposal(hub_root, "two\nlines", [], proposer="alice", source="s", reason="r", day=D)
    props = load_proposals(hub_root)
    assert [(p.pid, p.title) for p in props] == [(pid, "two lines")]
