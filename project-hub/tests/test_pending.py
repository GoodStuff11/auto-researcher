# project-hub/tests/test_pending.py
from datetime import date
import pytest
from hub.pending import (PendingError, add_proposal, apply_proposals, load_proposals, reject_proposals)
from hub.store import Store

D = date(2026, 11, 12)
NOOP = [{"op": "new", "type": "question", "title": "Placeholder"}]  # proposals need at least one op


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def test_add_and_load(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    pid = add_proposal(hub_root, "Close Q", [{"op": "set", "id": q.id, "fields": {"status": "answered"}}],
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
    pid = add_proposal(hub_root, "X", NOOP, proposer="alice", source="s", reason="r", day=D)
    reject_proposals(hub_root, store, [pid], author="bob", day=D)
    assert load_proposals(hub_root) == []
    assert f"Rejected {pid}: X" in (hub_root / "Status" / "Log" / "2026-11.md").read_text()


def test_unknown_pid(hub_root, store):
    with pytest.raises(PendingError, match="P-zzzz"):
        apply_proposals(hub_root, store, ["P-zzzz"], author="bob", day=D)


def _stage_raw(root, title, ops):
    """Write a proposal file directly, bypassing add_proposal's checks (as a hand edit would)."""
    from hub.pending import Proposal, _write
    pid = "P-raw" + str(len(load_proposals(root)))
    _write(root, Proposal(pid, title, {"proposer": "alice", "date": D, "source": "s", "reason": "r",
                                      "ops": ops}))
    return pid


def test_failed_apply_keeps_proposal(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    pid = add_proposal(hub_root, "Set", [{"op": "set", "id": q.id, "fields": {"status": "dropped"}}],
                       proposer="alice", source="s", reason="r", day=D)
    store.path(q.id).unlink()  # the target disappears between staging and applying
    with pytest.raises(PendingError, match="does not exist"):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_whitespace_source_rejected_at_add_and_apply(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    ops = [{"op": "entry", "id": q.id, "text": "t", "source": "meeting notes"}]
    with pytest.raises(PendingError, match="source"):
        add_proposal(hub_root, "Bad source", ops, proposer="alice", source="s", reason="r", day=D)
    pid = _stage_raw(hub_root, "Bad source", ops)
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_non_mapping_op_rejected_at_add_and_apply(hub_root, store):
    with pytest.raises(PendingError, match="mapping"):
        add_proposal(hub_root, "Oops", ["oops"], proposer="alice", source="s", reason="r", day=D)
    pid = _stage_raw(hub_root, "Oops", ["oops"])
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


@pytest.mark.parametrize("ops, match", [
    ([{"op": "new", "type": "action", "title": "Do", "links": {"spawned_by": ["D-20990101-zzz"]}}], "D-20990101-zzz"),
    ([{"op": "link", "id": "$d", "rel": "answers", "target": "Q-20990101-zzz"}], r"\$d"),
    ([{"op": "explode"}], "unknown op"),
    ([{"op": "entry", "text": "no id"}], "'id'"),
    ([{"op": "new", "type": "question", "title": "T", "fields": ["status"]}], "fields"),
    ([{"op": "new", "type": "question", "title": "T", "series": "group"}], "series"),
    ([{"op": "new", "type": "question", "title": "T", "links": {"spawned_by": "Q-1"}}], "links"),
    ([{"op": "new", "type": "question", "title": "T", "source": ""}], "source"),
])
def test_check_ops_rejects_at_add(hub_root, store, ops, match):
    with pytest.raises(PendingError, match=match):
        add_proposal(hub_root, "Bad", ops, proposer="alice", source="s", reason="r", day=D)
    assert load_proposals(hub_root) == []


def test_check_ops_lists_every_problem(hub_root, store):
    from hub.pending import check_ops
    errs = check_ops(store, [{"op": "entry", "id": "Q-20990101-zzz", "text": "x"}, {"op": "nope"},
                             {"op": "new", "ref": "$a", "type": "question", "title": "T"},
                             {"op": "link", "id": "$a", "rel": "answers", "target": "$b"}])
    assert len(errs) == 3 and any("$b" in e for e in errs)


def test_failed_apply_rolls_back_every_write(hub_root, store, monkeypatch):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    before_files = sorted(p.name for p in (hub_root / "Status" / "Items").iterdir())
    before_q = store.path(q.id).read_bytes()
    ops = [{"op": "new", "ref": "$d", "type": "decision", "title": "Do X", "links": {"answers": [q.id]}},
           {"op": "entry", "id": q.id, "text": "Decided.", "source": "m"},
           {"op": "set", "id": q.id, "fields": {"priority": 1}},
           {"op": "link", "id": "$d", "rel": "answers", "target": q.id}]
    pid = add_proposal(hub_root, "Record", ops, proposer="alice", source="s", reason="r", day=D)

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store, "link", boom)
    with pytest.raises(PendingError, match="disk on fire"):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert sorted(p.name for p in (hub_root / "Status" / "Items").iterdir()) == before_files
    assert store.path(q.id).read_bytes() == before_q
    assert [p.pid for p in load_proposals(hub_root)] == [pid]
    assert not (hub_root / "Status" / "Log").exists()


def test_new_ids_are_deterministic_per_proposal(hub_root, store):
    import shutil
    other = hub_root.parent / "other"
    shutil.copytree(hub_root, other)
    ops = [{"op": "new", "type": "question", "title": "One"}, {"op": "new", "type": "action", "title": "Two",
                                                                "owner": "bob"}]
    pid = add_proposal(hub_root, "Make two", ops, proposer="alice", source="s", reason="r", day=D)
    (other / "Status" / "Pending").mkdir(parents=True)
    shutil.copy(hub_root / "Status" / "Pending" / f"{pid}.md", other / "Status" / "Pending" / f"{pid}.md")
    made_here = apply_proposals(hub_root, store, [pid], author="alice", day=D)
    made_there = apply_proposals(other, Store(other), [pid], author="alice", day=D)
    assert made_here == made_there and len(set(made_here)) == 2


def test_unreadable_proposals_are_reported(hub_root):
    from hub.pending import load_unreadable
    good = add_proposal(hub_root, "Fine", NOOP, proposer="alice", source="s", reason="r", day=D)
    bad = hub_root / "Status" / "Pending" / "P-bad1.md"
    bad.write_text("## P-bad1 · Broken\n" + "`" * 3 + "yaml\nops: [unclosed\n" + "`" * 3 + "\n")
    junk = hub_root / "Status" / "Pending" / "P-junk.md"
    junk.write_text("just text\n")
    assert [p.pid for p in load_proposals(hub_root)] == [good]
    assert load_unreadable(hub_root) == [bad, junk]


def test_apply_reports_each_proposal(hub_root, store):
    p1 = add_proposal(hub_root, "One", [{"op": "new", "type": "question", "title": "A"}],
                      proposer="alice", source="s", reason="r", day=D)
    p2 = add_proposal(hub_root, "Two", [{"op": "new", "type": "question", "title": "B"}],
                      proposer="alice", source="s", reason="r", day=D)
    seen = []
    created = apply_proposals(hub_root, store, [p1, p2], author="alice", day=D,
                              on_applied=lambda pid, ids: seen.append((pid, list(ids))))
    assert [s[0] for s in seen] == [p1, p2] and [i for _, ids in seen for i in ids] == created


def test_bad_author_rejected_before_any_write(hub_root, store):
    pid = add_proposal(hub_root, "Create", [{"op": "new", "type": "question", "title": "Made", "owner": "alice"}],
                       proposer="alice", source="s", reason="r", day=D)
    before = store.ids()
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bad author", day=D)
    assert store.ids() == before
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_title_is_single_line(hub_root):
    pid = add_proposal(hub_root, "two\nlines", NOOP, proposer="alice", source="s", reason="r", day=D)
    props = load_proposals(hub_root)
    assert [(p.pid, p.title) for p in props] == [(pid, "two lines")]
