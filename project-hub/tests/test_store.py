from datetime import date
import pytest
from hub.store import Store, StoreError

D = date(2026, 11, 12)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def mk(store, t, title, **kw):
    kw.setdefault("owner", "alice")
    return store.create(t, title, author="alice", day=D, source="manual", **kw)


def test_create_writes_file_with_raise_entry(store, hub_root):
    q = mk(store, "question", "Why?", series=["group"], description="Because.")
    path = hub_root / "Status" / "Items" / f"{q.id}.md"
    assert path.is_file()
    got = store.get(q.id)
    assert got.title == "Why?" and got.description == "Because."
    assert got.entries[0].text == "Raised." and got.entries[0].source == "manual"
    assert got.meta["sources"] == ["manual"]


def test_create_rejects_bad_type_and_link(store):
    with pytest.raises(StoreError):
        mk(store, "idea", "x")
    with pytest.raises(StoreError, match="unknown link"):
        mk(store, "action", "x", links={"blocks": ["Q-1"]})
    with pytest.raises(StoreError, match="does not exist"):
        mk(store, "action", "x", links={"spawned_by": ["D-20990101-zzz"]})


def test_entry_appends(store):
    q = mk(store, "question", "Why?")
    store.entry(q.id, "More.", author="bob", source="commit:x@1", day=date(2026, 11, 13))
    assert [e.text for e in store.get(q.id).entries] == ["Raised.", "More."]


def test_set_fields_validates_and_coerces(store):
    q = mk(store, "question", "Why?")
    store.set_fields(q.id, {"status": "answered", "closed": "2026-11-14", "priority": "1"})
    got = store.get(q.id)
    assert got.status == "answered" and got.meta["closed"] == date(2026, 11, 14) and got.meta["priority"] == 1
    with pytest.raises(StoreError, match="status"):
        store.set_fields(q.id, {"status": "done"})
    with pytest.raises(StoreError, match="cannot set"):
        store.set_fields(q.id, {"id": "X"})


def test_link_and_children(store):
    q = mk(store, "question", "Why?")
    d = mk(store, "decision", "Do X")
    store.link(d.id, "answers", q.id)
    a = mk(store, "action", "Run X", links={"spawned_by": [d.id]})
    store.link(d.id, "answers", q.id)  # idempotent
    assert store.get(d.id).links("answers") == [q.id]
    assert [c.id for c in store.children(q.id)] == [d.id]
    assert [c.id for c in store.children(d.id)] == [a.id]


def test_query_filters(store):
    q = mk(store, "question", "Q1", series=["group"])
    mk(store, "action", "A1", owner="bob")
    store.entry(q.id, "late", author="alice", source="manual", day=date(2026, 12, 1))
    assert [i.title for i in store.query(type="action")] == ["A1"]
    assert [i.title for i in store.query(owner="bob")] == ["A1"]
    assert [i.title for i in store.query(series="group")] == ["Q1"]
    assert [i.title for i in store.query(since=date(2026, 11, 20))] == ["Q1"]
    assert {i.title for i in store.query(status="open")} == {"Q1", "A1"}


def test_log_appends_monthly(store, hub_root):
    store.log("Run finished.", author="alice", source="commit:x@1", day=D)
    store.log("Draft pushed.", author="bob", source="overleaf:2", day=date(2026, 11, 20))
    text = (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert text == ("# Log 2026-11\n"
                    "- 2026-11-12 · alice · commit:x@1 — Run finished.\n"
                    "- 2026-11-20 · bob · overleaf:2 — Draft pushed.\n")


def test_coerce_errors_are_store_errors(store):
    q = mk(store, "question", "Why?")
    with pytest.raises(StoreError, match="closed"):
        store.set_fields(q.id, {"closed": "tomorrow"})
    with pytest.raises(StoreError, match="priority"):
        store.set_fields(q.id, {"priority": "high"})


def test_children_skips_missing_pivot_and_dedupes(store, hub_root):
    a = mk(store, "question", "A")
    b = mk(store, "decision", "B")
    c = mk(store, "decision", "C")
    store.link(b.id, "answers", a.id)
    # link() refuses missing targets, so write the pivot list directly
    item = store.get(a.id)
    item.meta["links"]["pivoted_to"] = [b.id, "Q-20990101-zzz", c.id]
    store.save(item)
    assert [x.id for x in store.children(a.id)] == [b.id, c.id]
    with pytest.raises(KeyError):
        store.children("Q-20990101-zzz")


def test_log_invalid_author_creates_nothing(store, hub_root):
    with pytest.raises(ValueError):
        store.log("x", author="", source="manual", day=D)
    assert not (hub_root / "Status" / "Log").exists()


def test_malformed_ids_never_read_outside_items(store, hub_root):
    (hub_root / "Status" / "Items").mkdir(parents=True)
    (hub_root / "x.md").write_text("---\nid: X\ntype: question\ntitle: t\nstatus: open\n---\n")
    with pytest.raises(KeyError):
        store.get("../../x")
    q = mk(store, "question", "Why?")
    with pytest.raises(StoreError, match="does not exist"):
        store.link(q.id, "answers", "../../x")


def test_field_shape_validation(store):
    q = mk(store, "question", "Why?")
    with pytest.raises(StoreError, match="series"):
        store.set_fields(q.id, {"series": "group"})
    with pytest.raises(StoreError, match="sources"):
        store.set_fields(q.id, {"sources": "manual"})
    with pytest.raises(StoreError, match="title"):
        store.set_fields(q.id, {"title": "  "})
    with pytest.raises(StoreError, match="series"):
        store.create("question", "x", owner="alice", author="alice", day=D, source="manual",
                     fields={"series": "group"})
    store.set_fields(q.id, {"owner": None})
    assert store.get(q.id).owner is None
