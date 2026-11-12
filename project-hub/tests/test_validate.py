# project-hub/tests/test_validate.py
from datetime import date
import pytest
from hub.config import load_config
from hub.store import Store
from hub.validate import validate

D = date(2026, 11, 12)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    return Store(hub_root), load_config(hub_root)


def rules(store, cfg):
    return sorted(v.rule for v in validate(store, cfg))


def mk(store, t, title, **kw):
    kw.setdefault("owner", "alice")
    return store.create(t, title, author="alice", day=D, source="manual", **kw)


def test_clean_store(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    d = mk(store, "decision", "D", links={"answers": [q.id]})
    mk(store, "action", "A", links={"spawned_by": [d.id]})
    store.set_fields(d.id, {"status": "made"})
    assert rules(store, cfg) == []


def test_closing_rules(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    store.set_fields(q.id, {"status": "answered"})
    assert rules(store, cfg) == ["closed-date"]
    store.set_fields(q.id, {"closed": "2026-11-20"})
    assert rules(store, cfg) == ["closing-entry"]
    store.entry(q.id, "Answered by D-x.", author="bob", source="manual", day=date(2026, 11, 20))
    assert rules(store, cfg) == []


def test_pivot_needs_target(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    store.set_fields(q.id, {"status": "pivoted", "closed": "2026-11-12"})
    assert rules(store, cfg) == ["pivot-target"]


def test_made_decision_needs_answers_and_actions(env):
    store, cfg = env
    d = mk(store, "decision", "D")
    store.set_fields(d.id, {"status": "made"})
    assert rules(store, cfg) == ["decision-actions", "decision-answers"]
    store.set_fields(d.id, {"no_action": "Pure framing choice."})
    assert rules(store, cfg) == ["decision-answers"]


def test_owner_and_author_rules(env):
    store, cfg = env
    a = mk(store, "action", "A", owner=None)
    assert rules(store, cfg) == ["action-owner"]
    store.set_fields(a.id, {"owner": "carol"})
    assert rules(store, cfg) == ["action-owner", "unknown-owner"]
    store.set_fields(a.id, {"owner": "bob"})
    store.entry(a.id, "x", author="mallory", source="manual", day=D)
    assert rules(store, cfg) == ["entry-author"]


def test_broken_link_and_status(env, hub_root):
    store, cfg = env
    q = mk(store, "question", "Q")
    p = hub_root / "Status" / "Items" / f"{q.id}.md"
    p.write_text(p.read_text().replace("pivoted_to: []", "pivoted_to:\n  - Q-20990101-zzz")
                              .replace("status: open", "status: weird"))
    assert rules(store, cfg) == ["link-target", "status"]
