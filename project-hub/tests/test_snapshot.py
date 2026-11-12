# project-hub/tests/test_snapshot.py
from datetime import date
from hub.config import load_config
from hub.snapshot import last_meeting, render_snapshot, write_snapshot
from hub.store import Store
from hub.sync_state import get_watermark, load_state, save_state, set_watermark

D = date(2026, 11, 12)


def test_watermarks_round_trip(hub_root):
    assert get_watermark(hub_root, "clickup") is None
    set_watermark(hub_root, "clickup", "2026-11-12T10:00")
    assert get_watermark(hub_root, "clickup") == "2026-11-12T10:00"
    assert load_state(hub_root)["watermarks"] == {"clickup": "2026-11-12T10:00"}
    path = hub_root / "Status" / "sync-state" / "watermark.clickup.yaml"
    assert path.read_text() == "value: 2026-11-12T10:00\n"
    set_watermark(hub_root, "clickup", "2026-11-12T10:00:00")  # looks like a YAML timestamp: stays a string
    assert get_watermark(hub_root, "clickup") == "2026-11-12T10:00:00"
    assert not (hub_root / "Status" / "sync-state.yaml").exists()


def test_state_layout_round_trip(hub_root):
    import pytest
    save_state(hub_root, {"watermarks": {"clickup": "w1", "circleback": "w2"},
                          "next_meeting": {"group": date(2026, 11, 19)}, "flags": ["one", "two"]})
    d = hub_root / "Status" / "sync-state"
    assert sorted(p.name for p in d.iterdir()) == ["flags.md", "next_meeting.group.yaml",
                                                   "watermark.circleback.yaml", "watermark.clickup.yaml"]
    assert (d / "flags.md").read_text() == "- one\n- two\n"
    assert (d / "next_meeting.group.yaml").read_text() == "value: 2026-11-19\n"
    assert load_state(hub_root) == {"watermarks": {"circleback": "w2", "clickup": "w1"},
                                    "next_meeting": {"group": date(2026, 11, 19)}, "flags": ["one", "two"]}
    save_state(hub_root, {"watermarks": {"clickup": "w3"}, "next_meeting": {}, "flags": []})
    assert load_state(hub_root) == {"watermarks": {"clickup": "w3"}, "next_meeting": {}, "flags": []}
    with pytest.raises(ValueError):
        set_watermark(hub_root, "../evil", "x")


def test_last_meeting(hub_root):
    d = hub_root / "Status" / "Meetings" / "group"
    d.mkdir(parents=True)
    for s in ("2026-11-03", "2026-11-10"):
        (d / f"{s}.md").write_text("x")
    assert last_meeting(hub_root, "group") == date(2026, 11, 10)
    assert last_meeting(hub_root, "group", before=date(2026, 11, 10)) == date(2026, 11, 3)
    assert last_meeting(hub_root, "other") is None


def test_snapshot_content(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    store, cfg = Store(hub_root), load_config(hub_root)
    q = store.create("question", "Why?", owner="alice", author="alice", day=D, source="m", priority=1)
    d = store.create("decision", "Do X", owner="bob", author="alice", day=D, source="m",
                     links={"answers": [q.id]})
    store.create("action", "Run X", owner="bob", author="alice", day=D, source="m",
                 links={"spawned_by": [d.id]}, fields={"due": "2026-11-20"})
    state = load_state(hub_root)
    state["next_meeting"] = {"group": date(2026, 11, 19)}
    state["flags"] = ["ClickUp task 86x deleted but A-... open"]
    save_state(hub_root, state)
    text = render_snapshot(store, cfg, date(2026, 11, 15))
    assert text.startswith("# Snapshot — demo\n*Generated 2026-11-15 by `hub snapshot`.")
    assert "| group | — | 2026-11-19 |" in text
    assert f"| [[{q.id}]] | Why? | alice | 3 | 1 |" in text
    assert f"| [[{d.id}]] | Do X | {q.id} | bob |" in text
    assert "### bob\n" in text and "Run X (due 2026-11-20)" in text
    assert "- ClickUp task 86x deleted but A-... open" in text
    assert "## Validation\n0 issues." in text
    path = write_snapshot(store, cfg, date(2026, 11, 15))
    assert path == hub_root / "Status" / "Snapshot.md" and path.read_text() == text


def test_snapshot_survives_garbage_and_missing_fields(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    store, cfg = Store(hub_root), load_config(hub_root)
    q = store.create("question", "Bare", owner="alice", author="alice", day=D, source="m")
    item = store.get(q.id)
    del item.meta["raised"]
    del item.meta["priority"]
    store.save(item)
    (hub_root / "Status" / "Items" / "Q-20261112-bad.md").write_text("garbage\n")
    text = render_snapshot(store, cfg, D)
    assert f"| [[{q.id}]] | Bare | alice | 0 | — |" in text
    flags = text.split("## Flags\n")[1].split("\n## ")[0]
    assert "Q-20261112-bad.md" in flags


def test_snapshot_lists_deferred_decisions(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    store, cfg = Store(hub_root), load_config(hub_root)
    q = store.create("question", "Why?", owner="alice", author="alice", day=D, source="m")
    d = store.create("decision", "Later", owner="bob", author="alice", day=D, source="m",
                     links={"answers": [q.id]}, fields={"status": "deferred"})
    text = render_snapshot(store, cfg, D)
    assert "## Proposed decisions (0)" in text
    assert ("## Deferred decisions (1)\n| ID | Title | Answers | Owner |\n|---|---|---|---|\n"
            f"| [[{d.id}]] | Later | {q.id} | bob |") in text
