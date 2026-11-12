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
