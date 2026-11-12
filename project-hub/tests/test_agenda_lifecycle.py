# project-hub/tests/test_agenda_lifecycle.py
from datetime import date
import pytest
from openpyxl import load_workbook
from hub.agenda.build import SHEET_D
from hub.agenda.draft import save_draft, scaffold
from hub.agenda.lifecycle import AgendaError, build_agenda, live_path, read_agenda
from hub.config import load_config
from hub.pending import load_proposals
from hub.store import Store

MEET = date(2026, 11, 19)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    s, cfg = Store(hub_root), load_config(hub_root)
    q = s.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m", series=["group"])
    d1 = s.create("decision", "Adopt Y", owner="alice", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    d2 = s.create("decision", "Drop Z", owner="bob", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    d3 = s.create("decision", "Try W", owner="bob", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    draft = scaffold(s, hub_root, "group", MEET)
    draft["summary"] = "Status."
    save_draft(hub_root, draft)
    return s, cfg, hub_root, dict(q=q, d1=d1, d2=d2, d3=d3)


def fill(path, rows):
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    for ref, val in rows.items():
        ws[ref] = val
    wb.save(path)


def row_of(path, item_id):
    ws = load_workbook(path)[SHEET_D]
    for r in range(5, ws.max_row + 1):
        if ws.cell(r, 1).value == item_id:
            return r
    raise AssertionError(item_id)


def test_build_and_read_back(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    assert path == live_path(root, "group")
    r1, r2 = row_of(path, ids["d1"].id), row_of(path, ids["d2"].id)
    fill(path, {f"H{r1}": "Modify", f"I{r1}": "Only 4x3", f"J{r1}": "Rewrite ¶2 (alice)",
                f"H{r2}": "Reject", f"I{r2}": "Not worth it",
                "B8": "Drop UCJ", "H8": "approve", "I8": "Dropped", "J8": "Tell Tamra (bob)"})
    res = read_agenda(s, cfg, "group", author="bob", today=date(2026, 11, 19))
    d1 = s.get(ids["d1"].id)
    assert d1.status == "made" and d1.meta["meeting"] == "group/2026-11-19" and d1.meta["as_decided"] == "Only 4x3"
    assert d1.entries[-1].text == "Modified at group/2026-11-19: Only 4x3"
    d2 = s.get(ids["d2"].id)
    assert d2.status == "rejected" and d2.meta["closed"] == date(2026, 11, 19)
    assert s.get(ids["d3"].id).entries[-1].text == "Not reached at group/2026-11-19."
    assert res["not_reached"] == [ids["d3"].id]
    props = load_proposals(root)
    titles = sorted(p.title for p in props)
    assert titles == ["Actions from group/2026-11-19: Adopt Y", "Raised at group/2026-11-19: Drop UCJ"]
    raised = next(p for p in props if p.title.startswith("Raised"))
    assert raised.body["ops"][0]["fields"]["status"] == "made"
    assert raised.body["ops"][1]["owner"] == "bob"
    note = (root / "Status" / "Meetings" / "group" / "2026-11-19.md").read_text()
    assert note.startswith("# group — 2026-11-19\n- attendees: alice, bob\n")
    assert f"- [[{ids['d1'].id}]] modify — Only 4x3" in note
    assert (root / "Agenda" / "archive" / "group" / "2026-11-19.xlsx").is_file()
    with pytest.raises(AgendaError, match="already read back"):
        read_agenda(s, cfg, "group", author="bob", today=date(2026, 11, 19))


def test_build_refuses_unread_previous(env):
    s, cfg, root, _ = env
    build_agenda(root, "group")
    draft = scaffold(s, root, "group", date(2026, 11, 26))
    draft["summary"] = "Next."
    save_draft(root, draft)
    with pytest.raises(AgendaError, match="not been read back"):
        build_agenda(root, "group")
    assert build_agenda(root, "group", force=True).is_file()


def test_read_requires_live_file(env):
    s, cfg, root, _ = env
    with pytest.raises(AgendaError, match="no live agenda"):
        read_agenda(s, cfg, "group", author="bob", today=MEET)


def test_unrecognised_outcome_is_staged(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    r1 = row_of(path, ids["d1"].id)
    fill(path, {f"H{r1}": "approve, but only 4x3"})
    res = read_agenda(s, cfg, "group", author="bob", today=date(2026, 11, 19))
    d1 = s.get(ids["d1"].id)
    assert d1.status == "proposed"
    assert not any("Not reached" in e.text for e in d1.entries)
    assert ids["d1"].id not in res["not_reached"]
    props = load_proposals(root)
    p = next(x for x in props if x.title.startswith("Unrecognised"))
    assert p.title == "Unrecognised outcome at group/2026-11-19: Adopt Y"
    assert p.body["ops"] == [{"op": "entry", "id": ids["d1"].id, "source": "agenda:group/2026-11-19",
                              "text": "Outcome typed in meeting: 'approve, but only 4x3'; as decided: —"}]
    assert res["staged"] == [p.pid]
    assert p.pid in (root / "Status" / "Meetings" / "group" / "2026-11-19.md").read_text()
