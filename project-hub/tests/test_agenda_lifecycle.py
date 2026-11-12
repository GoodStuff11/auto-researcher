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


def test_second_read_refused_when_note_exists(env):
    from hub.agenda.build import set_read_back
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    r1 = row_of(path, ids["d1"].id)
    fill(path, {f"H{r1}": "approve"})
    read_agenda(s, cfg, "group", author="bob", today=MEET)
    before = {k: len(s.get(v.id).entries) for k, v in ids.items()}
    set_read_back(path, False)
    with pytest.raises(AgendaError, match="already exists"):
        read_agenda(s, cfg, "group", author="bob", today=MEET)
    assert before == {k: len(s.get(v.id).entries) for k, v in ids.items()}


def test_resolved_decision_not_overwritten(env):
    s, cfg, root, ids = env
    s.set_fields(ids["d2"].id, {"status": "made"})
    path = build_agenda(root, "group")
    r2 = row_of(path, ids["d2"].id)
    fill(path, {f"H{r2}": "Reject", f"I{r2}": "nope"})
    res = read_agenda(s, cfg, "group", author="bob", today=MEET)
    assert s.get(ids["d2"].id).status == "made"
    p = next(x for x in load_proposals(root) if x.title.startswith("Outcome for already-resolved"))
    assert p.title == "Outcome for already-resolved decision at group/2026-11-19: Drop Z"
    assert p.body["ops"][0]["op"] == "entry" and "nope" in p.body["ops"][0]["text"]
    assert p.pid in res["staged"]


def test_notes_without_outcome_staged(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    r1 = row_of(path, ids["d1"].id)
    fill(path, {f"I{r1}": "Leaning yes", f"J{r1}": "Email Bob (alice)"})
    res = read_agenda(s, cfg, "group", author="bob", today=MEET)
    assert s.get(ids["d1"].id).entries[-1].text == "Not reached at group/2026-11-19."
    assert ids["d1"].id in res["not_reached"]
    p = next(x for x in load_proposals(root) if x.title.startswith("Notes without outcome"))
    assert p.title == "Notes without outcome at group/2026-11-19: Adopt Y"
    op = p.body["ops"][0]
    assert op["op"] == "entry" and "Leaning yes" in op["text"] and "Email Bob (alice)" in op["text"]


def test_build_refuses_unreadable_live_date(env):
    s, cfg, root, _ = env
    path = build_agenda(root, "group")
    wb = load_workbook(path)
    del wb["_meta"]
    for ws in wb.worksheets:
        ws["B1"] = ""
    wb.save(path)
    draft = scaffold(s, root, "group", date(2026, 11, 26))
    draft["summary"] = "Next."
    save_draft(root, draft)
    with pytest.raises(AgendaError, match="no readable meeting date"):
        build_agenda(root, "group")
    assert build_agenda(root, "group", force=True).is_file()


def test_untitled_meeting_row_is_staged(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    text = "Use the 4x4 lattice for every production run from now on, agreed by all"
    fill(path, {"H8": "approve", "I8": text, "J8": "Rerun (bob)"})
    res = read_agenda(s, cfg, "group", author="bob", today=MEET)
    p = next(x for x in load_proposals(root) if x.title.startswith("Raised"))
    expected = f"(untitled: {text[:60]})"
    assert p.title == f"Raised at group/2026-11-19: {expected}"
    assert p.body["ops"][0]["title"] == expected
    assert p.pid in res["staged"]


def test_untitled_row_with_only_actions_is_read(tmp_path):
    from hub.agenda.build import build_workbook
    from hub.agenda.read import read_workbook
    draft = {"series": "group", "meeting_date": MEET, "previous_meeting": None, "summary": "S",
             "since_last": [], "questions": [], "decisions": []}
    p = tmp_path / "a.xlsx"
    build_workbook(draft, p)
    fill(p, {"J5": "Email Tamra (alice)"})
    rows = read_workbook(p).rows
    assert len(rows) == 1 and rows[0].decide == "" and rows[0].new_actions == "Email Tamra (alice)"


def test_reworded_planned_row_is_recorded(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    r1, r2 = row_of(path, ids["d1"].id), row_of(path, ids["d2"].id)
    fill(path, {f"B{r1}": "Adopt Y for 2D only", f"H{r1}": "approve",
                f"B{r2}": "Drop Z now", f"H{r2}": "reject", f"I{r2}": "Too costly"})
    read_agenda(s, cfg, "group", author="bob", today=MEET)
    d1 = s.get(ids["d1"].id)
    assert d1.meta["as_decided"] == "Adopt Y for 2D only"
    assert d1.entries[-1].text == "Approved at group/2026-11-19: Adopt Y for 2D only"
    d2 = s.get(ids["d2"].id)
    assert d2.entries[-1].text == 'Rejected at group/2026-11-19: Too costly (wording in meeting: "Drop Z now")'


def test_same_date_rebuild_refuses_over_meeting_input(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    build_agenda(root, "group")  # no input yet: re-prepping is fine
    fill(path, {f"I{row_of(path, ids['d1'].id)}": "Leaning yes"})
    with pytest.raises(AgendaError) as exc:
        build_agenda(root, "group")
    assert str(exc.value) == ("live agenda already contains meeting input; run `hub agenda read group` "
                              "or pass --force")
    build_agenda(root, "group", force=True)
    fill(path, {"B8": "Something new"})
    with pytest.raises(AgendaError, match="meeting input"):
        build_agenda(root, "group")


def test_same_date_rebuild_allowed_when_viewer_dropped_ids(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    wb = load_workbook(path)
    wb[SHEET_D].delete_cols(1)  # a viewer dropped the hidden id column; planned rows now look unplanned
    wb.save(path)
    assert build_agenda(root, "group").is_file()


def test_read_refuses_draft_workbook_date_mismatch(env):
    s, cfg, root, ids = env
    build_agenda(root, "group")
    draft = scaffold(s, root, "group", date(2026, 11, 26))
    draft["summary"] = "Next."
    save_draft(root, draft)
    with pytest.raises(AgendaError) as exc:
        read_agenda(s, cfg, "group", author="bob", today=MEET)
    assert str(exc.value) == "workbook is for 2026-11-19, draft is for 2026-11-26"


def test_partial_read_back_leaves_marker_and_retry_is_refused(env, monkeypatch):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    fill(path, {f"H{row_of(path, ids['d1'].id)}": "approve", f"H{row_of(path, ids['d2'].id)}": "defer"})
    real = s.set_fields
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash mid read-back")
        return real(*a, **k)

    monkeypatch.setattr(s, "set_fields", flaky)
    with pytest.raises(RuntimeError):
        read_agenda(s, cfg, "group", author="bob", today=MEET)
    note = root / "Status" / "Meetings" / "group" / "2026-11-19.md"
    assert note.read_text() == "# group — 2026-11-19\n- status: read-back in progress\n"
    monkeypatch.setattr(s, "set_fields", real)
    with pytest.raises(AgendaError, match="a previous read-back"):
        read_agenda(s, cfg, "group", author="bob", today=MEET)


def test_deferred_decisions_return_and_match_by_title(env):
    s, cfg, root, ids = env
    s.set_fields(ids["d3"].id, {"status": "deferred"})
    draft = scaffold(s, root, "group", MEET)
    assert [d["id"] for d in draft["decisions"]][-1] == ids["d3"].id
    draft["summary"] = "S."
    save_draft(root, draft)
    path = build_agenda(root, "group")
    fill(path, {"B8": "Try W", "H8": "approve"})  # retyped as a new row: matched by title
    read_agenda(s, cfg, "group", author="bob", today=MEET)
    assert s.get(ids["d3"].id).status == "made"


def test_raised_row_without_outcome_stages_proposed(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    fill(path, {"B8": "Maybe drop UCJ", "J8": "Check with Tamra (bob)"})
    read_agenda(s, cfg, "group", author="bob", today=MEET)
    p = next(x for x in load_proposals(root) if x.title.startswith("Raised"))
    assert p.body["ops"][0]["fields"]["status"] == "proposed"


def test_action_owner_mapped_to_people_key(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    r1 = row_of(path, ids["d1"].id)
    fill(path, {f"H{r1}": "approve", f"J{r1}": "Rerun (Bob); Plot (ALICE); Ask (Bob Smith); Mail (carol)"})
    read_agenda(s, cfg, "group", author="bob", today=MEET)
    p = next(x for x in load_proposals(root) if x.title.startswith("Actions"))
    assert [(o["title"], o["owner"]) for o in p.body["ops"]] == [
        ("Rerun", "bob"), ("Plot", "alice"), ("Ask (Bob Smith)", None), ("Mail (carol)", None)]
