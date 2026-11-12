from datetime import date

import pytest
from openpyxl import load_workbook

from hub.agenda.build import SHEET_D
from hub.agenda.draft import save_draft, scaffold
from hub.agenda.lifecycle import build_agenda, read_agenda
from hub.config import load_config
from hub.pending import load_proposals
from hub.snapshot import render_snapshot
from hub.store import Store
from hub.validate import validate

D = date(2026, 11, 12)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    return Store(hub_root), load_config(hub_root), hub_root


def test_wrong_shaped_links_are_reported_not_fatal(env):
    store, cfg, root = env
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    p = store.path(q.id)
    text = p.read_text()
    text = text.replace("links:\n  answers: []\n  spawned_by: []\n  pivoted_to: []", "links:\n- a")
    text = text.replace("series: []", "series: group")
    p.write_text(text)
    item = store.get(q.id)
    assert item.links("answers") == []
    rules = sorted(v.rule for v in validate(store, cfg))
    assert rules == ["shape", "shape"]
    assert "# Snapshot — demo" in render_snapshot(store, cfg, D)


def _meeting(store, root, rows):
    q = store.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m",
                     series=["group"])
    d = store.create("decision", "Adopt Y", owner="alice", author="alice", day=date(2026, 11, 2), source="m",
                     series=["group"], links={"answers": [q.id]})
    draft = scaffold(store, root, "group", date(2026, 11, 19))
    draft["summary"] = "Status."
    save_draft(root, draft)
    path = build_agenda(root, "group")
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    for ref, val in rows(d.id, ws).items():
        ws[ref] = val
    wb.save(path)
    return d


def test_untitled_row_title_is_single_line(env):
    store, cfg, root = env
    _meeting(store, root, lambda did, ws: {"H6": "approve", "J6": "Rerun HVA (bob)\nPlot it (alice)"})
    read_agenda(store, cfg, "group", author="bob", today=date(2026, 11, 19))
    titles = [p.title for p in load_proposals(root)]
    assert titles == ["Raised at group/2026-11-19: (untitled: Rerun HVA (bob) Plot it (alice))"]


def test_unmatched_owner_stays_in_title(env):
    store, cfg, root = env
    _meeting(store, root, lambda did, ws: {"H5": "approve", "J5": "Review draft (due Friday); Plot (Bob)"})
    read_agenda(store, cfg, "group", author="bob", today=date(2026, 11, 19))
    p = next(x for x in load_proposals(root) if x.title.startswith("Actions"))
    assert [(o["title"], o["owner"]) for o in p.body["ops"]] == [("Review draft (due Friday)", None),
                                                                  ("Plot", "bob")]
