# project-hub/tests/test_agenda_draft.py
from datetime import date
import pytest
from hub.agenda.draft import DraftError, draft_path, load_draft, save_draft, scaffold, validate_draft
from hub.store import Store

D = date(2026, 11, 5)


@pytest.fixture
def setup(hub_root):
    s = Store(hub_root)
    q1 = s.create("question", "Old high", owner="alice", author="alice", day=date(2026, 10, 1), source="m",
                  series=["group"], priority=1)
    q2 = s.create("question", "New low", owner="bob", author="alice", day=D, source="m", series=["group"], priority=3)
    s.create("question", "Other series", owner="bob", author="alice", day=D, source="m", series=["sub"])
    dprop = s.create("decision", "Adopt Y", owner="alice", author="alice", day=D, source="m", series=["group"],
                     links={"answers": [q2.id]}, fields={"rationale": "Cheaper."})
    dmade = s.create("decision", "Do X", owner="alice", author="alice", day=D, source="m", series=["group"],
                     links={"answers": [q1.id]},
                     fields={"status": "made", "meeting": "group/2026-11-05", "as_decided": "Do X at 4x4"})
    s.create("action", "Run X", owner="bob", author="alice", day=D, source="m", links={"spawned_by": [dmade.id]})
    mdir = hub_root / "Status" / "Meetings" / "group"
    mdir.mkdir(parents=True)
    (mdir / "2026-11-05.md").write_text("x")
    return s, hub_root, dict(q1=q1, q2=q2, dprop=dprop, dmade=dmade)


def test_scaffold(setup):
    s, root, ids = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    assert d["previous_meeting"] == date(2026, 11, 5) and d["summary"] == ""
    assert d["since_last"] == [{"id": ids["dmade"].id, "decision": "Do X at 4x4", "actions": "Run X (bob, open)",
                                "follow_up": "", "outcome": "in progress", "affected": ids["q1"].id}]
    assert [q["title"] for q in d["questions"]] == ["New low", "Old high"]  # has proposed decision first
    assert d["questions"][1]["age_days"] == 42 and d["questions"][0]["proposed_decision"] == ids["dprop"].id
    assert d["decisions"] == [{"id": ids["dprop"].id, "decide": "Adopt Y", "proposed_action": "",
                               "reason": "Cheaper.", "weigh_in": "", "minutes": 10, "link": ""}]


def test_validate_draft(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    assert validate_draft(d) == ["summary is empty"]
    d["summary"] = "Where we stand."
    d["since_last"][0]["outcome"] = "done"
    d["decisions"] = d["decisions"] * 8
    errs = validate_draft(d)
    assert any("outcome 'done'" in e for e in errs) and any("at most 7" in e for e in errs)


def test_save_load_round_trip(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    path = save_draft(root, d)
    assert path == root / "Agenda" / "group.draft.yaml"
    assert load_draft(root, "group") == d


def test_validate_summary_none(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    d["summary"] = None
    assert "summary is empty" in validate_draft(d)


def test_validate_non_mapping():
    assert validate_draft(None) == ["draft is not a mapping"]


def test_since_last_not_list(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    d["summary"] = "Where we stand."
    d["since_last"] = None
    errs = validate_draft(d)
    assert "'since_last' must be a list" in errs


def test_decision_empty_decide(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    d["summary"] = "Where we stand."
    d["decisions"][0]["decide"] = ""
    assert any("decide" in e for e in validate_draft(d))


def test_load_invalid_yaml(setup):
    s, root, _ = setup
    p = draft_path(root, "group")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("summary: [unclosed\n")
    with pytest.raises(DraftError):
        load_draft(root, "group")
    p.write_text("- just\n- a list\n")
    with pytest.raises(DraftError):
        load_draft(root, "group")


def test_since_last_outcomes(hub_root):
    s = Store(hub_root)

    def q(title):
        return s.create("question", title, owner="alice", author="alice", day=D, source="m", series=["group"])

    def made(title, target):
        return s.create("decision", title, owner="alice", author="alice", day=D, source="m", series=["group"],
                        links={"answers": [target.id]},
                        fields={"status": "made", "meeting": "group/2026-11-05"})

    qa, qp, qo = q("Answered"), q("Pivoted"), q("Still open")
    target = q("Pivot target")
    dA, dP, dN = made("Decide A", qa), made("Decide P", qp), made("Decide N", qo)

    s.set_fields(qa.id, {"status": "answered"})
    s.entry(qa.id, "Answered by decision.", author="alice", source="m", day=D)
    s.link(qp.id, "pivoted_to", target.id)
    s.set_fields(qp.id, {"status": "pivoted"})
    s.entry(qp.id, "Pivoted.", author="alice", source="m", day=D)
    s.create("question", "Follow-up", owner="alice", author="alice", day=D, source="m", series=["group"],
             links={"spawned_by": [dN.id]})

    mdir = hub_root / "Status" / "Meetings" / "group"
    mdir.mkdir(parents=True)
    (mdir / "2026-11-05.md").write_text("x")

    d = scaffold(s, hub_root, "group", date(2026, 11, 12))
    outcomes = {r["id"]: r["outcome"] for r in d["since_last"]}
    assert outcomes == {dA.id: "closed question", dP.id: "pivot", dN.id: "new question"}
