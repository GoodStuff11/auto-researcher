from datetime import date
import pytest
from openpyxl import load_workbook
from hub.agenda.build import (D_HEADERS, SHEET_D, SHEET_META, SHEET_Q, SHEET_SINCE, UNPLANNED_ROWS,
                              build_workbook, set_read_back)
from hub.agenda.draft import DraftError
from hub.agenda.read import parse_actions, parse_outcome, read_workbook

DRAFT = {
    "series": "group", "meeting_date": date(2026, 11, 19), "previous_meeting": date(2026, 11, 12),
    "summary": "We locked the framing.",
    "since_last": [{"id": "D-20261112-aaa", "decision": "Do X", "actions": "Run X (bob, done)",
                    "follow_up": "Ran at 4x4", "outcome": "closed question", "affected": "Q-20261101-qqq"}],
    "questions": [{"id": "Q-20261101-qqq", "title": "Why?", "description": "Because.", "owner": "alice",
                   "age_days": 18, "priority": 1, "proposed_decision": "D-20261115-bbb"}],
    "decisions": [{"id": "D-20261115-bbb", "decide": "Adopt Y", "proposed_action": "Rewrite ¶2",
                   "reason": "Cheaper", "weigh_in": "eunah", "minutes": 10, "link": "https://example.com/fig"}],
}


@pytest.fixture
def xlsx(tmp_path):
    p = tmp_path / "group.xlsx"
    build_workbook(DRAFT, p)
    return p


def test_structure(xlsx):
    wb = load_workbook(xlsx)
    assert wb.sheetnames == [SHEET_SINCE, SHEET_Q, SHEET_D, SHEET_META]
    assert wb[SHEET_META].sheet_state == "hidden"
    ws = wb[SHEET_D]
    assert [c.value for c in ws[4]] == D_HEADERS
    assert ws.column_dimensions["A"].hidden
    assert ws["A5"].value == "D-20261115-bbb" and ws["B5"].value == "Adopt Y"
    assert ws["A6"].value == "NEW-1" and ws[f"A{5 + UNPLANNED_ROWS}"].value == f"NEW-{UNPLANNED_ROWS}"
    assert len(ws.data_validations.dataValidation) == 1
    assert ws["G5"].hyperlink.target == "https://example.com/fig"
    assert wb[SHEET_SINCE]["B2"].value == "We locked the framing."
    assert wb[SHEET_SINCE]["B1"].value == "group — 2026-11-19 · Since last meeting"


def test_build_rejects_invalid(tmp_path):
    with pytest.raises(DraftError, match="summary"):
        build_workbook({**DRAFT, "summary": ""}, tmp_path / "x.xlsx")


def test_read_back_round_trip(xlsx):
    wb = load_workbook(xlsx)
    ws = wb[SHEET_D]
    ws["H5"], ws["I5"], ws["J5"] = "modify", "Adopt Y but only for 4x3", "Rewrite ¶2 (alice); Rerun fig (bob)"
    ws["B6"], ws["H6"], ws["I6"] = "Drop UCJ", "✓", "Dropped"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.series == "group" and rb.meeting_date == date(2026, 11, 19) and rb.read_back is False
    assert len(rb.rows) == 2
    r0, r1 = rb.rows
    assert (r0.id, r0.outcome, r0.planned, r0.as_decided) == ("D-20261115-bbb", "modify", True, "Adopt Y but only for 4x3")
    assert (r1.id, r1.decide, r1.outcome, r1.planned) == ("NEW-1", "Drop UCJ", "approve", False)


def test_read_without_meta_or_id_column(xlsx):
    wb = load_workbook(xlsx)
    del wb[SHEET_META]
    ws = wb[SHEET_D]
    ws.delete_cols(1)
    ws["G5"] = "Approve"
    ws["A6"] = "Drop UCJ"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.series == "group" and rb.meeting_date == date(2026, 11, 19)
    assert len(rb.rows) == 2
    r0, r1 = rb.rows
    assert (r0.id, r0.decide, r0.outcome, r0.planned) == (None, "Adopt Y", "approve", False)
    assert (r1.id, r1.decide, r1.planned) == (None, "Drop UCJ", False)


def test_read_meta_garbage_date_falls_back_to_title(xlsx):
    wb = load_workbook(xlsx)
    wb[SHEET_META]["B2"] = "garbage"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.meeting_date == date(2026, 11, 19) and rb.series == "group"


def test_literal_equals_text_is_not_a_formula(tmp_path):
    draft = {**DRAFT, "decisions": [{**DRAFT["decisions"][0], "decide": "=SUM(1,2)"}]}
    p = tmp_path / "eq.xlsx"
    build_workbook(draft, p)
    assert load_workbook(p)[SHEET_D]["B5"].data_type == "s"
    assert read_workbook(p).rows[0].decide == "=SUM(1,2)"


def test_set_read_back(xlsx):
    set_read_back(xlsx, True)
    assert read_workbook(xlsx).read_back is True


@pytest.mark.parametrize("text,expected", [
    ("Approve", "approve"), ("  a ", "approve"), ("✓", "approve"), ("yes", "approve"),
    ("Modify", "modify"), ("m", "modify"), ("Defer", "defer"), ("d", "defer"),
    ("Reject", "reject"), ("✗", "reject"), ("x", "reject"), ("no", "reject"), ("", None), (None, None)])
def test_parse_outcome(text, expected):
    assert parse_outcome(text) == expected


def test_parse_actions():
    assert parse_actions("Rewrite ¶2 (alice); Rerun fig (bob)\nThink more") == [
        ("Rewrite ¶2", "alice"), ("Rerun fig", "bob"), ("Think more", None)]
    assert parse_actions("") == []


@pytest.mark.parametrize("text", [
    "after next run", "another run", "no objection", "not now", "revise", "drop"])
def test_parse_outcome_rejects_free_text(text):
    assert parse_outcome(text) is None


@pytest.mark.parametrize("text,expected", [("approved.", "approve"), ("yes!", "approve")])
def test_parse_outcome_strips_trailing_punctuation(text, expected):
    assert parse_outcome(text) == expected


def test_summary_equals_text_is_literal(tmp_path):
    p = tmp_path / "sum.xlsx"
    build_workbook({**DRAFT, "summary": "=1+1"}, p)
    cell = load_workbook(p)[SHEET_SINCE]["B2"]
    assert cell.value == "=1+1" and cell.data_type == "s"


def test_outcome_text_keeps_raw_cell(xlsx):
    wb = load_workbook(xlsx)
    wb[SHEET_D]["H5"] = "approve, but only 4x3"
    wb.save(xlsx)
    r0 = read_workbook(xlsx).rows[0]
    assert r0.outcome is None and r0.outcome_text == "approve, but only 4x3"


def test_title_with_invalid_date_does_not_crash(xlsx):
    wb = load_workbook(xlsx)
    del wb[SHEET_META]
    wb[SHEET_D]["B1"] = "group — 2026-02-30 · Decisions"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.meeting_date is None and rb.series == "group"
