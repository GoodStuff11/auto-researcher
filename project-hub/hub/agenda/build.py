# project-hub/hub/agenda/build.py
"""Render an agenda draft into a fixed-format xlsx (the agent never styles cells itself)."""
from __future__ import annotations

import math
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .draft import DraftError, validate_draft

SHEET_SINCE, SHEET_Q, SHEET_D, SHEET_META = "Since last meeting", "Open questions", "Decisions", "_meta"
HEADER_ROW = 4
OUTCOME_CHOICES = ("Approve", "Modify", "Defer", "Reject")
UNPLANNED_ROWS = 5
SINCE_COLS = [("id", 16), ("Decision", 40), ("Actions · owners", 40), ("Follow-up", 50), ("Outcome", 16), ("Affected", 20)]
Q_COLS = [("id", 16), ("Title", 35), ("Description", 60), ("Owner", 12), ("Age (d)", 9), ("Priority", 9), ("Proposed decision", 18)]
D_COLS = [("id", 16), ("What to decide", 40), ("Proposed action", 40), ("Reason", 40), ("Who weighs in", 16),
          ("Min", 6), ("Figure / pre-read", 25), ("Outcome", 12), ("As decided", 40), ("New actions · owners", 40)]
D_HEADERS = [n for n, _ in D_COLS]

HEADER_FILL = PatternFill("solid", fgColor="404040")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=14)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Border(bottom=Side(style="thin", color="BFBFBF"))
OUTCOME_FILLS = {"closed question": "E2F0D9", "new question": "DDEBF7", "pivot": "FCE4D6"}
UNPLANNED_FILL = PatternFill("solid", fgColor="FFF2CC")
DECIDE_FILL = PatternFill("solid", fgColor="F2F2F2")


def _sheet(wb, title, cols, draft, first=False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws["B1"] = f"{draft['series']} — {draft['meeting_date'].isoformat()} · {title}"  # B: column A is hidden
    ws["B1"].font = TITLE_FONT
    for i, (name, width) in enumerate(cols, 1):
        c = ws.cell(HEADER_ROW, i, name)
        c.fill, c.font, c.alignment = HEADER_FILL, HEADER_FONT, WRAP
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.column_dimensions["A"].hidden = True
    ws.freeze_panes = f"B{HEADER_ROW + 1}"
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    return ws


def _row(ws, r, values, fill=None):
    for i, v in enumerate(values, 1):
        c = ws.cell(r, i, v)
        c.alignment, c.border = WRAP, THIN
        if fill:
            c.fill = fill
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            c.hyperlink, c.font = v, Font(color="0563C1", underline="single")


def build_workbook(draft: dict, out_path: Path) -> None:
    errs = validate_draft(draft)
    if errs:
        raise DraftError("; ".join(errs))
    wb = Workbook()
    ws = _sheet(wb, SHEET_SINCE, SINCE_COLS, draft, first=True)
    ws.merge_cells(start_row=2, start_column=2, end_row=2, end_column=len(SINCE_COLS))
    ws["B2"] = draft["summary"]
    ws["B2"].alignment = WRAP
    ws.row_dimensions[2].height = 15 * max(2, math.ceil(len(draft["summary"]) / 150) + 1)
    if draft["previous_meeting"] is None:
        _row(ws, HEADER_ROW + 1, ["", "First meeting of this series — nothing to follow up.", "", "", "", ""])
    for r, s in enumerate(draft["since_last"], HEADER_ROW + 1):
        fill = PatternFill("solid", fgColor=OUTCOME_FILLS[s["outcome"]]) if s["outcome"] in OUTCOME_FILLS else None
        _row(ws, r, [s["id"], s["decision"], s["actions"], s["follow_up"], s["outcome"], s["affected"]])
        if fill:
            ws.cell(r, 5).fill = fill
    ws = _sheet(wb, SHEET_Q, Q_COLS, draft)
    for r, q in enumerate(draft["questions"], HEADER_ROW + 1):
        _row(ws, r, [q["id"], q["title"], q["description"], q["owner"], q["age_days"], q["priority"],
                     q["proposed_decision"]])
    ws = _sheet(wb, SHEET_D, D_COLS, draft)
    r = HEADER_ROW + 1
    for d in draft["decisions"]:
        _row(ws, r, [d["id"], d["decide"], d["proposed_action"], d["reason"], d["weigh_in"], d["minutes"],
                     d["link"], "", "", ""])
        for col in (8, 9, 10):
            ws.cell(r, col).fill = DECIDE_FILL
        r += 1
    for k in range(1, UNPLANNED_ROWS + 1):
        _row(ws, r, [f"NEW-{k}", "", "", "", "", "", "", "", "", ""], fill=UNPLANNED_FILL)
        r += 1
    ws.cell(2, 2, f"Yellow rows: decisions raised in the meeting. Outcome: {' / '.join(OUTCOME_CHOICES)}.")
    dv = DataValidation(type="list", formula1='"' + ",".join(OUTCOME_CHOICES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H{HEADER_ROW + 1}:H{r - 1}")
    meta = wb.create_sheet(SHEET_META)
    for i, (k, v) in enumerate([("series", draft["series"]), ("meeting_date", draft["meeting_date"].isoformat()),
                                ("read_back", "false"), ("format", "1")], 1):
        meta.cell(i, 1, k)
        meta.cell(i, 2, v)
    meta.sheet_state = "hidden"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def set_read_back(path: Path, value: bool) -> None:
    wb = load_workbook(path)
    if SHEET_META not in wb.sheetnames:
        meta = wb.create_sheet(SHEET_META)
        meta.sheet_state = "hidden"
    meta = wb[SHEET_META]
    for row in meta.iter_rows(min_row=1, max_row=meta.max_row):
        if row[0].value == "read_back":
            row[1].value = "true" if value else "false"
            break
    else:
        meta.append(["read_back", "true" if value else "false"])
    wb.save(path)
