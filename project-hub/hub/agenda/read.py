# project-hub/hub/agenda/read.py
"""Read a meeting-edited agenda workbook back: matched by hidden id, tolerant of viewers that drop features."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from openpyxl import load_workbook

from .build import SHEET_D, SHEET_META

TITLE_RE = re.compile(r"^(\S+) — (\d{4}-\d{2}-\d{2}) · ")
ACTION_RE = re.compile(r"^(.*?)\s*\(([\w.-]+(?: [\w.-]+)?)\)\s*$")  # "title (owner)", owner raw: 1-2 words


@dataclass
class DecisionRow:
    id: str | None
    decide: str
    outcome: str | None
    as_decided: str
    new_actions: str
    planned: bool
    outcome_text: str = ""


@dataclass
class ReadBack:
    series: str | None
    meeting_date: date | None
    read_back: bool
    rows: list[DecisionRow]


OUTCOME_TOKENS = {
    "approve": {"approve", "approved", "a", "yes", "y", "✓", "✔"},
    "modify": {"modify", "modified", "m"},
    "defer": {"defer", "deferred", "d"},
    "reject": {"reject", "rejected", "r", "no", "n", "x", "✗", "✘"},
}


def parse_outcome(text) -> str | None:
    t = str(text or "").strip().lower().rstrip(".!")
    for outcome, tokens in OUTCOME_TOKENS.items():
        if t in tokens:
            return outcome
    return None


def parse_actions(text) -> list[tuple[str, str | None]]:
    out = []
    for part in re.split(r"[;\n]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        m = ACTION_RE.match(part)
        out.append((m[1].strip(), m[2]) if m else (part, None))
    return out


def _cell(v) -> str:
    return "" if v is None else str(v).strip()


def read_workbook(path) -> ReadBack:
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    series = meeting = None
    read_back = False
    if SHEET_META in wb.sheetnames:
        kv = {_cell(r[0].value): _cell(r[1].value) for r in wb[SHEET_META].iter_rows() if len(r) >= 2}
        series = kv.get("series") or None
        try:
            meeting = date.fromisoformat(kv["meeting_date"]) if kv.get("meeting_date") else None
        except ValueError:
            meeting = None  # invalid _meta date: fall back to the sheet title
        read_back = kv.get("read_back") == "true"
    if series is None or meeting is None:
        for ref in ("B1", "A1"):  # A1 if a viewer dropped the hidden id column
            m = TITLE_RE.match(_cell(ws[ref].value))
            if m:
                series = m[1]
                try:
                    meeting = date.fromisoformat(m[2])
                except ValueError:
                    meeting = None  # e.g. 2026-02-30: keep the series, leave the date unknown
                break
    header_row, cols = None, {}
    for r in range(1, 11):
        values = [_cell(c.value) for c in ws[r]]
        if "What to decide" in values:
            header_row = r
            cols = {v: i + 1 for i, v in enumerate(values) if v}
            break
    if header_row is None:
        raise ValueError(f"{path}: no 'What to decide' header in the first 10 rows of '{SHEET_D}'")

    def get(r, name):
        return _cell(ws.cell(r, cols[name]).value) if name in cols else ""

    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        rid = get(r, "id") or None
        decide = get(r, "What to decide")
        outcome_text, as_decided, new_actions = get(r, "Outcome"), get(r, "As decided"), get(r, "New actions · owners")
        if not (decide or outcome_text or as_decided or new_actions):
            continue  # an empty row; a row with any meeting input is kept even without a title
        planned = rid is not None and not rid.startswith("NEW")
        rows.append(DecisionRow(rid, decide, parse_outcome(outcome_text), as_decided, new_actions, planned,
                                outcome_text))
    return ReadBack(series, meeting, read_back, rows)
