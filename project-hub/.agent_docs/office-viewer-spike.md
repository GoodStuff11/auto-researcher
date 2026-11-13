# VS Code Office Viewer spike
Throwaway round-trip test of `cweijan.vscode-office` on an agenda-shaped xlsx, run 2026-11-12. Code: `project-hub/spike/make_fixture.py` and `check_fixture.py`.

## Setup
`make_fixture.py` wrote `~/spike.xlsx` with openpyxl. It holds:
- Sheet `Decisions`: title in B1; merged, wrapped summary B2:E2 (row 2 height 60); bold white header on row 4 with a dark fill; column A hidden and holding ids `D-20261112-abc` (row 5) and `D-20261112-def` (row 6); list dropdown `Approve,Modify,Defer,Reject` on C5:C10; freeze at B5; fill on B5.
- Hidden sheet `_meta`: `series` = `group`, `read_back` = `false`. No `meeting_date` row.

The user then typed `Approve` in C5 and `modify` in C6, saved, and `check_fixture.py` was run on the saved file (21050 bytes).

## Observed in Office Viewer
- No dropdown is shown on C5:C10. Outcomes have to be typed.
- Column A (the ids) is visible.
- The `_meta` tab is visible.

## Survived save
`check_fixture.py` output after save (controller's run on the user's saved file; the D5 "as proposed" step was not checked, since the checker does not read D5):
```
sheets: ['Decisions', '_meta']
_meta present: True state: visible
col A hidden: False
A5 id: D-20261112-abc | A6 id: D-20261112-def
C5 outcome: Approve | C6 outcome: modify
data validations: 1
freeze: B5 | merged: {<MergedCellRange B2:E2>}
B5 fill: FFE2F0D9 | header font bold: True
B2 wrap: True | row2 height: 60.0
```

Kept: both sheets, both ids, both typed outcomes, the data validation count (1, count only; its list contents were not checked), freeze pane, merge, fill, bold header, wrap, row height.
Changed: the hidden flags. `_meta` is now visible and column A is no longer hidden. The fill's alpha prefix moved from `00` to `FF`; the colour is the same.

`read_workbook` on the same saved file (controller's run) returned series `group`, meeting date `2026-11-19` (taken from the B1 title, since `_meta` has no `meeting_date`), `read_back` False, and rows `D-20261112-abc` with outcome `approve` and `D-20261112-def` with outcome `modify`.

## Consequences for the agenda
- `_meta`: kept; visibility changed. Office Viewer saves it visible (hidden flag lost), which is harmless to the reader. The reader takes the series from it. The date fell back to the title row here; the real build writes `meeting_date` into `_meta`, so the fallback is tested by this spike only.
- Hidden id column: kept; visibility changed. Office Viewer saves column A visible (hidden flag lost), which is harmless to the reader. The ids survive and the reader maps them.
- Dropdown: changed in use. The validation rule survives in the file, but Office Viewer does not show it, so outcomes must be typed. Typed tokens are read exactly as documented in [agenda-workbook.md](agenda-workbook.md) (`parse_outcome`); the Decisions hint row (B2) lists the choices.
- Styling: kept. No re-styling task is needed.
