# project-hub/spike/make_fixture.py
"""Throwaway: write spike.xlsx exercising every workbook feature the agenda relies on.

Usage: python make_fixture.py OUT.xlsx
  OUT.xlsx  path of the workbook to create (overwritten).
"""
import sys
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


def main(argv):
    out = argv[0]
    wb = Workbook()
    ws = wb.active
    ws.title = "Decisions"
    ws["B1"] = "group — 2026-11-19 · Decisions"
    ws["B1"].font = Font(bold=True, size=14)
    ws.merge_cells("B2:E2")
    ws["B2"] = "Merged, wrapped summary cell. " * 6
    ws["B2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 60
    headers = ["id", "What to decide", "Outcome", "As decided"]
    for col, name in enumerate(headers, 1):
        c = ws.cell(4, col, name)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="404040")
    ws.cell(5, 1, "D-20261112-abc")
    ws.cell(5, 2, "Row 5: pick Approve from the dropdown")
    ws.cell(6, 1, "D-20261112-def")
    ws.cell(6, 2, "Row 6: TYPE the word modify by hand")
    ws.cell(5, 2).fill = PatternFill("solid", fgColor="E2F0D9")
    ws.column_dimensions["A"].hidden = True
    ws.column_dimensions["B"].width = 45
    ws.freeze_panes = "B5"
    dv = DataValidation(type="list", formula1='"Approve,Modify,Defer,Reject"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("C5:C10")
    meta = wb.create_sheet("_meta")
    meta["A1"], meta["B1"] = "series", "group"
    meta["A2"], meta["B2"] = "read_back", "false"
    meta.sheet_state = "hidden"
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
