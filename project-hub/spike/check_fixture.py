# project-hub/spike/check_fixture.py
"""Throwaway: report which features survived an edit+save in Office Viewer.

Usage: python check_fixture.py EDITED.xlsx
"""
import sys
from openpyxl import load_workbook


def main(argv):
    wb = load_workbook(argv[0])
    print("sheets:", wb.sheetnames)
    print("_meta present:", "_meta" in wb.sheetnames,
          "state:", wb["_meta"].sheet_state if "_meta" in wb.sheetnames else None)
    ws = wb["Decisions"]
    print("col A hidden:", ws.column_dimensions["A"].hidden)
    print("A5 id:", ws["A5"].value, "| A6 id:", ws["A6"].value)
    print("C5 outcome:", ws["C5"].value, "| C6 outcome:", ws["C6"].value)
    print("data validations:", len(ws.data_validations.dataValidation))
    print("freeze:", ws.freeze_panes, "| merged:", ws.merged_cells.ranges)
    print("B5 fill:", ws["B5"].fill.fgColor.rgb, "| header font bold:", ws["A4"].font.bold)
    print("B2 wrap:", ws["B2"].alignment.wrap_text, "| row2 height:", ws.row_dimensions[2].height)


if __name__ == "__main__":
    main(sys.argv[1:])
