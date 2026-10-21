"""parser-B: print A and F for chosen rows of 'Evidence Lineage' (openpyxl). Usage: venv python -I rows_af.py XLSX ROW..."""
import sys, openpyxl
ws = openpyxl.load_workbook(sys.argv[1], read_only=True)["Evidence Lineage"]
want = set(int(x) for x in sys.argv[2:])
for r, row in enumerate(ws.iter_rows(min_row=1, max_row=max(want), max_col=16, values_only=True), start=1):
    if r in want: print(r, repr(row[0]), repr(row[5]), repr(row[1]))
