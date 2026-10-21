"""parser-B: cross-file cell diff of 'Evidence Lineage' and tie listing. Usage: venv python -I diff_and_ties.py XLSX..."""
import sys, collections, openpyxl
LIST11 = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION",
          "SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]
grids = {}
for p in sys.argv[1:]:
    ws = openpyxl.load_workbook(p, read_only=True, data_only=False)["Evidence Lineage"]
    g = {}
    for row in ws.iter_rows(min_row=1, max_row=2001, max_col=16):
        for c in row:
            if c.value is not None and str(c.value).strip() != "": g[(c.row, c.column)] = c.value
    grids[p.split("/")[-1]] = g
    lab = collections.Counter(str(v).strip() for (r, col), v in g.items() if col == 6 and r >= 2)
    out = collections.Counter({k: n for k, n in lab.items() if k not in LIST11})
    print("==", p.split("/")[-1], "labels with count>=3 outside list:", [(k, n) for k, n in out.most_common() if n >= 3])
    print("   outside labels with count==1:", sum(1 for n in out.values() if n == 1), " count==2:", sum(1 for n in out.values() if n == 2),
          " distinct outside:", len(out), " distinct in-list:", sum(1 for k in lab if k in LIST11))
names = list(grids)
for a, b in zip(names, names[1:]):
    A, B = grids[a], grids[b]
    keys = sorted(set(A) | set(B))
    diffs = [(k, A.get(k), B.get(k)) for k in keys if A.get(k) != B.get(k)]
    print(f"== diff {a} -> {b}: {len(diffs)} cells differ")
    rows = sorted({k[0] for k, _, _ in diffs})
    print("   rows touched:", rows[:60], len(rows))
    for (r, c), x, y in diffs:
        if c == 6 or r < 518: print("   ", (r, c), repr(x)[:80], "->", repr(y)[:80])
