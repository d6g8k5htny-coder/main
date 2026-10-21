#!/usr/bin/env python3
"""parser-A: Evidence Role counts in evidence_lineage.csv and the 2026-09-17 markdown export.
Usage: python3 -I evrole_csv_md.py CSV MD [XLSX_RESULTS_JSON]
Same whitespace rule as evrole_xlsx.py: empty iff value.strip()==""; membership is exact
case-sensitive match of value.strip() against the 11 options.
Markdown cells: split on unescaped '|', strip, then remove markdown backslash escapes
(backslash + ASCII punctuation -> punctuation).
"""
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter

OPTS = ["PRIMARY_DERIVATION", "EXECUTABLE_RECEIPT", "RAW_DATA", "FORMAL_PROOF",
        "INDEPENDENT_RECONSTRUCTION", "SUMMARY", "NAVIGATION", "BALLOT", "REGISTER_COPY",
        "REVIEW_OF_REVIEW", "GOVERNANCE_PRIMARY"]
ROLE_HEADERS = {"evidence role", "evidence_role", "role"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def count(rows, header, label):
    hl = [h.strip().lower() for h in header]
    role_idx = [i for i, h in enumerate(hl) if h in ROLE_HEADERS]
    print("== %s header (%d cols): %s" % (label, len(header), header))
    print("  role column candidates:", [(i, header[i]) for i in role_idx])
    ri = role_idx[0]
    first6 = list(range(6))
    any_AF = []
    any_all = []
    pop = []
    blank_other = []
    ragged = Counter(len(r) for r in rows)
    for n, r in enumerate(rows, start=2):
        cells = [(r[i] if i < len(r) else "") for i in range(len(header))]
        if any(c.strip() for c in cells):
            any_all.append(n)
        if any(cells[i].strip() for i in first6):
            any_AF.append(n)
            v = cells[ri].strip()
            if v:
                pop.append((n, v))
            elif any(cells[i].strip() for i in first6 if i != ri):
                blank_other.append(n)
    inl = [(n, v) for n, v in pop if v in OPTS]
    out = [(n, v) for n, v in pop if v not in OPTS]
    cnt = Counter(v for _, v in out)
    res = {"label": label, "role_col_index0": ri, "role_header": header[ri],
           "header_first6": header[:6],
           "data_records": len(rows), "row_length_histogram": dict(ragged),
           "records_with_any_first6": len(any_AF), "records_with_any_cell": len(any_all),
           "populated_roles": len(pop), "in_list": len(inl), "outside_list": len(out),
           "distinct_nonempty_labels": len({v for _, v in pop}),
           "blank_role_but_other_first6_populated": len(blank_other),
           "blank_role_rows": blank_other,
           "in_by_option": {o: sum(1 for _, v in inl if v == o) for o in OPTS},
           "top_outside": cnt.most_common(10),
           "fingerprint_ordinal": sha("\n".join(v for _, v in pop).encode())}
    for k, v in res.items():
        print("  %s: %s" % (k, json.dumps(v, ensure_ascii=False)))
    return res, pop


ESC = re.compile(r"\\([!-/:-@\[-`{-~])")


def md_split(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    parts = re.split(r"(?<!\\)\|", s)
    return [ESC.sub(r"\1", p.strip()) for p in parts]


def main():
    csv_path, md_path = sys.argv[1], sys.argv[2]
    xres = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    b = open(csv_path, "rb").read()
    print("CSV", csv_path, len(b), sha(b), "bom", b[:3] == b"\xef\xbb\xbf",
          "crlf", b.count(b"\r\n"), "physical_lines", b.count(b"\n"))
    rdr = list(csv.reader(io.StringIO(b.decode("utf-8-sig"), newline="")))
    csv_res, csv_pop = count(rdr[1:], rdr[0], "evidence_lineage.csv")

    mb = open(md_path, "rb").read()
    lines = mb.decode("utf-8").split("\n")
    print("MD", md_path, len(mb), sha(mb), "lines", len(lines))
    # find header lines containing Evidence Role
    hits = [i for i, l in enumerate(lines) if l.startswith("|") and "Evidence Role" in l]
    print("  md lines with a table row containing 'Evidence Role':", [h + 1 for h in hits])
    md_res = None
    for h in hits:
        hdr = md_split(lines[h])
        if "Evidence Role" not in hdr:
            continue
        # contiguous table block
        start = h
        while start > 0 and lines[start - 1].startswith("|"):
            start -= 1
        end = h
        while end + 1 < len(lines) and lines[end + 1].startswith("|"):
            end += 1
        print("  table block lines %d..%d (1-based); header line %d" % (start + 1, end + 1, h + 1))
        for j in range(start, h):
            print("   pre-header line %d: %r" % (j + 1, lines[j][:120]))
        data = [md_split(l) for l in lines[h + 1:end + 1]]
        print("  data lines %d..%d count %d" % (h + 2, end + 1, len(data)))
        md_res, md_pop = count(data, hdr, "md_table@line%d" % (h + 1))
        md_res["md_block_lines_1based"] = [start + 1, end + 1]
        md_res["md_header_line_1based"] = h + 1
        md_res["md_data_lines_1based"] = [h + 2, end + 1]
        # Compare MD rows with xlsx 09-18 F sequence (ordinal)
        if xres:
            x0 = [r for r in xres if r["file"].endswith("2026-09-18.xlsx")][0]
            print("  xlsx-0918 populated", x0["populated_roles"])
        md_res["md_records_ending_last_evidence_id"] = data[-1][0] if data else None
    # CSV vs xlsx ordinal comparison
    if xres:
        for r in xres:
            print("  compare csv to", r["file"].rsplit("/", 1)[-1], "csv populated", csv_res["populated_roles"],
                  "xlsx populated", r["populated_roles"])
    json.dump({"csv": csv_res, "md": md_res}, open(csv_path and sys.argv[0].rsplit("/", 1)[0] + "/csv_md_results.json", "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
