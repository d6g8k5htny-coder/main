#!/usr/bin/env python3
"""parser-A cross-check: regex/iterparse second path over the same members, and row-level F diff.
Usage: python3 -I crosscheck_regex.py XLSX [XLSX ...]
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
OPTS = ["PRIMARY_DERIVATION", "EXECUTABLE_RECEIPT", "RAW_DATA", "FORMAL_PROOF",
        "INDEPENDENT_RECONSTRUCTION", "SUMMARY", "NAVIGATION", "BALLOT", "REGISTER_COPY",
        "REVIEW_OF_REVIEW", "GOVERNANCE_PRIMARY"]


def sst_iter(z):
    out = []
    for ev, el in ET.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
        if el.tag == M + "si":
            # itertext includes rPh text; check rPh presence separately
            if el.find(M + "rPh") is not None:
                raise SystemExit("rPh present; itertext path invalid")
            out.append("".join(el.itertext()))
            el.clear()
    return out


def fcol(z, member, sst):
    s = z.read(member).decode("utf-8")
    vals = {}
    for m in re.finditer(r'<c r="([A-Z]+)(\d+)"([^>]*?)(/>|>(.*?)</c>)', s, re.S):
        col, row, attrs, _, body = m.groups()
        if col not in ("A", "B", "C", "D", "E", "F"):
            continue
        body = body or ""
        t = re.search(r'\bt="([^"]+)"', attrs)
        t = t.group(1) if t else None
        v = re.search(r"<v>(.*?)</v>", body, re.S)
        if t == "s" and v:
            val = sst[int(v.group(1))]
        elif t == "inlineStr":
            val = "".join(re.findall(r"<t[^>]*>(.*?)</t>", body, re.S))
        elif v:
            val = v.group(1)
        else:
            val = ""
        vals[(col, int(row))] = val
    return vals


def main():
    per = {}
    for p in sys.argv[1:]:
        z = zipfile.ZipFile(p)
        sst = sst_iter(z)
        vals = fcol(z, "xl/worksheets/sheet22.xml", sst)
        rows = sorted({r for (c, r) in vals if r >= 2 and vals[(c, r)].strip()})
        F = {r: vals.get(("F", r), "") for r in rows}
        pop = {r: v.strip() for r, v in F.items() if v.strip()}
        inl = sum(1 for v in pop.values() if v in OPTS)
        name = p.rsplit("/", 1)[-1]
        print(name, "rows_any_A_F", len(rows), "populated_F", len(pop), "in", inl,
              "out", len(pop) - inl, "distinct", len(set(pop.values())))
        per[name] = (pop, vals)
    names = list(per)
    for a, b in zip(names, names[1:]):
        pa, va = per[a]
        pb, vb = per[b]
        changed = [r for r in pa if r in pb and pa[r] != pb[r]]
        added = [r for r in pb if r not in pa]
        removed = [r for r in pa if r not in pb]
        # Evidence ID (col A) stability for common rows
        idchg = [r for r in pa if r in pb and va.get(("A", r)) != vb.get(("A", r))]
        print("DIFF %s -> %s: F changed rows %s; added rows %d %s; removed %s; colA changed %s" % (
            a, b, changed, len(added), added[:3] + ["..."] + added[-3:] if len(added) > 6 else added, removed, idchg))
        print("  added F values:", Counter(pb[r] for r in added).most_common())
        for r in added:
            print("   row %d A=%r F=%r" % (r, vb.get(("A", r)), pb[r]))


if __name__ == "__main__":
    main()
