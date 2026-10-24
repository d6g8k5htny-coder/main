#!/usr/bin/env python3
"""parser-A: align CSV records and MD table rows with the 2026-09-18 xlsx Evidence Lineage A:F.
Usage: python3 -I compare_sources.py XLSX_0918 CSV MD
"""
import csv
import io
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ESC = re.compile(r"\\([!-/:-@\[-`{-~])")


def sst(z):
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall(M + "si"):
        parts = []
        for ch in si:
            if ch.tag == M + "t":
                parts.append(ch.text or "")
            elif ch.tag == M + "r":
                parts += [(t.text or "") for t in ch.findall(M + "t")]
        out.append("".join(parts))
    return out


def col(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def xlsx_rows(p):
    z = zipfile.ZipFile(p)
    s = sst(z)
    root = ET.fromstring(z.read("xl/worksheets/sheet22.xml"))
    rows = {}
    for row in root.find(M + "sheetData").findall(M + "row"):
        r = int(row.get("r"))
        d = {}
        for c in row.findall(M + "c"):
            k = col(c.get("r"))
            if k > 16:
                continue
            v = c.find(M + "v")
            t = c.get("t")
            if v is None:
                val = ""
            elif t == "s":
                val = s[int(v.text)]
            elif t == "b":
                val = "TRUE" if v.text == "1" else "FALSE"
            else:
                val = v.text or ""
            d[k] = val
        rows[r] = [d.get(i, "") for i in range(1, 17)]
    return rows


def md_split(line):
    s = line.strip()[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [ESC.sub(r"\1", p.strip()) for p in re.split(r"(?<!\\)\|", s)]


def main():
    xr = xlsx_rows(sys.argv[1])
    nonempty = [r for r in sorted(xr) if r >= 2 and any(x.strip() for x in xr[r][:6])]
    b = open(sys.argv[2], "rb").read().decode("utf-8")
    cr = list(csv.reader(io.StringIO(b, newline="")))[1:]
    # CSV record i  <->  i-th non-empty xlsx row
    mism6 = []
    mism16 = []
    for i, rec in enumerate(cr):
        r = nonempty[i]
        if [x.strip() for x in rec[:6]] != [x.strip() for x in xr[r][:6]]:
            mism6.append((i + 2, r))
        if [x.strip() for x in rec[:16]] != [x.strip() for x in xr[r][:16]]:
            mism16.append((i + 2, r, [j for j in range(16) if rec[j].strip() != xr[r][j].strip()]))
    print("CSV records", len(cr), "xlsx0918 nonempty rows", len(nonempty))
    print("CSV vs xlsx A:F mismatches (csv_line, xlsx_row):", mism6[:20], "count", len(mism6))
    print("CSV vs xlsx A:P mismatches count", len(mism16), "first", mism16[:5])
    lines = open(sys.argv[3], encoding="utf-8").read().split("\n")
    data = [md_split(l) for l in lines[1779:1933]]  # 1-based lines 1780..1933
    mm = []
    for i, d in enumerate(data):
        r = i + 2
        if [x.strip() for x in d[:6]] != [x.strip() for x in xr.get(r, [""] * 16)[:6]]:
            mm.append((1780 + i, r, d[0], xr.get(r, [""])[0], d[5], xr.get(r, ["", "", "", "", "", ""])[5]))
    print("MD rows", len(data), "-> xlsx rows 2..%d; A:F mismatches %d" % (len(data) + 1, len(mm)))
    for m in mm[:20]:
        print("  ", m)
    fmm = [m for m in mm if m[4].strip() != m[5].strip()]
    print("MD vs xlsx F-only mismatches", len(fmm), fmm[:10])
    print("MD last data line 1933 Evidence ID:", data[-1][0], "| xlsx A155:", xr[155][0], "| xlsx A156:", xr[156][0])
    print("MD line 1934 repr:", repr(lines[1933][:80]), "line 1935:", repr(lines[1934][:80]), "line 1937:", repr(lines[1936][:120]))


if __name__ == "__main__":
    main()
