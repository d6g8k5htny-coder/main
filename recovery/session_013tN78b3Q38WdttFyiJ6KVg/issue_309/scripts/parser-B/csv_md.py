"""parser-B: Evidence Role counts in registers/csv/evidence_lineage.csv and in the
Evidence Lineage table of the 2026-09-17 markdown export. Usage: python3 -I csv_md.py CSV MD [XLSX_0918]"""
import sys, csv, re, collections, hashlib, html, zipfile
LIST11 = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION",
          "SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]
ne = lambda v: v is not None and v.strip() != ""
def report(name, header, rows, rownums):
    ci = header.index("Evidence Role")
    print(f"== {name}: role column index {ci} (letter {chr(65+ci)}), header[{ci}]={header[ci]!r}, ncols={len(header)}")
    any_af = [n for n, r in zip(rownums, rows) if any(ne(c) for c in r[:6])]
    pop = [(n, r[ci]) for n, r in zip(rownums, rows) if ne(r[ci])]
    blank_other = [n for n, r in zip(rownums, rows) if not ne(r[ci]) and any(ne(c) for c in r[:6])]
    lab = collections.Counter(v.strip() for _, v in pop)
    inl = sum(n for k, n in lab.items() if k in LIST11)
    raw_in = sum(1 for _, v in pop if v in LIST11)
    out = collections.Counter({k: n for k, n in lab.items() if k not in LIST11})
    print(f"   data rows={len(rows)} rows_any_A_F={len(any_af)} populated={len(pop)} in_list={inl} (raw-exact {raw_in}) outside={len(pop)-inl} distinct={len(lab)} blank_role_other_A_F={len(blank_other)} {blank_other[:10]}")
    print("   in-list by label:", {k: lab.get(k, 0) for k in LIST11})
    print("   top10 outside:", out.most_common(10))
    print("   lead/trail ws labels:", [v for _, v in pop if v != v.strip()][:5])
    return pop, lab

# CSV
cb = open(sys.argv[1], "rb").read()
print("CSV sha256", hashlib.sha256(cb).hexdigest(), "bytes", len(cb), "BOM", cb[:3] == b"\xef\xbb\xbf")
rd = list(csv.reader(cb.decode("utf-8-sig").splitlines(keepends=True) and __import__("io").StringIO(cb.decode("utf-8-sig"), newline="")))
header, body = rd[0], rd[1:]
print("CSV records incl header:", len(rd), "row widths:", collections.Counter(len(r) for r in rd))
csv_pop, csv_lab = report("CSV", header, body, list(range(2, len(body) + 2)))

# MD
mb = open(sys.argv[2], "rb").read()
print("MD sha256", hashlib.sha256(mb).hexdigest(), "bytes", len(mb))
lines = mb.decode("utf-8").split("\n")
hdr_idx = [i for i, l in enumerate(lines) if l.startswith("| Evidence ID | Recorded UTC |")]
print("MD lines starting with the Evidence Lineage header:", [i + 1 for i in hdr_idx])
def split_md(l):
    l = l.strip()
    assert l.startswith("|") and l.endswith("|")
    cells = re.split(r"(?<!\\)\|", l[1:-1])
    return [re.sub(r"\\(.)", r"\1", c.strip()) for c in cells]
md_pop = None
for i in hdr_idx:
    j = i + 1
    rows = []
    while j < len(lines) and lines[j].startswith("|"):
        rows.append(split_md(lines[j])); j += 1
    print(f"MD table header at line {i+1}; preceding lines {i-1}..{i}: {lines[i-2][:40]!r} / {lines[i-1][:40]!r}; data lines {i+2}..{j} ({len(rows)} lines); next line {j+1}: {lines[j][:40]!r}")
    print("   widths:", collections.Counter(len(r) for r in rows))
    hdr = split_md(lines[i])
    md_pop, md_lab = report("MD Evidence Lineage", hdr, rows, list(range(i + 2, j + 1)))
    md_rows = rows

# Compare with xlsx 09-18 rows (raw-regex) if given
if len(sys.argv) > 3:
    z = zipfile.ZipFile(sys.argv[3])
    def txt(f): return "".join(html.unescape(t) for t in re.findall(r"<t\b[^>]*>(.*?)</t>", re.sub(r"<rPh\b.*?</rPh>", "", f, flags=re.S), re.S))
    sst = [txt(si) for si in re.findall(r"<si\b[^>]*>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
    x = z.read("xl/worksheets/sheet22.xml").decode()
    F = {}
    for m in re.finditer(r'<c\b[^>]*?\br="F(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', x, re.S):
        r = int(m.group(1)); body = m.group(3) or ""
        v = re.search(r"<v>(.*?)</v>", body)
        if v and 't="s"' in m.group(0)[:200]: F[r] = sst[int(v.group(1))]
    # md data line k (0-based) -> xlsx row k+2
    mism = [(k + 2, r[5], F.get(k + 2)) for k, r in enumerate(md_rows) if (r[5] or None) != F.get(k + 2)]
    print("MD role vs xlsx 09-18 F (md data line k -> xlsx row k+2): mismatches", len(mism), mism[:10])
    print("MD covers xlsx rows 2..", len(md_rows) + 1)
    # CSV vs xlsx 09-18: csv record order vs xlsx non-blank rows
    xrows = sorted(r for r in F if r >= 2 and F[r].strip())
    cvals = [v for _, v in csv_pop]
    xvals = [F[r] for r in xrows]
    print("CSV populated roles == xlsx 09-18 non-blank F in order:", cvals == xvals, len(cvals), len(xvals))
