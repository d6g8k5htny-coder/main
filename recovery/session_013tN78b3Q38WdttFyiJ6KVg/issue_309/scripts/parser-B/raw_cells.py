"""parser-B raw-regex cross-check (no ElementTree, no openpyxl). Usage: python3 -I raw_cells.py XLSX..."""
import sys, zipfile, re, html, collections, hashlib
LIST11 = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION",
          "SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]
def txt(fragment):
    # concatenate all <t> runs (plain or rich text), excluding phonetic <rPh> runs
    fragment = re.sub(r"<rPh\b.*?</rPh>", "", fragment, flags=re.S)
    return "".join(html.unescape(t) for t in re.findall(r"<t\b[^>]*>(.*?)</t>", fragment, re.S))
def colnum(letters):
    n = 0
    for ch in letters: n = n*26 + ord(ch)-64
    return n
for p in sys.argv[1:]:
    z = zipfile.ZipFile(p)
    wb = z.read("xl/workbook.xml").decode(); rels = z.read("xl/_rels/workbook.xml.rels").decode()
    s = [x for x in re.findall(r"<sheet\b[^>]*/>", wb) if 'name="Evidence Lineage"' in x][0]
    rid = re.search(r'r:id="([^"]+)"', s).group(1)
    rel = [x for x in re.findall(r"<Relationship\b[^>]*/>", rels) if f'Id="{rid}"' in x][0]
    member = "xl/" + re.search(r'Target="([^"]+)"', rel).group(1)
    sstb = z.read("xl/sharedStrings.xml")
    sst = [txt(si) for si in re.findall(r"<si\b[^>]*>(.*?)</si>", sstb.decode(), re.S)]
    x = z.read(member).decode()
    cells = {}
    types = collections.Counter()
    for m in re.finditer(r"<c\b([^>]*?)(?:/>|>(.*?)</c>)", x, re.S):
        attrs, body = m.group(1), m.group(2) or ""
        ref = re.search(r'\br="([A-Z]+)(\d+)"', attrs)
        col, row = colnum(ref.group(1)), int(ref.group(2))
        t = re.search(r'\bt="([^"]+)"', attrs); t = t.group(1) if t else "n"
        v = re.search(r"<v>(.*?)</v>", body, re.S)
        if t == "s" and v: val = sst[int(v.group(1))]
        elif t == "inlineStr": val = txt(body)
        elif v: val = html.unescape(v.group(1))
        else: val = None
        if "<f" in body: types["formula"] += 1
        if col <= 6: types[(col, t)] += 1
        cells[(row, col)] = val
    ne = lambda v: v is not None and v.strip() != ""
    maxrow = max(r for r, _ in cells)
    rows_any = [r for r in range(2, maxrow+1) if any(ne(cells.get((r, c))) for c in range(1, 7))]
    pop = [(r, cells[(r, 6)]) for r in range(2, maxrow+1) if ne(cells.get((r, 6)))]
    blankF_other = [r for r in range(2, maxrow+1) if not ne(cells.get((r, 6))) and any(ne(cells.get((r, c))) for c in range(1, 6))]
    lastdata = max(rows_any)
    gaps = [r for r in range(2, lastdata+1) if r not in set(rows_any)]
    gaps_with_GP = [r for r in gaps if any(ne(cells.get((r, c))) for c in range(7, 40))]
    lab = collections.Counter(v.strip() for _, v in pop)
    inl = [(r, v) for r, v in pop if v.strip() in LIST11]
    print("===", p.split("/")[-1], hashlib.sha256(open(p,'rb').read()).hexdigest())
    print(" member", member, "sharedStrings sha256", hashlib.sha256(sstb).hexdigest(), "n_sst", len(sst))
    print(" F1 =", repr(cells.get((1, 6))), " A1..P1:", [cells.get((1, c)) for c in range(1, 17)])
    print(" max row with a <c> element:", maxrow, " cell type counts A:F:", dict(types))
    print(" rows_any_A_F", len(rows_any), "populated_F", len(pop), "in", len(inl), "out", len(pop)-len(inl),
          "distinct", len(lab), "blankF_other", len(blankF_other), "last data row", lastdata)
    print(" blank A:F rows inside 2..last:", len(gaps), gaps)
    print(" of those, rows with any G+ content:", gaps_with_GP)
    print(" in-list rows:", inl)
    print(" top10 out:", collections.Counter({k: n for k, n in lab.items() if k not in LIST11}).most_common(10))
    # Non-ASCII / odd characters in F
    odd = [(r, v) for r, v in pop if re.search(r"[^\x20-\x7e]", v)]
    print(" F values with non-printable-ASCII chars:", odd[:10], len(odd))
    # labels by shape
    shape = collections.Counter("UPPER_SNAKE" if re.fullmatch(r"[A-Z0-9_]+", v.strip()) else "other" for _, v in pop)
    print(" label shapes (rows):", dict(shape))
    # Evidence ID uniqueness
    ids = collections.Counter(cells.get((r, 1)) for r in rows_any)
    print(" duplicate Evidence IDs:", [(k, n) for k, n in ids.items() if n > 1][:10])
