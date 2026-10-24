"""parser-B: raw-regex scan of all worksheet parts for dataValidation / x14 / mentions of the 11-option list."""
import sys, zipfile, re
for p in sys.argv[1:]:
    z = zipfile.ZipFile(p)
    wb = z.read("xl/workbook.xml").decode(); rels = z.read("xl/_rels/workbook.xml.rels").decode()
    rmap = {re.search(r'Id="([^"]+)"', r).group(1): re.search(r'Target="([^"]+)"', r).group(1) for r in re.findall(r"<Relationship\b[^>]*/>", rels)}
    tot = 0; x14 = 0; hits = []
    for s in re.findall(r"<sheet\b[^>]*/>", wb):
        nm = html_nm = re.search(r'name="([^"]*)"', s).group(1)
        mem = "xl/" + rmap[re.search(r'r:id="([^"]+)"', s).group(1)]
        x = z.read(mem).decode()
        dvs = re.findall(r"<dataValidation\b[^>]*?(?:/>|>.*?</dataValidation>)", x, re.S)
        tot += len(dvs); x14 += len(re.findall(r"<x14:dataValidation\b", x))
        for d in dvs:
            if "PRIMARY_DERIVATION" in d or "Evidence" in d or "GOVERNANCE_PRIMARY" in d:
                hits.append((nm, mem, re.search(r'sqref="([^"]*)"', d).group(1), d[:200]))
    print("==", p.split("/")[-1], "total dataValidation elements:", tot, "x14:", x14)
    for h in hits: print("   ", h)
    # any non-validation occurrence of 'REVIEW_OF_REVIEW' or 'REGISTER_COPY' in shared strings
    sst = z.read("xl/sharedStrings.xml").decode()
    for tok in ["REVIEW_OF_REVIEW", "REGISTER_COPY", "NAVIGATION", "BALLOT"]:
        print("    sharedStrings occurrences of", tok, ":", len(re.findall(tok, sst)))
