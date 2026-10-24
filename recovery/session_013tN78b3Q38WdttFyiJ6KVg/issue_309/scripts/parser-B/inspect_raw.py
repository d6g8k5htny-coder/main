import sys, zipfile, re, hashlib
for p in sys.argv[1:]:
    print("=== ", p)
    print("sha256", hashlib.sha256(open(p,'rb').read()).hexdigest())
    z = zipfile.ZipFile(p)
    print("members", len(z.namelist()), "testzip", z.testzip())
    wb = z.read('xl/workbook.xml').decode('utf-8')
    rels = z.read('xl/_rels/workbook.xml.rels').decode('utf-8')
    sheets = re.findall(r'<sheet\b[^>]*/>', wb)
    print("n sheets", len(sheets))
    for s in sheets:
        if 'Lineage' in s or 'lineage' in s:
            print("SHEET", s)
            rid = re.search(r'r:id="([^"]+)"', s).group(1)
            m = [r for r in re.findall(r'<Relationship\b[^>]*/>', rels) if f'Id="{rid}"' in r]
            print("REL", m)
    # definedNames
    dn = re.findall(r'<definedName\b[^>]*>.*?</definedName>', wb, re.S)
    print("definedNames", len(dn))
    for d in dn[:50]: print("  ", d[:300])
