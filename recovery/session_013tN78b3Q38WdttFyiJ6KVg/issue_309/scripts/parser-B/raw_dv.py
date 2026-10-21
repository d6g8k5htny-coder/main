import sys, zipfile, re, hashlib
member = sys.argv[1]
for p in sys.argv[2:]:
    print("=== ", p)
    z = zipfile.ZipFile(p)
    b = z.read(member)
    print(member, "bytes", len(b), "sha256", hashlib.sha256(b).hexdigest())
    x = b.decode('utf-8')
    m = re.search(r'<dataValidations\b[^>]*>', x)
    print("dataValidations open tag:", m.group(0) if m else None)
    dvs = re.findall(r'<dataValidation\b[^>]*?(?:/>|>.*?</dataValidation>)', x, re.S)
    print("n <dataValidation>:", len(dvs))
    for d in dvs:
        print("  ", d[:1500])
    print("extLst present:", '<extLst' in x, " x14:dataValidation count:", len(re.findall(r'<x14:dataValidation\b', x)))
    for e in re.findall(r'<x14:dataValidations\b.*?</x14:dataValidations>', x, re.S):
        print("  X14:", e[:3000])
    print("dimension:", re.search(r'<dimension\b[^>]*/>', x).group(0) if re.search(r'<dimension\b[^>]*/>', x) else None)
    print("other ext tags:", sorted(set(re.findall(r'<(x1[0-9a-z]*:[A-Za-z]+)', x)))[:40])
    print("conditionalFormatting count:", len(re.findall(r'<conditionalFormatting\b', x)))
    print("mergeCells:", re.search(r'<mergeCells\b[^>]*>', x).group(0) if re.search(r'<mergeCells\b[^>]*>', x) else None)
