"""parser-B: byte offset / line of the dataValidations element in the Evidence Lineage part."""
import sys, zipfile
for p in sys.argv[1:]:
    b = zipfile.ZipFile(p).read("xl/worksheets/sheet22.xml")
    i = b.find(b"<dataValidations"); j = b.find(b"</dataValidations>")
    print(p.split("/")[-1], "lines in member:", b.count(b"\n") + 1, "dataValidations byte span:", i, j + len(b"</dataValidations>"), "line:", b[:i].count(b"\n") + 1)
