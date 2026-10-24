import sys, zipfile, re
for p in sys.argv[1:]:
    z = zipfile.ZipFile(p)
    print("==", p)
    for i in z.infolist():
        print(f"  {i.filename} {i.file_size}")
    wb = z.read('xl/workbook.xml').decode('utf-8')
    print(wb[:3000])
    print(z.read('xl/_rels/workbook.xml.rels').decode('utf-8')[:4000])
