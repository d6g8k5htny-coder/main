import sys, zipfile, re
p, member = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(p)
s = z.read(member).decode('utf-8')
print(len(s))
print(s[:2500])
print("....")
i = s.find('</sheetData>')
print(s[i:i+6000])
for tag in ['dataValidation','x14:dataValidation','extLst','mergeCell','<f>','<f ','t="inlineStr"','t="str"','t="s"','t="b"','t="n"','t="e"','<is>']:
    print(tag, s.count(tag))
