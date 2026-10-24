import sys, zipfile, re
import xml.etree.ElementTree as ET
M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PR = '{http://schemas.openxmlformats.org/package/2006/relationships}'
z = zipfile.ZipFile(sys.argv[1])
sx = z.read('xl/worksheets/sheet22.xml')
print('hidden rows', len(re.findall(rb'<row [^>]*hidden="1"', sx)), 'row elems', len(re.findall(rb'<row[ >]', sx)))
fcells = re.findall(rb'<c r="F(\d+)"([^>]*?)(/>|>(.*?)</c>)', sx)
print('F <c> elems', len(fcells), 'with <v>', sum(1 for f in fcells if b'<v>' in (f[3] or b'')), 'rows range', min(int(f[0]) for f in fcells), max(int(f[0]) for f in fcells))
rel = ET.fromstring(z.read('xl/worksheets/_rels/sheet22.xml.rels'))
for r in rel.findall(PR+'Relationship'):
    if 'drawing' in r.get('Type'): print('drawing target', r.get('Target'))
dn = [n for n in z.namelist() if n.startswith('xl/drawings/')]
print('drawing parts', len(dn))
d22 = z.read('xl/drawings/drawing22.xml') if 'xl/drawings/drawing22.xml' in z.namelist() else b''
print('drawing22 bytes', len(d22), 'anchors', d22.count(b'Anchor'), d22[:300])
print('comments parts', [n for n in z.namelist() if 'comment' in n.lower()])
# cell lookups in other sheets
def si_text(si):
    out = []
    for ch in si:
        if ch.tag == M+'t': out.append(ch.text or '')
        elif ch.tag == M+'r':
            for t in ch.findall(M+'t'): out.append(t.text or '')
    return ''.join(out)
sis = [si_text(si) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(M+'si')]
wb = ET.fromstring(z.read('xl/workbook.xml'))
rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')).findall(PR+'Relationship')}
want = {'Config': ['A66', 'F66', 'A217'], 'Definition Registry': ['A9', 'E9'], 'Metadata Schema': ['A59'], 'Operator Decisions': ['A2', 'A3'], 'Transition Register': ['AE1', 'AE3'], 'Run Log': ['A2', 'G2', 'A11', 'G11']}
for s in wb.iter(M+'sheet'):
    if s.get('name') in want:
        sh = ET.fromstring(z.read('xl/' + rels[s.get(R+'id')]))
        for c in sh.iter(M+'c'):
            if c.get('r') in want[s.get('name')]:
                v = c.find(M+'v'); t = c.get('t')
                val = sis[int(v.text)] if (t == 's' and v is not None) else (v.text if v is not None else None)
                print(repr(s.get('name')), c.get('r'), repr((val or '')[:200]))
