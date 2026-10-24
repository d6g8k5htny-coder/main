import sys, zipfile, re, html
import xml.etree.ElementTree as ET
M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PR = '{http://schemas.openxmlformats.org/package/2006/relationships}'
def si_text(si):
    out = []
    for ch in si:
        if ch.tag == M+'t': out.append(ch.text or '')
        elif ch.tag == M+'r':
            for t in ch.findall(M+'t'): out.append(t.text or '')
    return ''.join(out)
def cn(s):
    n = 0
    for ch in s: n = n*26 + ord(ch)-64
    return n
z = zipfile.ZipFile(sys.argv[1])
wb = ET.fromstring(z.read('xl/workbook.xml'))
rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')).findall(PR+'Relationship')}
sis = [si_text(si) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(M+'si')]
for s in wb.iter(M+'sheet'):
    part = 'xl/' + rels[s.get(R+'id')]
    b = z.read(part)
    if b'<dataValidation ' not in b: continue
    sh = ET.fromstring(b)
    cells = {}
    for c in sh.iter(M+'c'):
        t = c.get('t', 'n'); v = c.find(M+'v'); val = None
        if t == 's' and v is not None: val = sis[int(v.text)]
        elif t == 'b' and v is not None: val = 'TRUE' if v.text == '1' else 'FALSE'
        elif t == 'inlineStr' and c.find(M+'is') is not None: val = si_text(c.find(M+'is'))
        elif v is not None: val = v.text
        cells[c.get('r')] = val
    for dv in sh.iter(M+'dataValidation'):
        f1 = dv.find(M+'formula1').text or ''
        attrs = {k: v for k, v in dv.attrib.items() if k != 'sqref'}
        if not f1.startswith('"'):
            print(s.get('name'), dv.get('sqref'), 'NONLITERAL', f1, attrs); continue
        opts = [o.strip() for o in f1.strip('"').split(',')]
        tot = pin = 0; distinct = set()
        for rng in dv.get('sqref').split():
            m = re.match(r'([A-Z]+)(\d+):([A-Z]+)(\d+)$', rng.replace('$', ''))
            c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            for ref, val in cells.items():
                mm = re.match(r'([A-Z]+)(\d+)$', ref)
                if cn(c1) <= cn(mm.group(1)) <= cn(c2) and r1 <= int(mm.group(2)) <= r2 and val is not None and val.strip():
                    tot += 1; distinct.add(val.strip()); pin += val.strip() in opts
        print('%s!%s opts=%d populated=%d in=%d out=%d distinct=%d attrs=%s' % (s.get('name'), dv.get('sqref'), len(opts), tot, pin, tot-pin, len(distinct), attrs))
