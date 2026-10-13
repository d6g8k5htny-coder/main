import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
decl = re.compile(r'^(theorem|abbrev|instance|def|noncomputable def) ([A-Za-z][A-Za-z0-9_]*)', re.M)
bad = 0; n = 0
for m in decl.finditer(text):
    n += 1
    before = text[:m.start()]
    end = before.rstrip()
    if not end.endswith('-/'):
        print('NO DOCSTRING', m.group(2)); bad += 1; continue
    start = before.rfind('/--')
    doc = before[start:]
    if before[start:].count('-/') != 1 or before[start - 1:start] not in ('\n', ''):
        print('ODD DOCSTRING', m.group(2)); bad += 1
    lines = doc[3:-3].strip().split('\n') if doc.endswith('-/\n') else doc.strip()[3:-2].strip().split('\n')
    counts = {k: sum(1 for l in lines if l.strip().startswith(k)) for k in ('Proves:', 'Method:', 'Does not establish:')}
    occ = {k: doc.count(k) for k in ('Proves:', 'Method:', 'Does not establish:')}
    if any(v != 1 for v in counts.values()) or any(v != 1 for v in occ.values()):
        print('FIELD COUNT', m.group(2), counts, occ); bad += 1
    if not m.group(2).startswith('ordinaryH0Control_'):
        print('PREFIX', m.group(2)); bad += 1
print('declarations', n, 'defects', bad)
