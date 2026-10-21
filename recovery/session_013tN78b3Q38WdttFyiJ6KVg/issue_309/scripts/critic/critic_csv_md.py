import sys, csv, json, re, hashlib, collections, zipfile
OPTS = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION","SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]
csvp, mdp, xjson, x0918 = sys.argv[1:5]
def stats(vals):
    pop = [v.strip() for v in vals if v.strip()]
    inl = [v for v in pop if v in OPTS]
    return {'populated': len(pop), 'in': len(inl), 'out': len(pop)-len(inl), 'distinct': len(set(pop)),
            'per_option': {o: inl.count(o) for o in OPTS if inl.count(o)}}
b = open(csvp, 'rb').read()
print('csv sha256', hashlib.sha256(b).hexdigest(), 'bom', b[:3] == b'\xef\xbb\xbf', 'crlf', b.count(b'\r\n'))
rows = list(csv.reader(open(csvp, encoding='utf-8', newline='')))
hdr = rows[0]; ri = hdr.index('Evidence Role')
print('csv header idx', ri, 'records', len(rows)-1, 'widths', collections.Counter(len(r) for r in rows))
recs = rows[1:]
print('csv stats', stats([r[ri] for r in recs]))
for ln, r in enumerate(rows, start=1):
    if ln > 1 and r[ri].strip() in ('GOVERNANCE_PRIMARY', 'EXACT_DERIVATION'):
        print('csv line', ln, r[0], r[ri], 'G=', r[6])
x = json.load(open(xjson))[0]
xr = [x['_rowsAP'][k] for k in sorted(x['_rowsAP'], key=int)]
mism = sum(1 for a, c in zip(xr, recs) for i in range(16) if (a[i] or '').strip() != c[i].strip())
print('csv vs xlsx0918 nonempty rows: len', len(xr), len(recs), 'cell mismatches (stripped)', mism)
# G representation diffs
gd = collections.Counter((a[6], c[6]) for a, c in zip(xr, recs))
print('G xlsx->csv pairs', gd.most_common(8))
# markdown
L = open(mdp, encoding='utf-8').read().split('\n')
hdr_lines = [i+1 for i, l in enumerate(L) if 'Evidence Role' in l and l.startswith('|')]
print('md lines with Evidence Role header-ish', hdr_lines[:10])
def cells(line):
    parts = re.split(r'(?<!\\)\|', line)
    parts = parts[1:-1] if line.strip().startswith('|') else parts
    return [re.sub(r'\\([!-/:-@\[-`{-~])', r'\1', p).strip() for p in parts]
for h in hdr_lines:
    c = cells(L[h-1])
    if len(c) == 16 and c[0] == 'Evidence ID':
        print('md header line', h, 'prev lines:', repr(L[h-3][:60]), repr(L[h-2][:60]))
        j = h; data = []
        while j < len(L) and L[j].startswith('|'):
            data.append((j+1, cells(L[j]))); j += 1
        print('md data lines', data[0][0], '..', data[-1][0], 'n', len(data), 'next line', repr(L[j][:40]), 'widths', collections.Counter(len(d) for _, d in data))
        ne = [(ln, d) for ln, d in data if any(x.strip() for x in d[:6])]
        print('md nonempty', len(ne), 'empty', len(data)-len(ne))
        print('md stats', stats([d[5] for _, d in ne]))
        mm = 0
        for k, (ln, d) in enumerate(data):
            row = k + 2
            xv = x['_rowsAP'].get(str(row))
            xf = (xv[5] if xv else None) or ''
            if d[5] != xf.strip(): mm += 1
        print('md F vs xlsx F2..F%d mismatches' % (len(data)+1), mm, 'last A', data[-1][1][0])
