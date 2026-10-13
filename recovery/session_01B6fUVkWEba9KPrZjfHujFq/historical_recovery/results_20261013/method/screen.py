import sys, re, json, glob, os
S = sys.argv[1]; files = sys.argv[2:]
vals = []
for p in glob.glob(f'{S}/dbx/*sensitive_values*.json') + glob.glob(f'{S}/dbx/pii_values*'):
    try: d = json.load(open(p))
    except Exception: continue
    def walk(x, path):
        if isinstance(x, str):
            if len(x) >= 5 and not (x.isalpha() and x.islower()): vals.append((os.path.basename(p) + path, x))
        elif isinstance(x, dict): [walk(v, path + '/' + str(k)) for k, v in x.items()]
        elif isinstance(x, list): [walk(v, path + f'[{i}]') for i, v in enumerate(x)]
    walk(d, '')
pats = {
 'drive_id': re.compile(r'(?<![0-9A-Za-z_-])1[A-Za-z0-9_-]{24,43}(?![0-9A-Za-z_-])'),
 'drive_url': re.compile(r'(drive|docs)\.google\.com', re.I),
 'dropbox_id': re.compile(r'id:[A-Za-z0-9_-]{10,}'),
 'dropbox_url': re.compile(r'dropbox(usercontent)?\.com', re.I),
 'scratch_path': re.compile(r'/tmp/|scratchpad|/root/'),
 'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}'),
 'phone': re.compile(r'\b\d{3}[-. ]\d{3}[-. ]\d{4}\b'),
}
bad = 0
for f in files:
    t = open(f, errors='replace').read()
    for k, p in pats.items():
        hits = p.findall(t)
        if k == 'drive_id':  # exclude 40/64-hex and pure-hex tokens (git/sha ids)
            hits = [h for h in hits if not re.fullmatch(r'[0-9a-f]+', h)]
        if hits: bad += len(hits); print(os.path.basename(f), k, len(hits), [h[:12] + '…' for h in hits[:5]])
    sv = [path for path, v in vals if v in t]
    if sv: bad += len(sv); print(os.path.basename(f), 'private_value_categories', len(sv), sorted(set(s.split(':')[0].split('/')[1] if '/' in s else s for s in sv))[:8])
print('TOTAL_FLAGS', bad)
