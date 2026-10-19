"""C8's independent recomputation of the 50/14/2 classification for main#451 (R1 check d).

Reads the rerun record directory (argv[1]) and a directory holding the packet's committed
results at main 0a7e5145 (argv[2]); applies the ten regex substitutions transcribed from
comparison/timing_mask_rules.txt; reports per-file classes and totals. Standard library only.
"""
import hashlib, json, os, re, sys

REC, PKT = sys.argv[1], sys.argv[2]
RULES = [
    (r'(setup )[0-9.]+s', r'\1<T>s'),
    (r'(zeta zeros in )[0-9.]+s', r'\1<T>s'),
    (r'^time [0-9.eE+-]+\s*$', r'time <T>'),
    (r'(scan time )[0-9.eE+-]+', r'\1<T>'),
    (r'( time )[0-9.eE+-]+$', r'\1<T>'),
    (r'("seconds(?:_zero_finding|_total)?": )[0-9.eE+-]+', r'\1<T>'),
    (r'("sec_per_1e5_evals_at_9e5": )[0-9.eE+-]+', r'\1<T>'),
    (r'\([0-9.]+ ?s\)', r'(<T>s)'),
    (r'(samples in )[0-9.]+s', r'\1<T>s'),
    (r'^(init )[0-9.]+( N \d+)$', r'\1<T>\2'),
]
def mask(text):
    out = []
    for line in text.split('\n'):
        for pat, rep in RULES:
            line = re.sub(pat, rep, line)
        out.append(line)
    return '\n'.join(out)
sha = lambda b: hashlib.sha256(b).hexdigest()
H = json.load(open(os.path.join(REC, 'comparison/hashes.json')))
items = H['files']
classes = {}
problems = []
for e in items:
    p = e.get('path') or e.get('file')
    committed = open(os.path.join(PKT, p), 'rb').read()
    ch = e.get('committed_sha256') or e.get('committed') or e.get('original_sha256')
    rh = e.get('rerun_sha256') or e.get('rerun') or e.get('reproduced_sha256')
    if ch != sha(committed):
        problems.append(('committed hash in hashes.json != main bytes', p))
    if ch == rh:
        cls = 'IDENTICAL'
    else:
        o = os.path.join(REC, 'original', p); r = os.path.join(REC, 'reproduced', p)
        ob, rb = open(o, 'rb').read(), open(r, 'rb').read()
        if ob != committed: problems.append(('original/ copy != main bytes', p))
        if sha(rb) != rh: problems.append(('reproduced/ copy != rerun hash', p))
        cls = 'TIMING-ONLY' if mask(ob.decode()) == mask(rb.decode()) else 'DIFFERENT'
    if cls != e.get('class'): problems.append(('class differs from record', p, cls, e.get('class')))
    gitblob = hashlib.sha1(b'blob %d\0' % len(committed) + committed).hexdigest()
    if gitblob != e.get('committed_blob'): problems.append(('committed blob differs', p))
    classes.setdefault(cls, []).append(p)
for k in sorted(classes):
    print(k, len(classes[k]))
print('DIFFERENT:', classes.get('DIFFERENT'))
print('problems:', problems)
