import sys, re, hashlib, os, itertools, collections
p = sys.argv[1]; target = int(sys.argv[2]); declared = sys.argv[3]
raw = open(p, 'rb').read().decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
lines = raw.split('\n')
b = [i for i, l in enumerate(lines) if re.fullmatch(r'BEGIN_[A-Z0-9_]*FROZEN_BODY', l)][0]
e = [i for i, l in enumerate(lines) if re.fullmatch(r'END_[A-Z0-9_]*FROZEN_BODY', l)][0]
body = '\n'.join(lines[b+1:e]).rstrip('\n') + '\n'
na = collections.Counter(c for c in body if ord(c) > 127)
print('non-ascii:', {hex(ord(k)): v for k, v in na.items()})
print('utf8 bytes', len(body.encode()), 'target', target, 'diff', len(body.encode()) - target)
# hypotheses
cands = {}
cands['collapse_triple_blank'] = re.sub(r'\n\n\n+', '\n\n', body)
cands['nbsp_to_space'] = body.replace(' ', ' ')
cands['rstrip_lines'] = '\n'.join(l.rstrip() for l in body.split('\n'))
cands['emdash_to_--'] = body.replace('—', '--').replace('–', '--')
for k, v in list(cands.items()):
    for k2, f in [('+collapse', lambda s: re.sub(r'\n\n\n+', '\n\n', s))]:
        cands[k + k2] = f(v)
for k, v in cands.items():
    h = hashlib.sha256(v.encode()).hexdigest()
    print(k, len(v.encode()), h[:12], 'MATCH' if h.startswith(declared) else '')
