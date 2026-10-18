"""Scoping only: for each pin in d1_falsify_v4.py, locate candidate landed bytes by basename and apply the gate's own hash rule."""
import hashlib, re, subprocess, sys, json
G = '/home/user/main'
REFS = {'main': 'origin/main', 'hardening': 'a01c72f1937875b94bfd78e183a6dbbdb2b04da4', 'pr445': sys.argv[1]}
src = open('gate_v4.py').read()
def blocks(name):
    m = re.search(name + r' = \[(.*?)\n\]', src, re.S); return m.group(1) if m else ''
pins = []
for rel, h in re.findall(r'\("([^"]+)",\s*"([0-9a-f]{64})"\)', blocks('WHOLE')): pins.append((rel, h, 'whole'))
for rel, h, mode in re.findall(r'\("([^"]+)",\s*"([0-9a-f]{64})",\s*"(\w+)"\)', blocks('BODY')): pins.append((rel, h, 'body_' + mode))
for rel, h in re.findall(r'\("([^"]+)",\s*"([0-9a-f]{64})"\)', blocks('SEP')): pins.append((rel, h, 'sep'))
for rel, h in re.findall(r'\("([^"]+)",\s*"([0-9a-f]{64})"\)', blocks('BEFORE_HASHLINE')): pins.append((rel, h, 'before_hashline'))
for rel, h in re.findall(r'\("([^"]+)",\s*"([0-9a-f]{64})"\)', blocks('HIST')): pins.append(('D1_assembly/' + rel, h, 'd1_body'))
pins += [('D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md', 'c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470', 'body_strip'),
         ('H3_closure/H3_BAND_FLOOR.md', '281477c39412840bc9ca56a256884f3ddd5f0ae4283f90a017162676771382a7', 'excl_hashline'),
         ('H3_closure/H3_BAND_CEIL.md', 'cfe8a3a49e3281dfdc4ef5d925ae622b4f7aeea3d0462baae61b87dae699ede2', 'excl_hashline')]
def rule(t, kind):
    if kind == 'whole': return t
    if kind.startswith('body_') and kind != 'body_strip':
        a = t.find(b"BEGIN_FROZEN_BODY\n"); b = t.find(b"END_FROZEN_BODY")
        if a < 0 or b < 0: return None
        seg = t[a + len(b"BEGIN_FROZEN_BODY\n"):b]
        return seg if kind == 'body_raw' else seg.rstrip(b"\n") + b"\n"
    if kind == 'body_strip':
        a = t.find(b"BEGIN_FROZEN_BODY\n"); b = t.find(b"END_FROZEN_BODY")
        if a < 0 or b < 0: return None
        return t[a + len(b"BEGIN_FROZEN_BODY\n"):b].strip(b"\n") + b"\n"
    if kind == 'sep':
        i = t.find(b"\n---\n\n"); return t[:i] if i > 0 else None
    if kind == 'before_hashline':
        L = t.split(b"\n"); k = [i for i, l in enumerate(L) if l.startswith(b"body-sha256:")]
        return b"\n".join(L[:k[0]]) if len(k) == 1 else None
    if kind == 'excl_hashline':
        L = t.split(b"\n"); k = [i for i, l in enumerate(L) if b"Body hash of this document" in l]
        return b"\n".join(L[:k[0]] + L[k[0] + 1:]) if len(k) == 1 else None
    if kind == 'd1_body':
        L = t.decode().split("\n"); idx = [i for i, l in enumerate(L) if l in ("BEGIN_FROZEN_BODY", "END_FROZEN_BODY")]
        if len(idx) != 2: return None
        seg = L[idx[0] + 1:idx[1]]
        while seg and seg[0] == "": seg = seg[1:]
        while seg and seg[-1] == "": seg = seg[:-1]
        return ("\n".join(seg) + "\n").encode()
trees = {}
for k, ref in REFS.items():
    out = subprocess.run(['git', '-C', G, 'ls-tree', '-r', ref], capture_output=True).stdout.decode('utf-8', 'surrogateescape')
    trees[k] = [(l.split()[2], l.split('\t', 1)[1]) for l in out.splitlines() if '\t' in l]
res = []
for rel, h, kind in pins:
    base = rel.split('/')[-1]
    cands = []
    for k, items in trees.items():
        for blob, path in items:
            p = path.strip('"')
            if p.endswith('/' + base) or p.endswith('/' + base + '.txt') or p == base:
                cands.append((k, blob, p))
    seen = {}; hits = []
    for k, blob, p in cands:
        if blob not in seen:
            t = subprocess.run(['git', '-C', G, 'cat-file', 'blob', blob], capture_output=True).stdout
            seen[blob] = rule(t, kind)
        b = seen[blob]
        hits.append((k, blob[:8], p[-90:], None if b is None else hashlib.sha256(b).hexdigest() == h))
    ok = [x for x in hits if x[3]]
    res.append({'pin': rel, 'rule': kind, 'sha': h[:8], 'status': 'MATCH' if ok else ('MISMATCH' if hits else 'ABSENT'), 'candidates': hits})
    print(f"{('MATCH' if ok else ('MISMATCH' if hits else 'ABSENT')):8s} {kind:15s} {rel:52s} {h[:8]} " + '; '.join(f"{k}:{b}:{m}" for k, b, _, m in hits[:4]))
json.dump(res, open('pinscan.json', 'w'), indent=1)
print('MATCH', sum(r['status']=='MATCH' for r in res), 'MISMATCH', sum(r['status']=='MISMATCH' for r in res), 'ABSENT', sum(r['status']=='ABSENT' for r in res), 'of', len(res))
