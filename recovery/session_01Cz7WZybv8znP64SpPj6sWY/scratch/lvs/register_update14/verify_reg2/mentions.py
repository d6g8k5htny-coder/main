import json, re
o=json.load(open('REGISTER.committed.json')); n=json.load(open('REGISTER.new.json'))
R='38277458954'; H='9e305ea5'; M='1376175ce5536a6ac74cb03706d18e65fdb695fa'; B='a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc'
CAV=', the tested merge differing from the head only outside both packets'
TREE_CORE="differs from its head only outside both packets (main's changes from PRs #428 and #440: 13 files under companions/torus_filtration_20261011/ and .github/workflows/torus-filtration.yml; specifications/lean_statements_20261010 and specifications/lean_statement_proofs_20261011 are byte-identical between head and merge)"
def leaves(x, p=()):
    if isinstance(x, dict):
        for k,v in x.items(): yield from leaves(v, p+(k,))
    elif isinstance(x, list):
        for i,v in enumerate(x): yield from leaves(v, p+(i,))
    elif isinstance(x, str): yield p, x
tot=0; bad=[]
for p, s in leaves(n):
    if R not in s: continue
    # split into sentence units: '. ' followed by capital/backtick, or end
    units = re.split(r'(?<=[.)])\s+(?=[A-Z`])', s)
    for u in units:
        c = u.count(R)
        if not c: continue
        tot += c
        ok = (H in u) and (M in u) and (B in u) and (CAV in u or TREE_CORE in u) and ('same tree' not in u)
        # finer: semicolon-level clauses
        if not ok: bad.append((p, u[:300]))
    # per-occurrence: window from occurrence to next '. ' sentence boundary or end must contain caveat or tree
    for m in re.finditer(R, s):
        start = s.rfind('. ', 0, m.start()); start = 0 if start<0 else start
        end = s.find('. ', m.end()); end = len(s) if end<0 else end
        w = s[start:end]
print('total run mentions', tot)
print('bad units', len(bad))
for b in bad: print(b)
