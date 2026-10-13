"""Call-graph closure from theorem_v.compute (the #227 certificate) into theorem_b and the engines (static, name-based, conservative)."""
import ast, re
def load(fn):
    t = ast.parse(open(fn).read()); d = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): d[n.name] = n
        elif isinstance(n, ast.Assign):
            for x in n.targets:
                if isinstance(x, ast.Name): d[x.id] = n
    return d
V, B, E2, E3 = load('theorem_v.py'), load('theorem_b.py'), load('engine_e2.py'), load('engine_e3.py')
mods = {'V': V, 'B': B, 'E2': E2, 'E3': E3}
alias = {'V': {'TB': 'B', 'E2': 'E2', 'E3': 'E3', 'M': 'ENG', 'ENG': 'ENG'},
         'B': {'E2': 'E2', 'E3': 'E3', 'M': 'ENG', 'ENG': 'ENG'}, 'E2': {}, 'E3': {}}
def edges(m, f):
    node = mods[m][f]; out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and x.id in mods[m] and x.id != f: out.add((m, x.id))
        if isinstance(x, ast.Attribute):
            base = x.value
            bn = base.id if isinstance(base, ast.Name) else (base.value.id if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) else None)
            tgt = alias[m].get(bn)
            for t in (['E2', 'E3'] if tgt == 'ENG' else [tgt] if tgt else []):
                if x.attr in mods[t]: out.add((t, x.attr))
            if bn == 'self' or bn is None: pass
    return out
def closure(starts):
    seen, todo = set(), list(starts)
    while todo:
        k = todo.pop()
        if k in seen: continue
        seen.add(k); todo += list(edges(*k) - seen)
    return seen
s = closure([('V', 'compute')])
for m in ('B', 'E2', 'E3'):
    fs = sorted((mods[m][f].lineno, f) for (mm, f) in s if mm == m and not isinstance(mods[m][f], ast.Assign))
    print('==', m, len(fs)); print('  ' + ', '.join('%s:%d' % (f, l) for l, f in fs))

print('\n#### split by path')
def eng(s, m): return {f for (mm, f) in s if mm == m and not isinstance(mods[m][f], ast.Assign)}
# path 1: everything compute needs except eps0_215 (reference-kernel corollary + rate sweeps)
starts1 = [('V', f) for f in ('tv_law', 'tv_summary', 'rate_sweep_v', 'k_weights', 'assemble_v')] + [('B', 'reference')]
s1 = closure(starts1)
s2 = closure([('V', 'eps0_215')])
for m in ('B', 'E2', 'E3'):
    a, b = eng(s1, m), eng(s2, m) - eng(s1, m)
    print(m, 'main path', len(a), ':', ', '.join(sorted(a, key=lambda f: mods[m][f].lineno)))
    print(m, 'eps0-only', len(b), ':', ', '.join(sorted(b, key=lambda f: mods[m][f].lineno)))
# which V functions directly reach which engine names
print('\n#### direct engine names per theorem_v function')
for f in sorted(V, key=lambda f: V[f].lineno):
    if isinstance(V[f], ast.Assign): continue
    e = sorted({'%s.%s' % (m, g) for (m, g) in edges('V', f) if m in ('E2', 'E3', 'B')})
    if e: print(' ', f, '->', ', '.join(e))
