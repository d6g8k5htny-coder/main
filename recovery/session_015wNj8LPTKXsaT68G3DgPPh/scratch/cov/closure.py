import ast, re, sys
src_v = open('theorem_v.py').read(); src_b = open('theorem_b.py').read()
roots = set(re.findall(r'\b(?:E2|E3|M|ENG\[[a-z0-9]+\])\.([A-Za-z_]\w*)', src_v + src_b))
for eng in ('engine_e2.py', 'engine_e3.py'):
    tree = ast.parse(open(eng).read())
    defs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name): defs[t.id] = n
    def refs(node):
        return {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and x.id in defs} | \
               {x.attr for x in ast.walk(node) if isinstance(x, ast.Attribute) and x.attr in defs}
    seen, todo = set(), [r for r in roots if r in defs]
    while todo:
        f = todo.pop()
        if f in seen: continue
        seen.add(f); todo += list(refs(defs[f]) - seen)
    fns = sorted((defs[f].lineno, f, type(defs[f]).__name__) for f in seen if not isinstance(defs[f], ast.Assign))
    consts = sorted(f for f in seen if isinstance(defs[f], ast.Assign))
    allfns = [n for n in defs if not isinstance(defs[n], ast.Assign)]
    print('==', eng, 'used callables %d of %d' % (len(fns), len(allfns)))
    for ln, f, k in fns: print('  %5d %s%s' % (ln, f, ' (class)' if k == 'ClassDef' else ''))
    print('  constants:', ', '.join(consts))
    print('  unused:', ', '.join(sorted(set(allfns) - seen)))
