"""Independent leaf-by-leaf diff of REGISTER.json (committed 9e305ea5 vs worktree). Read-only."""
import json, sys, collections
V = sys.argv[1]
old = json.load(open(f'{V}/REGISTER.committed.json'))
new = json.load(open(f'{V}/REGISTER.new.json'))

diffs = []
def walk(a, b, path):
    if type(a) != type(b):
        diffs.append(('TYPE', path, a, b)); return
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            diffs.append(('KEYS', path, list(a.keys()), list(b.keys())))
        for k in a:
            if k in b: walk(a[k], b[k], path + [k])
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(('LEN', path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i])
    else:
        if a != b:
            diffs.append(('LEAF', path, a, b))
walk(old, new, [])
print('top-level keys', list(old.keys()))
print('n diffs', len(diffs), collections.Counter(d[0] for d in diffs))
rows = collections.OrderedDict()
for kind, path, a, b in diffs:
    assert kind == 'LEAF', (kind, path)
    assert path[0] == 'claims', path
    cid = old['claims'][path[1]].get('id') or old['claims'][path[1]].get('claim_id')
    rows.setdefault((path[1], cid), []).append(path[2:])
for k, v in rows.items():
    print(k, v)
json.dump([{'path': p, 'old': a, 'new': b} for _, p, a, b in diffs], open(f'{V}/changed_leaves2.json', 'w'), indent=1, ensure_ascii=False)
# serialization
raw = open(f'{V}/REGISTER.new.json', encoding='utf-8').read()
print('serialization reproduces:', json.dumps(new, indent=2, ensure_ascii=False) + '\n' == raw)
