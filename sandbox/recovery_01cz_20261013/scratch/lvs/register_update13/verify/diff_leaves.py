"""Independent leaf-by-leaf diff of REGISTER.json: committed blob at 4afcca6a vs worktree.
Read-only. Prints every changed/added/removed leaf path with old and new values."""
import json, subprocess, sys
REPO = sys.argv[1]
P = 'specifications/lean_statements_20261010/REGISTER.json'
old = json.loads(subprocess.check_output(['git', '-C', REPO, 'show', '4afcca6a:' + P]))
new = json.load(open(REPO + '/' + P))

def leaves(o, path=''):
    if isinstance(o, dict):
        if not o:
            yield path, o
        for k, v in o.items():
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(o, list):
        if not o:
            yield path, o
        for i, v in enumerate(o):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, o

lo = dict(leaves(old)); ln = dict(leaves(new))
out = {'changed': [], 'added': [], 'removed': []}
for k in lo:
    if k not in ln:
        out['removed'].append((k, lo[k]))
    elif lo[k] != ln[k] or type(lo[k]) != type(ln[k]):
        out['changed'].append((k, lo[k], ln[k]))
for k in ln:
    if k not in lo:
        out['added'].append((k, ln[k]))
json.dump(out, open(sys.argv[2], 'w'), indent=1, ensure_ascii=False)
print('changed', len(out['changed']), 'added', len(out['added']), 'removed', len(out['removed']))
for k, *_ in out['changed']: print('C', k)
for k, _ in out['added']: print('A', k)
for k, _ in out['removed']: print('R', k)
# key order check
def keyorder(o, path=''):
    if isinstance(o, dict):
        yield path, list(o.keys())
        for k, v in o.items(): yield from keyorder(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from keyorder(v, f'{path}[{i}]')
ko = dict(keyorder(old)); kn = dict(keyorder(new))
for k in ko:
    if k in kn and ko[k] != kn[k]: print('KEYORDER', k, ko[k], kn[k])
