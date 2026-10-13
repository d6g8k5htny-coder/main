import json, re, subprocess
R = '<SCRATCH>/repos/math_r5'
S = 'specifications/lean_statements_20261010'
show = lambda rev, p: subprocess.run(['git', '-C', R, 'show', f'{rev}:{p}'], capture_output=True, check=True).stdout.decode()
reg = json.load(open(f'{R}/{S}/REGISTER.json', encoding='utf-8'))
by = {c['claim_id']: c for c in reg['claims']}
esc = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')
for name in ('SCOPE.md', 'README.md'):
    t = open(f'{R}/{S}/{name}', encoding='utf-8').read()
    seen = {}
    for ln, line in enumerate(t.split('\n'), 1):
        m = re.match(r'^\| `([^`]+)` \| ', line)
        if not m or m.group(1) not in by:
            continue
        cid = m.group(1)
        cells = re.split(r'(?<!\\)\|', line)[1:-1]
        cells = [c[1:-1] if c.startswith(' ') and c.endswith(' ') else c for c in cells]
        seen[cid] = seen.get(cid, 0) + 1
        c = by[cid]
        if name == 'SCOPE.md':
            if len(cells) != 7:
                seen[cid] -= 1; continue
            for i, f in ((5, 'not_established'), (6, 'next_step')):
                if cells[i] != esc(c[f]):
                    print('MISMATCH', name, ln, cid, f)
            if cells[1] != c['kind'] or cells[3] != f"`{c['formal_progress']}`":
                print('KIND/FP MISMATCH', cid)
    print(name, 'claim rows (7-cell in SCOPE)', sum(1 for v in seen.values() if v > 0), 'dups', [k for k, v in seen.items() if v > 1])
# changed register leaves vs committed, and whether each changed not_established/next_step appears verbatim in SCOPE
old = json.loads(show('9e305ea5', f'{S}/REGISTER.json'))
oby = {c['claim_id']: c for c in old['claims']}
t = open(f'{R}/{S}/SCOPE.md', encoding='utf-8').read()
for cid in by:
    for f in ('not_established', 'next_step'):
        if by[cid][f] != oby[cid][f]:
            print('changed', cid, f, 'verbatim cell count in SCOPE:', t.count('| ' + esc(by[cid][f]) + ' |'))
