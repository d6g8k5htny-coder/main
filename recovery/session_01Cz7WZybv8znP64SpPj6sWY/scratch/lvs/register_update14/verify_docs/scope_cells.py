import json, re, subprocess
REPO = '<SCRATCH>/repos/math_r5'
S = 'specifications/lean_statements_20261010'
def show(rev, path):
    return subprocess.run(['git','-C',REPO,'show',f'{rev}:{path}'],capture_output=True,check=True).stdout.decode('utf-8')
reg = json.load(open(f'{REPO}/{S}/REGISTER.json',encoding='utf-8'))
oreg = json.loads(show('9e305ea5', f'{S}/REGISTER.json'))
def cell(s): return str(s).replace('|','\\|').replace('\n',' ')
by = {c['claim_id']: c for c in reg['claims']}
oby = {c['claim_id']: c for c in oreg['claims']}
for name in ('SCOPE.md','README.md'):
    t = open(f'{REPO}/{S}/{name}',encoding='utf-8').read()
    ot = show('9e305ea5', f'{S}/{name}')
    n_rows=0; bad=0
    for line in t.split('\n'):
        m = re.match(r'^\| `([^`]+)` \| (\w+) \|', line)
        if not m or m.group(1) not in by: continue
        cid = m.group(1); c = by[cid]
        cells = [x.strip() for x in re.split(r'(?<!\\)\|', line)[1:-1]]
        if name=='SCOPE.md':
            if len(cells)!=7: print('BAD COLS', cid, len(cells)); bad+=1; continue
            n_rows+=1
            if cells[5] != cell(c['not_established']).strip(): print('NE MISMATCH', cid); bad+=1
            if cells[6] != cell(c['next_step']).strip(): print('NS MISMATCH', cid); bad+=1
            if cells[3] != f"`{c['formal_progress']}`": print('FP MISMATCH', cid); bad+=1
            if cells[1] != c['kind']: print('KIND MISMATCH', cid); bad+=1
        else:
            n_rows+=1
            if len(cells)!=6: print('BAD COLS README', cid, len(cells)); bad+=1
            if cells[3] != f"`{c['formal_progress']}`": print('FP MISMATCH', cid); bad+=1
    print(name, 'claim rows checked', n_rows, 'bad', bad)
# also check exact substring presence of the escaped cells of the changed rows
t = open(f'{REPO}/{S}/SCOPE.md',encoding='utf-8').read()
for cid in ('side24-coefficient','math.side24-coefficient'):
    for f in ('not_established','next_step'):
        print(cid, f, 'escaped-verbatim-in-SCOPE:', t.count('| '+cell(by[cid][f])+' |'), 'changed-vs-old:', by[cid][f]!=oby[cid][f])
# changed register rows / fields
for cid in by:
    o, n = oby[cid], by[cid]
    if o != n:
        fields = [k for k in n if n[k]!=o.get(k)]
        print('changed row', cid, fields)
