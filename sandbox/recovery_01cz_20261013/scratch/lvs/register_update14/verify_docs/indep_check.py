"""Independent verification of the round-14 docs binding (read-only; not derived from the author's assertions)."""
import json, re, subprocess, sys, difflib
REPO = '<SCRATCH>/repos/math_r5'
S = 'specifications/lean_statements_20261010'
P = 'specifications/lean_statement_proofs_20261011'
def show(rev, path):
    return subprocess.run(['git','-C',REPO,'show',f'{rev}:{path}'],capture_output=True,check=True).stdout.decode('utf-8')
def wt(path): return open(f'{REPO}/{path}',encoding='utf-8').read()
DOCS = [f'{S}/README.md', f'{S}/SCOPE.md', f'{P}/README.md', f'{P}/SCOPE.md']
old = {d: show('9e305ea5', d) for d in DOCS}
r5 = {d: show('4afcca6a', d) for d in DOCS}
new = {d: wt(d) for d in DOCS}
reg = json.loads(wt(f'{S}/REGISTER.json'))
oreg = json.loads(show('9e305ea5', f'{S}/REGISTER.json'))
# run -> (head8, merge8)
RUNS = {'38116277597':('00d2ac6e','c4c66bb8'),'38121713840':('910809d6','5b198699'),'38122386459':('d9b8dd33','866136cc'),
        '38124262242':('79731b96','318ff194'),'38149888699':('0dcb0016','4363b692'),'38149524759':('88c21058','3306233b'),
        '38150363063':('12e072d6','86a6e8c8'),'38180705952':('48bfcbae','9833d6d9'),'38187538744':('be014794','79d3964c'),
        '38195810463':('c6ce0256','1320c5b2'),'38277458954':('9e305ea5','1376175c')}
NEWRUN='38277458954'
fails=[]
def units(text):
    """paragraphs; table rows split into cells"""
    out=[]
    for para in text.split('\n\n'):
        lines=para.split('\n')
        if all(l.startswith('|') for l in lines if l.strip()) and lines and lines[0].startswith('|'):
            for i,l in enumerate(lines):
                cells=re.split(r'(?<!\\)\|', l)[1:-1]
                for j,c in enumerate(cells): out.append((f'table-line{i} cell{j}', c))
        else:
            out.append(('para', para))
    return out
for d in DOCS:
    t=new[d]
    print(f'== {d}: daggers old={old[d].count("†")} new={t.count("†")} r5={r5[d].count("†")}')
    if '†' in t: fails.append((d,'dagger left'))
    for kind,u in units(t):
        for run in sorted(set(re.findall(r'\b38\d{9}\b',u))):
            if run=='38096827443': continue
            if run not in RUNS: fails.append((d,'unknown run',run,u[:100])); continue
            h,m=RUNS[run]
            if h not in u or m not in u:
                # report but compare with whether same unit existed in old
                existed = any(u==uu for _,uu in units(old[d]))
                fails.append((d,f'C422 {run} head/merge missing in {kind}', 'pre-existing' if existed else 'NEW', u[:200]))
        if NEWRUN in u:
            hascav = 'only outside both packets' in u
            if 'same tree' in u and NEWRUN in u:
                # check 'same tree' not attached to new run: crude - print context
                for mm in re.finditer('same tree', u):
                    ctx=u[max(0,mm.start()-200):mm.end()+50]
                    if NEWRUN in ctx or '9e305ea5' in ctx or '1376175c' in ctx:
                        fails.append((d,'same-tree near new run',ctx))
            if 'a0323b94' not in u: fails.append((d,'base a0323b94 missing',kind,u[:150]))
            print(f'   {kind:22s} run11 x{u.count(NEWRUN)} caveat={hascav} len={len(u)} :: ...{u[u.index(NEWRUN)-80:u.index(NEWRUN)+60]!r}')
    # counts
    for pat in ['77 of the 141','78 theorems','357','160','14 modules','75 theorems','74 of the 141','76 of','theorem_count 78','covered_props 77']:
        print(f'   count {pat!r}: old={old[d].count(pat)} new={t.count(pat)}')
print('FAILS:')
for f in fails: print('  ',f)
