#!/usr/bin/env python3
"""verify_docs: generic table parser for statement packet README/SCOPE: 'proved by' columns vs inventory, claims 'statements' vs register.
Scientific effect: NONE. Read-only."""
import json, re, collections
from pathlib import Path
M = Path('<SCRATCH>/repos/math')
OUT = Path('<SCRATCH>/lvs/register_update7/verify_docs')
wt = json.load(open(M/'specifications/lean_statements_20261010/REGISTER.json'))
PROOF_PKG = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
log = (OUT/'proof_check.log').read_text(); inv = json.loads(log[log.index('{'):log.rindex('}')+1])
proves = collections.defaultdict(set)
for t in inv['theorems']:
    proves[(t['statement_module'].split('/')[-1][:-5], t['proves'].split('.')[-1])].add(t['module'].split('/')[-1][:-5]+'.'+t['name'].split('.')[-1])
NEWMODS={'Side24Analysis','Side24Interfaces','RNCluster'}
def cells(row):
    return [p.strip() for p in re.split(r'(?<!\\)\|', row.strip())[1:-1]]
reg = {c['claim_id']:c for c in wt['claims']}
for docname in ['README.md','SCOPE.md']:
    lines=(M/'specifications/lean_statements_20261010'/docname).read_text().split('\n')
    probs=[]; decl_rows=0; claim_rows=0; module=None
    i=0
    while i<len(lines):
        m=re.match(r'^### `Specifications/(\w+)\.lean`', lines[i])
        if m: module=m.group(1)
        if lines[i].startswith('|') and i+1<len(lines) and re.match(r'^\|\s*-{3,}', lines[i+1]):
            hdr=cells(lines[i]); j=i+2
            pcol=next((k for k,h in enumerate(hdr) if h.startswith('proved by')), None)
            dcol=next((k for k,h in enumerate(hdr) if h=='declaration'), None)
            ccol=next((k for k,h in enumerate(hdr) if h=='claim_id'), None)
            scol=next((k for k,h in enumerate(hdr) if h=='statements'), None)
            while j<len(lines) and lines[j].startswith('|'):
                c=cells(lines[j])
                if pcol is not None and dcol is not None:
                    decl_rows+=1; decl=c[dcol].strip('`'); col=c[pcol]
                    exp=proves.get((module,decl), set())
                    names=re.findall(r'`([A-Za-z0-9]+\.[A-Za-z0-9]+)(†?)`', col)
                    got={n for n,_ in names}
                    if col=='—': got=set()
                    if got!=exp: probs.append(('proof col', module, decl, sorted(got), sorted(exp)))
                    for n,d in names:
                        if (d=='†')!=(n.split('.')[0] in NEWMODS): probs.append(('dagger', module, decl, n, d))
                if ccol is not None and scol is not None:
                    claim_rows+=1; cid=c[ccol].strip('`'); r=reg[cid]; col=c[scol]
                    regproof=sorted(s['declaration'].split('.',2)[-1] for s in r['statements'] if s['package']==PROOF_PKG)
                    m2=re.search(r'(\d+) proved \(proof packet\): (.*)$', col)
                    if m2:
                        names=re.findall(r'`([A-Za-z0-9]+\.[A-Za-z0-9]+)(†?)`', m2.group(2))
                        if int(m2.group(1))!=len(names) or sorted(n for n,_ in names)!=regproof: probs.append(('claim proof list', cid, m2.group(1), len(names), len(regproof)))
                        for n,d in names:
                            if (d=='†')!=(n.split('.')[0] in NEWMODS): probs.append(('claim dagger', cid, n, d))
                    elif regproof: probs.append(('claim missing proof list', cid, regproof))
                    fpcol=next((k for k,h in enumerate(hdr) if h=='formal_progress'), None)
                    if fpcol is not None and c[fpcol].strip('`')!=r['formal_progress']: probs.append(('fp', cid, c[fpcol]))
                j+=1
            i=j
        else: i+=1
    print(f'{docname}: declaration rows {decl_rows}, claim rows {claim_rows}, problems: {probs}')
