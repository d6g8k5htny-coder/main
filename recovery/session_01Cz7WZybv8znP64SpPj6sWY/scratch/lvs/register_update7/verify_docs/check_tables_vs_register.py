#!/usr/bin/env python3
"""verify_docs: statement-packet README/SCOPE tables vs REGISTER.json and the proof inventory; proof README table vs proved set.
Scientific effect: NONE. Read-only."""
import json, re, collections
from pathlib import Path
M = Path('<SCRATCH>/repos/math')
OUT = Path('<SCRATCH>/lvs/register_update7/verify_docs')
wt = json.load(open(M/'specifications/lean_statements_20261010/REGISTER.json'))
PROOF_PKG = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
log = (OUT/'proof_check.log').read_text()
inv = json.loads(log[log.index('{'):log.rindex('}')+1])
thms = inv['theorems']
assert inv['theorem_count']==73==len(thms)
proves = collections.defaultdict(list)  # Prop short -> [Module.thm]
for t in thms:
    mod = t['module'].split('/')[-1][:-5]; short = t['name'].split('.')[-1]
    proves[t['proves'].split('.')[-1]].append(f'{mod}.{short}')
print('inventory: covered props', len(proves), 'dups', [k for k,v in proves.items() if len(v)>1])
NEWMODS = {'Side24Analysis','Side24Interfaces','RNCluster'}
new13 = {f"{t['module'].split('/')[-1][:-5]}.{t['name'].split('.')[-1]}" for t in thms if t['module'].split('/')[-1][:-5] in NEWMODS}
assert len(new13)==13
def cells(row):
    parts = re.split(r'(?<!\\)\|', row.strip())
    return [p.strip() for p in parts[1:-1]]
for docname in ['README.md','SCOPE.md']:
    txt = (M/'specifications/lean_statements_20261010'/docname).read_text()
    # declaration tables: rows with 5 cells where cell[1] in kinds
    probs=[]; seen=set(); rows=0
    for line in txt.split('\n'):
        if not line.startswith('|'): continue
        c = cells(line)
        if len(c)==5 and c[1] in ('def_prop','structure'):
            rows+=1
            decl = c[0].strip('`'); seen.add(decl)
            col = c[4]
            if col=='—':
                if decl in proves: probs.append(('missing proof col', decl, proves[decl]))
            else:
                names = re.findall(r'`([A-Za-z0-9]+\.[A-Za-z0-9]+)(†?)`', col)
                got = {n for n,_ in names}
                exp = set(proves.get(decl, []))
                if got != exp: probs.append(('proof col mismatch', decl, sorted(got), sorted(exp)))
                for n,d in names:
                    if (d=='†') != (n in new13): probs.append(('dagger wrong', decl, n, d))
    print(f'{docname}: declaration rows {rows}, distinct {len(seen)}, problems {probs}')
    # claims table: rows with 6 cells where cell[1] is a kind
    kinds = {'landing_claim','graph_node','open_obligation','proof_index_result','existing_lean_package','main_experiment_proof','exact_certificate'}
    cprobs=[]; crow=0
    reg = {c['claim_id']:c for c in wt['claims']}
    for line in txt.split('\n'):
        if not line.startswith('|'): continue
        c = cells(line)
        if len(c)==6 and c[1] in kinds:
            crow+=1; cid=c[0].strip('`'); r=reg[cid]
            if c[3].strip('`')!=r['formal_progress']: cprobs.append(('fp', cid, c[3], r['formal_progress']))
            m = re.search(r'(\d+) proved \(proof packet\): (.*?)(?:$|; \d+ proved)', c[5])
            regproof = [s['declaration'].split('.',3)[-1] for s in r['statements'] if s['package']==PROOF_PKG]
            if m:
                n=int(m.group(1)); names=re.findall(r'`([A-Za-z0-9]+\.[A-Za-z0-9]+)(†?)`', m.group(2))
                if n!=len(names) or n!=len(regproof) or sorted(x for x,_ in names)!=sorted(regproof): cprobs.append(('proof list', cid, n, len(names), len(regproof)))
                for nme,d in names:
                    if (d=='†') != (nme in new13): cprobs.append(('dagger wrong', cid, nme, d))
            elif regproof: cprobs.append(('no proof list but register has', cid, regproof))
            m2 = re.search(r'(\d+) specified', c[5])
            nloc = len([s for s in r['statements'] if s['package'].endswith('lean_statements_20261010')])
            if nloc and (not m2 or int(m2.group(1))!=nloc): cprobs.append(('specified count', cid, c[5][:60], nloc))
            if not nloc and c[5]!='—' and not re.search('proved', c[5]): cprobs.append(('nonempty for empty', cid, c[5][:60]))
    print(f'{docname}: claim rows {crow}, problems {cprobs}')
# proof packet README table
txt = (M/'specifications/lean_statement_proofs_20261011/README.md').read_text()
proved_short = set(proves)
inrows=[]; both=[]; cnt=0
for line in txt.split('\n'):
    if not line.startswith('|'): continue
    c = cells(line)
    if len(c)==3 and c[0] not in ('group','---'):
        names = re.findall(r'`([A-Za-z0-9]+)`', c[1])
        for n in names:
            if n in proved_short: both.append((c[0], n))
        inrows.append((c[0], len(names), c[1]=='—'))
        cnt += len(names)
print('proof README table rows:', inrows)
print('names listed as without theorem but proved:', both)
print('explicit names in col 2 total:', cnt)
# full set of Props never proved (69)
allprops = set()
for c in wt['claims']:
    for s in c['statements']:
        if s['package'].endswith('lean_statements_20261010') and s['kind']=='def_prop': allprops.add(s['declaration'].split('.')[-1])
print('distinct def_prop in register', len(allprops), 'unproved', len(allprops-proved_short))
