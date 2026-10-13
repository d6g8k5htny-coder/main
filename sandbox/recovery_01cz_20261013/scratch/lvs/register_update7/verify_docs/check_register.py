#!/usr/bin/env python3
"""verify_docs: register counts and HEAD-vs-worktree structural diff. Scientific effect: NONE. Read-only."""
import json, re, subprocess, collections
from pathlib import Path
M = Path('<SCRATCH>/repos/math')
REG = 'specifications/lean_statements_20261010/REGISTER.json'
head = json.loads(subprocess.check_output(['git','-C',str(M),'show','HEAD:'+REG]))
wt = json.load(open(M/REG))
PROOF_PKG = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
def entries(reg):
    out=[]
    for c in reg['claims']:
        for s in c['statements']: out.append((c['claim_id'], s))
    return out
E = entries(wt); EH = entries(head)
print('statements total', len(E), '(HEAD', len(EH), ')')
print('by formal_progress', collections.Counter(s['formal_progress'] for _,s in E))
print('by package', collections.Counter(s['package'] for _,s in E))
proof = [(c,s) for c,s in E if s['package']==PROOF_PKG]
print('proof-packet entries', len(proof), 'distinct theorems', len({s['declaration'] for _,s in proof}))
props = set()
for c,s in proof:
    pass
print('claims', len(wt['claims']), collections.Counter(c['formal_progress'] for c in wt['claims']))
# claim formal_progress unchanged
hc = {c['claim_id']:c for c in head['claims']}
changed_fields = collections.Counter(); rows_changed = []
for c in wt['claims']:
    h = hc[c['claim_id']]
    diff = [k for k in set(c)|set(h) if c.get(k)!=h.get(k)]
    if diff:
        rows_changed.append((c['claim_id'], sorted(diff)))
        for k in diff: changed_fields[k]+=1
    assert c['formal_progress']==h['formal_progress'], c['claim_id']
    assert c['disposition_transcribed']==h['disposition_transcribed'], c['claim_id']
print('rows changed', len(rows_changed), 'fields', dict(changed_fields))
print([r for r in rows_changed if 'statements' in r[1] or 'kernel_evidence' in r[1] or 'not_established' in r[1]])
print('claim formal_progress and disposition unchanged on all 171 rows: OK')
# new statement entries
newE = [(c,s) for c,s in E if (c,json.dumps(s,sort_keys=True)) not in {(c2,json.dumps(s2,sort_keys=True)) for c2,s2 in EH}]
print('new/changed entries', len(newE))
for c,s in newE: print('  ', c, s['module'], s['declaration'].split('.')[-1], s['formal_progress'], 'anchor:', s['anchor'][:60])
removed = [(c,s) for c,s in EH if (c,json.dumps(s,sort_keys=True)) not in {(c2,json.dumps(s2,sort_keys=True)) for c2,s2 in E}]
print('removed entries', len(removed))
# kernel_evidence
ke_rows = [(c['claim_id'], c['kernel_evidence']) for c in wt['claims'] if c['kernel_evidence']]
print('kernel_evidence rows', len(ke_rows), 'entries', sum(len(k) for _,k in ke_rows))
for cid, kes in ke_rows:
    for ke in kes:
        if ke.get('package','').endswith('lean_statement_proofs_20261011') or 'proofs' in json.dumps(ke):
            pass
# print the keys of one kernel_evidence entry and the new OA row
for c in wt['claims']:
    if c['claim_id']=='OA-C6-RARE-CLUSTER-20260929-v1':
        print(json.dumps(c, indent=1, ensure_ascii=False)[:6000])
# run-id binding across all string fields
MERGE = {'38116277597':'c4c66bb8','38121713840':'5b198699','38122386459':'866136cc','38124262242':'318ff194',
         '38149888699':'4363b692','38149524759':'3306233b','38150363063':'86a6e8c8'}
unbound=[]
def walk(o, path):
    if isinstance(o, dict):
        for k,v in o.items(): walk(v, path+'/'+k)
    elif isinstance(o, list):
        for i,v in enumerate(o): walk(v, path+f'[{i}]')
    elif isinstance(o, str):
        for run in set(re.findall(r'\b(38\d{9})\b', o)):
            if run in MERGE:
                if MERGE[run] not in o: unbound.append((path, run, o[:120]))
            else: unbound.append((path, run, 'UNKNOWN '+o[:120]))
walk(wt, '')
print('unbound run mentions in register strings:', len(unbound))
for u in unbound[:20]: print('  ', u)
# vocabulary in changed string fields
voc = collections.Counter()
for c in wt['claims']:
    h = hc[c['claim_id']]
    for k in ('not_established','next_step'):
        if c[k]!=h[k]:
            for w in ['kernel-checked','closed','accepted','independent','proved']:
                voc[w]+=len(re.findall(r'\b'+w+r'\b', c[k]))
    if c['kernel_evidence']!=h['kernel_evidence']:
        s=json.dumps(c['kernel_evidence'])
        for w in ['kernel-checked','closed','accepted','independent','proved']:
            voc[w]+=len(re.findall(r'\b'+w+r'\b', s))
print('vocabulary counts in changed register text:', dict(voc))
