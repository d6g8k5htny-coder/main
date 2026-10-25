#!/usr/bin/env python3
"""Select files needing content_hash (phase 2) and write ns-path batches of 25 for download_link.
Priority: math subtrees first, then everything else; skips SKIP tier. Usage: phase2_batches.py <ns_id>"""
import sys, json
S='<SCRATCH>'
ns=sys.argv[1]
rows=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/phase1.tsv',encoding='utf-8')]
done=set(json.load(open(S+'/dbx/content_hashes.json')).keys()) if __import__('os').path.exists(S+'/dbx/content_hashes.json') else set()
MATH=('K3_SIDE24_LB','UPPER2D','LPW_','(withheld)','64_FORMAL','Q0_','rn-sector','(withheld)','research_archive','T2','master-final')
def prio(r):
    folder=r[0].split('/')[1] if r[0].count('/')>=2 else ''
    return 0 if any(folder.startswith(m) for m in MATH) else 1
todo=[r for r in rows if r[4]!='SKIP' and r[0] not in done]
todo.sort(key=lambda r:(prio(r), r[0]))
batches=[[f'ns:{ns}//'+r[0] for r in todo[i:i+25]] for i in range(0,len(todo),25)]
json.dump(batches, open(S+'/dbx/phase2_batches.json','w'))
print('todo',len(todo),'batches',len(batches),'math-first',sum(1 for r in todo if prio(r)==0))
