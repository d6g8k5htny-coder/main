#!/usr/bin/env python3
"""Merge subagent TSVs (rel<TAB>size<TAB>content_hash) from dbx/agentout into dbx/content_hashes.json.
A row is accepted only if the hash is 64 hex characters, the path is in inventory.tsv and the size equals the listed size
(a missing size is stored as null); rejected rows are counted."""
import glob, json, os, re
S='<SCRATCH>'
OUT=S+'/dbx/content_hashes.json'
inv={}
for l in open(S+'/dbx/inventory.tsv',encoding='utf-8'):
    c=l.rstrip('\n').split('\t')
    if len(c)>=6 and c[5]=='file': inv[c[0]]=int(c[2] or 0)
out=json.load(open(OUT)) if os.path.exists(OUT) else {}
new=bad=0
for p in sorted(glob.glob(S+'/dbx/agentout/*.tsv')):
    for line in open(p,encoding='utf-8'):
        c=line.rstrip('\n').split('\t')
        if len(c)!=3 or not re.fullmatch(r'[0-9a-f]{64}',c[2]): bad+=1; continue
        rel=c[0].split('//',1)[1] if c[0].startswith('ns:') else c[0]
        if rel not in inv: bad+=1; continue
        size=int(c[1]) if c[1].isdigit() else None
        if size is not None and size!=inv[rel]: bad+=1; continue
        if rel not in out: new+=1
        out[rel]={'size':size,'content_hash':c[2]}
json.dump(out, open(OUT,'w'), indent=0, sort_keys=True, ensure_ascii=False)
print('new',new,'rejected',bad,'total',len(out))
