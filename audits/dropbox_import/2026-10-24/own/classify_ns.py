#!/usr/bin/env python3
"""Phase 1: name+size classification of the Dropbox Github/ inventory against Git blobs (main all refs, Math- main)
and the two reassembled archives. Writes dbx/phase1.tsv. content_hash is added later (phase 2)."""
import json, collections, os, sys
S='<SCRATCH>'
git=json.load(open(S+'/dbx/idx/git_index.json'))
ns=collections.defaultdict(list)
for k,v in git.items():
    for p in v['paths']: ns[(os.path.basename(p).lower(), v['size'])].append(f"{v['repo']}:{p}")
for lab in ('a2136bc0','c4065c67'):
    for k,v in json.load(open(f'{S}/dbx/idx/{lab}_index.json')).items():
        ns[(os.path.basename(k.split(':',1)[1]).lower(), v['size'])].append(k)
rows=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/inventory.tsv',encoding='utf-8')]
files=[r for r in rows if r[5]=='file']
out=[]; summ=collections.Counter(); bytes_=collections.Counter(); top=collections.defaultdict(collections.Counter)
SKIP_EXT=('.pyc','.url','.ds_store')
for path,name,size,cm,mt,ot in files:
    size=int(size or 0); key=(name.lower(),size)
    folder=path.split('/')[1] if path.count('/')>=2 else '(Github root)'
    if name.lower().endswith(SKIP_EXT) or '__pycache__' in path: st='SKIP'
    elif key in ns: st='NAME_SIZE'
    else: st='NO_NAME_SIZE_MATCH'
    summ[st]+=1; bytes_[st]+=size; top[folder][st]+=1
    out.append((path,name,str(size),cm,st,';'.join(ns.get(key,[])[:3])))
with open(S+'/dbx/phase1.tsv','w',encoding='utf-8') as f:
    for r in out: f.write('\t'.join(r)+'\n')
print('files',len(files)); print({k:(v,bytes_[k]) for k,v in summ.items()})
for folder,c in sorted(top.items(), key=lambda x:-sum(x[1].values()))[:25]: print('%-55s %s'%(folder[:55], dict(c)))
