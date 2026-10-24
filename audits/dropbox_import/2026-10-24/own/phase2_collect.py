#!/usr/bin/env python3
"""Merge subagent TSVs (rel<TAB>size<TAB>content_hash) from dbx/agentout into dbx/content_hashes.json.
A row is accepted only if the hash is 64 hex characters, the path is in inventory.tsv and the size equals the listed size
(a missing size is accepted only for a listed size of 0 with the empty-content hash).  A path observed twice with different
hash or size is a CONFLICT: both observations are dropped and the path is listed in dbx/hash_conflicts.json, so the
classification fails closed.  dbx/hash_receipt.json counts observations, agreements and rejections (no paths)."""
import glob, json, os, re
S='<SCRATCH>'
OUT=S+'/dbx/content_hashes.json'
inv={}
for l in open(S+'/dbx/inventory.tsv',encoding='utf-8'):
    c=l.rstrip('\n').split('\t')
    if len(c)>=6 and c[5]=='file': inv[c[0]]=int(c[2] or 0)
EMPTY='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
prev=json.load(open(OUT)) if os.path.exists(OUT) else {}
obs={}; rej={'malformed':0,'not_in_listing':0,'size_mismatch':0,'size_missing':0}; agree=0
def observe(rel,size,h,src):
    if rel in obs and (obs[rel][0]!=h or obs[rel][1]!=size): conflicts.add(rel); return
    if rel in obs: 
        global agree; agree+=1
    obs[rel]=(h,size,src)
conflicts=set()
for rel,v in prev.items():   # observations harvested from the transcript (phase2_harvest.py) are observations too
    observe(rel,v.get('size'),v['content_hash'],'harvest')
for p in sorted(glob.glob(S+'/dbx/agentout/*.tsv')):
    for line in open(p,encoding='utf-8'):
        c=line.rstrip('\n').split('\t')
        if len(c)!=3 or not re.fullmatch(r'[0-9a-f]{64}',c[2]): rej['malformed']+=1; continue
        rel=c[0].split('//',1)[1] if c[0].startswith('ns:') else c[0]
        if rel not in inv: rej['not_in_listing']+=1; continue
        size=int(c[1]) if c[1].isdigit() else None
        if size is None:
            if inv[rel]==0 and c[2]==EMPTY: size=0
            else: rej['size_missing']+=1; continue
        if size!=inv[rel]: rej['size_mismatch']+=1; continue
        observe(rel,size,c[2],os.path.basename(p))
out={rel:{'size':s,'content_hash':h} for rel,(h,s,src) in obs.items() if rel not in conflicts and s is not None and s==inv.get(rel)}
json.dump(out, open(OUT,'w'), indent=0, sort_keys=True, ensure_ascii=False)
json.dump(sorted(conflicts), open(S+'/dbx/hash_conflicts.json','w'), indent=0, ensure_ascii=False)
receipt={'listing':'2026-10-24T18:31Z (11 pages)','observations_accepted':len(out),'repeat_observations_agreeing':agree,'conflicting_paths_dropped':len(conflicts),'rejected':rej,
         'rule':'hash observations are tied to the listing by path and byte count only; the connector returns no modified time or revision with link metadata, so a same-size edit between the listing and the hash read (18:31Z to 19:45Z) is not excluded by this receipt'}
json.dump(receipt, open(S+'/dbx/hash_receipt.json','w'), indent=1)
print('accepted',len(out),'agree',agree,'conflicts',len(conflicts),'rejected',rej)
