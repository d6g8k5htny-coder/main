#!/usr/bin/env python3
"""Classify every hashed file outside Github/ by content_hash: GIT_EXACT (a fetched Git blob of main or Math-), ARCHIVE_EXACT (member of a2136bc0 or c4065c67),
GITHUB_TREE_COPY (same bytes as a Dropbox Github/ file, main#594 table), ROOT_DUP (same bytes as another file outside Github/), ROOT_ONLY; EMPTY; NOT_HASHED
(personal or other material, counted only). Ledger name join by the main#594 rule, labelled STRONG / WEAK_ADJ / WEAK_PAIR / WEAK_LONE."""
import json, glob, re, collections, hashlib
S='<SCRATCH>'; D=S+'/dbxroot'
inv=[]
for l in open(D+'/inventory_root.tsv',encoding='utf-8'):
    c=l.rstrip('\n').split('\t')
    if c[0]=='path': continue
    inv.append({'path':c[0],'name':c[1],'size':int(c[2]),'client_modified':c[3],'folder':c[4],'personal':c[5]=='Y','hashable':c[6]=='Y'})
byp={r['path']:r for r in inv}
hashes={}; bad=[]
for fn in sorted(glob.glob(D+'/parts/hash_*.tsv')):
    for l in open(fn,encoding='utf-8'):
        c=l.rstrip('\n').split('\t')
        if len(c)<3 or not re.fullmatch(r'[0-9a-f]{64}',c[2]): bad.append(('malformed',fn,l[:60])); continue
        p,sz,h=c[0],c[1],c[2]
        if p not in byp: bad.append(('not_in_inventory',fn,p[-40:])); continue
        if not sz.isdigit(): bad.append(('size_missing',fn,p[-40:])); continue   # a hash row without a byte count is rejected, not accepted
        if int(sz)!=byp[p]['size']: bad.append(('size_mismatch',fn,p[-40:])); continue
        if p in hashes and hashes[p]!=h: bad.append(('conflict',fn,p[-40:])); continue
        hashes[p]=h
git=json.load(open(S+'/dbx/idx/git_index.json')); git_by_dbx=collections.defaultdict(list)
for k,v in git.items(): git_by_dbx[v['dbx']].append(v)
tip=json.load(open(S+'/dbx/idx/tip_blobs.json'))
arch=collections.defaultdict(list)
for a in ('a2136bc0','c4065c67'):
    for k,v in json.load(open(f'{S}/dbx/idx/{a}_index.json')).items(): arch[v['content_hash']].append(k)
gh=json.load(open(S+'/dbx/content_hashes.json')); gh_by_ch=collections.defaultdict(list)
for p,v in gh.items(): gh_by_ch[v['content_hash']].append(p)
EMPTY=hashlib.sha256(b'').hexdigest()
count=collections.Counter(hashes.values())
for r in inv:
    h=hashes.get(r['path']); r['content_hash']=h
    if r['size']==0: r['tier']='EMPTY'; r['flags']={}; continue
    if not h: r['tier']='NOT_HASHED' if not r['hashable'] else 'PENDING_HASH'; r['flags']={}; continue
    f={'in_git':bool(git_by_dbx.get(h)),'in_archive':bool(arch.get(h)),'in_github_tree':bool(gh_by_ch.get(h)),'root_copies':count[h]}
    if f['in_git']:
        ents=git_by_dbx[h]; f['git_blobs']=sorted({f"{e['repo']}:{e['blob']}" for e in ents})[:3]; f['git_paths']=sorted({f"{e['repo']}:{p}" for e in ents for p in e['paths']})[:4]
        allpaths=[p for e in ents for p in e['paths']]   # the drive/ test runs over every recorded path of the matching blobs, not the truncated locator list
        f['on_tip']=any(f"{e['repo']}:{e['blob']}" in tip for e in ents); f['git_where']='TIP' if f['on_tip'] else ('DRIVE_MIRROR_BRANCH' if allpaths and all(p.startswith('drive/') for p in allpaths) else 'BRANCH_OR_HISTORY')
    if f['in_archive']: f['archive_members']=arch[h][:3]
    r['tier']='GIT_EXACT' if f['in_git'] else 'ARCHIVE_EXACT' if f['in_archive'] else 'GITHUB_TREE_COPY' if f['in_github_tree'] else 'ROOT_DUP' if count[h]>1 else 'ROOT_ONLY'
    r['flags']=f
json.dump(inv,open(D+'/root_rows.json','w'),indent=0); json.dump({'accepted':len(hashes),'rejected':collections.Counter(b[0] for b in bad),'rejected_samples':[b[0] for b in bad[:5]]},open(D+'/hash_receipt.json','w'),indent=1)
print('files',len(inv),'hashed',len(hashes),'rejected',len(bad),dict(collections.Counter(b[0] for b in bad)))
print('tiers:',dict(collections.Counter(r['tier'] for r in inv).most_common()))
print('by folder x tier:'); agg=collections.Counter((r['folder'][:22],r['tier']) for r in inv)
for (fo,t),n in sorted(agg.items()): print('  %-24s %-18s %d'%(fo,t,n))
hashed=[r for r in inv if r.get('content_hash') and r['size']>0]
print('bytes: ROOT_ONLY',sum(r['size'] for r in hashed if r['tier']=='ROOT_ONLY'),'ROOT_DUP',sum(r['size'] for r in hashed if r['tier']=='ROOT_DUP'),'GITHUB_TREE_COPY',sum(r['size'] for r in hashed if r['tier']=='GITHUB_TREE_COPY'))
