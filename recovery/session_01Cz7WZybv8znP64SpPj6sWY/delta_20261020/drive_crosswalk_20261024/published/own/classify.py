#!/usr/bin/env python3
"""Drive byte-identity crosswalk (local, path-level). Inputs: the 2026-09-17 Drive publication (inventory.jsonl, Archive_Members.csv) and the
18-20 September deltas from the drive/ tree at a01c72f1; the session's Git blob index (sha256 + Dropbox content_hash per blob, all fetched refs
of main and Math-), the Dropbox Github/ content_hash table (main#594), the two archive member indexes; the DRV-H2/DRV-H3 maps on main; the ledger rows.
Rule: Dropbox content_hash of a file of 4 MiB or less = SHA-256 of its SHA-256 (tools/dropbox_reconcile.py); Git and archive matches are by SHA-256 directly."""
import json, csv, hashlib, re, collections, subprocess, sys
S='<SCRATCH>'; G=S+'/gdrive'; FOUR=4*1024*1024
inv=[json.loads(l) for l in open(G+'/baseline_inventory.jsonl')]
deltas=[json.loads(l) for l in open(G+'/baseline_deltas.jsonl')]
members=list(csv.DictReader(open(G+'/Archive_Members.csv',encoding='utf-8')))
git=json.load(open(S+'/dbx/idx/git_index.json')); tip=json.load(open(S+'/dbx/idx/tip_blobs.json'))
git_by_sha=collections.defaultdict(list)
for k,v in git.items(): git_by_sha[v['sha256']].append(v)
dbx=json.load(open(S+'/dbx/content_hashes.json'))     # dropbox path -> {content_hash,size}
dbx_by_ch=collections.defaultdict(list)
for p,v in dbx.items(): dbx_by_ch[v['content_hash']].append(p)
arch={}
for a in ('a2136bc0','c4065c67'):
    for k,v in json.load(open(f'{S}/dbx/idx/{a}_index.json')).items(): arch.setdefault(v['sha256'],[]).append(k)
def drv_csv(path):
    return list(csv.DictReader(subprocess.run(['git','-C','<MAIN_CHECKOUT>','show','origin/main:'+path],capture_output=True,text=True,check=True).stdout.splitlines()))
h2=drv_csv('docs/retro/DRV-H2/DRV-H2.map.csv'); h3=drv_csv('docs/retro/DRV-H3/DRV-H3_drive_map.csv')
pub_titles={r['name'].strip().lower():('DRV-H2',r) for r in h2}
for r in h3: pub_titles.setdefault(r['name'].strip().lower(),('DRV-H3',r))
EMPTY_SHA=hashlib.sha256(b'').hexdigest()
def derive_ch(sha, size=None):
    # Dropbox content_hash of an empty file is sha256 of nothing (the block list is empty); otherwise sha256 of the one block digest for files of 4 MiB or less
    return EMPTY_SHA if (size==0 or sha==EMPTY_SHA) else hashlib.sha256(bytes.fromhex(sha)).hexdigest()
def classify(sha,size):
    f={'in_git':bool(git_by_sha.get(sha)),'in_archive':bool(arch.get(sha)),'in_dbx':None,'on_tip':False,'git_paths':[],'dbx_count':0,'archive_members':[]}
    if f['in_git']:
        ents=git_by_sha[sha]; f['git_paths']=sorted({f"{e['repo']}:{p}" for e in ents for p in e['paths']})[:6]; f['git_blobs']=sorted({f"{e['repo']}:{e['blob']}" for e in ents})[:3]
        f['on_tip']=any(f"{e['repo']}:{e['blob']}" in tip for e in ents)
    if f['in_archive']: f['archive_members']=arch[sha][:4]
    if size is not None and size<=FOUR:
        ch=derive_ch(sha,size); f['in_dbx']=bool(dbx_by_ch.get(ch)); f['dbx_count']=len(dbx_by_ch.get(ch,[])); f['content_hash']=ch
    tier=('GIT_EXACT' if f['in_git'] else 'ARCHIVE_EXACT' if f['in_archive'] else 'DBX_EXACT' if f['in_dbx'] else ('DRIVE_ONLY_DBX_UNKNOWN_OVER_4MIB' if f['in_dbx'] is None else 'DRIVE_ONLY'))
    return tier,f
rows=[]; seen=set()
for r in inv:
    isfold=r['mimeType'].endswith('.folder'); hub=r['path'].split('/')[0]
    rec={'id':r['id'],'title':r['title'],'path':r['path'],'hub':hub,'context':r['context'],'access':r['access_status'],'mime':r['mimeType'],'bytes':r['bytes'],'sha256':r['sha256'],'source':'inventory_20260917','folder':isfold}
    if r.get('sha256'): rec['tier'],rec['flags']=classify(r['sha256'],r['bytes'])
    else: rec['tier']='NO_DIGEST' if not isfold else 'FOLDER'; rec['flags']={}
    rows.append(rec); seen.add(r['id'])
dn=0; byid={r['id']:r for r in rows}
for d in deltas:
    if 'id' not in d: continue   # manifest rows for stored blob destinations, not Drive objects
    if d['id'] in byid:          # F2: one record per Drive id; the delta listing is kept as an attribute
        byid[d['id']].setdefault('also_in_deltas',[]).append(d['_manifest'].split('/')[0]); continue
    sha=d.get('sha256') if (d.get('stored') and d.get('exact')) else None
    hubm=d['_manifest'].split('/')[0] if '/' in d['_manifest'] else (d.get('path') or '').split('/')[0]
    rec={'id':d['id'],'title':d['title'],'path':d.get('path') or '','hub':hubm,'manifest':d['_manifest'],'context':'DELTA_'+d['_delta'],'access':'DELTA_STORED' if d.get('stored') else 'DELTA_'+str(d.get('not_stored_reason') or 'not_stored'),'mime':d['mimeType'],'bytes':d.get('bytes_drive'),'sha256':sha,'source':'delta_'+d['_delta'],'folder':d['mimeType'].endswith('.folder'),'modifiedTime':d.get('modifiedTime'),'in_inventory':d['id'] in seen}
    if sha: rec['tier'],rec['flags']=classify(sha,d.get('bytes_drive') or d.get('bytes_stored'))
    else: rec['tier']='NO_DIGEST' if not rec['folder'] else 'FOLDER'; rec['flags']={}
    rows.append(rec); byid[d['id']]=rec; dn+=1
# DRV map join and personal exclusion
for rec in rows:
    k=rec['title'].strip().lower(); rec['drv_map']=pub_titles[k][0] if k in pub_titles else ''
    rec['personal_lane']=rec['hub'].startswith('03_PERSONAL')
# ledger join (same name rule as dbx/ledger_crosswalk.py), over Drive paths below the hub
sys.argv=['x']; src=open(S+'/dbx/ledger_crosswalk.py').read()
STOP=eval(src[src.index('STOP='):src.index('GENERIC_BASE=')][5:]); GENERIC_BASE=eval(src[src.index('GENERIC_BASE=')+13:src.index('CODE=re')])
CODE=re.compile(r'^(?=.*[a-z])(?=.*\d)[a-z0-9]{3,}$')
def stem(name):
    n=name.lower(); n=re.sub(r'( \(\d+\))+(\.[a-z0-9]{1,6})?$',lambda m:m.group(2) or '',n); n=re.sub(r'\.[a-z0-9]{1,6}$','',n); return n
def unit_parts(unit):
    u=unit.rstrip('/'); base=u.split('/')[-1]; parent=u.split('/')[-2].lower() if '/' in u else ''
    b=re.sub(r'\.[a-z0-9]{1,6}$','',base.lower()); b=re.sub(r'_(\d{8})(_r\d+)?$','',b); b=re.sub(r'_claude$','',b)
    return b,parent,[t for t in re.split(r'[_\-. ]+',b) if t]
files=[r for r in rows if not r['folder'] and not r['personal_lane']]
df=collections.Counter()
for st in {stem(r['title']) for r in files}:
    for t in set(re.split(r'[_\-. ]+',st)):
        if t: df[t]+=1
ledger=json.load(open(S+'/dbx/ledger/rows.json')); hits=[]
for L in ledger:
    b,parent,toks=unit_parts(L['unit'])
    if not toks: continue
    dist=[t for t in toks if t not in STOP and not t.isdigit() and len(t)>=3 and ((CODE.match(t) and df[t]<=150) or df[t]<60)]
    generic=b in GENERIC_BASE or toks[0] in GENERIC_BASE
    for r in files:
        segs=r['path'].lower().split('/') if r['path'] else []
        if generic and parent and parent not in segs: continue
        # the original matches below Github/<top-level folder> (two levels dropped); the Drive analogue drops the hub and the package folder under it (ledger lens F3);
        # a delta item without a reconstructed path is matched on its title (F4)
        low='/'.join(segs[2:]) if len(segs)>=3 else (segs[-1] if segs else r['title'].lower()); st=stem(r['title'])
        matched=[]
        if st==b: strength='STRONG'; where='title'
        else:
            matched=[t for t in dist if re.search(r'(?<![a-z0-9])'+re.escape(t)+r'(?![a-z0-9])',low)]
            adjacent=len(dist)>=2 and re.search(r'(?<![a-z0-9])'+r'[_\- .]*'.join(map(re.escape,dist))+r'(?![a-z0-9])',low)
            pairs=[(a,c) for a,c in zip(dist,dist[1:]) if re.search(r'(?<![a-z0-9])'+re.escape(a)+r'[_\- .]*'+re.escape(c)+r'(?![a-z0-9])',low)]
            lone=[m for m in matched if re.search(r'(?<![a-z0-9])'+re.escape(m)+r'(?![a-z0-9])',st)]
            if adjacent: strength='WEAK_ADJ'
            elif pairs: strength='WEAK_PAIR'
            elif len(lone)==1 and len(matched)==1 and ((CODE.match(lone[0]) and df[lone[0]]<=30) or df[lone[0]]<=3): strength='WEAK_LONE'   # the same rule as main#594; labelled apart because Drive titles are prose (ledger lens F1/F2)
            else: continue
            if not matched: matched=dist
            where='title' if any(re.search(r'(?<![a-z0-9])'+re.escape(m)+r'(?![a-z0-9])',st) for m in matched) else 'folder'
        hits.append({'row':L['n'],'repo':L['repo'],'unit':L['unit'],'missing_proof':L['proof']=='MISSING','match':strength,'where':where,'stem_tokens':len([x for x in re.split(r'[_\-. ]+',st) if x]),'tokens':'+'.join(matched) if matched else 'stem','id':r['id'],'title':r['title'],'hub':r['hub'],'tier':r['tier'],'drv_map':r['drv_map']})
# archive members
mrows=[]
for m in members:
    sha=m['Payload SHA-256'].strip(); 
    try: size=int(m['Bytes'] or 0)
    except ValueError: size=None
    if re.fullmatch(r'[0-9a-f]{64}',sha): tier,f=classify(sha,size)
    else: tier,f='NO_DIGEST',{}
    mrows.append({'carrier':m['Carrier title'],'carrier_id':m['Carrier ID'],'member':m['Member path'],'bytes':size,'sha256':sha,'context':m['Context'],'tier':tier,'flags':f})
carriers={}
for m in members:
    cs=m['Carrier SHA-256'].strip()
    if m['Carrier ID'] not in carriers:
        csz=next((r['bytes'] for r in rows if r['id']==m['Carrier ID']),None)   # F1: the carrier's byte count, so the 4 MiB rule applies to the carrier itself
        if re.fullmatch(r'[0-9a-f]{64}',cs): ct,cf=classify(cs,csz)
        else: ct,cf='NO_DIGEST',{}
        carriers[m['Carrier ID']]={'title':m['Carrier title'],'sha256':cs,'bytes':csz,'tier':ct,'git_paths':cf.get('git_paths',[]),'git_blobs':cf.get('git_blobs',[]),'archive_members':cf.get('archive_members',[])}
for mr in mrows:
    c=carriers.get(mr['carrier_id']); mr['carrier_tier']=c['tier'] if c else 'NO_DIGEST'
    mr['preserved']=('MEMBER' if mr['tier'] in ('GIT_EXACT','ARCHIVE_EXACT','DBX_EXACT') else 'VIA_CARRIER' if mr['carrier_tier'] in ('GIT_EXACT','ARCHIVE_EXACT','DBX_EXACT') else 'NOT_FOUND')
json.dump(carriers,open(G+'/carriers.json','w'),indent=0)
for r in rows:
    if r['sha256'] and r['flags'].get('in_git'):
        gp=r['flags']['git_paths']; r['git_where']=('TIP' if r['flags']['on_tip'] else 'DRIVE_MIRROR_BRANCH' if all(p.split(':',1)[1].startswith('drive/') for p in gp) else 'BRANCH_OR_HISTORY')
dbx_summ=json.load(open(S+'/dbx/ledger_crosswalk_summary.json'))
nodbx={int(k) for k,v in dbx_summ.items() if v.get('missing_proof')=='Y' and v['strong']==0 and v['weak']==0}
json.dump(sorted(nodbx),open(G+'/ledger_rows_without_dropbox_hit.json','w'))
json.dump(rows,open(G+'/drive_rows.json','w'),indent=0); json.dump(hits,open(G+'/ledger_drive_hits.json','w'),indent=0); json.dump(mrows,open(G+'/member_rows.json','w'),indent=0)
# aggregates (printed: counts only)
fil=[r for r in rows if not r['folder']]
print('items',len(rows),'files',len(fil),'from deltas',dn,'personal-lane files (excluded from joins)',sum(1 for r in fil if r['personal_lane']))
print('tiers (files):',dict(collections.Counter(r['tier'] for r in fil).most_common()))
hashed=[r for r in fil if r['sha256']]
print('hashed files',len(hashed),'in_git',sum(1 for r in hashed if r['flags'].get('in_git')),'on_tip',sum(1 for r in hashed if r['flags'].get('on_tip')),'in_archive',sum(1 for r in hashed if r['flags'].get('in_archive')),'in_dbx',sum(1 for r in hashed if r['flags'].get('in_dbx')),'dbx_unknown(>4MiB)',sum(1 for r in hashed if r['flags'].get('in_dbx') is None))
print('by hub x tier (hashed):'); agg=collections.Counter((r['hub'][:40],r['tier']) for r in hashed)
for (h,t),n in sorted(agg.items()): print('  %-42s %-32s %d'%(h,t,n))
print('DRV map title joins:',dict(collections.Counter(r['drv_map'] for r in fil if r['drv_map'])), 'of DRV-H2 names matched',sum(1 for k,v in pub_titles.items() if v[0]=='DRV-H2' and any(r['title'].strip().lower()==k for r in fil)),'/',len(h2),'DRV-H3',sum(1 for k,v in pub_titles.items() if v[0]=='DRV-H3' and any(r['title'].strip().lower()==k for r in fil)),'/',len(h3))
mp=[h for h in hits if h['missing_proof']]
print('ledger hits',len(hits),'by match',dict(collections.Counter(h['match'] for h in hits)),'on MISSING-proof rows',len(mp),'distinct MISSING-proof rows hit',len({h['row'] for h in mp}),'where',dict(collections.Counter(h['where'] for h in hits)))
print('archive members',len(mrows),'tiers:',dict(collections.Counter(m['tier'] for m in mrows).most_common()))

print('deltas hubs:',dict(collections.Counter(r['hub'][:30] for r in rows if r['source'].startswith('delta'))))
gw=collections.Counter(r.get('git_where') for r in fil if r.get('git_where')); print('GIT_EXACT where:',dict(gw))
pres=collections.Counter(m['preserved'] for m in mrows); print('archive members preserved:',dict(pres))
ct=collections.Counter(c['tier'] for c in carriers.values()); print('carriers',len(carriers),'tiers:',dict(ct))
nf=[m for m in mrows if m['preserved']=='NOT_FOUND']; print('members NOT_FOUND by carrier tier:',dict(collections.Counter(m['carrier_tier'] for m in nf)),'by context:',dict(collections.Counter(m['context'] for m in nf).most_common(6)))
hit_rows={h['row'] for h in mp}; print('MISSING-proof rows hit by Drive names',len(hit_rows),'of which had no Dropbox hit:',len(hit_rows & nodbx),'->',sorted(hit_rows & nodbx)[:40])
only=[r for r in fil if r['tier'].startswith('DRIVE_ONLY') and not r['personal_lane']]
print('DRIVE_ONLY files (non-personal):',len(only),'by hub:',dict(collections.Counter(r['hub'][:30] for r in only)),'by mime:',dict(collections.Counter(r['mime'].split('/')[-1][:20] for r in only).most_common(8)),'bytes total',sum(r['bytes'] or 0 for r in only))
print('NO_DIGEST files by access:',dict(collections.Counter(r['access'] for r in fil if r['tier']=='NO_DIGEST').most_common(8)))
