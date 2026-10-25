#!/usr/bin/env python3
"""Phase 3: byte-identity tiers for the Dropbox Github/ inventory.
Inputs: inventory.tsv, phase1.tsv, content_hashes.json, idx/git_index.json, idx/tip_blobs.json, idx/{a2136bc0,c4065c67}_index.json,
and the 2026-09-27 import's MANIFEST.csv / NOT_IMPORTED.csv (as flags).  Writes dbx/phase3.tsv and dbx/phase3_summary.json.
Tiers, strongest evidence of preservation first: GIT_EXACT, ARCHIVE_EXACT, EMPTY, then the not-preserved tiers DUP_IN_DROPBOX
(no Git or archive copy, but another Dropbox copy of the same bytes), NAME_SIZE_DIFF_BYTES, MISSING; SKIP and PENDING_HASH aside.
The 2026-09-27 import's listing of a hash is a flag (IMPORT_LISTED:...), not a tier: it never establishes preserved bytes."""
import csv, json, collections, os, re, sys
from pathlib import PurePosixPath
S='<SCRATCH>'
M='/home/user/main/audits/dropbox_import/2026-09-27'
IMPORT_DAY='2026-09-27T00:00:00Z'  # the import README records the day only, so 'on or after' the import day
EMPTY_HASH='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
PERSONAL=re.compile(r'(?i)r[eé]sum[eé]|curriculum.?vitae|(?<![a-z])cv(?![a-z])|(?<![a-z])tax(?:es)?(?![a-z])|(?<![a-z0-9])(?:w-?2|w-?9|1099)(?![a-z0-9])|bank|pay.?(?:stub|slip|roll)|passport|driv\w*.?licen[cs]e|invoice|medical|insurance|(?<![a-z])lease(?![a-z])|(?<![a-z])ssn(?![a-z])|id.?card|vehicle|policy\d|social.?security|birth.?cert|mortgage|(?<![a-z])loan(?![a-z])|credit.?card|(?<![a-z])visa(?![a-z])|\.dmg$')  # the import's personal-paperwork test, widened after a listing-page review (id cards, vehicle or policy numbers, installers)
git=json.load(open(S+'/dbx/idx/git_index.json'))
tip=json.load(open(S+'/dbx/idx/tip_blobs.json'))  # 'repo:blob' -> [paths] on origin/main (main) and main (Math-)
byhash=collections.defaultdict(list)
for k,v in git.items():
    for p in v['paths']: byhash[v['dbx']].append((v['repo'],v['blob'],p))
arch=collections.defaultdict(list)
for lab in ('a2136bc0','c4065c67'):
    for k,v in json.load(open(f'{S}/dbx/idx/{lab}_index.json')).items(): arch[v['content_hash']].append(k)
imp={}
for fn,kind in (('MANIFEST.csv','manifest'),('NOT_IMPORTED.csv','not_imported')):
    for r in csv.DictReader(open(f'{M}/{fn}',encoding='utf-8')):
        imp.setdefault(r['dropbox_content_hash'],[]).append((kind, r.get('disposition') or r.get('reason'), r['dropbox_path']))
imp_paths={p for v in imp.values() for _,_,p in v}
hashes=json.load(open(S+'/dbx/content_hashes.json'))
p1={r[0]:r for r in (l.rstrip('\n').split('\t') for l in open(S+'/dbx/phase1.tsv',encoding='utf-8'))}
inv=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/inventory.tsv',encoding='utf-8')]
files=[r for r in inv if r[5]=='file']
dup=collections.defaultdict(list)
for r in files:
    h=hashes.get(r[0],{}).get('content_hash')
    if h: dup[h].append(r[0])
rows=[]; summ=collections.Counter(); byfolder=collections.defaultdict(collections.Counter); bytes_=collections.Counter()
for path,name,size,cm,mt,ot in files:
    size=int(size or 0); folder=path.split('/')[1] if path.count('/')>=2 else '(Github root)'
    p1row=p1.get(path); p1tier=p1row[4] if p1row else '?'
    rec=hashes.get(path); h=rec.get('content_hash') if rec else None; loc=''; flags=[]
    if rec and rec.get('size') is not None and rec['size']!=size: flags.append('HASH_SIZE_CONFLICT'); h=None
    if p1tier=='SKIP': tier='SKIP'
    elif not h: tier='PENDING_HASH'
    elif h==EMPTY_HASH or size==0: tier='EMPTY'
    elif h in byhash:
        tier='GIT_EXACT'; hits=sorted(byhash[h]); loc=';'.join(f'{r}:{p}' for r,b,p in hits[:3])
        flags.append('ON_TIP' if any(f'{r}:{b}' in tip for r,b,p in hits) else 'HISTORY_OR_BRANCH_ONLY')
        if len(hits)>3: flags.append(f'LOCATORS:{len(hits)}')
    elif h in arch: tier='ARCHIVE_EXACT'; loc=';'.join(sorted(arch[h])[:3])
    elif len(dup.get(h,[]))>1: tier='DUP_IN_DROPBOX'; loc=str(len(dup[h]))+' copies'
    elif p1tier=='NAME_SIZE': tier='NAME_SIZE_DIFF_BYTES'; loc=p1row[5]
    else: tier='MISSING'
    if h and h in imp: flags+= [f'IMPORT_LISTED:{k}:{d}' for k,d,_ in imp[h][:2]]
    if cm>=IMPORT_DAY: flags.append('MODIFIED_ON_OR_AFTER_IMPORT_DAY')
    if PERSONAL.search(PurePosixPath(path).name): flags.append('PERSONAL_NAME')
    if path in imp_paths: flags.append('IN_IMPORT_LISTS')
    if tier=='PENDING_HASH' and p1tier=='NAME_SIZE': flags.append('NAME_SIZE_MATCH:'+p1row[5][:120])
    rows.append((path,name,str(size),cm,tier,h or '',loc,','.join(flags)))
    summ[tier]+=1; bytes_[tier]+=size; byfolder[folder][tier]+=1
with open(S+'/dbx/phase3.tsv','w',encoding='utf-8') as f:
    f.write('\t'.join(['path','name','bytes','client_modified','tier','content_hash','locator','flags'])+'\n')
    for r in rows: f.write('\t'.join(r)+'\n')
json.dump({'files':len(files),'tiers':{k:{'files':v,'bytes':bytes_[k]} for k,v in summ.items()},
           'by_folder':{k:dict(v) for k,v in byfolder.items()},
           'modified_on_or_after_import_day':sum(1 for r in rows if 'MODIFIED_ON_OR_AFTER_IMPORT_DAY' in r[7]),
           'personal_name':sum(1 for r in rows if 'PERSONAL_NAME' in r[7])}, open(S+'/dbx/phase3_summary.json','w'), indent=1)
print('files',len(files)); print({k:(v,bytes_[k]) for k,v in sorted(summ.items())})
print('modified on/after import day', sum(1 for r in rows if 'MODIFIED_ON_OR_AFTER_IMPORT_DAY' in r[7]), 'personal-name', sum(1 for r in rows if 'PERSONAL_NAME' in r[7]), 'on-tip', sum(1 for r in rows if 'ON_TIP' in r[7]), 'history/branch only', sum(1 for r in rows if 'HISTORY_OR_BRANCH_ONLY' in r[7]))
