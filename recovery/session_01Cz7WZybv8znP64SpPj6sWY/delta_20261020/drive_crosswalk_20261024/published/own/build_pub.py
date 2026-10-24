#!/usr/bin/env python3
"""Assemble the publishable subset of the Drive byte-identity crosswalk into build_out/ and run the guard.
Row-level rule: a per-file line only where the Drive title is already on main (DRV-H2/DRV-H3 maps) or the bytes are already public
(GIT_EXACT, ARCHIVE_EXACT); otherwise the file is counted in aggregates (hub x tier x class, no cell below 3 files). Digests are printed only for
public bytes. No Drive id, URL, owner address or 03_PERSONAL item appears. Byte totals are rounded to 100 KB."""
import json, csv, re, os, shutil, hashlib, collections, subprocess, sys
S='<SCRATCH>'; G=S+'/gdrive'; OUT=G+'/build_out/2026-10-24'
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT+'/own')
rows=json.load(open(G+'/drive_rows.json')); hits=json.load(open(G+'/ledger_drive_hits.json')); mrows=json.load(open(G+'/member_rows.json')); carriers=json.load(open(G+'/carriers.json'))
new=json.load(open(G+'/new_items.json')) if os.path.exists(G+'/new_items.json') else []; lsum=json.load(open(G+'/liveness_summary.json')) if os.path.exists(G+'/liveness_summary.json') else {}
nodbx=set(json.load(open(G+'/ledger_rows_without_dropbox_hit.json'))); ledger={r['n']:r for r in json.load(open(S+'/dbx/ledger/rows.json'))}
def gitshow(p): return subprocess.run(['git','-C','<MAIN_CHECKOUT>','show','origin/main:'+p],capture_output=True,text=True,check=True).stdout
h2=list(csv.DictReader(gitshow('docs/retro/DRV-H2/DRV-H2.map.csv').splitlines())); h3=list(csv.DictReader(gitshow('docs/retro/DRV-H3/DRV-H3_drive_map.csv').splitlines()))
PUB={r['name'].strip().lower() for r in h2}|{r['name'].strip().lower() for r in h3}
ROUND=100_000; rnd=lambda b:int(round((b or 0)/ROUND))*ROUND
def mclass(m):
    m=(m or '').lower()
    return 'folder' if 'google-apps.folder' in m else 'google-doc' if 'google-apps' in m else 'zip' if 'zip' in m else 'pdf' if 'pdf' in m else 'markdown' if 'markdown' in m else 'text' if m.startswith('text/') else 'json' if 'json' in m else 'image' if m.startswith('image/') else 'office' if ('officedocument' in m or 'msword' in m) else 'python' if 'python' in m else 'other'
def hublabel(h):
    h=h.strip(); 
    return h if h.startswith(('00_','01_','02_','04_','05_','06_','90_')) else '(2026-09-18..20 delta folders)'   # P-4: delta manifest folder names are Drive titles not on main
fil=[r for r in rows if not r['folder']]; personal=[r for r in fil if r['personal_lane']]; fil=[r for r in fil if not r['personal_lane']]
IDTOK=re.compile(r'(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,44}(?![A-Za-z0-9_-])')
def idbearing(s): return any(re.search('[0-9]',m) and re.search('[A-Z]',m) and re.search('[a-z]',m) and not re.fullmatch(r'[0-9a-f]+',m) and '_' not in m[1:-1].replace('_','',0) for m in IDTOK.findall(s)) if False else any((re.search('[0-9]',m) and re.search('[A-Z]',m) and re.search('[a-z]',m) and not re.fullmatch(r'[0-9a-f]+',m) and len(m) in (28,33,44)) for m in IDTOK.findall(s))
for r in fil:
    r['title_public']=(r['title'].strip().lower() in PUB) and not idbearing(r['title']); r['bytes_public']=r['tier'] in ('GIT_EXACT','ARCHIVE_EXACT')
    if not r['title_public']: r['drv_map']=''   # F7: the join is recorded only where the title is printed
NONPUB=[r['title'] for r in fil if not r['title_public'] and len(r['title'])>=8]+[r['title'] for r in personal if len(r['title'])>=4]   # a repo path is printed only if it carries none of these (basenames already in Git are allowed by the guard)
def locator(fl):
    blobs=fl.get('git_blobs',[]); paths=[p for p in fl.get('git_paths',[]) if not p.split(':',1)[1].startswith('drive/') and not any(n in p for n in NONPUB)]
    return ';'.join(blobs[:2]+paths[:2])
lrows=collections.defaultdict(list)
for h in hits: lrows[h['id']].append(h['row'])
# DRIVE_CROSSWALK.tsv
cols=['hub','title_if_public','class','bytes','sha256_if_public_bytes','tier','git_where','git_locator','archive_member','live_status','digest_state','drv_map','ledger_rows','source']
n_rows=0
with open(OUT+'/DRIVE_CROSSWALK.tsv','w',encoding='utf-8') as f:
    f.write('\t'.join(cols)+'\n')
    for r in sorted(fil,key=lambda r:(r['hub'],r['tier'],r['title'].lower())):
        if not (r['title_public'] or r['bytes_public']): continue
        fl=r['flags'] or {}
        f.write('\t'.join([hublabel(r['hub']), r['title'] if r['title_public'] else '', mclass(r['mime']), str(r['bytes'] if (r['bytes_public'] or r['title_public']) and r['bytes'] is not None else ''),
                            r['sha256'] if r['bytes_public'] and r['sha256'] else '', r['tier'], r.get('git_where',''), locator(fl), ((fl.get('archive_members') or [''])[0].split(':',1)[0] if fl.get('archive_members') else ''),
                            r.get('live_status',''), r.get('digest_state',''), r['drv_map'], ';'.join(str(x) for x in sorted(set(lrows.get(r['id'],[])))), r['source']])+'\n'); n_rows+=1
# aggregates hub x tier x class with min cell 3 (merge small classes into (other) per hub x tier; still-small cells print count only)
agg=collections.defaultdict(lambda:{'files':0,'bytes':0})
nonpub_files=[r for r in fil if not (r['title_public'] or r['bytes_public'])]   # P-3: the aggregate covers only files that have no per-file row
for r in nonpub_files:
    a=agg[(hublabel(r['hub']),r['tier'],mclass(r['mime']))]; a['files']+=1; a['bytes']+=r['bytes'] or 0
merged=collections.defaultdict(lambda:{'files':0,'bytes':0})
for (h,t,c),a in agg.items():
    k=(h,t,c) if a['files']>=3 else (h,t,'(other)')
    merged[k]['files']+=a['files']; merged[k]['bytes']+=a['bytes']
with open(OUT+'/DRIVE_AGGREGATE.tsv','w',encoding='utf-8') as f:
    f.write('hub\ttier\tclass\tfiles\tbytes_rounded_100KB\n')
    for (h,t,c),a in sorted(merged.items()): f.write('\t'.join([h,t,c,str(a['files']),str(rnd(a['bytes'])) if a['files']>=3 else ''])+'\n')
# LEDGER_DRIVE.tsv: one line per ledger unit with hits (counts by strength and tier); LEDGER_DRIVE_ROWS.tsv: per-item lines for public items only (P-1)
byrow=collections.defaultdict(list)
for h in hits:
    r=next((x for x in fil if x['id']==h['id']),None)
    if r is not None: byrow[h['row']].append((h,r))
with open(OUT+'/LEDGER_DRIVE.tsv','w',encoding='utf-8') as f:
    f.write('row\trepo\tunit\tmissing_proof\tno_dropbox_name_hit\thits\tstrong\tweak_adjacent\tweak_pair\tweak_lone\thits_in_title\thits_in_folder_only\thits_by_tier\thits_modified_after_snapshot_day\tpublic_items_listed\n')
    for n in sorted(byrow):
        hs=byrow[n]; h0=hs[0][0]
        f.write('\t'.join([str(n),h0['repo'],h0['unit'],'Y' if h0['missing_proof'] else 'N','Y' if n in nodbx else 'N',str(len(hs)),str(sum(1 for h,r in hs if h['match']=='STRONG')),str(sum(1 for h,r in hs if h['match']=='WEAK_ADJ')),str(sum(1 for h,r in hs if h['match']=='WEAK_PAIR')),str(sum(1 for h,r in hs if h['match']=='WEAK_LONE')),str(sum(1 for h,r in hs if h['where']=='title')),str(sum(1 for h,r in hs if h['where']=='folder')),
                            ';'.join(f'{k}:{v}' for k,v in sorted(collections.Counter(h['tier'] for h,r in hs).items())),str(sum(1 for h,r in hs if r.get('live_status')=='MODIFIED_AFTER_SNAPSHOT_DAY')),str(sum(1 for h,r in hs if r['title_public'] or r['bytes_public']))])+'\n')
with open(OUT+'/LEDGER_DRIVE_ROWS.tsv','w',encoding='utf-8') as f:
    f.write('row\trepo\tunit\tmissing_proof\tmatch\tmatched_tokens\twhere\ttitle_stem_tokens\tdrive_title_if_public\thub\ttier\tgit_locator\tlive_status\tdrv_map\n')
    for n in sorted(byrow):
        for h,r in sorted(byrow[n],key=lambda hr:(hr[0]['match']!='STRONG',hr[0]['title'].lower())):
            if not (r['title_public'] or r['bytes_public']): continue
            f.write('\t'.join([str(n),h['repo'],h['unit'],'Y' if h['missing_proof'] else 'N',h['match'],h['tokens'],h['where'],str(h['stem_tokens']),h['title'] if r['title_public'] else '',hublabel(h['hub']),h['tier'],locator(r['flags'] or {}),r.get('live_status',''),h['drv_map']])+'\n')
# ARCHIVE_CARRIERS.tsv: per-carrier lines only where the carrier title is on main or its bytes are public; other carriers aggregated by tier (P-2)
byc=collections.defaultdict(lambda:collections.Counter())
for m in mrows: byc[m['carrier_id']][m['preserved']]+=1; byc[m['carrier_id']]['members']+=1
title_of={r['id']:r for r in rows}
cagg=collections.defaultdict(lambda:collections.Counter()); n_car_rows=0
with open(OUT+'/ARCHIVE_CARRIERS.tsv','w',encoding='utf-8') as f:
    f.write('carrier_title_if_public\tcarrier_git_locator\thub\tcarrier_tier\tcarriers\tmembers\tmembers_found_elsewhere\tmembers_preserved_via_carrier\tmembers_not_found\n')
    for cid,c in sorted(carriers.items(),key=lambda kv:kv[1]['title'].lower()):
        r=title_of.get(cid)
        if r and r.get('personal_lane'): continue
        tpub=(c['title'].strip().lower() in PUB) and not idbearing(c['title']); bpub=c['tier'] in ('GIT_EXACT','ARCHIVE_EXACT')
        if tpub or bpub:
            f.write('\t'.join([c['title'] if tpub else '',locator(c),hublabel(r['hub']) if r else '(not in snapshot)',c['tier'],'1',str(byc[cid]['members']),str(byc[cid]['MEMBER']),str(byc[cid]['VIA_CARRIER']),str(byc[cid]['NOT_FOUND'])])+'\n'); n_car_rows+=1
        else:
            a=cagg[c['tier']]; a['carriers']+=1; a['members']+=byc[cid]['members']; a['MEMBER']+=byc[cid]['MEMBER']; a['VIA_CARRIER']+=byc[cid]['VIA_CARRIER']; a['NOT_FOUND']+=byc[cid]['NOT_FOUND']
    for tier,a in sorted(cagg.items()):
        f.write('\t'.join(['(carriers without a public title or public bytes)','','',tier,str(a['carriers']),str(a['members']),str(a['MEMBER']),str(a['VIA_CARRIER']),str(a['NOT_FOUND'])])+'\n')
# SUMMARY.json
hashed=[r for r in fil if r['sha256']]
summ={'snapshot':'drive/inventory.jsonl, drive/source_map/Archive_Members.csv and drive/deltas/2026-09-18..20 in the drive/ tree at a01c72f1 (Drive-GitHub hardening branch); the 2026-09-17 accessibility publication',
 'items':{'snapshot_items':4456,'delta_items':sum(1 for r in rows if r['source'].startswith('delta')),'files_non_personal':len(fil),'files_personal_lane_counted_only':len(personal),'folders':sum(1 for r in rows if r['folder'])},
 'hashed_files':{'count':len(hashed),'count_including_personal_lane':len(hashed)+sum(1 for r in personal if r['sha256']),'over_4mib_dropbox_not_derivable':sum(1 for r in hashed if (r['flags'] or {}).get('in_dbx') is None)},
 'drv_map_digests':{'title_public_files_with_digest':sum(1 for r in fil if r['title_public'] and r['sha256']),'of_which_digest_printed':sum(1 for r in fil if r['title_public'] and r['sha256'] and r['bytes_public']),'withheld_by_tier':dict(collections.Counter(r['tier'] for r in fil if r['title_public'] and r['sha256'] and not r['bytes_public']))},
 'archive_members_not_found_over_4mib':sum(1 for m in mrows if m['preserved']=='NOT_FOUND' and (m['bytes'] or 0)>4*1024*1024),
 'tiers':{t:{'files':sum(1 for r in fil if r['tier']==t),
             'bytes_rounded_published_parts':rnd(sum(r['bytes'] or 0 for r in fil if r['tier']==t and r['bytes_public']))+rnd(sum(a['bytes'] for (h,tt,c),a in merged.items() if tt==t and a['files']>=3)),
             'files_in_published_parts':sum(1 for r in fil if r['tier']==t and r['bytes_public'])+sum(a['files'] for (h,tt,c),a in merged.items() if tt==t and a['files']>=3)} for t in sorted({r['tier'] for r in fil})},
 'tier_bytes_rule':'a tier byte total covers only the parts that print bytes elsewhere in this directory (rows with public bytes, and aggregate cells of three files or more); files in count-only cells and title-public rows without public bytes are excluded, so no withheld size can be recovered by differencing (skeptic verdict on P-3)',
 'git_exact_where':dict(collections.Counter(r.get('git_where') for r in hashed if r.get('git_where'))),
 'dropbox_overlap':{'drive_files_byte_identical_to_a_dropbox_github_file':sum(1 for r in hashed if (r['flags'] or {}).get('in_dbx')),'dropbox_only_distinct_contents_also_in_drive':json.load(open(G+'/analysis2.json'))['both_dbx_drive'],'dropbox_only_distinct_contents_total':814},
 'archive_members':{'total':len(mrows),'preserved':dict(collections.Counter(m['preserved'] for m in mrows)),'not_found_by_carrier_tier':dict(collections.Counter(m['carrier_tier'] for m in mrows if m['preserved']=='NOT_FOUND')),'carriers':len(carriers),'carriers_by_tier':dict(collections.Counter(c['tier'] for c in carriers.values()))},
 'liveness':{'listing':'connector search of items modified after 2026-09-17T00:00Z in ten windows; metadata only','windows_complete':lsum.get('all_complete'),'live_items_seen':lsum.get('live_items'),
             'files_by_live_status':dict(collections.Counter(r.get('live_status','') for r in fil)),'hashed_files_by_digest_state':dict(collections.Counter(r.get('digest_state','') for r in hashed)),
             'new_since_snapshot':{'items':len(new),'folders':sum(1 for v in new if 'folder' in v['mime']),'by_class':dict(collections.Counter(mclass(v['mime']) for v in new)),'by_parent_hub_when_known':dict(collections.Counter(hublabel(v['parent_hub']) for v in new if v.get('parent_known'))),'parent_not_in_snapshot':sum(1 for v in new if not v.get('parent_known')),'titles_in_drv_maps':sum(1 for v in new if v['title'].strip().lower() in PUB)}},
 'ledger':{'rows':344,'missing_proof_rows':131,'dropbox_name_hit_rule':'a MISSING-proof row counts as having a Dropbox name hit when main#594 recorded a STRONG or WEAK hit for it at row level (30 rows) or in its aggregated ledger_hits (18 rows); 48 rows in all, 83 without','missing_proof_rows_without_dropbox_name_hit':len(nodbx),'name_hits':len(hits),'missing_proof_rows_hit':len({h['row'] for h in hits if h['missing_proof']}),
           'missing_proof_rows_hit_without_dropbox_name_hit':sorted({h['row'] for h in hits if h['missing_proof']} & nodbx),'strong':sum(1 for h in hits if h['match']=='STRONG'),
           'hits_by_match':dict(collections.Counter(h['match'] for h in hits)),'hits_by_where':dict(collections.Counter(h['where'] for h in hits)),'missing_proof_hits':sum(1 for h in hits if h['missing_proof']),
           'missing_proof_hits_by_match':dict(collections.Counter(h['match'] for h in hits if h['missing_proof'])),
           'rows_without_dropbox_name_hit_reached_only_by_lone_tokens':sorted(n for n in ({h['row'] for h in hits if h['missing_proof']} & nodbx) if all(h['match']=='WEAK_LONE' for h in hits if h['row']==n)),
           'lone_token_hits_on_prose_titles_over_8_stem_tokens':sum(1 for h in hits if h['match']=='WEAK_LONE' and h['stem_tokens']>8),
           'note':'the name rule is byte-for-byte the rule of main#594 (STRONG stem equality; WEAK adjacent tokens, a consecutive pair, or one rare token in the title stem); Drive titles are prose, so lone-token hits are labelled WEAK_LONE and should be read as the weakest class; the no_dropbox_name_hit flag is computed for MISSING-proof rows only',
           'hit_item_tiers':dict(collections.Counter(h['tier'] for h in hits if h['missing_proof']))},
 'drv_maps':{'drv_h2_rows':len(h2),'drv_h3_rows':len(h3),'snapshot_files_with_a_title_in_the_maps':sum(1 for r in fil if r['title_public'])},
 'row_level_rule':'per-file lines only where the title is already on main (DRV-H2/DRV-H3 maps) or the bytes are already public (GIT_EXACT, ARCHIVE_EXACT); digests only for public bytes; every other file is counted only, in DRIVE_AGGREGATE.tsv by hub x tier x class over those files alone, where classes below 3 files are merged into (other) per hub and tier and a cell still below 3 files prints its count and no bytes; the same rule for carriers and for ledger hits; 03_PERSONAL items counted only',
 'carrier_rows':n_car_rows,'aggregate_files':len(nonpub_files),
 'row_level_rows':n_rows,'bytes_rounding':'every byte total is rounded to the nearest 100 KB; tier totals cover published parts only (tier_bytes_rule)'}
json.dump(summ,open(OUT+'/SUMMARY.json','w'),indent=1,sort_keys=True)
# PINS.json (inputs by hash)
H=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
def blob(ref,path): return subprocess.run(['git','-C','<MAIN_CHECKOUT>','rev-parse',f'{ref}:{path}'],capture_output=True,text=True,check=True).stdout.strip()
pins={'drive_tree_commit':'a01c72f1','inventory_jsonl_blob':blob('a01c72f1','drive/inventory.jsonl'),'archive_members_csv_blob':blob('a01c72f1','drive/source_map/Archive_Members.csv'),
      'drv_h2_map_blob_at_origin_main':blob('origin/main','docs/retro/DRV-H2/DRV-H2.map.csv'),'drv_h3_map_blob_at_origin_main':blob('origin/main','docs/retro/DRV-H3/DRV-H3_drive_map.csv'),
      'git_index':{'blobs':32221,'sha256':H(S+'/dbx/idx/git_index.json'),'tip_commits_for_TIP_label':{'main':next(l.split()[0] for l in open(S+'/dbx/idx/refs_main.txt') if l.split()[1]=='refs/remotes/origin/main'),'Math-':next(l.split()[0] for l in open(S+'/dbx/idx/refs_math.txt') if l.split()[1]=='refs/heads/main')},'note':'the index was built at these tips; origin/main has moved since (8df77725 at build time of this directory), so TIP means on the tip as of the index'},'dropbox_content_hashes':{'entries':5724,'sha256':H(S+'/dbx/content_hashes.json')},
      'local_full_crosswalk':{'file':'gdrive/drive_rows.json (path and id level, held locally by session 01Cz; offered to the owner or the CoS)','sha256':H(G+'/drive_rows.json')},
      'local_live_listing':{'files':sorted(os.path.basename(p) for p in __import__('glob').glob(G+'/parts/*.tsv')),'sha256_of_concatenation':hashlib.sha256(b''.join(open(p,'rb').read() for p in sorted(__import__('glob').glob(G+'/parts/*.tsv')))).hexdigest()},
      'ledger':{'file':'data/hypergraph/math_coverage_ledger_20261023.md','rows':344}}
json.dump(pins,open(OUT+'/PINS.json','w'),indent=1)
SUBS=[(S,'<SCRATCH>'),('<MAIN_CHECKOUT>','<MAIN_CHECKOUT>')]
for fn in ('classify.py','analysis2.py','liveness.py','build_pub.py','readme_drive.py'):
    if os.path.exists(G+'/'+fn):
        t=open(G+'/'+fn,encoding='utf-8').read()
        for a,b in SUBS: t=t.replace(a,b)
        open(OUT+'/own/'+fn,'w',encoding='utf-8').write(t)
subprocess.run([sys.executable,'-I',G+'/readme_drive.py',OUT],check=True)
# guard
sys.path.insert(0,G+'/tools_scan'); import drive_id_scan as scan
bad=[]
IDSHAPE=re.compile(r'(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,44}(?![A-Za-z0-9_-])')
PAT=[r'https?://', r'drive\.google|docs\.google|dropbox\.com|dropboxusercontent|rlkey=', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}', r'\bns:\d', r'\bid:[A-Za-z0-9_-]{8,}']
GITNAMES={p.rsplit('/',1)[-1] for v in json.load(open(S+'/dbx/idx/git_index.json')).values() for p in v['paths']}   # basenames already in Git are public names
for dp,_,fns in os.walk(OUT):
    for fn in fns:
        t=open(os.path.join(dp,fn),encoding='utf-8').read()
        for rx in PAT:
            if fn in ('build_pub.py','classify.py') and rx in t: continue
            if re.search(rx,t): bad.append((fn,rx))
        for hit in scan.hits_in_text(t): bad.append((fn,'drive-id (repo scanner) '+str(hit)[:16]))   # tools/privacy/drive_id_scan.py at origin/main, the OP-PRIVACY check
        for m in IDSHAPE.findall(t):
            if len(m) in (28,33,44) and re.search('[0-9]',m) and re.search('[A-Z]',m) and re.search('[a-z]',m) and not re.fullmatch(r'[0-9a-f]+',m): bad.append((fn,'id-shape '+m[:12]))
        for r in personal:
            if r['title'] and r['title'] in t: bad.append((fn,'personal title'))
        for r in fil:
            if not r['title_public'] and len(r['title'])>=12 and r['title'] not in GITNAMES and r['title'] in t and fn!='classify.py': bad.append((fn,'non-public title '+r['title'][:20]))
assert not bad, sorted(set(bad))[:12]
print('build ok: rows',n_rows,'aggregate cells',len(merged),'guard passed')
