#!/usr/bin/env python3
"""Assemble the publishable subset of the Dropbox root-folders crosswalk into build_out/ and run the guard.
Nothing outside Github/ is import-named, so no Dropbox path, file name or time is published. Per-file lines only for bytes already public
(GIT_EXACT, ARCHIVE_EXACT): folder label, class, bytes, content_hash, Git blob or archive locator. Every other file is counted in aggregates by
folder label x class x tier with no cell below 3 files; byte totals rounded to 100 KB and tier totals over published parts only."""
import json, re, os, shutil, hashlib, collections, subprocess, sys, csv
S='<SCRATCH>'; D=S+'/dbxroot'; OUT=D+'/build_out/2026-10-25'
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT+'/own')
rows=json.load(open(D+'/root_rows.json')); hits_all=json.load(open(D+'/ledger_root_hits.json')) if os.path.exists(D+'/ledger_root_hits.json') else []
PUBTIERS=('GIT_EXACT','ARCHIVE_EXACT','GITHUB_TREE_COPY')   # ledger hits are published only for items whose bytes are public or are a Github/ copy; hits on other items stay local
hits=[h for h in hits_all if h['tier'] in PUBTIERS]; hits_local=len(hits_all)-len(hits)
ROUND=100_000; rnd=lambda b:int(round((b or 0)/ROUND))*ROUND
LOC=json.load(open(D+'/labels_local.json'))   # real folder names and the personal-name token live in this local file, never published
LABEL=LOC['labels']
def lab(f): return LABEL.get(f,'other root folder')
def cls(name):
    e=os.path.splitext(name)[1].lower()
    return {'.json':'json','.md':'markdown','.txt':'text','.pdf':'pdf','.py':'python','.zip':'zip','.docx':'office','.tex':'tex','.png':'image','.pkl':'binary','.url':'link','.sha256':'digest'}.get(e,'other')
pub=[r for r in rows if r['tier'] in ('GIT_EXACT','ARCHIVE_EXACT')]
NONPUB=[r['name'] for r in rows if len(r['name'])>=6]   # no Dropbox name is public here
def locator(fl):
    blobs=fl.get('git_blobs',[]); paths=[p for p in fl.get('git_paths',[]) if not p.split(':',1)[1].startswith('drive/') and not any(n in p for n in NONPUB)]
    arch=[a.split(':',1)[0] for a in fl.get('archive_members',[])][:1]
    return ';'.join(blobs[:2]+paths[:2]+[('archive:'+a) for a in arch])
lrows=collections.defaultdict(list)
for h in hits: lrows[h['path']].append(h['row'])
with open(OUT+'/ROOT_CROSSWALK.tsv','w',encoding='utf-8') as f:
    f.write('folder\tclass\tbytes\tcontent_hash\ttier\tgit_where\tlocator\tledger_rows\n')
    for r in sorted(pub,key=lambda r:(r['folder'],r['tier'],r['content_hash'])):
        f.write('\t'.join([lab(r['folder']),cls(r['name']),str(r['size']),r['content_hash'],r['tier'],(r['flags'] or {}).get('git_where',''),locator(r['flags'] or {}),';'.join(str(x) for x in sorted(set(lrows.get(r['path'],[]))))])+'\n')
nonpub=[r for r in rows if r not in pub]
agg=collections.defaultdict(lambda:{'files':0,'bytes':0})
for r in nonpub:
    a=agg[(lab(r['folder']),r['tier'],cls(r['name']))]; a['files']+=1; a['bytes']+=r['size']
merged=collections.defaultdict(lambda:{'files':0,'bytes':0})
for (fo,t,c),a in agg.items():
    k=(fo,t,c) if a['files']>=3 else (fo,t,'(other)')
    merged[k]['files']+=a['files']; merged[k]['bytes']+=a['bytes']
with open(OUT+'/ROOT_AGGREGATE.tsv','w',encoding='utf-8') as f:
    f.write('folder\ttier\tclass\tfiles\tbytes_rounded_100KB\n')
    for (fo,t,c),a in sorted(merged.items()): f.write('\t'.join([fo,t,c,str(a['files']),str(rnd(a['bytes'])) if a['files']>=3 else ''])+'\n')
# ledger: one line per unit with hits (counts by class and tier); per-item lines only for bytes-public items (no names)
byrow=collections.defaultdict(list)
for h in hits: byrow[h['row']].append(h)
with open(OUT+'/LEDGER_ROOT.tsv','w',encoding='utf-8') as f:
    f.write('row\trepo\tunit\tmissing_proof\thits\tstrong\tweak_adjacent\tweak_pair\tweak_lone\thits_by_tier\tpublic_items_listed\n')
    for n in sorted(byrow):
        hs=byrow[n]; h0=hs[0]
        f.write('\t'.join([str(n),h0['repo'],h0['unit'],'Y' if h0['missing_proof'] else 'N',str(len(hs)),str(sum(1 for h in hs if h['match']=='STRONG')),str(sum(1 for h in hs if h['match']=='WEAK_ADJ')),str(sum(1 for h in hs if h['match']=='WEAK_PAIR')),str(sum(1 for h in hs if h['match']=='WEAK_LONE')),';'.join(f'{k}:{v}' for k,v in sorted(collections.Counter(h['tier'] for h in hs).items())),str(sum(1 for h in hs if h['tier'] in ('GIT_EXACT','ARCHIVE_EXACT')))])+'\n')
with open(OUT+'/LEDGER_ROOT_ROWS.tsv','w',encoding='utf-8') as f:
    f.write('row\trepo\tunit\tmissing_proof\tmatch\tfolder\ttier\tcontent_hash\tlocator\n')   # matched tokens are fragments of Dropbox names and stay local
    byp={r['path']:r for r in rows}
    for n in sorted(byrow):
        for h in byrow[n]:
            r=byp.get(h['path'])
            if r and r['tier'] in ('GIT_EXACT','ARCHIVE_EXACT'): f.write('\t'.join([str(n),h['repo'],h['unit'],'Y' if h['missing_proof'] else 'N',h['match'],lab(r['folder']),r['tier'],r['content_hash'],locator(r['flags'] or {})])+'\n')
hashed=[r for r in rows if r.get('content_hash') and r['size']>0]
rec=json.load(open(D+'/hash_receipt.json'))
def year(s): return s[:4] if s else ''
pubfolders={r['folder'] for r in pub}
def years(fo):
    if fo in pubfolders: return []   # a folder with per-file rows prints no time at all
    c=collections.Counter(year(r['client_modified']) for r in rows if r['folder']==fo and r['client_modified'])
    return sorted(y for y,n in c.items() if n>=3)
summ={'scope':'every file in the owner\'s Dropbox outside the Github/ subtree that main#594 covered: six top-level folders (labelled A to F), the root of the shared namespace that also carries Github/, and loose files at the account root; listed 2026-10-24 21:4xZ to 2026-10-25 00:0xZ; Github/ entries reached by the recursive listing were excluded',
 'files':{'total':len(rows),'hashed':len(hashed),'counted_only_personal_or_other_material':sum(1 for r in rows if r['tier']=='NOT_HASHED'),'empty':sum(1 for r in rows if r['tier']=='EMPTY'),'pending_hash':sum(1 for r in rows if r['tier']=='PENDING_HASH')},
 'by_folder':{lab(fo):{'files':n,'hashed':sum(1 for r in rows if r['folder']==fo and r.get('content_hash')),'years':years(fo)} for fo,n in collections.Counter(r['folder'] for r in rows).items()},
 'by_folder_rule':'labels are synthetic (the label-to-folder map is local); years list only years with three or more files and are omitted for a folder that has per-file rows; no month, day or time is published',
 'tiers':{t:{'files':sum(1 for r in rows if r['tier']==t),'files_in_published_parts':sum(1 for r in pub if r['tier']==t)+sum(a['files'] for (fo,tt,c),a in merged.items() if tt==t and a['files']>=3),
             'bytes_rounded_published_parts':rnd(sum(r['size'] for r in pub if r['tier']==t))+rnd(sum(a['bytes'] for (fo,tt,c),a in merged.items() if tt==t and a['files']>=3))} for t in sorted({r['tier'] for r in rows})},
 'tier_bytes_rule':'a tier byte total covers only the parts that print bytes elsewhere in this directory (per-file rows and aggregate cells of three files or more); files in count-only cells are excluded, so no withheld size can be recovered by differencing',
 'git_exact_where':dict(collections.Counter((r['flags'] or {}).get('git_where') for r in rows if r['tier']=='GIT_EXACT')),
 'github_tree_overlap':{'files_byte_identical_to_a_Github_file':sum(1 for r in hashed if (r['flags'] or {}).get('in_github_tree'))},
 'root_only':{'files':sum(1 for r in rows if r['tier']=='ROOT_ONLY'),'distinct_contents':len({r['content_hash'] for r in rows if r['tier']=='ROOT_ONLY'}),'by_folder':dict(collections.Counter(lab(r['folder']) for r in rows if r['tier']=='ROOT_ONLY')),'by_class':dict(collections.Counter(cls(r['name']) for r in rows if r['tier']=='ROOT_ONLY'))},
 'root_dup':{'files':sum(1 for r in rows if r['tier']=='ROOT_DUP'),'distinct_contents':len({r['content_hash'] for r in rows if r['tier']=='ROOT_DUP'})},
 'hash_receipt':rec,
 'ledger':{'rows':344,'name_hits':len(hits),'rows_hit':len(byrow),'missing_proof_rows_hit':sorted({h['row'] for h in hits if h['missing_proof']}),'hits_by_match':dict(collections.Counter(h['match'] for h in hits)),'hits_by_tier':dict(collections.Counter(h['tier'] for h in hits)),'hits_on_other_tiers_kept_local':hits_local,'note':'the main#594 name rule applied to the path below the folder; names are prose and identifiers mixed; WEAK_LONE is the weakest class; name evidence only; published hits cover only items in the GIT_EXACT, ARCHIVE_EXACT and GITHUB_TREE_COPY tiers, and the matched tokens stay local'},
 'row_level_rule':'no Dropbox path, name or time is published (nothing outside Github/ is import-named); per-file lines only for bytes already public (GIT_EXACT, ARCHIVE_EXACT) with hash, size and locator; everything else counted by folder label x class x tier over the non-listed files with classes below 3 merged and a cell still below 3 printing its count only',
 'row_level_rows':len(pub),'aggregate_files':len(nonpub),'bytes_rounding':'every byte total is rounded to the nearest 100 KB; tier totals cover published parts only'}
json.dump(summ,open(OUT+'/SUMMARY.json','w'),indent=1,sort_keys=True)
H=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
pins={'git_index':{'blobs':32221,'sha256':H(S+'/dbx/idx/git_index.json'),'tip_commits':{'main':next(l.split()[0] for l in open(S+'/dbx/idx/refs_main.txt') if l.split()[1]=='refs/remotes/origin/main'),'Math-':next(l.split()[0] for l in open(S+'/dbx/idx/refs_math.txt') if l.split()[1]=='refs/heads/main')}},
      'dropbox_github_hashes':{'entries':5724,'sha256':H(S+'/dbx/content_hashes.json')},'archives':{a:{'members':len(json.load(open(f'{S}/dbx/idx/{a}_index.json'))),'sha256_of_index':H(f'{S}/dbx/idx/{a}_index.json')} for a in ('a2136bc0','c4065c67')},
      'local_inventory':{'file':'dbxroot/inventory_root.tsv (paths, held locally by session 01Cz; offered to the owner or the CoS)','rows':len(rows),'sha256':H(D+'/inventory_root.tsv')},
      'local_rows':{'file':'dbxroot/root_rows.json','sha256':H(D+'/root_rows.json')},'ledger':{'file':'data/hypergraph/math_coverage_ledger_20261023.md','rows':344}}
json.dump(pins,open(OUT+'/PINS.json','w'),indent=1)
SUBS=[(S,'<SCRATCH>'),('/home/user/main','<MAIN_CHECKOUT>')]
for fn in ('build_inventory.py','classify_root.py','ledger_root.py','build_root_pub.py','readme_root.py'):
    if os.path.exists(D+'/'+fn):
        t=open(D+'/'+fn,encoding='utf-8').read()
        for a,b in SUBS: t=t.replace(a,b)
        open(OUT+'/own/'+fn,'w',encoding='utf-8').write(t)
subprocess.run([sys.executable,'-I',D+'/readme_root.py',OUT],check=True)
# guard
sys.path.insert(0,S+'/gdrive/tools_scan'); import drive_id_scan as scan
bad=[]; PAT=[r'https?://', r'drive\.google|docs\.google|dropbox\.com|dropboxusercontent|rlkey=', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}', r'\bns:\d', r'\bid:[A-Za-z0-9_-]{8,}', r'(?<![0-9A-Fa-f])\d{10,12}(?![0-9A-Fa-f])']
NSIDS={r['path'].split('//',1)[0].split(':',1)[1] for r in rows}   # the namespace ids, read from the local inventory, never written into this script
GITNAMES={p.rsplit('/',1)[-1] for v in json.load(open(S+'/dbx/idx/git_index.json')).values() for p in v['paths']}
for dp,_,fns in os.walk(OUT):
    for fn in fns:
        t=open(os.path.join(dp,fn),encoding='utf-8').read()
        for rx in PAT:
            if fn in ('build_root_pub.py','build_inventory.py','classify_root.py') and rx in t: continue
            if re.search(rx,t): bad.append((fn,rx))
        for hit in scan.hits_in_text(t): bad.append((fn,'drive-id '+str(hit)[:12]))
        for ns in NSIDS:
            if ns in t: bad.append((fn,'namespace id'))
        for r in rows:   # every published file, the scripts included: no Dropbox file name, no real folder name, no personal-name token
            if len(r['name'])>=6 and r['name'] not in GITNAMES and r['name'] in t: bad.append((fn,'dropbox name '+r['name'][:14]))
        for seg in LOC['folder_name_fragments']+[x for x in LOC['extra_personal_pattern'].split('|') if '\\' not in x]:
            if seg in t: bad.append((fn,'folder or personal name '+seg[:6]))
assert not bad, sorted(set(bad))[:12]
print('build ok: per-file rows',len(pub),'aggregate cells',len(merged),'ledger hits published',len(hits),'kept local',hits_local,'guard passed')
