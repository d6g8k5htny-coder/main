#!/usr/bin/env python3
"""Assemble audits/dropbox_import/2026-10-24/ from the local crosswalk data, in a scratch directory first; it is copied
into the PR worktree only after the publication guard passes.
Publication rule (01Fi's caveat, main#539 comment 6387941726): a full Dropbox path appears only for a file the 2026-09-27
import already named (MANIFEST.csv / NOT_IMPORTED.csv).  Every other file is published as (folder label, extension, bytes,
client_modified, content_hash, tier, locator, flags).  The folder label is the real top-level folder name only when that name
is already public: it is a path segment of an import-list path or of a Git path in main or Math-, or (names of six characters
or more) it occurs literally in a file tracked by origin/main or Math- main; otherwise "(withheld)".  Archive locators are
redacted component by component with the same test.  Files with personal-looking names appear in counts only.
No Dropbox id, rev, namespace id or URL anywhere."""
import csv, json, collections, hashlib, os, re, shutil, sys
from pathlib import PurePosixPath
S='<SCRATCH>'
SD='<SESSION_DIR>'
WT=S+'/dbxpr/wt'; FINAL=WT+'/audits/dropbox_import/2026-10-24'; OUT=S+'/dbxpr/build_out/2026-10-24'; M='<MAIN_CHECKOUT>/audits/dropbox_import/2026-09-27'
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT+'/own')
p3=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/phase3.tsv',encoding='utf-8')][1:]
# --- public names
imp_paths=set(); seg_pub=set()
for fn in ('MANIFEST.csv','NOT_IMPORTED.csv'):
    for r in csv.DictReader(open(f'{M}/{fn}',encoding='utf-8')):
        imp_paths.add(r['dropbox_path']); seg_pub.update(r['dropbox_path'].split('/')[:-1])
git=json.load(open(S+'/dbx/idx/git_index.json'))
for v in git.values():
    for p in v['paths']: seg_pub.update(p.split('/')[:-1])
gitgrep=json.load(open(S+'/dbx/idx/folder_gitgrep.json'))   # literal occurrences of each top-level base name in tracked files
base=lambda t: re.sub(r'( \(\d+\))+$','',t)
def public_name(t):
    b=base(t)
    if t in seg_pub or b in seg_pub: return True
    g=gitgrep.get(b)
    return bool(g and len(b)>=6 and (g['main_files'] or g['math_files']))
tops=collections.Counter((r[0].split('/')[1] if r[0].count('/')>=2 else '(Github root)') for r in p3)
pub={t for t in tops if t=='(Github root)' or public_name(t)}
withheld_bases={base(t) for t in tops if t not in pub}
reasons={t:('(Github root)' if t=='(Github root)' else 'segment of an import-list or Git path' if (t in seg_pub or base(t) in seg_pub) else 'literal in a tracked file' if t in pub else 'withheld') for t in tops}
def label(path):
    t=path.split('/')[1] if path.count('/')>=2 else '(Github root)'
    return base(t) if t in pub else '(withheld)'
def redact_locator(loc):
    out=[]
    for one in [x for x in loc.split(';') if x]:
        if ':' in one and one.split(':',1)[0] in ('a2136bc0','c4065c67'):
            lab,member=one.split(':',1)
            comps=[('(withheld)' if base(c) in withheld_bases else c) for c in member.split('/')]
            out.append(lab+':'+'/'.join(comps))
        else: out.append(one)
    return ';'.join(out)
def ext(name):
    n=name.lower()
    if n.endswith('.export.txt'): return '.export.txt'
    e=PurePosixPath(n).suffix; return e if e else '(none)'
rows=[]
for path,name,size,cm,tier,h,loc,flags in p3:
    rows.append(dict(path=path,name=name,size=int(size),cm=cm,tier=tier,hash=h,loc=redact_locator(loc),flags=flags,personal='PERSONAL_NAME' in flags,named=path in imp_paths,label=label(path),ext=ext(name)))
# --- CROSSWALK.tsv: row level only where nothing new is disclosed: the 2026-09-27 import already named the path, or the bytes
#     are already public (GIT_EXACT, ARCHIVE_EXACT, EMPTY: the hash is the hash of a public blob).  Everything else is aggregated.
PUBLIC_BYTES=('GIT_EXACT','ARCHIVE_EXACT','EMPTY')
def rowlevel(r): return (not r['personal']) and (r['named'] or r['tier'] in PUBLIC_BYTES)
with open(OUT+'/CROSSWALK.tsv','w',encoding='utf-8') as f:
    f.write('\t'.join(['dropbox_path_if_already_named','folder','extension','bytes','client_modified','content_hash','tier','git_or_archive_locator','flags'])+'\n')
    for r in sorted(rows,key=lambda r:(r['tier'],r['label'],r['hash'],r['size'])):
        if not rowlevel(r): continue
        f.write('\t'.join([r['path'] if r['named'] else '', r['label'], r['ext'], str(r['size']), r['cm'], r['hash'], r['tier'], r['loc'], r['flags']])+'\n')
agg=collections.defaultdict(lambda:{'files':0,'bytes':0,'distinct_contents':set(),'modified_on_or_after_import_day':0})
for r in rows:
    if rowlevel(r) or r['personal']: continue
    a=agg[(r['label'],r['ext'],r['tier'])]; a['files']+=1; a['bytes']+=r['size']; a['distinct_contents'].add(r['hash'] or r['path']); a['modified_on_or_after_import_day']+=('MODIFIED_ON_OR_AFTER_IMPORT_DAY' in r['flags'])
MIN_CELL=3   # no aggregate cell may describe fewer than three files (a single file's exact size is per-file data)
agg2={}
for (lab,ex,ti),a in agg.items():
    key=(lab,ex,ti) if a['files']>=MIN_CELL else (lab,'(other)',ti)
    b=agg2.setdefault(key,{'files':0,'bytes':0,'distinct_contents':set(),'modified_on_or_after_import_day':0,'merged_extensions':set()})
    b['files']+=a['files']; b['bytes']+=a['bytes']; b['distinct_contents']|=a['distinct_contents']; b['modified_on_or_after_import_day']+=a['modified_on_or_after_import_day']
    if key[1]=='(other)': b['merged_extensions'].add(ex)
with open(OUT+'/CROSSWALK_AGGREGATE.tsv','w',encoding='utf-8') as f:
    f.write('folder\textension\ttier\tfiles\tdistinct_contents\tbytes\tmodified_on_or_after_import_day\n')
    for (lab,ex,ti),a in sorted(agg2.items()):
        small=a['files']<MIN_CELL
        f.write('\t'.join([lab,ex,ti,str(a['files']),'withheld' if small else str(len(a['distinct_contents'])),'withheld' if small else str(a['bytes']),'withheld' if small else str(a['modified_on_or_after_import_day'])])+'\n')
summary_agg_note=f'cells with fewer than {MIN_CELL} files are merged into an (other) extension row per folder and tier; a cell still below {MIN_CELL} files prints its file count only'
# --- SUMMARY.json
tiers=collections.defaultdict(lambda:{'files':0,'bytes':0})
for r in rows: tiers[r['tier']]['files']+=1; tiers[r['tier']]['bytes']+=r['size']
byfolder=collections.defaultdict(collections.Counter)
for r in rows: byfolder[r['label']][r['tier']]+=1
mod=[r for r in rows if 'MODIFIED_ON_OR_AFTER_IMPORT_DAY' in r['flags']]
DO=('MISSING','DUP_IN_DROPBOX'); donly={r['hash']:r for r in rows if r['tier'] in DO}
EX=('GIT_EXACT','ARCHIVE_EXACT')
summary={'listed_at':'2026-10-24T18:31Z (list_folder, 11 pages, has_more=false)','files':len(rows),'bytes':sum(r['size'] for r in rows),
 'tiers':dict(tiers),'folders':{'public_named':sorted(t for t in tops if t in pub),'withheld_count':sum(1 for t in tops if t not in pub),'files_in_withheld_folders':sum(c for t,c in tops.items() if t not in pub),'reason_by_folder':{(t if t in pub else '(withheld folder %d)'%i):reasons[t] for i,t in enumerate(sorted(tops),1)}},
 'by_folder':{k:dict(v) for k,v in sorted(byfolder.items())},
 'modified_on_or_after_2026-09-27':{'files':len(mod),'by_tier':dict(collections.Counter(r['tier'] for r in mod))},
 'personal_name_files_counted_only':sum(1 for r in rows if r['personal']),
 'named_by_import_lists':sum(1 for r in rows if r['named']),
 'import_listed_hashes':{'files':sum(1 for r in rows if 'IMPORT_LISTED:' in r['flags']),'by_tier':dict(collections.Counter(r['tier'] for r in rows if 'IMPORT_LISTED:' in r['flags']))},
 'git_exact_on_tip':{'ON_TIP':sum(1 for r in rows if 'ON_TIP' in r['flags']),'HISTORY_OR_BRANCH_ONLY':sum(1 for r in rows if 'HISTORY_OR_BRANCH_ONLY' in r['flags'])},
 'dropbox_only':{'note':'MISSING plus DUP_IN_DROPBOX: bytes found in no Git blob (fetched refs of main, main and PR refs of Math-) and in neither archive; DUP means at least one other Dropbox copy of the same bytes',
                 'files':sum(1 for r in rows if r['tier'] in DO),'distinct_contents':len(donly),'bytes_distinct':sum(r['size'] for r in donly.values()),
                 'by_extension_distinct':dict(collections.Counter(r['ext'] for r in donly.values()).most_common(30)),
                 'by_folder_distinct':dict(collections.Counter(r['label'] for r in donly.values()))},
 'exact_in_git_or_archive':{'files':sum(1 for r in rows if r['tier'] in EX),'distinct_contents':len({r['hash'] for r in rows if r['tier'] in EX})}}
json.dump(summary,open(OUT+'/SUMMARY.json','w'),indent=1,sort_keys=True)
# --- LEDGER_CROSSWALK.tsv: per-file rows only for files whose folder is named or that the import named; withheld-folder hits are counted per unit
lc=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/ledger_crosswalk.tsv',encoding='utf-8')][1:]
bypath={r['path']:r for r in rows}; agg2=collections.defaultdict(collections.Counter)
with open(OUT+'/LEDGER_CROSSWALK.tsv','w',encoding='utf-8') as f:
    f.write('\t'.join(['ledger_row','repo','unit','missing_proof','missing_impl','match','matched_tokens','kind','dropbox_path_if_already_named','folder','extension','bytes','client_modified','tier','content_hash','git_or_archive_locator','locator_in_unit','locator_count','flags'])+'\n')
    for row,repo,unit,mp,mi,match,kind,path,size,cm,tier,h,loc,flags,mt,inu,nloc in lc:
        r=bypath.get(path)
        if r is None or r['personal']: continue
        if not rowlevel(r) or (r['label']=='(withheld)' and not r['named']): agg2[(row,unit)][f'{match}:{kind}:{r["tier"]}']+=1; continue
        f.write('\t'.join([row,repo,unit,mp,mi,match,mt,kind,r['path'] if r['named'] else '',r['label'],r['ext'],str(r['size']),r['cm'],r['tier'],r['hash'],r['loc'],inu,nloc,r['flags']])+'\n')
summary['ledger_hits_aggregated']={f'{row} {unit}':dict(c) for (row,unit),c in sorted(agg2.items(),key=lambda kv:int(kv[0][0]))}
summary['row_level_rule']='per-file rows only for paths the 2026-09-27 import already named or for files whose bytes are already public (GIT_EXACT, ARCHIVE_EXACT, EMPTY); all other files are in CROSSWALK_AGGREGATE.tsv by folder, extension and tier, with their row-level hashes held locally pending an owner or CoS publication decision'
summary['hash_receipt']=json.load(open(S+'/dbx/hash_receipt.json'))
summary['row_level_rows']=sum(1 for r in rows if rowlevel(r))
summary['aggregate_rule']=summary_agg_note
json.dump(summary,open(OUT+'/SUMMARY.json','w'),indent=1,sort_keys=True)
# --- HASH_UNAVAILABLE.tsv
import glob
with open(OUT+'/HASH_UNAVAILABLE.tsv','w',encoding='utf-8') as f:
    f.write('dropbox_path_if_already_named\tfolder\textension\tbytes\treason\n')
    seen=set()
    for fp in sorted(glob.glob(S+'/dbx/agentout/*.fail.txt')):
        for l in open(fp,encoding='utf-8'):
            c=l.rstrip('\n').split('\t'); q=c[0].split('//',1)[1] if '//' in c[0] else c[0]; r=bypath.get(q)
            if r and not r['personal'] and q not in seen:
                seen.add(q); f.write('\t'.join([r['path'] if r['named'] else '', r['label'], r['ext'], str(r['size']), 'NOT_ALLOWED' if 'NOT_ALLOWED' in l else 'error'])+'\n')
# --- own scripts, with local path roots replaced
SUBS=[(S,'<SCRATCH>'),(SD,'<SESSION_DIR>'),('<SESSION_PROJECTS>','<SESSION_PROJECTS>'),('<MAIN_CHECKOUT>','<MAIN_CHECKOUT>')]
for fn in ('classify_ns.py','classify_hash.py','ledger_crosswalk.py','phase2_harvest.py','phase2_collect.py','phase2_batches.py','nextbatches.py','idx/zip_index.py','pages/merge.py'):
    src=S+'/dbx/'+fn
    if os.path.exists(src):
        t=open(src,encoding='utf-8').read()
        for a,b in SUBS: t=t.replace(a,b)
        for wb in withheld_bases: t=t.replace("'"+wb+"'","'(withheld)'")   # a withheld folder name used as a priority prefix
        open(OUT+'/own/'+os.path.basename(fn),'w',encoding='utf-8').write(t)
for fn in ('build_pr.py','readme_gen.py'):
    t=open(S+'/dbxpr/'+fn,encoding='utf-8').read()
    for a,b in SUBS: t=t.replace(a,b)
    open(OUT+'/own/'+fn,'w',encoding='utf-8').write(t)
H=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
refs_main=[l.split() for l in open(S+'/dbx/idx/refs_main.txt')]; refs_math=[l.split() for l in open(S+'/dbx/idx/refs_math.txt')]
pins={'local_full_crosswalk':{'file':'dbx/phase3.tsv (path-level, held locally by session 01Cz; offered to the owner or the CoS)','rows':len(p3),'sha256':H(S+'/dbx/phase3.tsv')},
      'local_inventory':{'file':'dbx/inventory.tsv','rows':sum(1 for _ in open(S+'/dbx/inventory.tsv')),'sha256':H(S+'/dbx/inventory.tsv')},
      'local_content_hashes':{'file':'dbx/content_hashes.json','sha256':H(S+'/dbx/content_hashes.json'),'entries':len(json.load(open(S+'/dbx/content_hashes.json')))},
      'git_index':{'blobs':len(git),'sha256':H(S+'/dbx/idx/git_index.json'),'main':{'refs':len(refs_main),'origin_main':next(c for c,r in refs_main if r=='refs/remotes/origin/main'),'history':'full after git fetch --unshallow (1,493 commits on origin/main)'},
                   'Math-':{'refs':len(refs_math),'main':next(c for c,r in refs_math if r=='refs/heads/main'),'other_refs':'32 refs/pr/* heads'}},
      'archives':{'a2136bc0':{'members':len(json.load(open(S+'/dbx/idx/a2136bc0_index.json'))),'sha256_of_index':H(S+'/dbx/idx/a2136bc0_index.json')},
                  'c4065c67':{'members':len(json.load(open(S+'/dbx/idx/c4065c67_index.json'))),'sha256_of_index':H(S+'/dbx/idx/c4065c67_index.json')}},
      'ledger':{'file':'data/hypergraph/math_coverage_ledger_20261023.md','last_commit':'6c604f34','read_at':'9c03589b (unchanged through 08c98c8f)'}}
json.dump(pins,open(OUT+'/PINS.json','w'),indent=1)
# README is written by readme_gen.py into OUT before the guard runs (it imports nothing from here)
import subprocess; subprocess.run([sys.executable,'-I',S+'/dbxpr/readme_gen.py',OUT],check=True)
# --- guard, then copy into the worktree
bad=[]
PAT=[r'ns:\d{3,}', r'1465\d{6,}', r'dropboxusercontent', r'dropbox\.com', r'rlkey=', r'https?://(?!github\.com/d6g8k5htny-coder/)', r'\bid:[A-Za-z0-9_-]{8,}', r'"rev"', r'(?:drive|docs)\.google\.com']  # session ids in Git paths are public paths
for dp,_,fns in os.walk(OUT):
    for fn in fns:
        t=open(os.path.join(dp,fn),encoding='utf-8').read()
        for rx in PAT:
            if fn=='build_pr.py' and rx in t: continue   # the guard's own pattern strings
            if re.search(rx,t): bad.append((fn,rx))
        for wb in withheld_bases:
            if len(wb)>=4 and wb in t and fn!='SUMMARY.json': bad.append((fn,'withheld base '+wb))
assert not bad, bad
if os.path.exists(FINAL): shutil.rmtree(FINAL)
shutil.copytree(OUT,FINAL)
print('tiers',dict(tiers)); print('withheld folders',sorted(withheld_bases)); print('dropbox-only',summary['dropbox_only']['files'],summary['dropbox_only']['distinct_contents'],summary['dropbox_only']['bytes_distinct']); print('guard passed; copied to worktree')
