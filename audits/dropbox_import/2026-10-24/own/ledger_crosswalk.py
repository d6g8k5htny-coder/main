#!/usr/bin/env python3
"""Name crosswalk: ledger units (data/hypergraph/math_coverage_ledger_20261023.md) vs the Dropbox Github/ inventory.
STRONG: the Dropbox basename stem (lowercased, ' (n)' copy suffix and extension removed) equals the unit's basename stem;
for generic basenames (README, manifest, gate...) the unit's parent directory name must also be a path segment.
WEAK: two consecutive distinctive tokens of the unit occur adjacently in the path (as in the unit name), or one CODE token
(letters+digits, e.g. lm006, iba2, c026) with distinct-basename frequency at most 30, or one token with frequency at most 3.
Tokens in the stop list or with distinct-basename frequency >= 60 (CODE tokens: > 150) carry no evidence.  Name matching is weak evidence; the tier column (phase 3) is the byte evidence.
Writes dbx/ledger_crosswalk.tsv (with matched_tokens, locator_in_unit, locator_count) and dbx/ledger_crosswalk_summary.json."""
import json, re, collections
S='<SCRATCH>'
rows=json.load(open(S+'/dbx/ledger/rows.json'))
p3={}
for l in open(S+'/dbx/phase3.tsv',encoding='utf-8'):
    c=l.rstrip('\n').split('\t')
    if c[0]!='path': p3[c[0]]=c
STOP={'readme','md','py','json','protocol','review','reviews','numerics','recon','audit','workflow','repair','coverage','trigger','legacy','exact','check','checks',
      'test','tests','frontiers','formal','imports','coefficients','certificates','repairs','claims','experiments','lean','gitignore','toml','data','source','all',
      'planar','dims','constant','constants','value','laws','law','rate','bridge','gate','round2','external','direct','import','diagnostic','evidence','originals',
      'blueprint','manifest','lake','lakefile','toolchain','compatibility','scope','glossary','lineage','validation','universallaw','v0','v1','v2','embedded','small',
      'certified','closed','form','integrity','file','files','numeric','protocols','candidates','pending','hole','substitutes','substitute','completion','mechanism',
      'transfer','difference','floor','floors','remainder','vanishing','window','coefficient','failure','elder','cap','torus','three','fronts','claude','the','and'}
GENERIC_BASE={'readme','scope','manifest','gate','lakefile','evidence','originals','blueprint','tests','gitignore','compatibility','glossary','lake-manifest','lean-toolchain'}
CODE=re.compile(r'^(?=.*[a-z])(?=.*\d)[a-z0-9]{3,}$')
def stem(name):
    n=name.lower()
    n=re.sub(r'\.export\.txt$','',n); n=re.sub(r'( \(\d+\))+(\.[a-z0-9]{1,6})?$',lambda m:m.group(2) or '',n)
    n=re.sub(r'\.[a-z0-9]{1,6}$','',n); return n
def unit_parts(unit):
    u=unit.rstrip('/'); base=u.split('/')[-1]; parent=u.split('/')[-2].lower() if '/' in u else ''
    b=re.sub(r'\.[a-z0-9]{1,6}$','',base.lower()); b=re.sub(r'_(\d{8})(_r\d+)?$','',b); b=re.sub(r'_claude$','',b)
    toks=[t for t in re.split(r'[_\-. ]+',b) if t]
    return b, parent, toks
inv=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/inventory.tsv',encoding='utf-8')]
files=[(r[0],r[0].lower(),int(r[2] or 0),r[3],stem(r[1])) for r in inv if r[5]=='file' and not r[0].lower().endswith(('.pyc','.url','.ds_store')) and '__pycache__' not in r[0]]
# document frequency over distinct basename stems
df=collections.Counter()
for st in {f[4] for f in files}:
    for t in set(re.split(r'[_\-. ]+',st)):
        if t: df[t]+=1
DOC=('.md','.pdf','.tex','.txt','.docx','.rtf','.html','.ipynb'); CODEEXT=('.py','.lean','.sage','.jl','.m','.c','.cpp','.rs','.sh','.yml','.yaml')
out=[]; summ={}
for r in rows:
    b,parent,toks=unit_parts(r['unit'])
    if not toks: continue
    dist=[t for t in toks if t not in STOP and not t.isdigit() and len(t)>=3 and ((CODE.match(t) and df[t]<=150) or df[t]<60)]
    dropped=[t for t in toks if t not in dist and t not in STOP and not t.isdigit()]
    generic=b in GENERIC_BASE or toks[0] in GENERIC_BASE
    hits=[]
    for path,low,size,cm,st in files:
        segs=low.split('/')
        if generic and parent and parent not in segs: continue
        low=('/'.join(segs[2:]) if len(segs)>=3 else segs[-1])  # match below the top-level folder: folder names like K3_SIDE24_LB are not file evidence
        matched=[]
        if st==b: strength='STRONG'
        else:
            matched=[t for t in dist if re.search(r'(?<![a-z0-9])'+re.escape(t)+r'(?![a-z0-9])', low)]
            adjacent=len(dist)>=2 and re.search(r'(?<![a-z0-9])'+r'[_\- .]*'.join(map(re.escape,dist))+r'(?![a-z0-9])', low)
            pairs=[(a,b) for a,b in zip(dist,dist[1:]) if re.search(r'(?<![a-z0-9])'+re.escape(a)+r'[_\- .]*'+re.escape(b)+r'(?![a-z0-9])', low)]
            lone=[m for m in matched if re.search(r'(?<![a-z0-9])'+re.escape(m)+r'(?![a-z0-9])', st)]  # a lone token counts only in the basename
            if adjacent or pairs or (len(lone)==1 and len(matched)==1 and ((CODE.match(lone[0]) and df[lone[0]]<=30) or df[lone[0]]<=3)): strength='WEAK'
            else: continue
            if not matched: matched=dist
        name=low.rsplit('/',1)[-1]
        kind='doc' if name.endswith(DOC) or name.endswith('.export.txt') else ('code' if name.endswith(CODEEXT) else 'other')
        t=p3.get(path); tier=t[4] if t else '?'; h=t[5] if t else ''; loc=t[6] if t else ''; flags=t[7] if t else ''
        locs=[x for x in loc.split(';') if x]
        in_unit='Y' if any(x.split(':',1)[-1].startswith(r['unit'].rstrip('/')+'/') or x.split(':',1)[-1]==r['unit'] for x in locs) else 'N'
        nloc=next((int(f.split(':')[1]) for f in flags.split(',') if f.startswith('LOCATORS:')), len(locs))
        hits.append((strength,kind,path,size,cm,tier,h,loc,flags,'+'.join(matched) if matched else 'stem',in_unit,str(nloc)))
    hits.sort(key=lambda x:(x[0]!='STRONG',x[1]!='doc',x[2]))
    mp='Y' if r['proof']=='MISSING' else 'N'; mi='Y' if r['impl'] in ('MISSING','—','-','') else 'N'
    summ[r['n']]={'unit':r['unit'],'repo':r['repo'],'missing_proof':mp,'missing_impl':mi,'strong':sum(1 for h in hits if h[0]=='STRONG'),'weak':sum(1 for h in hits if h[0]=='WEAK'),
                  'docs':sum(1 for h in hits if h[1]=='doc'),'code':sum(1 for h in hits if h[1]=='code'),'tokens':dist,'generic_tokens_dropped':dropped}
    for h in hits: out.append((str(r['n']),r['repo'],r['unit'],mp,mi)+tuple(map(str,h)))
with open(S+'/dbx/ledger_crosswalk.tsv','w',encoding='utf-8') as f:
    f.write('\t'.join(['row','repo','unit','missing_proof','missing_impl','match','kind','dropbox_path','bytes','client_modified','tier','content_hash','locator','flags','matched_tokens','locator_in_unit','locator_count'])+'\n')
    for o in out: f.write('\t'.join(o)+'\n')
json.dump(summ,open(S+'/dbx/ledger_crosswalk_summary.json','w'),indent=0)
mp=[v for v in summ.values() if v['missing_proof']=='Y']
print('units',len(summ),'hit rows',len(out),'strong',sum(1 for o in out if o[5]=='STRONG'),'weak',sum(1 for o in out if o[5]=='WEAK'))
print('missing-proof units',len(mp),'with any hit',sum(1 for v in mp if v['strong']+v['weak']),'with STRONG',sum(1 for v in mp if v['strong']),'with doc hit',sum(1 for v in mp if v['docs']),'zero tokens',sum(1 for v in mp if not v['tokens']))
mi=[v for v in summ.values() if v['missing_impl']=='Y']; print('missing-impl units',len(mi),'with code hit',sum(1 for v in mi if v['code']))
for n,v in sorted(summ.items()):
    if v['strong']+v['weak']: print(n, v['unit'][:55], 'S',v['strong'],'W',v['weak'],'docs',v['docs'],'code',v['code'], v['tokens'], 'MP' if v['missing_proof']=='Y' else '')
