#!/usr/bin/env python3
"""Ledger name join for the files outside Github/ (the main#594 rule: STRONG stem equality; WEAK_ADJ all distinctive tokens adjacent; WEAK_PAIR a consecutive pair;
WEAK_LONE one rare token in the basename stem), applied to the path below the folder. Writes ledger_root_hits.json (local; paths kept)."""
import json, re, collections
S='<SCRATCH>'; D=S+'/dbxroot'
rows=json.load(open(D+'/root_rows.json'))
src=open(S+'/dbx/ledger_crosswalk.py').read()
STOP=eval(src[src.index('STOP='):src.index('GENERIC_BASE=')][5:]); GENERIC_BASE=eval(src[src.index('GENERIC_BASE=')+13:src.index('CODE=re')])
CODE=re.compile(r'^(?=.*[a-z])(?=.*\d)[a-z0-9]{3,}$')
def stem(name):
    n=name.lower(); n=re.sub(r'( \(\d+\))+(\.[a-z0-9]{1,6})?$',lambda m:m.group(2) or '',n); n=re.sub(r'\.[a-z0-9]{1,6}$','',n); return n
def unit_parts(unit):
    u=unit.rstrip('/'); base=u.split('/')[-1]; parent=u.split('/')[-2].lower() if '/' in u else ''
    b=re.sub(r'\.[a-z0-9]{1,6}$','',base.lower()); b=re.sub(r'_(\d{8})(_r\d+)?$','',b); b=re.sub(r'_claude$','',b)
    return b,parent,[t for t in re.split(r'[_\-. ]+',b) if t]
files=[r for r in rows if r['size']>0 and not r['personal']]
df=collections.Counter()
for st in {stem(r['name']) for r in files}:
    for t in set(re.split(r'[_\-. ]+',st)):
        if t: df[t]+=1
ledger=json.load(open(S+'/dbx/ledger/rows.json')); hits=[]
for L in ledger:
    b,parent,toks=unit_parts(L['unit'])
    if not toks: continue
    dist=[t for t in toks if t not in STOP and not t.isdigit() and len(t)>=3 and ((CODE.match(t) and df[t]<=150) or df[t]<60)]
    generic=b in GENERIC_BASE or toks[0] in GENERIC_BASE
    for r in files:
        rel=r['path'].split('//',1)[1].lower(); segs=rel.split('/')
        if generic and parent and parent not in segs: continue
        low='/'.join(segs[1:]) if len(segs)>=2 else segs[-1]; st=stem(r['name'])
        matched=[]
        if st==b: strength='STRONG'
        else:
            matched=[t for t in dist if re.search(r'(?<![a-z0-9])'+re.escape(t)+r'(?![a-z0-9])',low)]
            adjacent=len(dist)>=2 and re.search(r'(?<![a-z0-9])'+r'[_\- .]*'.join(map(re.escape,dist))+r'(?![a-z0-9])',low)
            pairs=[(a,c) for a,c in zip(dist,dist[1:]) if re.search(r'(?<![a-z0-9])'+re.escape(a)+r'[_\- .]*'+re.escape(c)+r'(?![a-z0-9])',low)]
            lone=[m for m in matched if re.search(r'(?<![a-z0-9])'+re.escape(m)+r'(?![a-z0-9])',st)]
            if adjacent: strength='WEAK_ADJ'
            elif pairs: strength='WEAK_PAIR'
            elif len(lone)==1 and len(matched)==1 and ((CODE.match(lone[0]) and df[lone[0]]<=30) or df[lone[0]]<=3): strength='WEAK_LONE'
            else: continue
            if not matched: matched=dist
        hits.append({'row':L['n'],'repo':L['repo'],'unit':L['unit'],'missing_proof':L['proof']=='MISSING','match':strength,'tokens':'+'.join(matched) if matched else 'stem','path':r['path'],'folder':r['folder'],'tier':r['tier']})
json.dump(hits,open(D+'/ledger_root_hits.json','w'),indent=0)
mp=[h for h in hits if h['missing_proof']]
print('hits',len(hits),'by match',dict(collections.Counter(h['match'] for h in hits)),'rows hit',len({h['row'] for h in hits}),'MISSING-proof rows hit',len({h['row'] for h in mp}),'STRONG',sum(1 for h in hits if h['match']=='STRONG'))
