import json, collections, re
S='<SCRATCH>'; G=S+'/gdrive'
rows=json.load(open(G+'/drive_rows.json')); hits=json.load(open(G+'/ledger_drive_hits.json')); nodbx=set(json.load(open(G+'/ledger_rows_without_dropbox_hit.json')))
ledger={r['n']:r for r in json.load(open(S+'/dbx/ledger/rows.json'))}
fil=[r for r in rows if not r['folder']]
# 1. the 13 ledger rows: units and hits
print('== MISSING-proof rows with a Drive name hit and no Dropbox hit')
for n in sorted({h['row'] for h in hits if h['missing_proof']} & nodbx):
    hs=[h for h in hits if h['row']==n]
    print(f"row {n} {ledger[n]['repo']}:{ledger[n]['unit']}  hits={len(hs)} tiers={dict(collections.Counter(h['tier'] for h in hs))} tokens={sorted({h['tokens'] for h in hs})[:3]}")
    for h in hs[:3]: print('     ', h['hub'][:28], '|', h['title'][:90], '|', h['tier'], '|', h['drv_map'])
# 6. Drive files whose bytes are in Dropbox but not Git: what Dropbox tier are those paths?
p3={}
for l in open(S+'/dbx/phase3.tsv',encoding='utf-8'):
    c=l.rstrip('\n').split('\t')
    if c[0]!='path': p3[c[0]]=c
dbx=json.load(open(S+'/dbx/content_hashes.json')); by_ch=collections.defaultdict(list)
for p,v in dbx.items(): by_ch[v['content_hash']].append(p)
dbxonly_ch={v['content_hash'] for p,v in dbx.items() if p in p3 and p3[p][4] in ('MISSING','DUP_IN_DROPBOX')}
drive_ch={r['flags'].get('content_hash') for r in fil if r['sha256'] and r['flags'].get('content_hash')}
both=dbxonly_ch & drive_ch
print('== Dropbox-only distinct contents (MISSING/DUP) that also exist in Drive (digest-level):',len(both),'of',len(dbxonly_ch))
# how many Drive DBX_EXACT files correspond to Dropbox tiers
tiers=collections.Counter()
for r in fil:
    if r['tier']=='DBX_EXACT':
        ps=by_ch.get(r['flags']['content_hash'],[]); t=sorted({p3[p][4] for p in ps if p in p3}); tiers[','.join(t)]+=1
print('   Drive DBX_EXACT files by the Dropbox tier of their copies:',dict(tiers))
# 2. publication: how many non-personal file rows have a public title (DRV maps) or public bytes
pub=[r for r in fil if not r['personal_lane'] and (r['drv_map'] or r['tier'] in ('GIT_EXACT','ARCHIVE_EXACT'))]
print('== rows eligible for a per-file line (title in DRV maps or bytes public):',len(pub),'of',len([r for r in fil if not r['personal_lane']]))
print('   DRIVE_ONLY files with a public title (DRV maps):',sum(1 for r in fil if r['tier'].startswith('DRIVE_ONLY') and r['drv_map'] and not r['personal_lane']),'; DBX_EXACT with public title:',sum(1 for r in fil if r['tier']=='DBX_EXACT' and r['drv_map']))
# 3. aggregate for publication: hub x tier x mime-class, min cell 3
def mclass(m):
    m=m.lower()
    return 'google-doc' if 'google-apps' in m else 'zip' if 'zip' in m else 'pdf' if 'pdf' in m else 'markdown' if 'markdown' in m else 'text' if 'text/' in m else 'json' if 'json' in m else 'image' if m.startswith('image/') else 'office' if 'officedocument' in m or 'msword' in m else 'other'
agg=collections.Counter((r['hub'][:45],r['tier'],mclass(r['mime'])) for r in fil if not r['personal_lane'])
small=sum(1 for k,v in agg.items() if v<3); print('== hub x tier x class cells:',len(agg),'cells below 3:',small)
# DRIVE_ONLY by hub and sub-lane (second path segment) for a sense of where they are
sub=collections.Counter((r['hub'][:30], (r['path'].split('/')[1][:45] if r['path'].count('/')>=2 else '(hub root)')) for r in fil if r['tier'].startswith('DRIVE_ONLY') and not r['personal_lane'])
print('== DRIVE_ONLY by hub/sub-lane (top 12):'); [print('   ',n,'|',h,'|',s) for (h,s),n in sub.most_common(12)]
json.dump({'both_dbx_drive':len(both),'dbxonly_ch':len(dbxonly_ch)},open(G+'/analysis2.json','w'))
