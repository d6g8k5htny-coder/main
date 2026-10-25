#!/usr/bin/env python3
"""Merge the live connector listing (items modified after 2026-09-17, written by the listing agents as parts/w*.tsv) into drive_rows.json.
live_status: CHANGED_AFTER_SNAPSHOT (seen in the live listing with a later modifiedTime; a digest from the snapshot is then STALE unless a dated
delta re-hashed it at or after that time), UNCHANGED (not in the listing; valid only if every window completed), NEW_SINCE_SNAPSHOT (live id unknown
to the snapshot and deltas). Ids stay local."""
import json, glob, os, collections, sys
G='<SCRATCH>/gdrive'
WINDOWS=['w%02d'%i for i in range(1,11)]
present={os.path.basename(f)[:-4] for f in glob.glob(G+'/parts/w*.tsv')}
complete=json.load(open(G+'/parts/complete.json')) if os.path.exists(G+'/parts/complete.json') else {}
live={}
dups=0
for f in sorted(glob.glob(G+'/parts/w*.tsv')):
    for l in open(f,encoding='utf-8'):
        c=l.rstrip('\n').split('\t')
        if len(c)<6 or not c[0]: continue
        rec={'id':c[0],'title':c[1],'mime':c[2],'size':int(c[3]) if c[3].isdigit() else None,'created':c[4],'modified':c[5],'parent':c[6] if len(c)>6 else '','window':os.path.basename(f)[:-4]}
        if c[0] in live:
            dups+=1
            if rec['modified']>live[c[0]]['modified']: live[c[0]]=rec
        else: live[c[0]]=rec
rows=json.load(open(G+'/drive_rows.json')); known={r['id'] for r in rows}
folders={r['id']:r for r in rows if r['folder']}
all_complete=all(complete.get(w) for w in WINDOWS)
for r in rows:
    L=live.get(r['id'])
    if L:
        r['live_modified']=L['modified']; r['live_size']=L['size']
        # the search cutoff (2026-09-17T00:00Z) precedes the snapshot taken that day, so a modifiedTime on 17 September is ambiguous;
        # an item whose modifiedTime equals its createdTime was never modified after creation, so a snapshot digest for it is still current
        if L['created'] and L['modified'] and L['created'][:19]==L['modified'][:19]: r['live_status']='SAME_AS_CREATED'
        elif L['modified']<'2026-09-18T00:00:00Z': r['live_status']='MODIFIED_ON_SNAPSHOT_DAY'
        else: r['live_status']='MODIFIED_AFTER_SNAPSHOT_DAY'
        rehashed=bool(r.get('modifiedTime') and r['modifiedTime']>=L['modified'])   # a dated delta re-hashed it at or after its last modification
        if not r['sha256']: r['digest_state']='NONE'
        elif r['live_status']=='SAME_AS_CREATED' or rehashed: r['digest_state']='CURRENT'
        elif r['live_status']=='MODIFIED_ON_SNAPSHOT_DAY': r['digest_state']='UNCERTAIN_SNAPSHOT_DAY'
        else: r['digest_state']='STALE_SAME_SIZE' if (r['bytes'] is not None and L['size'] is not None and L['size']==r['bytes']) else 'STALE'
    else:
        r['live_status']='NOT_MODIFIED_SINCE_2026-09-17' if all_complete else 'NOT_MODIFIED_IF_LISTING_COMPLETE'; r['digest_state']='CURRENT' if r['sha256'] else 'NONE'
new=[v for k,v in live.items() if k not in known]
for v in new:
    p=folders.get(v['parent']); v['parent_known']=bool(p); v['parent_hub']=p['hub'] if p else ''; v['parent_title']=p['title'] if p else ''
json.dump(rows,open(G+'/drive_rows.json','w'),indent=0); json.dump(new,open(G+'/new_items.json','w'),indent=0)
json.dump({'windows_present':sorted(present),'complete':complete,'all_complete':all_complete,'live_items':len(live),'duplicate_lines':dups,'new_since_snapshot':len(new)},open(G+'/liveness_summary.json','w'),indent=1)
fil=[r for r in rows if not r['folder']]
print('windows present',sorted(present),'all complete',all_complete,'live items',len(live),'dup lines',dups)
print('live_status (files):',dict(collections.Counter(r['live_status'] for r in fil)))
print('digest_state for hashed files:',dict(collections.Counter(r['digest_state'] for r in fil if r['sha256'])))
print('new since snapshot:',len(new),'parent known',sum(1 for v in new if v['parent_known']),'by mime class:',dict(collections.Counter(v['mime'].split('/')[-1][:22] for v in new).most_common(8)))
print('new by parent hub (known parents):',dict(collections.Counter(v['parent_hub'][:30] for v in new if v['parent_known']).most_common(8)))
