#!/usr/bin/env python3
"""Validate a dated, additive folder-membership reconciliation.

An omitted/null metadata field is unknown, not an empty value or a move.
Absent membership means 'not observed', never deletion. No global account,
historical-revision, native-behavior or scientific completeness is asserted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.manifest_integrity_check import local_file

DEFAULT='drive/deltas/2026-09-20/DG-RECON-20260920'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


def metadata_value(record,key):
    """Keep missing, reported null, observed empty and observed values distinct."""
    if key not in record:return ('MISSING',None)
    if record[key] is None:return ('NULL',None)
    return ('OBSERVED',record[key])


def index_observations(folders):
    by_id={}; memberships={}; folder_ids=set()
    for folder in folders:
        fid=folder['id']
        require(fid not in folder_ids,'duplicate folder observation')
        folder_ids.add(fid)
        require(folder.get('complete') is True,'incomplete folder observation')
        items=folder['items']
        require(type(folder.get('count')) is int and folder['count']==len(items)<1000,'folder cap or count mismatch')
        require(len({x['id'] for x in items})==len(items),'duplicate item within folder')
        for item in items:
            ident=item['id'];require(isinstance(ident,str) and ident,'empty source identity')
            if ident in by_id:
                require(item==by_id[ident],'conflicting observations within snapshot')
            by_id[ident]=item;memberships.setdefault(ident,set()).add(fid)
    return by_id,memberships,folder_ids


def reconcile(previous,current):
    old,om,of=index_observations(previous)
    new,nm,nf=index_observations(current)
    require(of==nf,'different folder universes require explicit new-scope review')
    changed=[];unknown=[];membership=[]
    for ident in sorted(set(old)&set(new)):
        fields=[]
        for key in ('title','mime_type','size','modified_time'):
            a,b=metadata_value(old[ident],key),metadata_value(new[ident],key)
            if a[0]=='OBSERVED' and b[0]=='OBSERVED':
                if a[1]!=b[1]:fields.append(key)
            elif a!=b:unknown.append({'id':ident,'field':key,'before':a[0],'after':b[0]})
        if fields:changed.append({'id':ident,'fields':fields})
        if om[ident]!=nm[ident]:
            membership.append({'id':ident,'before':sorted(om[ident]),'after':sorted(nm[ident])})
    return {'folders':len(nf),'observed_unique':len(new),'added':sorted(set(new)-set(old)),
            'not_observed':sorted(set(old)-set(new)), 'changed':changed,
            'membership_changes':membership,'unknown_fields':unknown,
            'parent_metadata_states':dict(Counter(metadata_value(x,'parent_ids')[0] for x in new.values()))}


def check(root=ROOT,bundle=DEFAULT):
    base=local_file(root,bundle+'/reconciliation.json').parent
    spec=json.loads((base/'reconciliation.json').read_text())
    require(spec['schema_version']==1,'wrong reconciliation schema')
    expected_pins={spec['baseline_coverage'].rsplit('/',1)[0]+'/'+name for name in ('scope.json','coverage.jsonl','objects.json','_MANIFEST.jsonl')}
    require(len(spec['baseline_pins'])==4 and {p['path'] for p in spec['baseline_pins']}==expected_pins,'incomplete baseline identity pins')
    for pin in spec['baseline_pins']:
        require(sha(local_file(root,pin['path']).read_bytes())==pin['sha256'],'baseline identity changed')
    previous=json.loads((base/'previous-membership.json').read_text())
    current=json.loads((base/'current-membership.json').read_text())
    result=reconcile(previous,current)
    require(spec['membership_cutoff']==max(r['observed_at'] for r in current),'cutoff is not the final membership observation')
    require(result==spec['comparison'],'declared reconciliation differs from observations')
    require(not result['not_observed'],'prior membership not observed; explicit disposition required')
    require(not result['membership_changes'],'changed membership needs review')
    baseline=[json.loads(x) for x in local_file(root,spec['baseline_coverage']).read_text().splitlines()]
    rows={r['id']:r for r in baseline}
    require(len(rows)==len(baseline),'duplicate baseline identity')
    additions=set(result['added'])
    required=additions|{r['id'] for r in result['changed']}
    payloads=spec['payloads']
    require(len({r['id'] for r in payloads})==len(payloads),'duplicate payload identity')
    require({r['id'] for r in payloads}==required,'added or changed source lacks exact payload disposition')
    for row in payloads:
        data=local_file(root,row['path']).read_bytes()
        require(len(data)==row['bytes'] and sha(data)==row['sha256'],'refreshed payload identity mismatch')
        require(row['status'] in {'RAW_SOURCE_SNAPSHOT','NATIVE_EXPORT'},'incorrect refreshed payload type')
        if row['id'] not in additions:
            require(rows[row['id']]['status']=='NATIVE_EXPORT' and row['status']=='NATIVE_EXPORT','unexpected replacement of frozen raw source')
            require(row.get('revision_observation',{}).get('revision_pinned') is False,'export represented as revision-pinned')
        else:require(row['id'] not in rows,'addition already in baseline')
        rows[row['id']]=row
    counts=dict(sorted(Counter(x['status'] for x in rows.values()).items()))
    require(counts==spec['current_counts'],'current counts are not derived from baseline plus delta')
    require(len(rows)==spec['scoped_objects'],'current scope count mismatch')
    observed=set(index_observations(current)[0])
    exceptions=spec['not_in_folder_membership']
    require({r['id'] for r in exceptions}==set(rows)-observed,'unobserved scoped identities not explicitly accounted')
    require(all(r.get('reason') for r in exceptions),'unexplained unobserved identity')
    require(spec.get('whole_account_complete') is False,'global completeness overclaim')
    require(spec.get('historical_revisions_complete') is False,'revision-history overclaim')
    return {'scoped_objects':len(rows),'counts':counts,'folders_relisted':result['folders'],
            'new_payloads':len(additions),'refreshed_payloads':len(payloads)-len(additions),
            'missing_previous_membership':len(result['not_observed']),
            'cutoff':spec['membership_cutoff'],'scope':'Declared research snapshot only; no whole-account, revision-history or scientific completeness.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--bundle',default=DEFAULT)
    a=p.parse_args()
    try:print(json.dumps(check(a.root,a.bundle),sort_keys=True))
    except (ValueError,KeyError,OSError) as e:raise SystemExit('Drive reconciliation failed: '+str(e))


if __name__=='__main__':main()
