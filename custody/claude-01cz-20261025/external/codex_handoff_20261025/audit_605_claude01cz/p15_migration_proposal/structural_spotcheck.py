# Independent spot-check (Claude 01Cz) of the obligation-level rules touched by the migration,
# on the successor proposal. Not the published validator.
import json
d=json.load(open('dataset_successor_PROPOSAL.json',encoding='utf-8'))
rec={r['id']:r for r in d['records']}; obl={o['conclusion']:o for o in d['obligations']}
for n in ('P15-T1','P15-T2','P15-T3'):
    o=obl[n]; p=o['premises']
    assert p==sorted(p) and len(p)==len(set(p)) and n not in p and all(x in rec for x in p), n
    assert rec[n]['source'].endswith('/PROOF.md') and o['proof_bundle']['source'].endswith('/PROOF.md'), n
    assert o['status']=='side_conditions_open' and not o['full_proof_verified'], n
migs=d['metadata']['documented_link_migrations']
total=sum(len(o['premises']) for o in obl.values())
ok=all(m['added_premise'] in obl[m['obligation'][4:]]['premises'] and m['link_count_after']-m['link_count_before']==1 for m in migs)
print('affected obligations well-formed; migrations:',len(migs),'present:',ok,'| links',total,'== 108+migs:',total==108+len(migs))
print('records:',len(rec),'primary:',sum(r['project_acceptance']=='primary' for r in rec.values()),'local_evidence:',len(d['local_evidence']))
