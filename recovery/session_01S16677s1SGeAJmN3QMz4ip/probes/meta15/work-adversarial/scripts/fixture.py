import json,tempfile,subprocess,sys
from pathlib import Path
TREE=Path('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/meta15/work-adversarial/tree')
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}
def make(root:Path, generated:dict):
    for name in REPOS:(root/name).mkdir(parents=True,exist_ok=True)
    registry={'schema_version':1,'scientific_status_authority':False,'repositories':{name:{'full_name':'d6g8k5htny-coder/'+name,'role':name,'visibility':'private' if name=='sandbox' else 'public'} for name in REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(registry))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True,exist_ok=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
    g=root/'meta-framework/generated'
    for rel,raw in generated.items():
        p=g/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw if isinstance(raw,bytes) else raw.encode())
def cli(root,checker=None,opt=False):
    script=str(checker or TREE/'tools/architecture_conformance.py')
    flags=['-B','-S']+(['-O'] if opt else [])
    return subprocess.run([sys.executable,*flags,script,'--workspace',str(root)],capture_output=True,text=True,timeout=60)
