import json,subprocess,sys,tempfile,hashlib
from pathlib import Path
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
M='"scientific_status_authority"'
def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}
def ws(root,text):
    for n in REPOS:(root/n).mkdir(parents=True)
    reg={'schema_version':1,'scientific_status_authority':False,'repositories':{n:{'full_name':'d6g8k5htny-coder/'+n,'role':n,'visibility':'private' if n=='sandbox' else 'public'} for n in REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(reg))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
    g=root/'meta-framework/generated';g.mkdir();(g/'x.json').write_text(text,encoding='utf-8')
def run(src,py,text,opt=False):
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);ws(root,text)
        h=hashlib.sha256((root/'meta-framework/generated/x.json').read_bytes()).hexdigest()
        flags=['-B','-S']+(['-O'] if opt else [])
        p=subprocess.run([py,*flags,str(Path(src)/'tools/architecture_conformance.py'),'--workspace',str(root)],capture_output=True,text=True,timeout=120)
        assert h==hashlib.sha256((root/'meta-framework/generated/x.json').read_bytes()).hexdigest()
        try: v=json.loads(p.stdout)['violations']
        except Exception: v=None
        tail=p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''
        return p.returncode,v,tail,p.stdout
def short(r):
    rc,v,t,_=r
    if v is None: return f'rc={rc} stdout_empty={_==""} tail={t!r}'
    cats=[x.split(':')[0] for x in v]
    return f'rc={rc} '+('|'.join(cats) if cats else 'none')
def deep(n,inner): return '{"a":'*n+inner+'}'*n
