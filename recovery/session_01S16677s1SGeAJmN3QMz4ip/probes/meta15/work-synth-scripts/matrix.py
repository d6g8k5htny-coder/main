import json,subprocess,sys,tempfile,hashlib
from pathlib import Path
SRC=Path(sys.argv[1]); PY=sys.argv[2]
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}
def ws(root,text):
    for n in REPOS:(root/n).mkdir(parents=True)
    reg={'schema_version':1,'scientific_status_authority':False,'repositories':{n:{'full_name':'d6g8k5htny-coder/'+n,'role':n,'visibility':'private' if n=='sandbox' else 'public'} for n in REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(reg))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
    g=root/'meta-framework/generated';g.mkdir();(g/'x.json').write_text(text,encoding='utf-8')
def run(text,opt):
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);ws(root,text)
        h=hashlib.sha256((root/'meta-framework/generated/x.json').read_bytes()).hexdigest()
        flags=['-B','-S']+(['-O'] if opt else [])
        p=subprocess.run([PY,*flags,str(SRC/'tools/architecture_conformance.py'),'--workspace',str(root)],capture_output=True,text=True,timeout=60)
        assert h==hashlib.sha256((root/'meta-framework/generated/x.json').read_bytes()).hexdigest()
        try: v=json.loads(p.stdout)['violations']
        except Exception: v=None
        tail=p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''
        return p.returncode,v,tail
M='"scientific_status_authority"'
vals=['true','false','0','1','0.0','1.0','-0.0','""','"true"','"false"','null','[]','[true]','{}','{'+M+':true}','NaN']
places={'root':lambda d:d,'nested_dict':lambda d:'{"a":{"b":'+d+'}}','list_in_dict':lambda d:'{"items":['+d+']}','root_list':lambda d:'['+d+']'}
def short(r):
    rc,v,t=r
    if v is None: return f'CRASH rc={rc} {t}'
    cats=[x.split(':')[0] for x in v]
    return f'rc={rc} '+('|'.join(cats) if cats else 'none')
for val in ['ABSENT']+vals:
    D='{"k":"v"}' if val=='ABSENT' else '{"k":"v",'+M+':'+val+'}'
    row=[]
    for pn,f in places.items():
        a=run(f(D),False);b=run(f(D),True)
        row.append(pn+'='+short(a)+('' if short(a)==short(b) else ' /O:'+short(b)))
    print(val,'::',' ; '.join(row))
extra={'dup_true_then_false':'{'+M+':true,'+M+':false}','dup_false_then_true':'{'+M+':false,'+M+':true}',
 'malformed_list_with_true':'{'+M+':[{'+M+':true}]}','marker_list_bare_true':'{'+M+':[true]}',
 'root_scalar_true':'true','bare_true_in_list':'[true]'}
for k,t in extra.items():
    a=run(t,False);b=run(t,True);print(k,'::',short(a),'' if short(a)==short(b) else '/O:'+short(b))
def deep(n,inner):
    return '{"a":'*n+inner+'}'*n
for n in (300,331,332,400,497,498,499,600,1000,5000,9997,9998,20000):
    t=deep(n,'{'+M+':true}')
    a=run(t,False);print('deep_true',n,'::',short(a))
