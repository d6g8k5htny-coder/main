import json,subprocess,sys,tempfile,os
from pathlib import Path
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}
CASES={
 'dup_true_then_false': '{"nested":{"scientific_status_authority":true,"scientific_status_authority":false}}',
 'dup_false_then_true': '{"nested":{"scientific_status_authority":false,"scientific_status_authority":true}}',
 'dup_true_then_null':  '{"nested":{"scientific_status_authority":true,"scientific_status_authority":null}}',
 'dup_true_then_obj_with_true': '{"m":{"scientific_status_authority":true,"scientific_status_authority":{"scientific_status_authority":true}}}',
 'unrelated_dup_plus_separate_true': '{"x":1,"x":1,"layer":[{"scientific_status_authority":true}]}',
 'nan_marker': '{"scientific_status_authority":NaN}',
 'obj_marker_containing_true': '{"scientific_status_authority":{"scientific_status_authority":true}}',
 'list_marker_containing_true': '{"scientific_status_authority":[{"scientific_status_authority":true}]}',
}
src=Path(sys.argv[1]).resolve()
for name,text in CASES.items():
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        for r in REPOS:(root/r).mkdir()
        reg={'schema_version':1,'scientific_status_authority':False,'repositories':{n:{'full_name':'d6g8k5htny-coder/'+n,'role':n,'visibility':'private' if n=='sandbox' else 'public'} for n in REPOS},'artifacts':[]}
        (root/'meta-framework/registry.json').write_text(json.dumps(reg))
        a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
        g=root/'meta-framework/generated';g.mkdir();(g/'p.json').write_text(text)
        before=(g/'p.json').read_bytes()
        outs=[]
        for flags in (['-B','-S'],['-B','-O','-S']):
            env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None)
            p=subprocess.run([sys.executable,*flags,str(src/'tools/architecture_conformance.py'),'--workspace',str(root)],capture_output=True,text=True,env=env,timeout=20)
            rep=json.loads(p.stdout)
            outs.append((p.returncode,rep['ok'],[v.replace(str(root),'<ws>') for v in rep['violations']],p.stderr))
        assert outs[0][:3]==outs[1][:3],outs
        assert (g/'p.json').read_bytes()==before
        print(f'{name:36s} exit={outs[0][0]} ok={outs[0][1]} viol={outs[0][2]} stderr={outs[0][3]!r}')
