import json,subprocess,sys,tempfile
from pathlib import Path
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
def ws(root):
    for n in REPOS:(root/n).mkdir(parents=True)
    reg={'schema_version':1,'scientific_status_authority':False,'repositories':{n:{'full_name':'d6g8k5htny-coder/'+n,'role':n,'visibility':'private' if n=='sandbox' else 'public'} for n in REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(reg))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True)
    (a/'AUTHORITY_MAP.json').write_text(json.dumps({'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}))
    (root/'meta-framework/generated').mkdir()
if __name__=="__main__":
  py,script=sys.argv[1],sys.argv[2]
  for n in map(int,sys.argv[3:]):
      with tempfile.TemporaryDirectory() as t:
          root=Path(t);ws(root)
          (root/'meta-framework/generated/x.json').write_text('{"a":'*n+'{"scientific_status_authority":true}'+'}'*n)
          r=subprocess.run([py,'-B','-S',script,'--workspace',str(root)],capture_output=True,text=True)
          v=json.loads(r.stdout)['violations'] if r.stdout else None
          print(py,n,r.returncode,(v[0][:45] if v else v),r.stderr.strip().splitlines()[-1][:60] if r.stderr else '')
  