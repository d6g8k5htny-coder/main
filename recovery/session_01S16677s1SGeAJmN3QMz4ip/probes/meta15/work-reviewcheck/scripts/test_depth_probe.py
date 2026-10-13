import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.architecture_conformance import check_workspace
REPOS=['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
K='scientific_status_authority'
DEPTH=int(os.environ.get('DEPTH','900'))
def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}
class Depth(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in REPOS:(self.root/name).mkdir(parents=True)
        registry={'schema_version':1,'scientific_status_authority':False,'repositories':{name:{'full_name':'d6g8k5htny-coder/'+name,'role':name,'visibility':'private' if name=='sandbox' else 'public'} for name in REPOS},'artifacts':[]}
        (self.root/'meta-framework/registry.json').write_text(json.dumps(registry))
        a=self.root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
    def _run(self, inner, expected_v):
        g=self.root/'meta-framework/generated';g.mkdir()
        (g/'x.json').write_text('{"a":'*DEPTH+inner+'}'*DEPTH)
        exp={'ok':not expected_v,'repositories_checked':sorted(REPOS),'violations':expected_v,'scientific_effect':'NONE'}
        self.assertEqual(check_workspace(self.root),exp)
        script=Path(sys.modules['tools.architecture_conformance'].__file__)
        env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None)
        for flags in (['-B','-S'],['-B','-S','-O']):
            r=subprocess.run([sys.executable,*flags,str(script),'--workspace',str(self.root)],capture_output=True,text=True,env=env,timeout=30)
            self.assertEqual(r.stderr,'',flags);self.assertEqual(r.returncode,0 if not expected_v else 2,flags)
            self.assertEqual(r.stdout,json.dumps(exp,indent=2,sort_keys=True)+'\n')
    def test_true_innermost(self):
        self._run('{"%s":true}'%K,['generated scientific-status authority: meta-framework/generated/x.json'])
    def test_malformed_innermost(self):
        self._run('{"%s":1}'%K,['generated scientific-status marker type invalid: meta-framework/generated/x.json'])
    def test_no_marker(self):
        self._run('{"k":"v"}',[])
if __name__=='__main__':unittest.main()
