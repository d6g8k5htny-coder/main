"""Finite source-bound campaign; runs only changed Lean metadata integration."""
import ast, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE/'proposed'
OUT=HERE/'evidence'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def check(cond, why):
    if not cond: raise RuntimeError(why)

sys.path.insert(0,str(ROOT))
import mutation_test
prior=json.loads((ROOT/'evidence/lexer698-admission.json').read_text())
old_methods=prior['affected_results']['selected_methods']
old_ids=[x['id'] if isinstance(x,dict) else x for x in old_methods]
module=ast.parse((ROOT/'tests/test_auditor.py').read_text())
new_ids=['test_auditor.NumericDeclarationRules.'+x.name for cls in module.body if isinstance(cls,ast.ClassDef) and cls.name=='NumericDeclarationRules' for x in cls.body if isinstance(x,ast.FunctionDef) and x.name.startswith('test_')]
selected=old_ids+new_ids
check(len(old_ids)==45 and len(new_ids)==7 and len(set(selected))==52,'wrong selected population')
source_files=['source_checks.py','auditor.py','canonical_lean.py','mutation_test.py','tests/test_auditor.py','tests/test_mutation_runner.py']
bindings={x:sha(ROOT/x) for x in source_files}
mode='optimized' if sys.flags.optimize else 'normal'
mode_dir=OUT/mode;mode_dir.mkdir(parents=True,exist_ok=True)
start=datetime.now(timezone.utc).isoformat()
cmd=[sys.executable]+(['-O'] if sys.flags.optimize else [])+['-B','-S',str(ROOT/'mutation_test.py'),'--internal-test-runner','--test-root',str(ROOT/'tests'),'--output',str(mode_dir/'affected-baseline.json')]
for test in selected:cmd.extend(['--test',test])
env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
(mode_dir/'affected-baseline.log').write_text(p.stdout)
v=json.loads((mode_dir/'affected-baseline.json').read_text());outcome=mutation_test.classify_test_result(v,p.returncode,expected_ids=sorted(selected))
check(outcome=='SURVIVED' and v['tests_run']==52,'affected baseline did not pass')
new_names=[row[0] for row in json.loads((HERE/'new-mutants.json').read_text())]
mutants=[]
for name,filename,old,new in mutation_test.MUTATIONS:
 if name not in new_names:continue
 def change(s,old=old,new=new):
  check(s.count(old)==1,'mutation target count '+name)
  return s.replace(old,new,1)
 result=mutation_test.run_one(name,filename,change,mode_dir/'mutants',tests=selected)
 mutants.append(result)
 check(result['outcome']=='KILLED' and result['errors']==0,'source mutant not killed '+name)
check(len(mutants)==3,'wrong new mutation count')
check(bindings=={x:sha(ROOT/x) for x in source_files},'source changed during campaign')
result={'mode':mode,'started_utc':start,'ended_utc':datetime.now(timezone.utc).isoformat(),'selected_methods':selected,'new_methods':new_ids,'fresh_methods':52,'affected_predecessor_methods':45,'new_method_count':7,'bindings':bindings,'baseline':{'command':cmd,'returncode':p.returncode,'outcome':outcome,'receipt':'affected-baseline.json','receipt_sha256':sha(mode_dir/'affected-baseline.json'),'log_sha256':sha(mode_dir/'affected-baseline.log')},'new_source_mutants':mutants,'mutant_count':3,'declared_mutant_count':len(mutation_test.MUTATIONS),'claim':'affected Lean metadata methods and3new guards only; no full G0/Lean/project replay'}
(mode_dir/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'mode':mode,'tests':v['tests_run'],'new_guards':[{k:m[k] for k in ['name','outcome','tests_run','assertion_failures','errors']} for m in mutants]}))
