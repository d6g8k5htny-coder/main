#!/usr/bin/env python3
"""Bounded actual runner-source guard omissions; no scanner/Lean mutations."""
import ast, hashlib, importlib.util, json, os
from pathlib import Path
import shutil, subprocess, sys
ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

GUARDS=[('allow-static-descriptor-subclasses', 'type(value) in (staticmethod,classmethod)', 'isinstance(value,(staticmethod,classmethod))', 'test_callable_factory_selector_is_refused_before_factory_or_body'), ('allow-named-descriptor-subclasses', 'type(declared) in (type,staticmethod,classmethod)', 'isinstance(declared,(type,staticmethod,classmethod))', 'test_callable_factory_selector_is_refused_before_factory_or_body'), ('invoke-special-descriptor-during-profile-inspection', 'inspect.getattr_static(testCaseClass,name,None) is not inspect.getattr_static(unittest.TestCase,name,None)', 'getattr(testCaseClass,name,None) is not getattr(unittest.TestCase,name,None)', 'test_callable_factory_selector_is_refused_before_factory_or_body')]

report=[]
for tool in ['kernel_replay','layer_a']:
 parent=ROOT/tool/'proposed';filename='run_tests.py' if tool=='kernel_replay' else 'mutation_test.py'
 original=(parent/filename).read_text(); healthy=load(parent/('mutations.py' if tool=='kernel_replay' else 'mutation_test.py'),'healthy_'+tool)
 for mode in ['normal','optimized']:
  for name,before,after,method in GUARDS:
   if original.count(before)!=1:raise ValueError((tool,name,'nonunique anchor',original.count(before)))
   changed=original.replace(before,after,1);ast.parse(changed)
   work=ROOT/tool/'evidence'/'guard-mutants'/mode/name
   work.mkdir(parents=True,exist_ok=True)
   for p in parent.glob('*.py'):(work/p.name).write_bytes(p.read_bytes())
   shutil.copytree(parent/'tests',work/'tests',dirs_exist_ok=True)
   (work/filename).write_text(changed)
   out=work/'receipt.json';env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONOPTIMIZE']='1' if mode=='optimized' else '0'
   command=[sys.executable,'-B','-S']+(['-O'] if mode=='optimized' else [])+[str(work/filename)]
   cls='test_mutation_harness.IdentityHarnessTests' if tool=='kernel_replay' else 'test_mutation_runner.IdentityHarnessRules'
   if tool=='layer_a':command+=['--internal-test-runner','--test-root',str(work/'tests')]
   command+=['--output',str(out),'--test',cls+'.'+method]
   p=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=20)
   (work/'stdout.log').write_text(p.stdout);(work/'stderr.log').write_text(p.stderr)
   value=json.loads(out.read_text()) if out.exists() else {}
   outcome=(healthy.classify(value,p.returncode) if tool=='kernel_replay' else healthy.classify_test_result(value,p.returncode))
   report.append({'tool':tool,'mode':mode,'name':name,'source_file':filename,'original_sha256':sha(original.encode()),'mutant_sha256':sha(changed.encode()),'test':cls+'.'+method,'outcome':outcome,'exit_code':p.returncode,'assertion_failures':value.get('assertion_failures'),'errors':value.get('errors'),'receipt':str(out.relative_to(ROOT)),'command':command})
   print(tool,mode,name,outcome,flush=True)
(ROOT/'EXACT_DESCRIPTOR_GUARD_MUTATIONS.json').write_text(json.dumps({'scope':'3 exact-descriptor/static-inspection refusal safeguards per tool/mode; prior11themes remain separate','results':report,'all_assertion_killed':all(x['outcome']=='KILLED' and x['errors']==0 for x in report)},indent=2)+'\n')
if any(x['outcome']!='KILLED' or x['errors']!=0 for x in report):raise SystemExit(1)
