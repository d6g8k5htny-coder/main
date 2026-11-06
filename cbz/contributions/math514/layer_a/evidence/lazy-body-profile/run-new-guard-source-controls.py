"""Run actual lazy-body grader source mutants against only new contract controls."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'work/layer_a'
runner=ROOT/'mutation_test.py'
spec=importlib.util.spec_from_file_location('trusted_lazy_body_runner',runner)
trusted=importlib.util.module_from_spec(spec);spec.loader.exec_module(trusted)
source=runner.read_text()
prefix='test_mutation_runner.LazyBodyHarnessRules.'
mutants=[
 ('accept-non-none-return-objects','if value is not None:','if False:','test_sync_bodies_require_none_and_refuse_other_returns_without_iteration'),
 ('accept-falsey-non-none-returns','if value is not None:','if value:','test_sync_bodies_require_none_and_refuse_other_returns_without_iteration'),
 ('iterate-unknown-return-objects','                    # Closing a fresh native coroutine',"                    if hasattr(value,'__iter__'): list(value)\n                    # Closing a fresh native coroutine",'test_sync_bodies_require_none_and_refuse_other_returns_without_iteration'),
]

records=[]
for mode in ('normal','optimized'):
 for name,before,after,method in mutants:
  if source.count(before)!=1:raise ValueError((name,source.count(before)))
  changed=source.replace(before,after,1);selection=prefix+method
  with tempfile.TemporaryDirectory(prefix='cbz-lazy-layer-guard-') as temporary:
   root=Path(temporary);(root/'tests').mkdir()
   (root/'mutation_test.py').write_text(changed)
   (root/'tests/test_mutation_runner.py').write_bytes((ROOT/'tests/test_mutation_runner.py').read_bytes())
   result=root/'trusted-result.json'
   command=[sys.executable,'-B','-S']+(['-O'] if mode=='optimized' else [])+[str(runner),'--internal-test-runner','--output',str(result),'--test-root',str(root/'tests'),'--test',selection]
   env=dict(os.environ);env.pop('PYTHONPATH',None)
   if mode=='optimized':env['PYTHONOPTIMIZE']='1'
   else:env.pop('PYTHONOPTIMIZE',None)
   env['CBZ_LAZY_CONTROL_EVIDENCE']=str(HERE/'guard-fixtures'/mode/name)
   p=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
   value=json.loads(result.read_text()) if result.is_file() else {}
   outcome=trusted.classify_test_result(value,p.returncode,expected_ids=[selection])
  log=HERE/(mode+'-'+name+'.log');log.write_text(p.stdout)
  record={'name':name,'mode':mode,'file':'mutation_test.py','before':before,'after':after,'original_sha256':hashlib.sha256(source.encode()).hexdigest(),'mutated_sha256':hashlib.sha256(changed.encode()).hexdigest(),'selected_test':selection,'outcome':outcome,'returncode':p.returncode,'tests':value,'log':str(log),'log_sha256':hashlib.sha256(p.stdout.encode()).hexdigest()}
  records.append(record);print(mode,name,outcome,flush=True)
(HERE/'new-guard-source-controls.json').write_text(json.dumps({'scope':'three final None-return profile source mutants in two modes; unchanged lazy-shape and disposal evidence reused','planned':6,'executed':len(records),'killed':sum(r['outcome']=='KILLED' for r in records),'results':records},sort_keys=True,indent=2)+'\n')
if not all(r['outcome']=='KILLED' for r in records):raise RuntimeError('a lazy safeguard source control did not produce a clean assertion kill')
