"""Targeted actual runner-source mutants, graded by unchanged phase-aware runner."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'work/layer_a'
runner=ROOT/'mutation_test.py'
spec=importlib.util.spec_from_file_location('trusted_lifecycle_runner',runner)
trusted=importlib.util.module_from_spec(spec);spec.loader.exec_module(trusted)
source=runner.read_text()
malformed='test_mutation_runner.LifecycleHarnessRules.test_missing_forged_or_noninteger_receipts_fail_closed'
lifecycle='test_mutation_runner.LifecycleHarnessRules.test_setup_teardown_cleanup_assertions_cannot_kill'
mutants=[
 ('drop-schema-requirement',"if value.get('receipt_schema') != 'cbz-unittest-lifecycle-v1':","if False:",malformed),
 ('allow-boolean-counts',"type(value.get(k)) is not int","type(value.get(k)) not in (int,bool)",malformed),
 ('ignore-completed-population',"if any(value[k] != selected for k in lists[1:5]):","if False:",malformed),
 ('allow-duplicate-method-ids',"if key != 'body_assertion_ids' and len(items) != len(set(items)):","if False:",'test_mutation_runner.LifecycleHarnessRules.test_empty_duplicate_synthetic_and_rewritten_method_ids_are_rejected'),
 ('credit-lifecycle-assertions',"is_body=(phase == 'body' and issubclass(err[0],AssertionError)\n                     and err[1] is getattr(test,'_cbz_body_exception',None))","is_body=issubclass(err[0],AssertionError)",lifecycle),
 ('credit-synthetic-body-callback',"\n                     and err[1] is getattr(test,'_cbz_body_exception',None)","",'test_mutation_runner.LifecycleHarnessRules.test_empty_duplicate_synthetic_and_rewritten_method_ids_are_rejected'),
 ('ignore-subtest-phase'," or event['phase'] != 'body'","",malformed),
 ('ignore-subtest-event-count'," or value['subtests'] != len(events)","",malformed),
 ('lose-method-decorators','        @functools.wraps(original)\n','', 'test_mutation_runner.LifecycleHarnessRules.test_errors_skips_expected_failures_and_subtests_are_ambiguous'),
 ('drop-body-entry-accounting',"            receipt['body_entered_ids'].append(_id)\n","",'test_mutation_runner.LifecycleHarnessRules.test_body_assertion_kills_and_identity_survives'),
]
receipt_path=HERE/'harness-source-controls.json'
prior={r['name']:r for r in json.loads(receipt_path.read_text())['results']} if receipt_path.exists() else {}
records=[]
for name,before,after,selection in mutants:
 assert source.count(before)==1,(name,source.count(before))
 changed=source.replace(before,after,1)
 existing=prior.get(name)
 if (existing and existing['outcome']=='KILLED' and existing['original_sha256']==hashlib.sha256(source.encode()).hexdigest() and existing['mutated_sha256']==hashlib.sha256(changed.encode()).hexdigest() and existing['selected_test']==selection):
  records.append(existing);print(name,'REUSED KILLED',flush=True);continue
 with tempfile.TemporaryDirectory(prefix='cbz-layer-lifecycle-control-') as temporary:
  root=Path(temporary);(root/'tests').mkdir()
  (root/'mutation_test.py').write_text(changed)
  (root/'tests/test_mutation_runner.py').write_bytes((ROOT/'tests/test_mutation_runner.py').read_bytes())
  result=root/'trusted-result.json'
  command=[sys.executable,'-B','-S',str(runner),'--internal-test-runner','--output',str(result),'--test-root',str(root/'tests'),'--test',selection]
  env=dict(os.environ);env.pop('PYTHONPATH',None)
  p=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
  value=json.loads(result.read_text()) if result.is_file() else {}
  outcome=trusted.classify_test_result(value,p.returncode)
 log=HERE/(name+'.log');log.write_text(p.stdout)
 record={'name':name,'file':'mutation_test.py','original_sha256':hashlib.sha256(source.encode()).hexdigest(),'mutated_sha256':hashlib.sha256(changed.encode()).hexdigest(),'selected_test':selection,'outcome':outcome,'returncode':p.returncode,'tests':value,'log':str(log),'log_sha256':hashlib.sha256(p.stdout.encode()).hexdigest()}
 records.append(record);print(name,outcome,flush=True)
(HERE/'harness-source-controls.json').write_text(json.dumps({'scope':'changed lifecycle harness safeguards only','planned':len(mutants),'executed':len(records),'killed':sum(r['outcome']=='KILLED' for r in records),'results':records},sort_keys=True,indent=2)+'\n')
assert all(r['outcome']=='KILLED' for r in records)
