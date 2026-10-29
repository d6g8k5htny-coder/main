import importlib.util,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'tests/test_atlas_browser_regressions_runner.py'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
trusted=load('trusted_browser_origin',PATH);source=PATH.read_text();rows=[]
cases=[
 ('origin_frames','and all(code in _ASSERTION_CODES for code in codes[positions[0]+1:])','and True','test_subject_helper_callback_assertions_cannot_kill'),
 ('phase',"and state.get('phase') in ('body', 'body-exited')",'and True','test_later_phase_calling_selected_body_cannot_kill'),
 ('origin_admission','and all(v for _, v in result.origins)','and True','test_subject_helper_callback_assertions_cannot_kill'),
 ('lazy_bodies',' or inspect.iscoroutinefunction(selected) or inspect.isgeneratorfunction(selected) or inspect.isasyncgenfunction(selected)','', 'test_deferred_bodies_refused'),
 ('baseline','result.oracle_baseline = complete and result.wasSuccessful() and not result.origins','result.oracle_baseline = True','test_error_skip_expected_failure_refused'),
 ('terminal_accounting','complete = (complete and accounted == set(expected)\n                and len(result.successful) == len(set(result.successful))\n                and not (set(result.successful) & {identity for identity, unused in result.origins}))','complete = complete','test_missing_or_duplicate_terminal_success_refused'),
]
for name,before,after,method in cases:
 if source.count(before) != 1: raise RuntimeError('nonunique mutation target: '+name)
 baseline=trusted.run_oracle_controls(trusted.OracleOriginControls,(method,))
 if not baseline.oracle_baseline:raise RuntimeError('bad baseline '+name)
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'mutant.py';changed=source.replace(before,after,1)
  if name=='lazy_bodies': changed=changed.replace('if returned is not None:', 'if False:')
  p.write_text(changed);mutant=load('browser_origin_'+name,p)
  result=trusted.run_oracle_controls(mutant.OracleOriginControls,(method,))
  if not result.oracle_killed:raise RuntimeError('not intended assertion kill '+name+repr(result.failures)+repr(result.errors))
  rows.append({'name':name,'method':method,'intended_assertion_kill':True,'failures':len(result.failures),'errors':len(result.errors)})
print(json.dumps({'optimized':sys.flags.optimize,'source_sha256':__import__('hashlib').sha256(source.encode()).hexdigest(),'results':rows},indent=2))
