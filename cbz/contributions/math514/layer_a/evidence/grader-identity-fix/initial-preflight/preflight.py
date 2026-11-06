#!/usr/bin/env python3
"""Finite, isolated new identity controls; no production/tool source edits."""
import hashlib, importlib.util, json, os
from pathlib import Path
import shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent
ACTIVE=Path('/workspace/cbz-0/tools')
SOURCES={'kernel_replay':['run_tests.py','mutations.py'], 'layer_a':['mutation_test.py']}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

FIXTURE='''import os\nfrom pathlib import Path\nimport unittest\ndef trace(label):\n    with Path(os.environ['CBZ_PREFLIGHT_TRACE']).open('a') as out: out.write(label+'\\n')\nclass AliasTests(unittest.TestCase):\n    def test_first(self):\n        trace('original-first')\n        if os.environ.get('CBZ_PREFLIGHT_FAIL_FIRST') == '1': self.fail('original first body assertion')\n    def test_second(self):\n        trace('original-second')\n        self.fail('original second body must remain covered')\n'''
HOOK_ALIAS='''\ndef load_tests(loader, tests, pattern):\n    AliasTests.test_second = AliasTests.test_first\n    return tests\n'''
HOOK_CLASS='''\ndef load_tests(loader, tests, pattern):\n    def replacement(self):\n        trace('replacement')\n    Replacement = type('AliasTests', (AliasTests,), {'test_first':replacement,'test_second':replacement,'__module__':__name__,'__qualname__':'AliasTests'})\n    globals()['AliasTests'] = Replacement\n    return loader.loadTestsFromTestCase(Replacement)\n'''

baseline={}
for tool, names in SOURCES.items():
 for name in names:
  raw=(ACTIVE/tool/name).read_bytes();baseline[f'{tool}/{name}']={'sha256':sha(raw),'bytes':len(raw)}
(ROOT/'SOURCE_BINDINGS.json').write_text(json.dumps(baseline,indent=2)+'\n')
results=[]
for tool in SOURCES:
 runner_name='run_tests.py' if tool=='kernel_replay' else 'mutation_test.py'
 for case,hook,fail in [('alias-pass',HOOK_ALIAS,False),('alias-kill',HOOK_ALIAS,True),('same-name-subclass',HOOK_CLASS,False)]:
  work=ROOT/'fixtures'/tool/case; (work/'tests').mkdir(parents=True,exist_ok=True)
  for name in SOURCES[tool]: (work/name).write_bytes((ACTIVE/tool/name).read_bytes())
  fixture_name='test_alias.py' if tool=='kernel_replay' else 'test_auditor.py'
  (work/'tests'/fixture_name).write_text(FIXTURE+hook)
  env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['CBZ_PREFLIGHT_TRACE']=str(work/'body-trace.txt');env['CBZ_PREFLIGHT_FAIL_FIRST']='1' if fail else '0'
  args=[sys.executable,'-B','-S',str(work/runner_name)]
  if tool=='layer_a':args+=['--internal-test-runner','--test-root',str(work/'tests')]
  cmds=[]
  for phase in ('list','run'):
   output=work/(phase+'.json');cmd=args+['--output',str(output)]+(['--list'] if phase=='list' else [])
   p=subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,timeout=15)
   (work/(phase+'.stdout')).write_text(p.stdout);(work/(phase+'.stderr')).write_text(p.stderr)
   cmds.append({'phase':phase,'command':cmd,'exit_code':p.returncode})
  population=json.loads((work/'list.json').read_text());value=json.loads((work/'run.json').read_text())
  grader=load(work/('mutations.py' if tool=='kernel_replay' else 'mutation_test.py'),f'grader_{tool}_{case.replace("-","_")}')
  if tool=='kernel_replay':outcome=grader.classify(value,cmds[-1]['exit_code'],['discover:test_*.py'],population['selected_ids'])
  else:outcome=grader.classify_test_result(value,cmds[-1]['exit_code'],population['selected_ids'])
  trace=(work/'body-trace.txt').read_text().splitlines()
  row={'tool':tool,'case':case,'commands':cmds,'source_sha256':sha((work/runner_name).read_bytes()),'fixture_sha256':sha((work/'tests'/fixture_name).read_bytes()),'outcome':outcome,'selected_ids':value['selected_ids'],'body_entered_ids':value['body_entered_ids'],'body_finished_ids':value['body_finished_ids'],'invalid_population':value['invalid_population'],'assertion_failures':value['assertion_failures'],'errors':value['errors'],'body_trace':trace,'original_second_executed':'original-second' in trace}
  results.append(row)
  # Class-selector negative control: unittest bypasses the module load_tests hook.
  if case=='alias-pass':
   env['CBZ_PREFLIGHT_TRACE']=str(work/'named-body-trace.txt')
   module=fixture_name[:-3];output=work/'named.json';cmd=args+['--output',str(output),'--test',module+'.AliasTests']
   p=subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,timeout=15)
   (work/'named.stdout').write_text(p.stdout);(work/'named.stderr').write_text(p.stderr)
   nv=json.loads(output.read_text()); nt=(work/'named-body-trace.txt').read_text().splitlines()
   results.append({'tool':tool,'case':'named-class-control','command':cmd,'exit_code':p.returncode,'selected_ids':nv['selected_ids'],'body_trace':nt,'assertion_failures':nv['assertion_failures'],'errors':nv['errors'],'original_second_executed':'original-second' in nt})
report={'new_obligation':'Ordinary synchronous None-return load_tests method alias/same-name class replacement after discovery; original declared bodies retained but omitted','execution_mode':'normal only; no old tests rerun','source_bindings':baseline,'results':results}
(ROOT/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{'tool':r['tool'],'case':r['case'],'outcome':r.get('outcome'),'exit_code':r.get('exit_code',r.get('commands',[{}])[-1].get('exit_code')),'trace':r['body_trace'],'second_executed':r['original_second_executed'],'failures':r['assertion_failures']} for r in results],indent=2))
