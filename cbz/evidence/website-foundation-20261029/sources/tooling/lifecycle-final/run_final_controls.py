import hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/release/mutation-review');MODE='optimized-asyncfix' if sys.flags.optimize else 'normal-asyncfix'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helper=load(ROOT/'tools/cbz_mutation.py','grader')
summary={'mode':sys.flags.optimize,'source_sha256':{},'portable':[],'classifier_mutants':[]}
for kind in ('ref','baseline'):
 p=ROOT/'cbz/evidence/website-foundation-20261029/verification'/f'{kind}_controls.py';runner=load(p,'runner_'+kind);out=OUT/('fixed-'+kind+'-'+MODE);out.mkdir(exist_ok=False)
 positive=runner.positive(ROOT,out,runner.TEST_NAME,runner.CLASS_NAME);rows=runner.mutations(ROOT,out)
 summary['portable'].append({'runner':kind,'positive':positive,'mutants':rows})
source=(ROOT/'tools/cbz_mutation.py').read_text()
mutations=[
 ('legacy_counter_kill',"body_kill=bool(complete and clean_lifecycle and accounted and body_failures)","body_kill=bool(result.testsRun and result.failures and not result.errors)"),
 ('legacy_positive',"baseline_success=bool(complete and clean_lifecycle and result.wasSuccessful())","baseline_success=bool(result.wasSuccessful())"),
 ('ignore_registered_population','complete=(population_valid and events==expected and result.testsRun==len(expected)\n              and all(state[\'body_entries\']==1 for state in states.values()))','complete=True'),
 ('drop_body_exception_identity',"_state['assertions'].add(id(error));raise","pass;raise"),
 ('erase_expected_failure_decoration','body=functools.wraps(method)(body)','body=body'),
 ('misreport_child_optimization','optimization_args=optimization_flags()','optimization_args=[]'),
 ('ignore_lifecycle_failures',"clean_lifecycle=(not invalid and not result.errors and not result.skipped\n                     and not result.expectedFailures and not result.unexpectedSuccesses)",'clean_lifecycle=True'),
 ('leak_case_patches','for case,name,existed,value in reversed(originals):','for case,name,existed,value in []:'),
 ('hide_unsupported_body_kind',"if unsupported: state['invalid'].append('unsupported_named_body')",'if False: pass'),
 ('accept_deferred_return','if value is not None:','if False:'),
]
for name,old,new in mutations:
 if source.count(old)!=1: raise AssertionError('mutation source site missing/ambiguous:'+name)
 with tempfile.TemporaryDirectory(prefix='cbz-lifecycle-source-mutant-') as tmp:
  p=Path(tmp)/'mutant.py';p.write_text(source.replace(old,new));tests=load(ROOT/'tests/test_cbz_mutation_lifecycle.py','tests_lifecycle_'+name);tests.TOOL=p
  suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
  with (OUT/('classifier-'+MODE+'-'+name+'.log')).open('w') as stream:
   result=helper.run_suite(suite,stream)
  summary['classifier_mutants'].append({'name':name,'killed':result['body_kill'],'lifecycle':result})
for p in [ROOT/'tools/cbz_mutation.py',ROOT/'tools/cbz_audit.py',ROOT/'tests/test_cbz_mutation_lifecycle.py',ROOT/'tests/test_cbz_audit.py',ROOT/'cbz/evidence/website-foundation-20261029/verification/ref_controls.py',ROOT/'cbz/evidence/website-foundation-20261029/verification/baseline_controls.py']:
 summary['source_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
summary['result']='PASS' if all(x['positive']['passed'] and all(r['detected'] for r in x['mutants']) for x in summary['portable']) and all(x['killed'] for x in summary['classifier_mutants']) else 'FAIL'
(OUT/('fixed-controls-'+MODE+'.json')).write_text(json.dumps(summary,indent=2)+'\n');print(summary['result']);print([(x['name'],x['killed']) for x in summary['classifier_mutants']]);raise SystemExit(0 if summary['result']=='PASS' else 1)
