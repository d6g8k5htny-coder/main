import importlib.util, json, sys, types, unittest
from pathlib import Path
repo=Path.cwd()
source=repo/'experiments/universality/two_parameter_cusp_population/run_population_certificate.py'
sys.path.insert(0,str(source.parent))
import test_population_certificate as tests
old="                elif not sound:\n                    row['status'] = 'FAIL_IMPLEMENTATION'\n"
raw=source.read_text()
if raw.count(old)!=1: raise RuntimeError('Mutation anchor not unique')
module=types.ModuleType('mutated_population_runner'); module.__file__=str(source)
exec(compile(raw.replace(old,''),str(source),'exec'),module.__dict__)
tests.runner=module
suite=unittest.defaultTestLoader.loadTestsFromName('MassControlRouteTests.test_real_campaign_withholds_acceptance_after_mass_control_fault',tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests_run':result.testsRun,'failures':[test.id() for test,_ in result.failures],'errors':[test.id() for test,_ in result.errors],'skipped':len(result.skipped)}))
sys.exit(0 if result.wasSuccessful() else 1)
