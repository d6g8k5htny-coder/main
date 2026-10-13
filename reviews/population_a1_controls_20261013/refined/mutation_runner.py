import json, sys, types, unittest
from pathlib import Path
source=Path(sys.argv[1]); mutant=Path(sys.argv[2])
sys.path.insert(0,str(source.parent))
module=types.ModuleType("population_certificate"); module.__file__=str(source)
exec(compile(mutant.read_text(),str(mutant),"exec"),module.__dict__)
sys.modules["population_certificate"]=module
controls=module.negative_controls()
print(json.dumps({"status":controls["status"],"gates":controls["gates"],"mass":str(controls["whole_rectangle_mass"]),"missing_sign_mass":str(controls["missing_sign_mass"]),"conditioned_mass":str(controls["conditioned_mass"])},sort_keys=True),flush=True)
suite=unittest.defaultTestLoader.loadTestsFromName("test_population_certificate.MassControlRouteTests")
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({"tests_run":result.testsRun,"failures":[test.id() for test,trace in result.failures],"errors":[test.id() for test,trace in result.errors],"skipped":len(result.skipped)},sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
