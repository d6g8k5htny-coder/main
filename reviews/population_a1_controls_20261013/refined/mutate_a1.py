import hashlib, json, subprocess, sys
from pathlib import Path
repo=Path.cwd()
out=Path(__file__).resolve().parent
source=repo/'experiments/universality/two_parameter_cusp_population/population_certificate.py'
original=source.read_text(); before=hashlib.sha256(source.read_bytes()).hexdigest()
changes=[
('A1_LITERAL_SATURATION','    tau = min(tau, Q(9, 4))\n    s = root_bracket(tau) if tau < Q(9, 4) else (Q(1), Q(1))','    if tau >= Q(9, 4):\n        return Q(1, 5), Q(1, 5)\n    s = root_bracket(tau)'),
('A1_LITERAL_MISSING_SIGN','    missing_sign = _scale(saturation, Q(1, 2))','    missing_sign = (Q(1, 10), Q(1, 10))'),
('A1_LITERAL_CONDITIONED','    conditioned = _scale(saturation, Q(5))','    conditioned = (Q(1), Q(1))'),
('A1_WHOLE_GATE_TRUE',"'whole_rectangle_mass': rectangle_mass_control(saturation)","'whole_rectangle_mass': True"),
('A1_MISSING_GATE_TRUE',"'missing_sign_detected': missing_sign[0] <= Q(1, 10) <= missing_sign[1]\n             and not rectangle_mass_control(missing_sign)","'missing_sign_detected': True"),
('A1_CONDITIONED_GATE_TRUE',"'conditioning_detected': conditioned[0] <= Q(1) <= conditioned[1]\n             and not rectangle_mass_control(conditioned)","'conditioning_detected': True"),
('A1_STATUS_ALWAYS_PASS',"return {'status': 'PASS' if all(gates.values()) else 'FAIL_IMPLEMENTATION',","return {'status': 'PASS',"),
('A1_RECTANGLE_EQUALITY','    return observed[0] <= Q(1, 5) <= observed[1]','    return observed == (Q(1, 5), Q(1, 5))'),
('A1_HALF_ROUTE_NORMALIZATION','    result = _scale(_add(first, _multiply(coefficient, second)), Q(1, 15))','    result = _scale(_add(first, _multiply(coefficient, second)), Q(1, 30))'),
('A1_OMIT_HALF_SCALING','    missing_sign = _scale(saturation, Q(1, 2))','    missing_sign = saturation'),
('A1_OMIT_CONDITIONING','    conditioned = _scale(saturation, Q(5))','    conditioned = saturation')]
runner='''import json, sys, types, unittest\nfrom pathlib import Path\nsource=Path(sys.argv[1]); mutant=Path(sys.argv[2])\nsys.path.insert(0,str(source.parent))\nmodule=types.ModuleType("population_certificate"); module.__file__=str(source)\nexec(compile(mutant.read_text(),str(mutant),"exec"),module.__dict__)\nsys.modules["population_certificate"]=module\ncontrols=module.negative_controls()\nprint(json.dumps({"status":controls["status"],"gates":controls["gates"],"mass":str(controls["whole_rectangle_mass"]),"missing_sign_mass":str(controls["missing_sign_mass"]),"conditioned_mass":str(controls["conditioned_mass"])},sort_keys=True),flush=True)\nsuite=unittest.defaultTestLoader.loadTestsFromName("test_population_certificate.MassControlRouteTests")\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nprint(json.dumps({"tests_run":result.testsRun,"failures":[test.id() for test,trace in result.failures],"errors":[test.id() for test,trace in result.errors],"skipped":len(result.skipped)},sort_keys=True),flush=True)\nsys.exit(0 if result.wasSuccessful() else 1)\n'''
(out/'mutation_runner.py').write_text(runner)
results=[]
for name,old,new in changes:
    if original.count(old)!=1: raise RuntimeError((name,original.count(old)))
    mutated=original.replace(old,new); path=out/(name+'.py'); path.write_text(mutated)
    for mode in ('normal','optimized'):
        cmd=[sys.executable,'-B','-E']+(['-O'] if mode=='optimized' else [])+['-S',str(out/'mutation_runner.py'),str(source),str(path)]
        process=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
        (out/(name+'.'+mode+'.stdout.log')).write_text(process.stdout)
        (out/(name+'.'+mode+'.stderr.log')).write_text(process.stderr)
        lines=process.stdout.splitlines(); controls=json.loads(lines[0]); result=json.loads(lines[-1])
        result.update({'name':name,'mode':mode,'exit_code':process.returncode,'controls':controls,'source_sha256':hashlib.sha256(mutated.encode()).hexdigest(),'old':old,'new':new})
        results.append(result)
        if process.returncode!=1 or result['tests_run']!=7 or not result['failures'] or result['errors'] or result['skipped']:
            raise RuntimeError(result)
        print(name,mode,'KILLED',len(result['failures']),'FAILURES',flush=True)
(out/'a1_mutation_results.json').write_text(json.dumps({'source_before':before,'source_after':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'11 targeted A1 mutants; not S’s original 22-mutant or pending v3 campaign','results':results},indent=2)+'\n')
