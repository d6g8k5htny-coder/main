"""Line coverage of run_population_certificate.py while the PR's own suite runs. Usage: cov_driver.py <root>"""
import sys, trace, unittest
from pathlib import Path
root = Path(sys.argv[1]).resolve()
pop = root / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
runner_file = str(pop / 'run_population_certificate.py')
tracer = trace.Trace(count=1, trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])
def go():
    suite = unittest.defaultTestLoader.discover(str(pop), pattern='test_population_certificate.py')
    r = unittest.TextTestRunner(verbosity=0).run(suite)
    print('ran', r.testsRun, 'failures', len(r.failures), 'errors', len(r.errors))
tracer.runfunc(go)
counts = tracer.results().counts
hit = {ln for (f, ln), c in counts.items() if f == runner_file and c > 0}
import ast
src = Path(runner_file).read_text()
tree = ast.parse(src)
# executable statement lines (first line of each statement) within _campaign and _overall and run
def stmt_lines(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.stmt) and not isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.lineno)
    return out
for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_overall', '_campaign', 'run')]:
    lines = stmt_lines(fn)
    missing = sorted(l for l in lines if l not in hit)
    executed = sorted(l for l in lines if l in hit)
    print(f'{fn.name}: {len(executed)} executed, {len(missing)} never executed')
    print('  never executed:', missing)
