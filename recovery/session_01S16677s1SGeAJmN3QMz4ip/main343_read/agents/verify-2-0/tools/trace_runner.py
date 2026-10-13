"""Run the PR's unittest suite exactly as CI does (discover in the population dir) under
sys.settrace and record every executed line of run_population_certificate.py."""
import sys, unittest, io
from pathlib import Path
root = Path(sys.argv[1]).resolve()
pop = root / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
target = str(pop / 'run_population_certificate.py')
executed = set()
def tracer(frame, event, arg):
    if frame.f_code.co_filename == target:
        if event == 'line':
            executed.add(frame.f_lineno)
        return tracer
    return tracer
sys.settrace(tracer)
suite = unittest.defaultTestLoader.discover(str(pop), pattern='test_population_certificate.py', top_level_dir=str(pop))
buf = io.StringIO()
res = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
sys.settrace(None)
print('tests ran=%d failures=%d errors=%d' % (res.testsRun, len(res.failures), len(res.errors)))
src = Path(target).read_text().splitlines()
def report(lo, hi):
    missing = [n for n in range(lo, hi + 1) if n not in executed and src[n-1].strip() and not src[n-1].strip().startswith('#')]
    hit = [n for n in range(lo, hi + 1) if n in executed]
    print(f'lines {lo}-{hi}: executed={hit}')
    print(f'lines {lo}-{hi}: never executed (nonblank)={missing}')
report(102, 108)
report(169, 244)
report(312, 348)
