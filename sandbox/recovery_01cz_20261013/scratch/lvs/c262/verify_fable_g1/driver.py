"""Observe the mode of the REAL Fixture.run children (source, emit, finish) from inside the child."""
import importlib.util, sys, tempfile, pathlib
side = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('tm', side / 'test_torus_filtration.py')
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
print('outer sys.flags.optimize =', sys.flags.optimize, '| child_python() =', tm.child_python()[1:])
with tempfile.TemporaryDirectory() as tmp:
    f = tm.Fixture(tm.checker(), tmp)
    r = f.run('source'); print('source rc', r.returncode, '|', r.stderr.strip().splitlines()[0])
    out = f.evidence()  # runs source+emit via Fixture.run
    r = f.run('finish', str(out)); print('finish rc', r.returncode, '|', r.stderr.strip().splitlines()[0])
    (out / 'receipt.json').unlink(); (out / 'leanchecker.status').write_text('1\n')
    r = f.run('finish', str(out)); print('finish(refused) rc', r.returncode, '|', ' / '.join(r.stderr.strip().splitlines()[:2]))
