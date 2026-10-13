import importlib.util, sys, tempfile, pathlib, subprocess
side = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('tm', side / 'test_torus_filtration.py')
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
print('outer optimize', sys.flags.optimize)
with tempfile.TemporaryDirectory() as tmp:
    f = tm.Fixture(tm.checker(), tmp)
    r = subprocess.run([*tm.child_python(), '-c', tm.CHECK_SOURCE_OBSERVED, str(f.side / 'check.py')], cwd=f.root, capture_output=True, text=True, env=tm.child_env())
    print('rc', r.returncode, '| stderr:', r.stderr.strip().splitlines()[:1], '| stdout has checked_commit:', '"checked_commit"' in r.stdout)
    (f.side / 'README.md').write_text('# edited\n')
    r = subprocess.run([*tm.child_python(), '-c', tm.CHECK_SOURCE_OBSERVED, str(f.side / 'check.py')], cwd=f.root, capture_output=True, text=True, env=tm.child_env())
    print('refused rc', r.returncode, '| stderr:', r.stderr.strip().splitlines()[:2])
