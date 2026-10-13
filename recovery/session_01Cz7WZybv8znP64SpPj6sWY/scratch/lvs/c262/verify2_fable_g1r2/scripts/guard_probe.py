"""Does Fixture.run itself refuse a mis-moded child?  Monkeypatch child_python to the OTHER mode and call f.run('source')."""
import importlib.util, sys, tempfile, pathlib, io, contextlib
tree = pathlib.Path(sys.argv[1])
test_path = tree / 'companions/torus_filtration_20261011/test_torus_filtration.py'
sys.argv = [str(test_path)]
with contextlib.redirect_stderr(io.StringIO()):
    spec = importlib.util.spec_from_file_location('tt', test_path)
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
outer = sys.flags.optimize
other = [sys.executable, '-B', '-S'] + ([] if outer else ['-O'])
tt.child_python = lambda: other
with tempfile.TemporaryDirectory() as tmp:
    f = tt.Fixture(tt.checker(), tmp)
    try:
        r = f.run('source')
        print('%s outer=%d: Fixture.run ACCEPTED a mis-moded child (rc=%d, stderr=%r)' % (tree.name, outer, r.returncode, r.stderr.strip()))
    except AssertionError as e:
        print('%s outer=%d: Fixture.run REFUSED the mis-moded child: %s' % (tree.name, outer, str(e)[:90]))
