import shutil, sys, pathlib
D = pathlib.Path(sys.argv[1])
SUB = 'companions/torus_filtration_20261011'
tree = D / 'tree'

def mutant(name, file, old, new, count=1):
    dst = D / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(tree, dst)
    p = dst / SUB / file
    text = p.read_text()
    n = text.count(old)
    if n != count:
        raise SystemExit('%s: expected %d occurrence(s) of %r, found %d' % (name, count, old[:60], n))
    p.write_text(text.replace(old, new))
    print('built', name)

RUN_NEW = """        result = subprocess.run([*child_python(), str(self.side / 'check.py'), *args], cwd=self.root,
                                capture_output=True, text=True, env=child_env(env))
        # Every child reports the mode it actually runs in (check.py main); it must be the outer mode, on every
        # call, accept or refuse. A fixture that stopped propagating -O would fail here, not pass silently (C262 G1).
        if result.stderr.count(PYTHON_LINE + '\\n') != 1 or 'TORUS_FILTRATION_PYTHON: optimize=' in result.stderr.replace(PYTHON_LINE, ''):
            raise AssertionError('check.py child did not report the outer interpreter mode %d: %r' % (sys.flags.optimize, result.stderr[:300]))
        return result
"""
RUN_OLD = """        environment = {k: v for k, v in os.environ.items() if not k.startswith('GITHUB_')}
        environment.update(env or {})
        return subprocess.run([sys.executable, '-B', '-S', str(self.side / 'check.py'), *args], cwd=self.root,
                              capture_output=True, text=True, env=environment)
"""
# A: Fixture.run reverted to the pre-C262 invocation (no -O, old env incl. PYTHONOPTIMIZE passthrough, no stderr check)
mutant('mutA_revert_run', 'test_torus_filtration.py', RUN_NEW, RUN_OLD)
# B: drop the -O flag only (child_python never appends -O); stderr check intact
mutant('mutB_no_O_flag', 'test_torus_filtration.py',
       "    if sys.flags.optimize:\n        flags.append('-' + 'O' * sys.flags.optimize)\n",
       "    if False:\n        flags.append('-' + 'O' * sys.flags.optimize)\n")
# C: drop the stderr check only
CHECK = """        if result.stderr.count(PYTHON_LINE + '\\n') != 1 or 'TORUS_FILTRATION_PYTHON: optimize=' in result.stderr.replace(PYTHON_LINE, ''):
            raise AssertionError('check.py child did not report the outer interpreter mode %d: %r' % (sys.flags.optimize, result.stderr[:300]))
"""
mutant('mutC_no_stderr_check', 'test_torus_filtration.py', CHECK, "        pass\n")
# D1/D2: fixture env sets PYTHONOPTIMIZE=1 / =2 instead of popping it
mutant('mutD1_env_opt1', 'test_torus_filtration.py',
       "    environment.pop('PYTHONOPTIMIZE', None)\n", "    environment['PYTHONOPTIMIZE'] = '1'\n")
mutant('mutD2_env_opt2', 'test_torus_filtration.py',
       "    environment.pop('PYTHONOPTIMIZE', None)\n", "    environment['PYTHONOPTIMIZE'] = '2'\n")
# D3: no pop at all (ambient passes through); run with ambient PYTHONOPTIMIZE=2
mutant('mutD3_no_pop', 'test_torus_filtration.py',
       "    environment.pop('PYTHONOPTIMIZE', None)\n", "    pass\n")
# E: check.py stops printing the diagnostic
mutant('mutE_checkpy_silent', 'check.py',
       "    print('TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize, file=sys.stderr)\n", "    pass\n")
# F: Fixture.run propagates and checks, but the check compares against a constant 0 (wrong reference)
mutant('mutF_ref_const0', 'test_torus_filtration.py',
       "PYTHON_LINE = 'TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize\n",
       "PYTHON_LINE = 'TORUS_FILTRATION_PYTHON: optimize=%d' % 0\n")
