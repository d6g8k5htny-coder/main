"""C262 G1 + G2 + G3 edits to test_torus_filtration.py (exact-string replacements; every anchor must match once)."""
from pathlib import Path
p = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011/test_torus_filtration.py')
s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)

# ---- G1: child interpreter invocation and environment, shared by the fixture and the observation control
rep("""def checker():
    spec = importlib.util.spec_from_file_location('torus_filtration_checker', SIDE / 'check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
""",
"""def checker():
    spec = importlib.util.spec_from_file_location('torus_filtration_checker', SIDE / 'check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_python():
    \"\"\"Interpreter prefix for every check.py child: `-B -S` plus the OUTER interpreter's optimization level.

    A child spawned through sys.executable does not inherit the parent's -O; until C262 (Math- #428 comment
    6112740935, finding G1) the -O test run therefore exercised check.py's CLI paths unoptimized.\"\"\"
    flags = ['-B', '-S']
    if sys.flags.optimize:
        flags.append('-' + 'O' * sys.flags.optimize)
    return [sys.executable, *flags]


def child_env(extra=None):
    \"\"\"Child environment: no GITHUB_* (host identity is set explicitly where a test needs it) and no PYTHONOPTIMIZE,
    ambient or injected, so the child's mode is exactly the one child_python() selects.\"\"\"
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GITHUB_')}
    environment.update(extra or {})
    environment.pop('PYTHONOPTIMIZE', None)
    return environment


OPTIMIZE_PROBE = 'import sys; print(sys.flags.optimize)'
# Runs the real check.py `source` inside a child that first reports the mode it actually executes in.
CHECK_SOURCE_OBSERVED = ('import sys, runpy; print("child optimize", sys.flags.optimize, file=sys.stderr); '
                         'sys.argv = [sys.argv[1], "source"]; runpy.run_path(sys.argv[0], run_name="__main__")')
""")

rep("""    def run(self, *args, env=None):
        environment = {k: v for k, v in os.environ.items() if not k.startswith('GITHUB_')}
        environment.update(env or {})
        return subprocess.run([sys.executable, '-B', '-S', str(self.side / 'check.py'), *args], cwd=self.root,
                              capture_output=True, text=True, env=environment)
""",
"""    def run(self, *args, env=None):
        return subprocess.run([*child_python(), str(self.side / 'check.py'), *args], cwd=self.root,
                              capture_output=True, text=True, env=child_env(env))
""")

# ---- G3: the fixture evidence directory carries the two protocol-test transcripts replay.sh writes
rep("""        (out / 'toolchain.log').write_text(LEAN_VERSION_LINE)
        (out / 'packages.log').write_text(''.join('%s %s\\n' % kv for kv in self.revisions.items()))
""",
"""        (out / 'toolchain.log').write_text(LEAN_VERSION_LINE)
        (out / 'packages.log').write_text(''.join('%s %s\\n' % kv for kv in self.revisions.items()))
        (out / 'tests.log').write_text('Ran 1 test\\n\\nOK\\n')
        (out / 'tests-optimized.log').write_text('Ran 1 test\\n\\nOK\\n')
""")

# ---- G1: observation controls of the child mode (a new test class before InventoryTests)
rep("""class InventoryTests(unittest.TestCase):
""",
"""class ChildInterpreterMode(unittest.TestCase):
    \"\"\"C262 G1: the mode a check.py child actually runs in is observed, not inferred from an argument string.\"\"\"

    def test_children_run_in_the_outer_optimization_mode(self):
        outer = sys.flags.optimize
        probe = subprocess.run([*child_python(), '-c', OPTIMIZE_PROBE], capture_output=True, text=True, env=child_env())
        self.assertEqual((probe.returncode, probe.stdout.strip()), (0, str(outer)), probe.stderr)
        # Ambient or injected PYTHONOPTIMIZE never reaches the child: the mode is the outer one in both cases.
        saved = os.environ.get('PYTHONOPTIMIZE')
        os.environ['PYTHONOPTIMIZE'] = '2'
        try:
            ambient = subprocess.run([*child_python(), '-c', OPTIMIZE_PROBE], capture_output=True, text=True, env=child_env())
            injected = subprocess.run([*child_python(), '-c', OPTIMIZE_PROBE], capture_output=True, text=True,
                                      env=child_env({'PYTHONOPTIMIZE': '1'}))
        finally:
            if saved is None:
                del os.environ['PYTHONOPTIMIZE']
            else:
                os.environ['PYTHONOPTIMIZE'] = saved
        self.assertEqual(ambient.stdout.strip(), str(outer))
        self.assertEqual(injected.stdout.strip(), str(outer))
        self.assertNotIn('PYTHONOPTIMIZE', child_env({'PYTHONOPTIMIZE': '1'}))
        # The invocation the fixture used before C262 (no propagation) runs the child unoptimized whatever the outer
        # mode is; under `-O` that is exactly the G1 observation, under the normal mode the two agree at 0.
        plain = subprocess.run([sys.executable, '-B', '-S', '-c', OPTIMIZE_PROBE], capture_output=True, text=True, env=child_env())
        self.assertEqual(plain.stdout.strip(), '0')
        if outer:
            self.assertNotEqual(plain.stdout.strip(), str(outer))
        self.assertEqual(child_python()[1:], ['-B', '-S'] + (['-' + 'O' * outer] if outer else []))

    def test_check_py_source_child_reports_the_outer_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(checker(), tmp)
            run = subprocess.run([*child_python(), '-c', CHECK_SOURCE_OBSERVED, str(f.side / 'check.py')], cwd=f.root,
                                 capture_output=True, text=True, env=child_env())
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('child optimize %d' % sys.flags.optimize, run.stderr)
            self.assertEqual(json.loads(run.stdout)['checked_commit'], f.git('rev-parse', 'HEAD').strip())
            # The CLI refusal path is exercised in the same observed mode.
            (f.side / 'README.md').write_text('# edited without a commit\\n')
            refused = subprocess.run([*child_python(), '-c', CHECK_SOURCE_OBSERVED, str(f.side / 'check.py')], cwd=f.root,
                                     capture_output=True, text=True, env=child_env())
            self.assertEqual(refused.returncode, 1)
            self.assertIn('child optimize %d' % sys.flags.optimize, refused.stderr)
            self.assertIn('TORUS_FILTRATION_FAIL: dirty execution/source bytes', refused.stderr)


class InventoryTests(unittest.TestCase):
""")

# ---- G2: scanner controls for the uninventoried alias (indented right after the namespace opening, same-line, column 0)
rep("""            (inject('  theorem indented_hidden : True := trivial'), command),
""",
"""            (inject('  theorem indented_hidden : True := trivial'), command),
            (good.replace('namespace TorusFiltration\\n', 'namespace TorusFiltration\\n  alias hidden := Nat.add_comm\\n'), command),
            (inject('theorem a_hidden : True := trivial alias hidden := Nat.add_comm'), command),
            (inject('alias hidden := Nat.add_comm'), unexpected),
            (inject('  irreducible_def hidden : Nat := 0'), command),
            (inject('  simproc hidden (Nat.succ _) := fun _ => return .continue'), command),
            (inject('  notation3 "hidden" => Nat.zero'), command),
            (inject('  unif_hint (x : Nat) where ⊢ x =?= x'), command),
            (inject('  recall superlevel_set : Nat := 0'), command),
            (inject('  add_decl_doc superlevel_set'), command),
            (inject('  nonrec def hidden : Nat := 0'), command),
""")
rep("""                     (original.replace('\\nend TorusFiltration', '\\n  theorem hidden : True := trivial\\nend TorusFiltration'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
""",
"""                     (original.replace('\\nend TorusFiltration', '\\n  theorem hidden : True := trivial\\nend TorusFiltration'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     # C262 G2: an indented imported `alias` immediately after the namespace opening, refused by the source CLI
                     (original.replace('namespace TorusFiltration\\n', 'namespace TorusFiltration\\n  alias hidden := Nat.add_comm\\n'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('theorem superlevel_mono : True := trivial',
                                       'theorem superlevel_mono : True := trivial alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
""")

# ---- G3: receipt immutability and exact membership controls
rep("""            self.assertEqual(set(receipt['log_sha256']), {p.name for p in out.iterdir() if p.is_file()} - {'receipt.json'})
            self.assertIn('axioms.log', receipt['log_sha256'])
            self.assertIn('leanchecker.cmd', receipt['log_sha256'])
            self.assertIn('Positive.lean', receipt['log_sha256'])
            # An operator's --fresh run is recorded as such, with leanchecker's own progress line tolerated.
            (out / 'receipt.json').unlink()
""",
"""            self.assertEqual(set(receipt['log_sha256']), {p.name for p in out.iterdir() if p.is_file()} - {'receipt.json'})
            self.assertEqual(set(receipt['log_sha256']), set(self.c.EVIDENCE_FILES))
            self.assertEqual(len(receipt['log_sha256']), 41)
            self.assertIn('axioms.log', receipt['log_sha256'])
            self.assertIn('leanchecker.cmd', receipt['log_sha256'])
            self.assertIn('Positive.lean', receipt['log_sha256'])
            self.assertIn('tests-optimized.log', receipt['log_sha256'])
            # C262 G3: a receipt is never replaced. A second finish on the same directory is refused and leaves the
            # old bytes intact, with or without a newly failing input; no digest of the receipt itself appears.
            frozen = (out / 'receipt.json').read_bytes()
            self.assertFails(f.run('finish', str(out)), 'receipt.json already present')
            self.assertEqual((out / 'receipt.json').read_bytes(), frozen)
            (out / 'leanchecker.status').write_text('1\\n')
            self.assertFails(f.run('finish', str(out)), 'receipt.json already present')
            self.assertEqual((out / 'receipt.json').read_bytes(), frozen)
            (out / 'leanchecker.status').write_text('0\\n')
            self.assertNotIn('receipt.json', receipt['log_sha256'])
            # An operator's --fresh run is recorded as such, with leanchecker's own progress line tolerated.
            (out / 'receipt.json').unlink()
""")

rep("""            for name, message in (('leanchecker.cmd', 'leanchecker invocation record absent'),
                                  ('Positive.lean', 'emitted Lean file changed across execution: Positive.lean'),
                                  ('RejectSuperlevelBelowThreshold.status', '[Errno 2] No such file or directory')):
                with self.subTest(name=name, message=message):
                    saved = (out / name).read_text()
                    (out / name).unlink()
                    try:
                        self.assertFails(f.run('finish', str(out)), message)
                        self.assertFalse((out / 'receipt.json').exists(), 'receipt written despite refusal')
                    finally:
                        (out / name).write_text(saved)
            self.assertEqual(f.run('finish', str(out)).returncode, 0)
""",
"""            # C262 G3: every one of the 41 required evidence members is required; removing any one withholds the receipt.
            self.assertEqual({p.name for p in out.iterdir()}, set(self.c.EVIDENCE_FILES))
            for name in sorted(self.c.EVIDENCE_FILES):
                with self.subTest(removed=name):
                    saved = (out / name).read_bytes()
                    (out / name).unlink()
                    try:
                        self.assertFails(f.run('finish', str(out)), "evidence membership differs: missing ['%s'], unexpected []" % name)
                        self.assertFalse((out / 'receipt.json').exists(), 'receipt written despite refusal')
                    finally:
                        (out / name).write_bytes(saved)
            # An unexpected member, a symlinked member and a stray subdirectory are refused too.
            (out / 'extra.log').write_text('')
            self.assertFails(f.run('finish', str(out)), "evidence membership differs: missing [], unexpected ['extra.log']")
            (out / 'extra.log').unlink()
            (out / 'stray').mkdir()
            self.assertFails(f.run('finish', str(out)), "evidence membership differs: missing [], unexpected ['stray']")
            (out / 'stray').rmdir()
            saved = (out / 'tests.log').read_bytes()
            (out / 'tests.log').unlink()
            os.symlink('tests-optimized.log', out / 'tests.log')
            self.assertFails(f.run('finish', str(out)), 'evidence member is not a regular file: tests.log')
            self.assertFalse((out / 'receipt.json').exists())
            (out / 'tests.log').unlink()
            (out / 'tests.log').write_bytes(saved)
            # A pre-existing receipt (even a symlink to nothing) is never replaced.
            os.symlink('absent-target', out / 'receipt.json')
            self.assertFails(f.run('finish', str(out)), 'receipt.json already present')
            (out / 'receipt.json').unlink()
            self.assertEqual(f.run('finish', str(out)).returncode, 0)
""")
p.write_text(s, encoding='utf-8')
print('tests patched')
