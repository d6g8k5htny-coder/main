from pathlib import Path
p = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011/test_torus_filtration.py')
s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100])
    s = s.replace(old, new)

rep("""EMITTED = ('Audit.lean', 'Types.lean', 'Positive.lean', 'RejectSuperlevelSignFlip.lean', 'RejectSuperlevelBelowThreshold.lean',
           'RejectConstantNotPeriodic.lean', 'RejectSorry.lean', 'RejectCustomAxiom.lean', 'RejectNativeDecide.lean')
""",
"""EMITTED = ('Audit.lean', 'Types.lean', 'Positive.lean', 'Census.lean', 'RejectSuperlevelSignFlip.lean',
           'RejectSuperlevelBelowThreshold.lean', 'RejectConstantNotPeriodic.lean', 'RejectSorry.lean', 'RejectCustomAxiom.lean',
           'RejectNativeDecide.lean')
PYTHON_LINE = 'TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize
# The outer interpreter mode goes into the test transcript (tests.log / tests-optimized.log), so the hosted
# evidence records which mode each suite ran in (C262 G1).
print('TORUS_FILTRATION_TESTS: outer optimize=%d' % sys.flags.optimize, file=sys.stderr)
""")
rep("""OPTIMIZE_PROBE = 'import sys; print(sys.flags.optimize)'
# Runs the real check.py `source` inside a child that first reports the mode it actually executes in.
CHECK_SOURCE_OBSERVED = ('import sys, runpy; print("child optimize", sys.flags.optimize, file=sys.stderr); '
                         'sys.argv = [sys.argv[1], "source"]; runpy.run_path(sys.argv[0], run_name="__main__")')
""",
"""OPTIMIZE_PROBE = 'import sys; print(sys.flags.optimize)'
""")
rep("""    def run(self, *args, env=None):
        return subprocess.run([*child_python(), str(self.side / 'check.py'), *args], cwd=self.root,
                              capture_output=True, text=True, env=child_env(env))
""",
"""    def run(self, *args, env=None):
        result = subprocess.run([*child_python(), str(self.side / 'check.py'), *args], cwd=self.root,
                                capture_output=True, text=True, env=child_env(env))
        # Every child reports the mode it actually runs in (check.py main); it must be the outer mode, on every
        # call, accept or refuse. A fixture that stopped propagating -O would fail here, not pass silently (C262 G1).
        if result.stderr.count(PYTHON_LINE + '\\n') != 1 or 'TORUS_FILTRATION_PYTHON: optimize=' in result.stderr.replace(PYTHON_LINE, ''):
            raise AssertionError('check.py child did not report the outer interpreter mode %d: %r' % (sys.flags.optimize, result.stderr[:300]))
        return result
""")
rep("""        (out / 'tests.log').write_text('Ran 1 test\\n\\nOK\\n')
        (out / 'tests-optimized.log').write_text('Ran 1 test\\n\\nOK\\n')
""",
"""        (out / 'tests.log').write_text('TORUS_FILTRATION_TESTS: outer optimize=0\\nRan 1 test\\n\\nOK\\n')
        (out / 'tests-optimized.log').write_text('TORUS_FILTRATION_TESTS: outer optimize=1\\nRan 1 test\\n\\nOK\\n')
""")
rep("""        for control in self.c.INJECTIONS:
            (out / (control + '.log')).write_text(good_injection_log(self.c, control))
            (out / (control + '.status')).write_text('0\\n')
        return out
""",
"""        for control in self.c.INJECTIONS:
            (out / (control + '.log')).write_text(good_injection_log(self.c, control))
            (out / (control + '.status')).write_text('0\\n')
        listed = sorted(targets + list(meta['census_auxiliary']))
        (out / 'census.log').write_text(''.join(n + '\\n' for n in listed) + 'CENSUS_END %d\\n' % len(listed))
        return out
""")
# G1 controls: replace the runpy-based test by the CLI diagnostic observation
start = s.index("    def test_check_py_source_child_reports_the_outer_mode(self):")
end = s.index("class InventoryTests(unittest.TestCase):")
s = s[:start] + """    def test_every_check_py_child_reports_the_outer_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(checker(), tmp)
            accepted = f.run('source')                      # Fixture.run itself requires the diagnostic on every call
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(accepted.stderr.strip(), PYTHON_LINE)
            (f.side / 'README.md').write_text('# edited without a commit\\n')
            refused = f.run('source')
            self.assertEqual(refused.returncode, 1)
            self.assertEqual(refused.stderr.splitlines()[0], PYTHON_LINE)
            self.assertIn('TORUS_FILTRATION_FAIL: dirty execution/source bytes', refused.stderr)
            f.git('checkout', '--', '.')
            # The invocation the fixture used before C262 runs the real check.py unoptimized under an outer -O, and
            # check.py now says so itself; under the normal outer mode both report 0.
            plain = subprocess.run([sys.executable, '-B', '-S', str(f.side / 'check.py'), 'source'], cwd=f.root,
                                   capture_output=True, text=True, env=child_env())
            self.assertEqual(plain.returncode, 0, plain.stderr)
            self.assertEqual(plain.stderr.strip(), 'TORUS_FILTRATION_PYTHON: optimize=0')
            if sys.flags.optimize:
                self.assertNotEqual(plain.stderr.strip(), PYTHON_LINE)
            # A child forced into another mode is refused by the fixture's own check.
            with self.assertRaises(AssertionError):
                saved = sys.flags.optimize
                other = [sys.executable, '-B', '-S'] + ([] if saved else ['-O'])
                result = subprocess.run([*other, str(f.side / 'check.py'), 'source'], cwd=f.root, capture_output=True, text=True, env=child_env())
                if result.stderr.count(PYTHON_LINE + '\\n') != 1:
                    raise AssertionError('other mode')


""" + s[end:]
# G2 scanner controls
rep("""            (inject('  notation3 hidden => Nat.zero'), command),
""",
"""            (inject('  notation3 hidden => Nat.zero'), command),
            (inject('  run_meta pure ()'), command),
            (inject('  mk_iff_of_inductive_prop Nat.le hidden'), command),
            (inject('  cbv_simproc hidden (Nat.succ _) := fun _ => return .continue'), command),
            (inject('  builtin_simproc_decl hidden (Nat.succ _) := fun _ => return .continue'), command),
            (inject('  add_aesop_rules safe hidden'), command),
            (inject('  register_label_attr hidden'), command),
            (inject('  grind_propagator hidden ↓Nat.le := fun _ => pure ()'), command),
            (inject('  declare_uint_theorems Nat'), command),
            (inject('  name_poly_vars X over Nat'), command),
            (inject('  with_weak_namespace Hidden theorem hidden : True := trivial'), command),
            (inject('  meta def hidden : Nat := 0'), command),
            (inject('  public theorem hidden : True := trivial'), command),
            # lexical splices found by the verification of the G2 repair: a closed block comment is a token separator
            # (as in Lean), and a numeral ends at the first non-digit, so `0alias` is `0` then the keyword
            (inject('  theorem/- -/hidden : True := trivial'), command),
            (inject('  alias/- -/hidden := Nat.add_comm'), command),
            (inject('theorem a_hidden : True := trivial theorem/- -/b_hidden : True := trivial'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0alias hidden := Nat.add_comm'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0theorem hidden : True := trivial'), command),
            (good.replace(':= trivial', ':= by\\n  exact/- -/sorry', 1), 'admission token in ClosedSuperlevel: sorry'),
            (good.replace(':= trivial', ':= by/- -/native_decide', 1), 'admission token in ClosedSuperlevel: native_decide'),
""")
rep("""        # Every command keyword the scanner knows is refused somewhere other than the start of a whitelisted line.
        for keyword in self.c.COMMANDS:
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                self.c.scan_module(inject('theorem k_hidden : True := trivial ' + keyword), 'ClosedSuperlevel')
""",
"""        # Every command keyword the scanner knows is refused somewhere other than the start of a whitelisted line,
        # indented after the namespace opening included (the C262 G2 position).
        for keyword in self.c.COMMANDS:
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                self.c.scan_module(inject('theorem k_hidden : True := trivial ' + keyword), 'ClosedSuperlevel')
            with self.subTest(keyword=keyword, position='indented after namespace'), self.assertRaises(ValueError):
                self.c.scan_module(good.replace('namespace TorusFiltration\\n', 'namespace TorusFiltration\\n  ' + keyword + ' hidden\\n'), 'ClosedSuperlevel')
        # Identifiers that merely contain a keyword, or dotted components, are not keywords (no false refusal).
        for text in ('theorem superlevel_mono : True := by\\n  exact (Nat.alias2 : True)'.replace('Nat.alias2', 'trivial'),
                     good.replace(':= trivial', ':= (fun (foo2alias : True) => foo2alias) trivial', 1),
                     good.replace(':= trivial', ':= (Classical.byContradiction fun h => h trivial)', 1)):
            self.assertEqual(self.c.scan_module(text, 'ClosedSuperlevel'), INVENTORY['ClosedSuperlevel'])
        # A `where` clause adds a dotted auxiliary declaration that no keyword list refuses: the scanner accepts it
        # and the elaborated census (Census.lean, `finish`) is what refuses it. Recorded here so the boundary is explicit.
        where_clause = good.replace('theorem superlevel_mono : True := trivial', 'theorem superlevel_mono : True := trivial\\n  where hidden : True := trivial')
        self.assertEqual(self.c.scan_module(where_clause, 'ClosedSuperlevel'), INVENTORY['ClosedSuperlevel'])
""")
# CLI controls
rep("""                     (original.replace('theorem superlevel_mono : True := trivial',
                                       'theorem superlevel_mono : True := trivial alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
""",
"""                     (original.replace('theorem superlevel_mono : True := trivial',
                                       'theorem superlevel_mono : True := trivial alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('\\nend TorusFiltration', '\\nalias hidden := Nat.add_comm\\nend TorusFiltration'),
                      'unexpected column-0 line in ClosedSuperlevel:'),
                     (original.replace('\\nend TorusFiltration', '\\n  theorem/- -/hidden : True := trivial\\nend TorusFiltration'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
""")
# emit test
rep("""            self.assertEqual({p.name for p in out.iterdir()},
                             {'Audit.lean', 'Types.lean', 'Positive.lean'} | {n + '.lean' for n in self.c.CONTROLS})
""",
"""            self.assertEqual({p.name for p in out.iterdir()},
                             {'Audit.lean', 'Types.lean', 'Positive.lean', 'Census.lean'} | {n + '.lean' for n in self.c.CONTROLS})
            census = (out / 'Census.lean').read_text()
            self.assertTrue(census.startswith('import TorusFiltration\\n'))
            for needed in ('isInternalDetail', 'getModuleIdxFor?', '`TorusTopology', '`ClosedSuperlevel', '`TorusFiltration', 'CENSUS_END'):
                self.assertIn(needed, census)
""")
# finish tests: 44 members, census controls, directory member
rep("""            self.assertEqual(set(receipt['log_sha256']), set(self.c.EVIDENCE_FILES))
            self.assertEqual(len(receipt['log_sha256']), 41)
""",
"""            self.assertEqual(set(receipt['log_sha256']), set(self.c.EVIDENCE_FILES))
            self.assertEqual(len(receipt['log_sha256']), 44)
            self.assertEqual(receipt['census'], {'constants': 58 + len(self.c.metadata()['census_auxiliary']),
                                                 'auxiliary': sorted(self.c.metadata()['census_auxiliary'])})
            self.assertIn('census.log', receipt['log_sha256'])
""")
rep("""                ('types.log', types_log + 'error: crash\\n', 'type diagnostic'),
""",
"""                ('types.log', types_log + 'error: crash\\n', 'type diagnostic'),
                # C262 G2 closure: the elaborated census must equal the inventory plus the named auxiliaries exactly
                ('census.log', (out / 'census.log').read_text().replace('CENSUS_END', 'TorusFiltration.superlevel_mono.hidden\\nCENSUS_END'),
                 'census count mismatch'),
                ('census.log', (out / 'census.log').read_text().replace('\\nCENSUS_END %d' % (58 + len(meta['census_auxiliary'])),
                                                                        '\\nTorusFiltration.superlevel_mono.hidden\\nCENSUS_END %d' % (59 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra ['TorusFiltration.superlevel_mono.hidden'], missing []"),
                ('census.log', (out / 'census.log').read_text().replace('TorusFiltration.superlevel_mono\\n', '').replace(
                    'CENSUS_END %d' % (58 + len(meta['census_auxiliary'])), 'CENSUS_END %d' % (57 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra [], missing ['TorusFiltration.superlevel_mono']"),
                ('census.log', (out / 'census.log').read_text().replace('TorusFiltration.torus_periodBasis.congr_simp\\n', '').replace(
                    'CENSUS_END %d' % (58 + len(meta['census_auxiliary'])), 'CENSUS_END %d' % (57 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra [], missing ['TorusFiltration.torus_periodBasis.congr_simp']"),
                ('census.log', ''.join((out / 'census.log').read_text().splitlines(True)[:-1]), 'census terminator absent'),
                ('census.log', 'warning: noise\\n' + (out / 'census.log').read_text(), 'unrecognized census output'),
                ('census.status', '1\\n', 'missing successful process: census'),
""")
rep("""            (out / 'stray').rmdir()
            saved = (out / 'tests.log').read_bytes()
""",
"""            (out / 'stray').rmdir()
            # A required member that is a directory is not a regular file.
            saved = (out / 'tests.log').read_bytes()
            (out / 'tests.log').unlink()
            (out / 'tests.log').mkdir()
            self.assertFails(f.run('finish', str(out)), 'evidence member is not a regular file: tests.log')
            self.assertFalse((out / 'receipt.json').exists())
            (out / 'tests.log').rmdir()
            (out / 'tests.log').write_bytes(saved)
            saved = (out / 'tests.log').read_bytes()
""")
# metadata schema: census_auxiliary rules
rep("""            (setter(('extra',), 1), 'SOURCE_FILES schema keys'),
            (deleter(('provenance',)), 'SOURCE_FILES schema keys'),
""",
"""            (setter(('extra',), 1), 'SOURCE_FILES schema keys'),
            (deleter(('provenance',)), 'SOURCE_FILES schema keys'),
            (deleter(('census_auxiliary',)), 'SOURCE_FILES schema keys'),
            (setter(('census_auxiliary',), 'TorusFiltration.torus_periodBasis.congr_simp'), 'census_auxiliary must list dotted auxiliary names'),
            (setter(('census_auxiliary',), ['TorusFiltration.hidden']), 'census_auxiliary must list dotted auxiliary names'),
            (setter(('census_auxiliary',), ['Other.torus_periodBasis.congr_simp']), 'census_auxiliary must list dotted auxiliary names'),
            (setter(('census_auxiliary',), ['TorusFiltration.torus_periodBasis.congr_simp'] * 2), 'census_auxiliary must list dotted auxiliary names'),
""")
# replay contract
rep("""                         'run_lean positive lean -DwarningAsError=true', 'run_lean axioms lean -DwarningAsError=true',
                         'run_lean types lean -DwarningAsError=true'):
            self.assertIn(required, self.replay, required)
""",
"""                         'run_lean positive lean -DwarningAsError=true', 'run_lean axioms lean -DwarningAsError=true',
                         'run_lean types lean -DwarningAsError=true',
                         'run_lean census lean -DwarningAsError=true --root="$TORUS_OUT" "$TORUS_OUT/Census.lean"',
                         'set -euo pipefail\\n# The outer Python mode is the one named on each command line below, never an ambient PYTHONOPTIMIZE (C262 G1).\\nunset PYTHONOPTIMIZE\\n'):
            self.assertIn(required, self.replay, required)
""")
p.write_text(s, encoding='utf-8')
print('tests second-round patch applied')
