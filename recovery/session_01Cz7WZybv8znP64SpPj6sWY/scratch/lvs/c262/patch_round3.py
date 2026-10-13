"""Leftover wording and list fixes from the first-round docs verifier (apply only when no replay is running)."""
from pathlib import Path
side = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011')
def patch(name, pairs):
    p = side / name; s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:80])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print(name, 'patched')
patch('check.py', [
    ("""            'declare_eval_bin', 'declare_eval_bin_bitwise', 'declare_eval_bin_bool_pred', 'name_poly_vars',
            'name_power_vars', 'with_weak_namespace', 'meta', 'public')
""",
     """            'declare_eval_bin', 'declare_eval_bin_bitwise', 'declare_eval_bin_bool_pred', 'name_poly_vars',
            'name_power_vars', 'with_weak_namespace', 'meta', 'public', 'aux_def', 'lrat_proof')
"""),
])
patch('test_torus_filtration.py', [
    ("""            (inject('  public theorem hidden : True := trivial'), command),
""",
     """            (inject('  public theorem hidden : True := trivial'), command),
            (inject('  aux_def hidden : Nat := 0'), command),
            (inject('  lrat_proof hidden "p cnf 1 1 1 0" "1 0"'), 'string literal in code: ClosedSuperlevel:'),
            (inject('  lrat_proof hidden'), command),
"""),
    ("""            # A pre-existing receipt (even a symlink to nothing) is never replaced.
""",
     """            # A symlinked evidence directory is refused before anything is read.
            os.symlink(out, Path(tmp) / 'evidence-link')
            self.assertFails(f.run('finish', str(Path(tmp) / 'evidence-link')), 'evidence directory absent, not a directory, or a symlink')
            (Path(tmp) / 'evidence-link').unlink()
            # A pre-existing receipt (even a symlink to nothing) is never replaced.
"""),
])
patch('README.md', [
    ("""`skipKernelTC`, `autoImplicit`) in comment-free code, and the declaration-producing commands of
Batteries, Aesop and Mathlib at the pin (`alias`, `irreducible_def`, `simproc`, `simproc_decl`,
`dsimproc`, `dsimproc_decl`, `builtin_simproc`, `builtin_dsimproc`, `notation3`, `unif_hint`,
`recall`, `binder_predicate`, `add_decl_doc`, `library_note`, `assert_not_exists`,
`register_simp_attr`, `declare_aesop_rule_sets`, `compile_inductive`, `compile_def`, `nonrec`).
""",
     """`skipKernelTC`, `autoImplicit`) in comment-free code, and the declaration-producing commands of
Lean core, Batteries, Aesop and Mathlib found at the pin (the list `COMMANDS` in `check.py`:
`alias`, `irreducible_def`, the `simproc`/`dsimproc`/`cbv_simproc` families with their `_decl`
and `builtin_` forms, `notation3`, `unif_hint`, `recall`, `binder_predicate`, `add_decl_doc`,
`library_note`, `assert_not_exists`, `register_simp_attr`, `register_builtin_option`,
`register_linter_set`, `register_label_attr`, `declare_aesop_rule_sets`, `add_aesop_rules`,
`erase_aesop_rules`, `compile_inductive`, `compile_def`, `nonrec`, `mk_iff_of_inductive_prop`,
`run_meta`, `grind_propagator`, `builtin_grind_propagator`, `declare_config_elab` and its legacy
forms, `gen_injective_theorems`, `declare_uint_theorems`, `declare_int_theorems`,
`declare_bitwise_uint_theorems`, `declare_bitwise_int_theorems`, `elab_stx_quot`, the
`declare_eval_bin` family, `name_poly_vars`, `name_power_vars`, `with_weak_namespace`, `meta`,
`public`, `aux_def`, `lrat_proof`).
"""),
    ("""ten emitted Lean files, a `.log` and `.status` pair for each of the eight Lean processes
(three module builds, `positive`, `axioms`, `types`, `census`, `leanchecker`), the six controls'
`.lean`/`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
""",
     """ten emitted Lean files (`Audit.lean`, `Types.lean`, `Positive.lean`, `Census.lean` and the six
controls' `.lean`), a `.log` and `.status` pair for each of the eight Lean processes (three
module builds, `positive`, `axioms`, `types`, `census`, `leanchecker`), the six controls'
`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
"""),
    ("""which mode each run was. Until head `a898bb44` the children ran unoptimized whatever the outer
mode was (C262 finding G1): the `tests-optimized.log` transcripts of hosted runs 38165155316 and
""",
     """which mode each run was. Until head `a898bb44` the children ran unoptimized whatever the outer
mode was in a clean environment, and in the mode of an ambient `PYTHONOPTIMIZE` otherwise (C262
finding G1); the hosted runners are clean, so the `tests-optimized.log` transcripts of hosted runs 38165155316 and
"""),
])
print('round-3 wording patch ready')

# ---- additions from the second-round G1 verifier
patch('test_torus_filtration.py', [
    ("""        self.assertNotIn('PYTHONOPTIMIZE', child_env({'PYTHONOPTIMIZE': '1'}))
""",
     """        self.assertFalse('PYTHONOPTIMIZE' in child_env({'PYTHONOPTIMIZE': '1'}))   # assertFalse: never print an environment
"""),
    ("""            # A child forced into another mode is refused by the fixture's own check.
            with self.assertRaises(AssertionError):
                saved = sys.flags.optimize
                other = [sys.executable, '-B', '-S'] + ([] if saved else ['-O'])
                result = subprocess.run([*other, str(f.side / 'check.py'), 'source'], cwd=f.root, capture_output=True, text=True, env=child_env())
                if result.stderr.count(PYTHON_LINE + '\\n') != 1:
                    raise AssertionError('other mode')
""",
     """            # Negative control of the fixture's own guard: with the child prefix rebound to the OTHER mode, Fixture.run
            # itself must refuse (so a fixture that stopped propagating -O, or dropped the guard, is caught here).
            other = [sys.executable, '-B', '-S'] + ([] if sys.flags.optimize else ['-O'])
            saved = globals()['child_python']
            globals()['child_python'] = lambda: other
            try:
                with self.assertRaises(AssertionError):
                    f.run('source')
            finally:
                globals()['child_python'] = saved
            self.assertEqual(f.run('source').returncode, 0)
"""),
    ("""            # C262 G3: every one of the 41 required evidence members is required; removing any one withholds the receipt.
""",
     """            # C262 G3: every one of the 44 required evidence members is required; removing any one withholds the receipt.
"""),
])
patch('README.md', [
    ("""  replay unsets `PYTHONOPTIMIZE` and records the outer mode in both test transcripts; the receipt
  is created with `O_EXCL`/`O_NOFOLLOW` and a directory as a required member is refused. The
  replay step and workflow changed by the one census invocation and the `unset`; no Lean byte,
  declaration or control statement changed. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
""",
     """  replay unsets `PYTHONOPTIMIZE` and records the outer mode in both test transcripts; the receipt
  is created with `O_EXCL`/`O_NOFOLLOW` and a directory as a required member is refused. The
  replay script changed by the one census invocation and the `unset`; the Actions workflow file,
  every Lean byte, declaration and control statement are unchanged. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
"""),
    ("""  a receipt and requires the exact 41-member evidence set of regular files, with controls for a
""",
     """  a receipt and requires the exact evidence set of regular files (44 members once the census
  below was added), with controls for a
"""),
    ("""(`TORUS_FILTRATION_PYTHON: optimize=N`) to the job log.
""",
     """(`TORUS_FILTRATION_PYTHON: optimize=N`) to the job log (an invocation that argparse rejects
exits 2 before that line; the replay never issues one).
"""),
])
print('round-3 patch extended')
