from pathlib import Path
side = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011')
def patch(name, pairs):
    p = side / name; s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:80])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print(name, 'patched')

patch('test_torus_filtration.py', [
    ("""        for text in ('theorem superlevel_mono : True := by\\n  exact (Nat.alias2 : True)'.replace('Nat.alias2', 'trivial'),
                     good.replace(':= trivial', ':= (fun (foo2alias : True) => foo2alias) trivial', 1),
                     good.replace(':= trivial', ':= (Classical.byContradiction fun h => h trivial)', 1)):
""",
     """        for text in (good.replace(':= trivial', ':= (fun (_ : Foo.alias = 0) => trivial) rfl', 1),
                     good.replace(':= trivial', ':= (fun (foo2alias : True) => foo2alias) trivial', 1),
                     good.replace(':= trivial', ':= (fun (alias_like theorem_like : True) => theorem_like) trivial trivial', 1),
                     good.replace(':= trivial', ':= (Classical.byContradiction fun h => h trivial)', 1)):
"""),
])

patch('README.md', [
    # scanner paragraph: bounded, census is the guarantee
    ("""The refused keywords are the explicit enumeration `COMMANDS` in `check.py`, not a Lean grammar:
a declaration-producing command absent from that enumeration would pass an indented or
same-line position uncounted, and the `#print axioms` audit covers the inventoried names only.
The C262 gate review (Math- #428 comment 6112740935, finding G2) found `alias` absent from the
enumeration at head `a898bb44`; the enumeration now lists it and the other commands above, and
the census claim is bounded to that list. A compile fix that needs a refused form must extend
the scanner and the inventory explicitly.
""",
     """The refused keywords are the explicit list `COMMANDS` in `check.py`: a first filter, not a Lean
grammar and not a complete census. A declaration-producing command or form absent from that list
passes an indented or same-line position uncounted: the C262 gate review (Math- #428 comment
6112740935, finding G2) found `alias` absent at head `a898bb44`, the pre-push verification of
the repair found further commands at the pin (`run_meta`, `mk_iff_of_inductive_prop`, the
`cbv_simproc` family, `grind_propagator`, `register_builtin_option`, `register_linter_set`,
`register_label_attr`, `declare_config_elab`, `gen_injective_theorems%`, `declare_uint_theorems`,
`name_poly_vars`, `add_aesop_rules`, and more, all now listed), and the `where` clause of a
declaration adds a dotted auxiliary declaration that no keyword list can refuse. The lexer also
treats a closed block comment as a token separator (`theorem/- -/hidden` is `theorem hidden`)
and ends a numeral at its first non-digit (`0alias` is `0` then `alias`), both found by that
verification. **The declaration census the receipt stands on is therefore not the scanner but
`Census.lean`**, emitted by `check.py emit` and run by the replay: Lean itself lists every
constant the three companion modules add to the environment that is not an internal detail
(`Name.isInternalDetail`: `_`-prefixed, `eq_n`, `match_n`, `proof_n`, `omega_n` or numeric
components), and `check.py finish` requires that list to equal the 58 inventoried names plus
the compiler-generated auxiliaries named in `SOURCE_FILES.json` (`census_auxiliary`, today
exactly `TorusFiltration.torus_periodBasis.congr_simp`, the simp-congruence lemma Lean generates
for `torus_periodBasis`), nothing more and nothing less. A declaration the scanner missed is
refused by the receipt whatever its spelling. A compile fix that needs a refused form must
extend the scanner, the inventory and, for a new auxiliary, `census_auxiliary` explicitly.
"""),
    # run description: census step
    ("""concrete applications of the companion's theorems), `Audit.lean` (58 `#print axioms`),
`Types.lean` (41 `#check` with `pp.explicit`), runs
`leanchecker TorusTopology ClosedSuperlevel TorusFiltration`, and runs six negative controls.
""",
     """concrete applications of the companion's theorems), `Audit.lean` (58 `#print axioms`),
`Types.lean` (41 `#check` with `pp.explicit`), `Census.lean` (the elaborated declaration census
above), runs `leanchecker TorusTopology ClosedSuperlevel TorusFiltration`, and runs six negative
controls. The replay unsets `PYTHONOPTIMIZE` first, so each Python invocation runs in exactly the
mode its command line names, and every `check.py` invocation prints the mode it runs in
(`TORUS_FILTRATION_PYTHON: optimize=N`) to the job log.
"""),
    # finish paragraph: 44 members
    ("""directory) and requires exactly the 41 evidence files, each a regular file and nothing else: the
nine emitted Lean files, a `.log` and `.status` pair for each of the seven Lean processes
(three module builds, `positive`, `axioms`, `types`, `leanchecker`), the six controls'
`.lean`/`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
`tests.log` and `tests-optimized.log` (`EVIDENCE_FILES` in `check.py`; C262 finding G3). It then
re-runs the source binding, re-derives the nine emitted Lean files and requires byte equality
with the ones Lean consumed, checks every status, the toolchain line, the installed package
HEADs against `formal/lake-manifest.json`, the recorded `leanchecker` invocation, the axiom and
type inventories and the six controls, and writes `receipt.json` with `log_sha256` over exactly
those 41 files. The evidence directory `.lake/torus-filtration-evidence` must not pre-exist.
""",
     """directory) and requires exactly the 44 evidence files, each a regular file and nothing else: the
ten emitted Lean files, a `.log` and `.status` pair for each of the eight Lean processes
(three module builds, `positive`, `axioms`, `types`, `census`, `leanchecker`), the six controls'
`.lean`/`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
`tests.log` and `tests-optimized.log` (`EVIDENCE_FILES` in `check.py`; C262 finding G3). It then
re-runs the source binding, re-derives the ten emitted Lean files and requires byte equality
with the ones Lean consumed, checks every status, the toolchain line, the installed package
HEADs against `formal/lake-manifest.json`, the recorded `leanchecker` invocation, the axiom and
type inventories, the declaration census and the six controls, and writes `receipt.json` with
`log_sha256` over exactly those 44 files, created exclusively (`O_EXCL`, `O_NOFOLLOW`). The
evidence directory `.lake/torus-filtration-evidence` must not pre-exist.
"""),
    # protocol tests paragraph: structural observation
    ("""inventory-exact placeholder modules; they execute no Lean. The fixture spawns its `check.py`
children with the outer interpreter's optimization level (`-O` under `-O`) and strips
`PYTHONOPTIMIZE` from their environment, and two controls observe the mode the children actually
run in (`sys.flags.optimize` reported from inside the child, including from a child running the
real `check.py source`). Until head `a898bb44` the children ran unoptimized whatever the outer
mode was (C262 finding G1): the `tests-optimized.log` transcripts of hosted runs 38165155316 and
38165763430 therefore cover the imported helpers under `-O` and the CLI paths unoptimized; that
narrower attribution stands for those runs and is not revised.
""",
     """inventory-exact placeholder modules; they execute no Lean. The fixture spawns its `check.py`
children with the outer interpreter's optimization level (`-O` under `-O`) and strips
`PYTHONOPTIMIZE` from their environment; every `check.py` invocation reports the mode it actually
runs in on stderr, and the fixture refuses any child whose report differs from the outer mode,
on every call, accept or refuse (so a fixture that stopped propagating `-O` fails, rather than
passing with unoptimized children). The suite prints its own outer mode into the transcript
(`TORUS_FILTRATION_TESTS: outer optimize=N`), so `tests.log` and `tests-optimized.log` record
which mode each run was. Until head `a898bb44` the children ran unoptimized whatever the outer
mode was (C262 finding G1): the `tests-optimized.log` transcripts of hosted runs 38165155316 and
38165763430 therefore cover the imported helpers under `-O` and the CLI paths unoptimized; that
narrower attribution stands for those runs and is not revised.
"""),
    # amendments: add the verification round
    ("""  member and a symlinked member. No Lean byte, declaration, control statement or workflow line
  changed. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
""",
     """  member and a symlinked member. Before the push, this lane's own adversarial verification of
  the repair (same account, credit 0) refuted its first form and the fixes were folded into the
  same commit: the child-mode control now observes every fixture child through the CLI's own
  report, not a separate probe; the keyword list gained the further declaration-producing
  commands found at the pin and the lexer no longer splices tokens across a closed block comment
  or a numeral; the elaborated declaration census (`Census.lean`) replaces the scanner as the
  census guarantee, with `census_auxiliary` naming the one compiler-generated constant; the
  replay unsets `PYTHONOPTIMIZE` and records the outer mode in both test transcripts; the receipt
  is created with `O_EXCL`/`O_NOFOLLOW` and a directory as a required member is refused. The
  replay step and workflow changed by the one census invocation and the `unset`; no Lean byte,
  declaration or control statement changed. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
"""),
])

patch('SCOPE.md', [
    ("""`alias` indented after the namespace opening or on a declaration line, and a `check.py` child
running in a different optimization mode from the outer test interpreter).
""",
     """`alias` indented after the namespace opening, on a declaration line or at column 0, a keyword
spliced to a closed block comment or to a numeral, every refused command keyword indented after
the namespace opening, a declaration census (`Census.lean`) with an extra, missing or unlisted
constant, a wrong census count or terminator, and a `check.py` child running in a different
optimization mode from the outer test interpreter, which the fixture checks on every child).
"""),
    ("""A green hosted run establishes, for these 58 declarations at this toolchain and dependency
lock, that Lean elaborated them with warnings as errors, that
""",
     """A green hosted run establishes, for these 58 declarations at this toolchain and dependency
lock, that Lean elaborated them with warnings as errors, that Lean's own census of the
constants the three modules add to the environment (`Census.lean`, non-internal names) is
exactly these 58 plus the one named compiler-generated auxiliary, that
"""),
])
