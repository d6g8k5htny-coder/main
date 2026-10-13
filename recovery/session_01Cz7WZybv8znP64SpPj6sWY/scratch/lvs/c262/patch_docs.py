from pathlib import Path
side = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011')
def patch(name, pairs):
    p = side / name; s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:80])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print(name, 'patched')

# test: the notation3 case must not carry a string literal (the string check fires first)
patch('test_torus_filtration.py', [
    ("""            (inject('  notation3 "hidden" => Nat.zero'), command),\n""",
     """            (inject('  notation3 hidden => Nat.zero'), command),\n"""),
])

patch('README.md', [
    # G2: bounded scanner grammar
    ("""`skipKernelTC`, `autoImplicit`) in comment-free code. A compile fix that needs a refused form
must extend the scanner and the inventory explicitly; nothing escapes the `#print axioms` audit
by being written in a shape the scanner does not count.
""",
     """`skipKernelTC`, `autoImplicit`) in comment-free code, and the declaration-producing commands of
Batteries, Aesop and Mathlib at the pin (`alias`, `irreducible_def`, `simproc`, `simproc_decl`,
`dsimproc`, `dsimproc_decl`, `builtin_simproc`, `builtin_dsimproc`, `notation3`, `unif_hint`,
`recall`, `binder_predicate`, `add_decl_doc`, `library_note`, `assert_not_exists`,
`register_simp_attr`, `declare_aesop_rule_sets`, `compile_inductive`, `compile_def`, `nonrec`).
The refused keywords are the explicit enumeration `COMMANDS` in `check.py`, not a Lean grammar:
a declaration-producing command absent from that enumeration would pass an indented or
same-line position uncounted, and the `#print axioms` audit covers the inventoried names only.
The C262 gate review (Math- #428 comment 6112740935, finding G2) found `alias` absent from the
enumeration at head `a898bb44`; the enumeration now lists it and the other commands above, and
the census claim is bounded to that list. A compile fix that needs a refused form must extend
the scanner and the inventory explicitly.
"""),
    # G3: finish contract
    ("""control, not a passed one. `check.py finish` re-runs the source binding, re-derives the nine
emitted Lean files (`Audit.lean`, `Types.lean`, `Positive.lean`, six controls) and requires byte
equality with the ones Lean consumed, checks every status, the toolchain line, the installed
package HEADs against `formal/lake-manifest.json`, the recorded `leanchecker` invocation, the
axiom and type inventories and the six controls, and writes `receipt.json` with `log_sha256`
over every file in the evidence directory. The evidence directory
`.lake/torus-filtration-evidence` must not pre-exist.
""",
     """control, not a passed one. `check.py finish` refuses an evidence directory that already holds
`receipt.json` (a receipt is written once and never replaced; a new attempt needs a fresh
directory) and requires exactly the 41 evidence files, each a regular file and nothing else: the
nine emitted Lean files, a `.log` and `.status` pair for each of the seven Lean processes
(three module builds, `positive`, `axioms`, `types`, `leanchecker`), the six controls'
`.lean`/`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
`tests.log` and `tests-optimized.log` (`EVIDENCE_FILES` in `check.py`; C262 finding G3). It then
re-runs the source binding, re-derives the nine emitted Lean files and requires byte equality
with the ones Lean consumed, checks every status, the toolchain line, the installed package
HEADs against `formal/lake-manifest.json`, the recorded `leanchecker` invocation, the axiom and
type inventories and the six controls, and writes `receipt.json` with `log_sha256` over exactly
those 41 files. The evidence directory `.lake/torus-filtration-evidence` must not pre-exist.
"""),
    # G1: protocol tests
    ("""Local protocol tests: `python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py`
and the same with `-O`. They exercise `check.py` against a temporary Git fixture with
inventory-exact placeholder modules; they execute no Lean.
""",
     """Local protocol tests: `python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py`
and the same with `-O`. They exercise `check.py` against a temporary Git fixture with
inventory-exact placeholder modules; they execute no Lean. The fixture spawns its `check.py`
children with the outer interpreter's optimization level (`-O` under `-O`) and strips
`PYTHONOPTIMIZE` from their environment, and two controls observe the mode the children actually
run in (`sys.flags.optimize` reported from inside the child, including from a child running the
real `check.py source`). Until head `a898bb44` the children ran unoptimized whatever the outer
mode was (C262 finding G1): the `tests-optimized.log` transcripts of hosted runs 38165155316 and
38165763430 therefore cover the imported helpers under `-O` and the CLI paths unoptimized; that
narrower attribution stands for those runs and is not revised.
"""),
    # Amendment record
    ("""## What this companion does not do
""",
     """## Amendments

- C262 (Math- #428, nonauthor engineering gate review [6112740935](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112740935),
  disposition [6112743879](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112743879), AMEND_ENGINEERING_GATE on
  head `a898bb44`): G1 the test fixture now propagates the outer optimization level to its `check.py`
  children, strips `PYTHONOPTIMIZE` and observes the child mode; G2 the scanner refuses `alias` and
  the other declaration-producing commands listed above, with indented, same-line and column-0
  negative controls, and the census claim is bounded to the enumeration; G3 `finish` never replaces
  a receipt and requires the exact 41-member evidence set of regular files, with controls for a
  repeat `finish`, an existing receipt beside a newly failing input, every removed member, an extra
  member and a symlinked member. No Lean byte, declaration, control statement or workflow line
  changed. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
  [6112740740](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112740740)) and hosted-evidence verdict
  (APPROVE_OBSERVED_EXECUTION_SCOPE, [6112741132](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112741132))
  were of head `a898bb44`; organizational-independence credit 0 throughout.

## What this companion does not do
"""),
])

patch('SCOPE.md', [
    ("""unrecognized `leanchecker` invocation, leanchecker output in modules mode, wrong installed
dependency).
""",
     """unrecognized `leanchecker` invocation, leanchecker output in modules mode, wrong installed
dependency, a second `finish` on a directory already holding a receipt, an existing receipt
beside a newly failing input, any missing, extra or symlinked evidence member, an uninventoried
`alias` indented after the namespace opening or on a declaration line, and a `check.py` child
running in a different optimization mode from the outer test interpreter).
"""),
])
