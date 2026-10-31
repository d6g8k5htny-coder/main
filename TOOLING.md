# CBZ-0 tooling registry

Every tool built or changed for the CBZ-0 mission, with what it does, how to run
it, how it is itself verified, and its known limits. All tools live in
[`tools/cbz/`](tools/cbz/), use only the Python 3.11 standard library and Git,
and run from a clone of this repository. Rules for tools are statute B10 of
[GOVERNANCE.md](GOVERNANCE.md). Maintenance owner of every tool: none; each is
self-contained. Scientific effect of every tool: NONE.

## Certifying tools (mutation-covered, gate G0)

### `layer_a_scan.py` — Layer A static scanner (G1; invariants A1–A3)

- **Does.** Reads every Git-tracked `.lean` file at an exact commit through
  `git cat-file` (never the working tree). It masks comments (including nested
  block comments), string text, raw strings and character literals. It keeps
  string-interpolation code. Identifiers follow Lean 4's own `isIdFirst` and
  `isIdRest` rules. It reports each placeholder (`sorry`, `admit`, `sorryAx`,
  `stop`, sorry constructors, proof-search tactics, `#exit`), each axiom or
  trust token (`axiom`, `native_decide`, `+native`, `ofReduceBool`,
  `implemented_by`, `extern`, `skipKernelTC`, `addDecl`) and any floating point
  (`Float`, `Float32`).
- **Fails closed.** An undecodable file, an unterminated comment, string or
  quoted name, a symlinked `.lean` path or a submodule is an error.
- **Exemptions.** These go only through `layer_a_exemptions.json`, keyed by the
  exact repository, path, token and count. A stale count exempts nothing. An
  `isolated` fixture must not be imported. The registry does not exist yet,
  which means nothing is exempt.
- **Run.** `python3 -B tools/cbz/layer_a_scan.py --repo NAME=PATH@REV ... [--json FILE]`.
- **Verified by.** `tests/test_cbz_layer_a_scan.py` (59 tests), and the G0
  mutation target `layer_a_scan`: 58 mutants, being 27 rule deletions and 31
  logic mutants, all killed.
- **Limits.**
  - The checks are token checks; semantic placeholders (a vacuous statement, an
    unused hypothesis) need G6 and review.
  - Files that are not tracked at the commit are out of scope.
  - The kernel axiom audit (G2) remains the authority on axioms.

### `fuzz_bounds.py` — empirical fuzzing harness (G6; invariants A3, A6)

- **Does.** For every theorem of every Lean package at an exact commit, it
  translates the statement when it lies in the evaluable fragment:
  - numerals and `+ - * / % ^`;
  - comparisons, `∧ ∨ ¬ → ↔` and `∣`;
  - `∀`/`∃` over `ℕ ℤ ℚ ℝ`, with binder predicates;
  - `Set`/`Finset` intervals and finite `∑`/`∏`;
  - `|x|`, `min` and `max`;
  - exact integer functions;
  - non-recursive package definitions, including Prop-valued ones, also across
    the packages of one repository.

  It then evaluates exactly, with Python integers and fractions under Lean's
  own semantics: truncated `ℕ` subtraction, floor `ℕ` division, Euclidean `ℤ`
  division, and `x / 0 = 0`. Finite domains are evaluated exhaustively.
  Universal variables over infinite domains are sampled: pathological edge
  values first, then values near hypothesis bounds, then seeded random values
  at many scales. An equality hypothesis is repaired soundly: an exact rational
  root of a fitted polynomial of degree at most 2 is used only after every
  hypothesis has been re-evaluated exactly.
- **Statuses.**
  - `PASS-exact` and `PASS-sampled`;
  - `FAIL`, with an exact counterexample;
  - `VACUITY_SUSPECT`, when no sample satisfied the hypotheses;
  - `NA`, outside the fragment, with the exact reason;
  - `ERROR`, an internal fault. This fails closed.
- **Never certifies.** A pass means only that no counterexample was found in
  the recorded search. Acknowledging a vacuity needs
  `fuzz_acknowledged_vacuity.json`, keyed by the statement's SHA-256.
- **Run.** `python3 -B tools/cbz/fuzz_bounds.py --repo NAME=PATH@REV ... [--samples N] [--seed S] [--json FILE]`.
- **Verified by.** `tests/test_cbz_fuzz_bounds.py` (27 tests: semantic traps,
  edge-only counterexamples, vacuity, witness repair, definition expansion, NA
  boundaries, the CLI gate, acknowledgements, fail-closed faults), and the G0
  mutation target `fuzz_bounds`: 24 logic mutants, all killed. Its self-tests
  have already caught and fixed two real faults. One was a false pass when every
  sample exceeded the size limit. The other was identifier arguments that never
  reached a definition.
- **Limits.**
  - The fragment is lexical. Statements over measures, integrals, norms,
    functions, matrices or abstract types are `NA` with a recorded reason.
  - Transcendental functions are `NA`, since no rigorous interval layer is
    built.
  - Elaborated types need a Lean toolchain.

### `drift_crawler.py` — dynamic federation drift crawler (G7; criterion 8)

- **Discovers the topology at run time.** It starts from the federation root
  and follows:
  - submodule gitlinks;
  - Lean manifest git packages of the same owner;
  - JSON pin records (an object naming a repository and a 40-hex commit,
    optionally with a path and a blob).

  It keeps following until closure. No repository is listed in its code.
- **Checks.**
  - **P1:** every pinned object exists.
  - **P2:** every pinned commit lies on the default branch or on an archive
    spine.
  - **P3:** every recorded blob matches `commit:path`.
  - **P4:** there is one Mathlib revision and one toolchain.
  - **D1:** namespaced Lean declarations shared across files have one
    statement, unless one file is a provable older snapshot or the pair is
    allowlisted.
  - **L:** root-pin lag. It blocks with `--require-current-root-pins`, which is
    the baseline lock.

  Test-data JSON is excluded by statute B4 and counted.
- **Run.** `python3 -B tools/cbz/drift_crawler.py [--root URL] [--local URL=PATH ...] [--head URL=REV ...] [--emit-repos FILE] [--require-current-root-pins] [--json FILE]`.
- **Verified by.** `tests/test_cbz_drift_crawler.py` (19 tests on a synthetic
  federation behind made-up URLs, exercising every check in both directions),
  and the G0 mutation target `drift_crawler`: 18 logic mutants, all killed.
- **Limits.**
  - Signatures are lexical (`lean_decls.py`), not elaborated types.
  - A repository that the crawler cannot reach is a blocking P1 finding.

### `mutation.py` — mutation engine (G0; invariant A8)

- **Does.** For each certifying tool above, it builds mutants: literal
  replacements whose target text must occur exactly once, plus, for the
  scanner, one deletion per rule-table entry. It runs the tool's own suite
  against each mutant from a temporary copy of `tools/cbz/`. The gate passes
  only when every unmutated tool passes and 100% of mutants are killed. A
  missing or ambiguous mutation target is an error, never a kill.
- **Run.** `python3 -B tools/cbz/mutation.py [--target NAME] [--jobs N] [--json FILE]`.
- **Verified by.** `tests/test_cbz_mutation.py`. For each tool it shows that a
  behavior-neutral mutant survives and a real mutant dies, so "all killed"
  cannot come out by construction. It also shows that every target resolves,
  and that a missing target raises an error rather than counting as a kill.

## Support module

- **`lean_decls.py`** extracts Lean 4 declarations lexically for the fuzzer and
  the crawler, working on the scanner's masked text. For each declaration it
  returns:
  - its kind and namespace-qualified name;
  - its binders, type, value and a statement hash.

  It recognizes a pattern-matching arm only after a non-empty type and before
  an `=>`, so a statement that starts with `|x|` parses correctly. It is
  exercised through the fuzzer and crawler suites, which the mutation engine
  reruns.

## Known tooling gaps

The gap list, with what each gate can and cannot run today, is maintained in
[INVENTORY.md](INVENTORY.md).
