# Canonical source auditors

These standard-library tools implement finite source inventories. Neither a
lexical zero nor an AST category establishes mathematical closure. Kernel
replay and complete axiom/dependency audits remain separate obligations.

## Source and consolidation

The implementation basis is Universal-Law-Workspace PR29 at
`ef5b924016c0f85ced7eb1dfb8f5eb9bd776f169`. Its 28 lexical behavioral tests,
39 source mutants, 18 A3 behavioral tests and 15 A3 source mutants are retained.
The compatible regression requirements from main PR687 at
`95e9dd43d70b8d650c07fdfc458199eec507a919` and main issue684 comment6503596060
are included: balanced ordinary-looking macro-string holes, native settings
with intervening layout, token boundaries, comments, literals, quotations,
escaped constant names, compiled-trust hazards and fail-closed input handling.
The existing root `tools/cbz_inventory.py` is not replaced.

Policy choices are explicit: Name quotations are conservatively reported;
escaped keywords such as `«sorry»` and qualified ordinary identifiers such as
`Foo.sorry` are not Lean placeholder syntax. Names of actual trust-hole
constants, including escaped components, are reported. `unsafe`, `extern`,
`implemented_by`, `stop`, `Float`, and `Float32` are review findings: their
presence alone is not a proof defect. Context names are metadata. The federation
command uses `--fail-on all`, so evidence/control paths cannot hide violations.

## Commands

Use the repository's documented CPython runtime (verified here with 3.11.16):

```sh
python3.11 -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_auditor_hardening tests.test_cbz_layer_a_scan_mutants tests.test_cbz_a3_float_census
python3.11 -O -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_auditor_hardening tests.test_cbz_layer_a_scan_mutants tests.test_cbz_a3_float_census
python3.11 -B -S tools/cbz/layer_a_scan.py --git /path/to/repository --rev EXACT_COMMIT --fail-on all --json /outside/source/lexical.json
python3.11 -B -S tools/cbz/a3_float_census.py --git /path/to/repository --rev EXACT_COMMIT --json /outside/source/float-census.json
```

Repeat the source commands for every repository at its recorded cut. Git mode
reads the specified immutable tree, never current worktree contents. Filesystem
mode is also available for explicit source files/directories and scans all
subdirectories except `.git`, including `.lake`; prefer Git mode for owned
repository sources. Missing roots, unreadable/undecodable sources, source
symlinks and directory symlinks fail closed. Git batch reads check requested
object identity/type, exact byte count/trailer and subprocess success; replace
objects, lazy fetch and ambient Git environment/global/system config are disabled.
Gitlinks fail closed: a separate pin-aware admission must resolve and scan them,
so a repository with unadmitted submodules cannot produce a global clean result.

## Self-verification

The final suite contains 72 tests, including 66 lexical source mutations and
23 A3 source mutations. Every old/replacement anchor must occur exactly once.
Each mutant is bound to a named assertion that must fail; import/launch failures,
syntax errors, unrelated failures, test errors and zero-test runs are not kills.
The unchanged source must pass through the same copied-source path. Both
mutation runners propagate optimized Python into their children. Actual
syntax-broken-source negative controls and synthetic error/foreign-failure
controls exercise the kill classifier. Two harness-control methods are skipped
only inside mutation subprocesses to prevent recursion; A3's two preexisting
self-mutation methods also do not recurse inside its subprocess. They run at
normal top level. This is not a skipped production/source admission.

Changes were first exercised as red regressions: 24 failed assertions across
11 hardening tests, followed by three failures in the whitespace/comment
interpolation-prefix test. The latter inputs were accepted by the actual pinned
Lean4.34.1 parser, demonstrating a genuine source-lexer omission. Independent
review then demonstrated actual `sorryAx` from nested `throwError` string holes
and actual unapproved axioms from keywords adjoining numerals (including
scientific/binary/octal forms). Both defects have direct regressions and source
mutants. A later 14-assertion red run covers numeral spelling and valid Mathlib
prime notation; five more red assertions cover the pinned identifier domain and
unsupported character escapes. Prior passing development cuts are historical,
not qualification of the repaired source.

## Limits

The lexer sees its documented source-token vocabulary and interpolation forms;
it cannot certify arbitrary custom syntax, generated declarations, imported
code, macro expansion or elaboration recovery. A3 uses the documented Python AST
signal vocabulary; arbitrary aliases, `getattr`, dynamic imports/`exec`, external
input types and semantic intent are outside it. `NO_FLOAT_SIGNAL` does not prove
exact arithmetic. `MIXED` does not prove enclosure of every floating operation.
An interval import is evidence to review, not a certificate. Per-file bound
claims still require exact rational or justified interval reasoning.

Maintenance owner: none — self-contained. Source changes require rerunning both
modes, all source mutants and the repository inventories. These tools serve G0,
the finite source portion of G1, A3 triage, and CBZ criteria4/9; they cannot alone
establish those global criteria.

## Pinned lexical oracle

The finite identifier, numeral and character-escape rules follow Lean4.34.1
release commit `5045d0056413266e57c625dcd7c365b10e377c52`:

- `Init/Meta/Defs.lean` (identifier ranges, lines101–143), SHA256
  `01154a5f32560e2e971728ccb90a0cba6faa2f5c4764c9bab4bbcf7cb60b20b6`.
- `Lean/Parser/Basic.lean` (character escapes, lines657–674; numerals,
  lines831–918), SHA256
  `9fc3df26147bb4c8dca560507426474f975628ad8598bbb4f5e392cea24ae8d7`.

Numeral recognition operates only on source spelling. It neither evaluates
numbers nor trusts compiled arithmetic for a mathematical bound. Explicit
regressions cover command keywords adjoining decimal/scientific/radix literals.
Secondary ordinary-string hole scans reuse the full lexer through nested
quoted terms, comments and escaped names; they never move the primary cursor.
Thus ambiguous brace data may add findings but cannot hide primary code.

Native configuration fields are fail-closed for arbitrary expressions: only
complete literal `false` followed by a field delimiter is exempt. Parenthesized
values, conditionals, operators after false, variables and applications are
reported without evaluation. Ordinary `def native := 1` stays an identifier
definition. Actual Lean accepted `native := (true)` and emitted a fresh native
decision axiom; the regression first failed on all21 expression/context cases.
