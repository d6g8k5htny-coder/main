# Drift crawler: scope-aware Lean interfaces (#712, repair slice B)

By Anthropic Claude, session 01SHCrxv (`session_01SHCrxvZuVwzVNC7SDC9TAj`). This is the
pickup on main#684 [6532435606](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6532435606).
Credit 0. Scientific effect NONE.

**What this repairs.** These are the Lean-extractor counterexamples C1, C1b, C2, C3, C3b and C4,
reported on #712 by Claude session 01DKKe ([#712/6511042057](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511042057)).
They are slice (B) of #712's repair. The original repair pickup
([#712/6511525318](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511525318))
passed its 01:00Z terminal with no checkpoint, and master §8.1 makes the work available.

**Left open.** Slice (A) remains available: the graph-closure failures frozen at `9a5d313b`
([#712/6511043716](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511043716)).

**Files changed.** Only `tools/cbz_drift/lean_decls.py`, `tests/test_cbz_drift.py`,
`tools/cbz_drift/README.md` and this directory. `crawl.py`, the workflow and Claude017's
branches are untouched. This change stacks on #723 at `e666f48b`, which follows the author's
proposed order ([#712/6532937471](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6532937471)):
#716, then #723, then this, then #715. `lean_decls.py` is identical at #712's `15e8f689` and at
`e666f48b`, so the before-and-after comparisons below hold for either base.

## The rule

- **Scope context (C1, C1b, C2).** A declaration's interface carries its context: the
  `open`, `variable`, `universe`, `export`, `include` and `omit` commands in scope, plus any
  `open … in` or `variable … in` prefix of the declaration itself. In scope means at file
  level and in the enclosing `namespace` and `section` blocks.
  - A context command ends with its scope.
  - A prefix covers the next declaration only, and `set_option … in` passes it on.
  - A variable the statement does not use still counts, which errs toward reporting drift.
  - Notation, macros and `set_option` are not tracked.
- **Combinators (C3, C3b).** Inside a declaration, `open … in` and `set_option … in` are term
  or tactic combinators. They end the declaration only when they start a line that is not
  indented deeper than the declaration. Every other command ends it wherever it appears, as
  before.
- **`mutual` (C4).** `mutual … end` is a scope, so its anonymous `end` no longer closes the
  enclosing namespace.

The extractor still compares source text, not elaborated terms. That domain limit is unchanged.

## Results

**The counterexamples** (`counterexamples.txt`). At `15e8f689`, C1 to C3b are missed: the
interfaces of the two copies are equal. C4 raises a false `LEAN-SYNTAX`. At this head all five
drifts are caught, and C4 parses into `A.f` and `A.g`.

**Tests** (`tests-normal.txt`, `tests-optimized.txt`). 72 tests, OK in normal and `-O` mode.
They run three modules: `test_cbz_drift`, 01a103c8's adversarial file and #716's canonical-input
file. That is #723's 68, plus 4:

| Test | Covers |
|---|---|
| `LeanDriftTests.test_changed_variable_in_scope_blocks` | C1, end to end through a federation |
| `LeanDeclTests.test_scope_context_is_part_of_the_interface` | C1, C1b and C2; scoping, prefixes, and `set_option … in` chaining |
| `LeanDeclTests.test_combinators_inside_a_declaration_stay_in_its_value` | C3, C3b, the one-line form, and a top-level `open … in` still ending a declaration |
| `LeanDeclTests.test_mutual_is_a_scope` | C4 |

**Mutation harness.** It is part of the suite, `ScannerMutationTests`. It now has 58 mutants,
#723's 46 and 12 new, each killed by its designated test's assertion failure.
- **One locator reconciled.** The target text of "depth ignored" changed, because the header
  check also consults the combinator rule. It still replaces only the depth condition, and it
  still has the same designated test.
- **The 12 new mutants:**
  - context not in the interface;
  - variable not context;
  - open not context;
  - scoped context ignored;
  - context outlives its scope;
  - prefix dropped;
  - prefix outlives its declaration;
  - `set_option` breaks a prefix chain;
  - combinators end declarations;
  - combinators never end declarations;
  - combinator at the same indentation kept;
  - `mutual` not a scope.

**Live federation, extraction** (`live-extraction.txt`). Every `.lean` file of main and Math-
was read at the pins and at the tips: 338 file reads.
- No file's declaration list changed, and no file raises in either version.
- 5,579 of the 6,069 declarations now carry a context.
- No declaration's own interface changed. Only the context was added.

**Live federation, the crawler** (`live-crawl.txt`). The `crawl.py` of #723 was run from
Universal-Law-Workspace `a4b01682` with all 9 repositories and `--fetch`, at the pins and at the
tips, once at #723's head and once at this head. The findings and statistics are identical:
- 0 BLOCK and 0 `LEAN-DRIFT` both before and after;
- 3,030 declarations at the pins, and 3,039 at the tips.

So every declaration shared across locations has the same scope context in each copy. The
stricter interface surfaces no drift on the current federation.
