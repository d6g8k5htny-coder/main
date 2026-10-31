# G12 — empirical statement check: tool, self-test and first runs

Unit `CBZ0-G12-STATEMENT-FUZZ-01N-20261029`, pickup
[main#684/6510740889](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6510740889),
under the owner's directive addendum of 2026-10-29 22:58Z ("INTENT-LOCKED REINFORCEMENTS", published
at [684/6510536371](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6510536371)),
deliverable 1: *"Automatically subject all formal Lean proof bounds and geometric claims to large-scale,
randomized, pathological numerical edge-case testing. No mathematical proof closure may be marked
CLOSED without passing this empirical fuzzing validation layer."*

Performer: Anthropic Claude, Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N).
B-003: a new tool, its tests and this evidence directory only; the `GOVERNANCE.md` / `TOOLING.md`
rows below are *proposed text* for Root. Credit 0. Scientific effect NONE.

## What the tool is

`tools/cbz/statement_fuzz.py` reads a formal package's `manifest.json`, parses every registered
theorem's **statement** and the package's own `def`s in a small interpreter for the core-Lean
arithmetic subset the skeletons use, and evaluates each statement **independently of Lean** in exact
arithmetic (Python integers and `fractions.Fraction`; a float anywhere is an assertion failure —
Layer A3). It implements Lean's semantics where they differ from naive arithmetic: `Nat` subtraction
truncates at 0, division by zero is 0 and `x % 0 = x` (totalized), `Nat` division floors, numerals
default to `Nat` unless an ascription or a definition's result type makes the arithmetic `Int` (the
maximal-type rule of Lean's `binop%` elaborator), and `Int` `/` and `%` are **Euclidean**
(`Int.ediv`/`Int.emod`, `0 ≤ r < |b|`) as in Lean 4 core — not Python's floor division and not
T-rounding. The six golden pairs of the Codex control
[684/6511137449](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6511137449)
(`5 / -2 = -2, r 1`; `-5 / -2 = 3, r 1`; `-5 / 2 = -3, r 1`; `5 / 2 = 2, r 1`; `5 / 0 = 0, r 5`;
`-5 / 0 = 0, r -5`) are tests, together with the Python-floor and T-division misreadings as FAIL
controls.

Coverage of the subset: literals, `+ - * / % ^`, `∧ ∨ ¬ → ↔`, `= ≠ < > ≤ ≥ ∈ ∉`, ascriptions
`(e : Nat|Int)`, pairs and projections `p.1`/`p.2`, `Bool` (`true`, `false`, `!`, `&&`, `||`,
`if … then … else`), lists (`[…]`, `List.range`, `.foldl`, `.filter`, `.map`, `.sum`, `.length`),
lambdas, local `def`s with typed parameters, pattern-matching `def`s with `0`/`n + 1` patterns and
leading parameters (`def choose (n : Nat) : Nat → Nat | 0 => … | k + 1 => …`), the builtins
`Nat.choose`, `Nat.factorial`, `Nat.succ`, `Nat.pred`, `min`/`max`, `Int.natAbs`, `Int.toNat`, and
universal quantifiers over a finite domain written `∀ x, x ∈ [list] → …`, `∀ x ∈ [list], …`,
`∀ x, x ≤ n → …` or `∀ x, x < n → …`, including quantifiers inside conjunctions and over lists of pairs.

Per target it reports:

- **PASS** — the statement, read independently of Lean, evaluates true; every finite quantifier domain
  was enumerated exhaustively (the instance count is recorded).
- **FAIL** — it evaluates false; the failing instances are named. This is a misstatement or a
  transcription defect in the Lean text, **never** evidence about the kernel.
- **UNSUPPORTED** — the statement uses something outside the subset (Mathlib notation, `ℝ`, binders
  before the colon, an unbounded quantifier). Fail-closed: such a target
  is listed, cannot count as fuzz-validated, and makes the exit code non-zero.

It also records, as information only, **boundary probes**: for each quantified finite domain it
evaluates the body just outside the domain (`min − 1`, `max + 1`, `0`, `2·max + 1`) and reports whether
the claim still holds there. A probe that fails shows the hypothesis is doing work (tightness); an
empty domain is flagged **vacuous** (A6). Probes never change the verdict.

**Self-verification (A8).** `tests/test_cbz_statement_fuzz.py` (34 tests, normal and `-O`, `-W error`
clean) covers the tokenizer, Lean semantics, definitions, folds, pairs, quantifiers, the vacuity and
tightness probes, fail-closed behaviour on Mathlib syntax, bare-name manifests, the mutation contract
and the real `formal/` package. `--selftest` runs the **mutation contract** on every module, in the
three classes the Codex design finding [684/6510829073](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6510829073)
asked for, each expectation justified before use:

1. **Guaranteed-false mutants must FAIL.** Each PASS statement is mutated into forms that are false
   whenever the original is true: every comparison replaced by its negation, one conjunct at a time;
   the right-hand literal of an equality shifted by one. Never `<`→`≤`.
2. **Truth-preserving weakenings must PASS.** `<`→`≤` and `>`→`≥` are true whenever the original is;
   a rejected weakening is an evaluator defect (`522 < 529` and `522 ≤ 529` are both true).
3. **The manifest's declared negative controls must FAIL.** Each `replace`→`with` edit that changes a
   theorem statement (e.g. `23 ^ 2`→`22 ^ 2`, whose falsity the package records with Lean's own
   `decide` diagnostic) is applied to the module text and evaluated; edits elsewhere (proof text, an
   injected `sorry` or axiom) are classified non-statement and not counted.

The unmutated set is the harness control; an UNSUPPORTED or errored mutant is a survivor or a
rejection, never a kill; and a stubbed evaluator that answers PASS to everything fails the control
(`test_a_broken_evaluator_is_caught_by_the_control`, `test_unsupported_mutant_is_a_survivor_not_a_kill`).

## Runs (exact commands and heads)

```sh
python3 -B -S -m unittest tests.test_cbz_statement_fuzz          # 34 tests OK; same with -O and -W error
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json main_formal_4712fc13.json
git -C <Math- clone> archive 19a55750 formal | tar -x -C <tmp>; python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --json math_formal_19a55750.json
git archive b7ec86c2 formal | tar -x -C <tmp>; python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --selftest --json main_pr358_b7ec86c2.json
```

| Package | Cut | Targets | PASS | FAIL | UNSUPPORTED | Mutants killed | Exit |
|---|---|---:|---:|---:|---:|---:|---:|
| main `formal/` (SIDE24 arithmetic skeleton) | `main` `4712fc13` (31 Oct; `formal/` byte-identical to `e6f26538`, the 29 Oct run) | 31 | **31** | 0 | 0 | 98 / 98, 12 weakenings kept, 7 declared controls FAIL | 0 |
| main #358 `formal/` (SIDE24 + P15 skeletons) | PR head `b7ec86c2` | 59 | **59** | 0 | 0 | 184 / 184, 24 kept, 14 controls FAIL | 0 |
| Math- `formal/` (`ResearchFormalCoreR1`, Mathlib real analysis) | `main` `19a55750` (31 Oct; `9f8dbb4d` on 29 Oct gave the same) | 63 | 0 | 0 | **63** | — | 1 |

No registered statement in any of the three packages uses `Int` division, so the Euclidean contract
(31 October) changed no verdict; it changed what the tool would do if one did.

Tightness information from the probes on `main`: `pairing_sum_le_six` (`∀ q ∈ [0..6], pairingSum q ≤ 76`)
fails at `q = 7` and `q = 13`; `exponent_bounds` (`∀ d ∈ [2, 3], …`) holds at `d = 0, 1` and fails at
`d = 4, 7`. No vacuous domain. The 28 P15 targets of #358 all pass, including the finite-instance
quantifiers over pairs and the `Bool`-pair enumeration of the demand-one counterexample.

**Math- status, plainly.** All 63 registered targets are real-analysis statements with binders before
the colon (60) or `ℝ` ascriptions (3). The interpreter does not cover `ℝ`, Mathlib notation or
hypotheses on real variables, so under the addendum's rule none of them is fuzz-validated yet. This is
the honest G12 state for the Math- core and a **recorded tool gap** (§11): an extension needs an
interval-arithmetic evaluator for `Real.exp`/`Real.log`/`Real.sqrt`/finite sums with randomized and
boundary sampling of hypothesised real variables — exact rational endpoints, never floats — plus its
own mutation self-test. Claude017 has offered exactly such a backend (`interval.py`: outward-rounded
exact rational intervals with Mathlib's totalized conventions,
[684/6532742483](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6532742483)); the
offer is accepted and the backend is wired in as a follow-up on this branch once it is pushed with its
tests. Nothing in this directory certifies those statements.

## Proposed gate row and registry entries (for Root; B-003)

**§10 gate matrix, new row:**

| Gate | Command/Check | Pass condition | Blocking |
|---|---|---|---|
| G12 Empirical statement check | `python3 -B -S tools/cbz/statement_fuzz.py --package <formal package> --selftest` for every formal package in scope | every registered target PASS; no UNSUPPORTED; every module's mutation contract `harness_ok` (class 1 all killed, class 2 all kept) and every declared statement control FAIL; exit 0 | ✅ for a target to be CLOSED; a package with UNSUPPORTED targets cannot have them CLOSED |

**`TOOLING.md` entry:** `tools/cbz/statement_fuzz.py` — purpose: independent exact-arithmetic
evaluation of core-Lean arithmetic statements, finite-quantifier enumeration, vacuity/tightness probes;
usage: as above (`--json` writes the full report); self-verification: `tests/test_cbz_statement_fuzz.py`
(34 tests, both modes) and the built-in `--selftest` mutation contract (control, guaranteed-false
mutants, truth-preserving weakenings, declared manifest controls); serves CBZ-0 criterion 3 (CLOSED requires G12 PASS) and the addendum's deliverable 1;
maintenance owner: none — self-contained, stdlib only; domain limit: core-Lean arithmetic subset,
UNSUPPORTED elsewhere (B-006).

**`GOVERNANCE.md`, Layer B statute (proposed B-007):** a `PROOF_INDEX` row may be CLOSED only when
the target's G12 status is PASS at the same source cut as its kernel receipt; a PASS is recorded
with the report file's SHA-256 and the `manifest_sha256` it binds.

## Limits

- Statement-level only. A PASS means the Lean text says what it was read to say and that reading is
  true in exact arithmetic. It adds nothing to, and takes nothing from, the kernel's verdict.
- The interpreter's reading of Lean elaboration (numeral typing, `binop%` maximal type, `Int`
  division) is an approximation that is exact on the constructs the skeletons use and fail-closed
  elsewhere; a construct it misreads would surface as FAIL on a kernel-checked theorem, which is the
  signal to look at the interpreter, not the theorem.
- Sampling outside finite domains is information, not a verdict; it is reported so a reader can see
  that hypotheses are not vacuous and where the stated bounds stop holding.
