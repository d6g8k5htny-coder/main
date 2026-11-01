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
universal quantifiers over a finite domain written `∀ x, x ∈ [list] → …`, `∀ x ∈ [list], …`, or — for an
**explicit `Nat`/`ℕ` binder only** — `∀ x : Nat, x ≤ n → …` / `∀ x : ℕ, x < n → …`, including quantifiers
inside conjunctions and over lists of pairs. (An `Int`, `ℤ` or untyped binder with only an upper bound
is unbounded below and stays UNSUPPORTED: 01DKKe's P1 finding
[726/6547986788](https://github.com/d6g8k5htny-coder/main/pull/726#issuecomment-6547986788), fixed by
its follow-on with a regression test that fails on the previous code.)

Per target it reports:

- **PASS** — the statement, read independently of Lean, evaluates true; every finite quantifier domain
  was enumerated exhaustively (the instance count is recorded).
- **FAIL** — it evaluates false; the failing instances are named. This is a misstatement or a
  transcription defect in the Lean text, **never** evidence about the kernel.
- **CORROBORATED** (since the ℝ backend of 1 November, below) — the statement has binders before the
  colon; no sampled instance satisfying its hypotheses refuted it. Admitted evidence of the statement,
  never a proof, never PASS; the instance count, seed and rejected-draw count are recorded.
- **UNSUPPORTED** — the statement uses something outside the subset (Mathlib structures, measures and
  functions, transcendental functions, an unbounded quantifier inside the statement, a binder that is
  neither a sampled numeric type nor a decidable hypothesis). Fail-closed: such a target
  is listed, cannot count as fuzz-validated, and makes the exit code non-zero.

It also records, as information only, **boundary probes**: for each quantified finite domain it
evaluates the body just outside the domain (`min − 1`, `max + 1`, `0`, `2·max + 1`) and reports whether
the claim still holds there. A probe that fails shows the hypothesis is doing work (tightness); an
empty domain is flagged **vacuous** (A6). Probes never change the verdict.

**Self-verification (A8).** `tests/test_cbz_statement_fuzz.py` (35 tests at #726, 66 with the ℝ backend; normal and `-O`, `-W error`
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
python3 -B -S -m unittest tests.test_cbz_statement_fuzz          # 35 tests OK; same with -O and -W error
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
(31 October) changed no verdict; it changed what the tool would do if one did. Likewise every quantified
published target quantifies by list membership, so the bound-domain follow-on changed no verdict; the
three runs above were regenerated after it and are identical per target.

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

## ℝ backend and sampled binders (1 November 2026; receipts `*_r2*`)

The gap recorded above is closed for the algebraic part of the Math- core. The tool now:

- **Parses binders before the colon.** `(x y : ℝ)`, `{n : ℕ}` and the other numeric types (ℕ ℤ ℚ ℝ)
  declare *sampled variables*; `(h : p)` whose type parses as a proposition over names declared so far
  (variables, the module's `def`s, the interpreter's builtins) is a *hypothesis*, with Mathlib interval
  membership desugared (`x ∈ Set.Icc a b` ↦ `a ≤ x ∧ x ≤ b`, likewise `Ico Ioc Ioo Ici Iic Ioi Iio`).
  Anything else — `[IsProbabilityMeasure μ]`, `(μ : Measure Ω)`, `(f : Ω → ℝ)`, `(A : Set Ω)`, a
  predicate the interpreter does not define such as `Nat.Prime p` — makes the theorem UNSUPPORTED
  **naming the binder**. A quantifier with an infinite domain *inside* a statement stays UNSUPPORTED as
  before (the landed contract of #726 is unchanged there).
- **Samples pathologically, exactly, deterministically.** Variables are drawn in declaration order from
  a fixed pool (0, ±1, ±½, ±2, ±10⁶, ±10⁻⁶, squares and non-squares, repeating decimals such as 1/3 and
  22/7, 2³¹, 10¹²…), from the **boundary values the hypotheses name** (`0 ≤ r`, `r ≤ 1`, `m2 ^ 2 < m4`,
  `|a - b| ≤ δ` contribute 0, 1, m2², b ± δ and their ±1, ±10⁻⁶ neighbours) and from seeded random
  rationals (numerators to 10⁹, denominators to 10⁶); every hypothesis is checked as soon as its
  variables are assigned, so six-hypothesis statements are sampled without rejection explosions. The
  first points are structured (every variable 0, then 1, −1, ½, …), so a refutation witness is small
  (`x = 0` for `x / x = 1`, `cW = -1` for `√cW · √cW = cW`). Python's `random` chooses *which* exact
  rationals to try; no float exists anywhere (the A3 assertion guard is unchanged).
- **Evaluates ℝ exactly.** Rationals are `Fraction`s; `Real.sqrt` of a rational is an element of the
  multi-quadratic field ℚ(√d₁, …, √dₖ), kept in canonical form over squarefree radicands (`√8 = 2√2`,
  `√(1/2) = √2/2`, `√12·√3 = 6`), so **equality is decided exactly** (the √d for distinct squarefree d
  are ℚ-linearly independent: an element is zero iff every coefficient is), division rationalizes the
  denominator by conjugation over a coprime base of the radicands, and the **sign of a nonzero element
  is decided by refining integer-square-root enclosures** (64 → 4096 bits). `√2 + √3 < √10` is PASS and
  `√2 + √3 > √10` is FAIL (the difference is −0.016); `(√2 + √3 + √5 + √30)⁻¹ · (√2 + √3 + √5 + √30) = 1`
  is PASS. Mathlib's totalized conventions hold: `x / 0 = 0`, `0⁻¹ = 0`, `Real.sqrt x = 0` for `x ≤ 0`
  (`Real.sqrt_eq_zero'`); `a ^ n` types its exponent on its own as Lean's `rightact%` does, so
  `x ^ (1/2)` is `x ^ 0 = 1` exactly as in Lean, `x ^ (-2 : ℤ)` is `zpow`, and `x ^ (1/2 : ℝ)` (rpow)
  is UNSUPPORTED; an ascription elaborates its body at the ascribed type (`(n - 1 : ℝ)` is `↑n - 1`,
  `((n - 1 : ℕ) : ℝ)` casts a truncated Nat, `((7 : ℕ) / 2 : ℝ)` is `7/2` because `binop%` pushes the
  cast to the leaves). Nested radicals (`√(√2)`), `Real.exp`/`log`/`pi`/trig and real exponents are
  UNSUPPORTED with a message naming them. `|e|`, `√e`, `e⁻¹`, scientific literals (`0.5`, only in a
  rational or real context — alone, Lean would default them to `Float`, and the tool refuses) are parsed.
- **Caps instead of approximating.** An exact result over 2·10⁶ bits (`2 ^ (2 ^ 63)`), a factorial or
  binomial loop over 5 000, a `List.range` over 200 000 raise `Infeasible`: a sampled instance that is
  infeasible is skipped and *counted* (`sampling.infeasible`); a closed statement that is infeasible is
  UNSUPPORTED.
- **Keeps the mutation contract on sampled statements.** Mutants carry the theorem's binders: a class-1
  mutant of a true statement is false at every instance, so sampling must refute it; a class-2
  weakening is true at every instance, so sampling must not. The harness control accepts PASS or
  CORROBORATED for the unmutated set.
- **Exit code.** The registered contract is unchanged by default — exit 0 only when every target is
  PASS. `--admit-corroborated` admits CORROBORATED as well (recorded in the report under
  `sampling.admit_corroborated`); `--seed` and `--samples` (default 1 and 64) are recorded too.

Self-verification grew to **66 tests** (`tests/test_cbz_statement_fuzz.py`, normal and `-O`, `-W error`
clean): the 35 landed cases unchanged except one expectation (`lemma c (x : ℝ) : x ^ 2 ≥ 0` is now
CORROBORATED, not UNSUPPORTED; the package it sits in still exits 1 on its Mathlib target), plus the real
binders (`RealBinderTests`, 14: identities corroborated with counts, refutations with exact witnesses,
hypothesis filtering, totalized division, unsatisfiable hypotheses as UNSUPPORTED "no sample", exact and
totalized `Real.sqrt`, non-square radicands decided, `|·|`/`max`/`≠`, local ℝ `def`s with parameters,
non-numeric binders UNSUPPORTED, hypothesis boundaries sampled, seed determinism, the mutation contract),
exact algebra (`ExactRealAlgebraTests`, 6), sampled-binder semantics (`SampledBinderSemanticsTests`, 10:
ℕ/ℤ binders with truncation and Euclidean division, infeasible instances counted not approximated,
interval membership, boundaries between variables, inner finite quantifiers under a sampled binder,
undefined predicates naming the binder, type tokens such as `ℝ≥0` inside a binder type reported as types, a
comparison of bare scientific literals refused as Float, `|·|` as an application argument, exact witnesses at
which the hypotheses hold) and the exit-code flag (`AdmitCorroboratedPackageTests`, 1). Two of the author's own
first expectations were wrong and the tool was right — `((7 : ℕ) / 2 : ℝ) = 3` (Lean gives 7/2) and
`∀ k ≤ 3, k ≤ n + 2` at `n = 0` — which is the behaviour one wants from an independent reader.

```sh
python3 -B -S -m unittest tests.test_cbz_statement_fuzz                         # 66 tests OK; same with -O -W error
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json main_formal_ec6962ff_r2.json
git -C <Math- clone> archive 19a55750 formal | tar -x -C <tmp>
python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --selftest --json math_formal_19a55750_r2.json
python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --admit-corroborated --seed 2 --samples 256 --json math_formal_19a55750_r2_seed2_s256.json
```

| Package | Cut | Targets | PASS | CORROBORATED | FAIL | UNSUPPORTED | Mutants killed | Exit |
|---|---|---:|---:|---:|---:|---:|---|---:|
| main `formal/` (SIDE24 + P15 skeletons, after #740) | `main` `ec6962ff` (`formal/` tree `e32c6bcb`) | 59 | **59** | 0 | 0 | 0 | 184 / 184, 24 weakenings kept, 15 declared statement controls FAIL | 0 |
| Math- `formal/` (`ResearchFormalCoreR1`) | `main` `19a55750` | 63 | **3** | **29** | 0 | **31** | 45 / 45, 6 weakenings kept (the three all-supported modules `harness_ok`; the five with UNSUPPORTED targets fail the control by design) | 1 |
| Math- `formal/`, seed 2, 256 samples, `--admit-corroborated` | `19a55750` | 63 | 3 | 29 | 0 | 31 | — | 1 (the 31 UNSUPPORTED) |

The ℝ backend changes **no verdict** on the core-Lean packages (59/59 PASS before and after, identical
per target). On the Math- core it moves 32 of 63 targets from UNSUPPORTED to a decided verdict — 3 PASS
(the closed counterexample controls `p02_lm008_sqrt_counterexample`, `p02_lm008_upper_normalizer_counterexample`,
`d2_degenerate_control`, the last of which rests on `x / 0 = 0` through `d2Schur 1 1 (1/4) = 2`) and 29
CORROBORATED (the `ec0xx` polynomial identities, the `p02_lm008`/`p02_lm009` real inequalities including
the six-hypothesis `p02_lm008_cross_multiplied` and `_quotient_bound` with `Real.sqrt`, `weightPerturbation_quotient_bound`
with two absolute-value hypotheses, and the 13 sampled `D2Schur` statements). **Zero FAIL on 29
kernel-proved statements is itself a soundness check of the backend**: a false refutation would have
been a reader defect. Seeds 1, 2, 3 and 64/128/256 samples give the same distribution. The 31 that stay
UNSUPPORTED all carry a measure-theoretic binder — `(μ : Measure Ω)` (27), `(R : Ω → ℝ)` (2),
`(μ : ℝ → Measure Ω)` (2) — and are named as such in the stdout receipt; no sampler reads a measure, and
the tool says so rather than approximating.

## Finite discrete models for measure-theoretic statements (1 November 2026; receipts `*_r3*`)

The 31 statements the ℝ backend left UNSUPPORTED are all over a carrier `variable {Ω : Type*} [MeasurableSpace Ω]`
with binders `(μ : Measure Ω)`, `(W R V : Ω → ℝ)`, `(A : Set Ω)`, instance hypotheses `[IsProbabilityMeasure μ]` /
`[IsFiniteMeasure μ]`, integrals `∫ x, f x ∂μ` and `∫ x in A, …`, event masses `(μ A).toReal`, a.e. relations
`0 ≤ᵐ[μ] W` / `W =ᵐ[μ] V`, set-builders `{x | p x}`, `⊆`, `≪`, and two local definitions
(`weightedLaw μ W := (ENNReal.ofReal (∫ x, W x ∂μ))⁻¹ • μ.withDensity (fun x => ENNReal.ofReal (W x))`,
`badJetEvent R r ε := {x | ε < r * R x ^ 5}`). The tool now samples them on **finite discrete models**:
Ω = Fin n with n ∈ {1, 2, 3, 4} and the discrete σ-algebra, μ a vector of non-negative rational weights
(normalised to total mass 1 when `IsProbabilityMeasure μ` is hypothesised, allowed to be zero otherwise),
functions as tables of exact rationals, sets as subsets, with neighbour sampling (W equal to or one entry
away from an already drawn V) so that perturbation hypotheses such as `∫ |W x - V x| ∂μ ≤ δ` are hit.

Why this is sound: every such model *is* an instance of the Lean statement. The Bochner integral of a
table against a finite measure is the finite sum Σ wᵢ f(i) (every function is integrable); `∫ x in A, …`
is the sum over A; `μ A` is the partial sum; `0 ≤ᵐ[μ] W` holds iff W ≥ 0 at every point of nonzero weight;
`MeasurableSet`, `Measurable`, `Integrable`, `MemLp` hold for everything the model carries (discrete
σ-algebra, finite measure); `IsProbabilityMeasure ν` is total mass 1; `ν ≪ μ` is "every μ-null point is
ν-null"; measures are equal iff their weight vectors are; `ENNReal` follows Mathlib (`0⁻¹ = ∞`, `∞⁻¹ = 0`,
`∞ · 0 = 0`, truncated subtraction, `ENNReal.toReal ∞ = 0`, `ENNReal.ofReal` clips at 0); `c • μ` and
`μ.withDensity f` are pointwise products; a quantifier `∀ A : Set Ω, …` enumerates all 2ⁿ subsets. A
refutation on a finite model is therefore a refutation of the Lean text's reading — and the test
`weightedLaw_apply` without `0 ≤ᵐ[μ] W` shows one: with a signed W, `ofReal` clips on one side only and
`∞ · 0 = 0` makes the two sides differ. Points of infinite mass are not generated (every statement in scope
hypothesises finiteness or integrability that would exclude them), and a conclusion that is only a
structural predicate (`MeasurableSet (badJetEvent R r ε)`) is reported UNSUPPORTED as trivial on discrete
models rather than CORROBORATED. Real-indexed families `(μ : ℝ → Measure Ω)` with `∀ r, 0 < r → r ≤ r0 → …`
hypotheses stay UNSUPPORTED (an infinite quantifier inside a hypothesis). Definitions and the carrier are
shared across a package's modules (`weightedLaw` is defined in `WeightedLaw.lean` and used in
`WeightPerturbation.lean`). The class-1 literal-shift mutant is now typing-aware: `weightedLaw μ W = 0`
compares measures, and `= 1` there is not a Lean term, so it is not generated.

Tests: 66 → **77** (`FiniteModelTests`, 10: Markov with and without its subset hypothesis, integral
Cauchy–Schwarz with exact square roots and its strict misstatement, set integrals and event masses, the
`weightedLaw` identity and its signed-W refutation, probability/zero/one/≪/a.e.-congruence, set-builder
events and the moment tail with and without the moment bound, `∀ A : Set Ω` enumeration, structural and
family UNSUPPORTED, exact seeded model witnesses, the mutation contract). The 58 adversarial probes still
match; main `formal/` is unchanged.

```sh
python3 -B -S -m unittest tests.test_cbz_statement_fuzz                         # 77 tests OK; same with -O -W error
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json main_formal_ec6962ff_r3.json
python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --selftest --json math_formal_19a55750_r3.json
python3 -B -S tools/cbz/statement_fuzz.py --package <tmp>/formal --admit-corroborated --seed 2 --samples 128 --json math_formal_19a55750_r3_seed2_s128.json
```

| Package | Cut | Targets | PASS | CORROBORATED | FAIL | UNSUPPORTED | Mutants killed | Exit |
|---|---|---:|---:|---:|---:|---:|---|---:|
| main `formal/` (SIDE24 + P15) | `ec6962ff` | 59 | **59** | 0 | 0 | 0 | 184 / 184, 24 weakenings kept, 15 declared statement controls FAIL | 0 |
| Math- `formal/` | `19a55750` | 63 | **3** | **56** | 0 | **4** | 72 / 72, 6 weakenings kept; the two modules with UNSUPPORTED targets fail the control by design | 1 |
| Math- `formal/`, seed 2, 128 samples, `--admit-corroborated` | `19a55750` | 63 | 3 | 56 | 0 | 4 | — | 1 (the 4) |

Seeds 1, 2 and 3 give the same distribution, 72/72 kills and zero survivors; every CORROBORATED statement
reached the full 64 accepted instances, and no draw was infeasible (`sampling_totals` in the report and the
CLI summary — a8's #742 nit). Radicands are split by exact factorisation (trial division, deterministic
Miller–Rabin below 3.3·10²⁴, Brent's rho); a root whose radicand lies beyond that range is kept exact but
*non-canonical*: orderings are still decided by enclosures, while an exact zero between such roots cannot be
recognised and is reported Infeasible/UNSUPPORTED rather than guessed (`ExactRealAlgebraTests`). The 4 UNSUPPORTED are named in the stdout receipt:
`p02_lm009_badJet_measurable` (structural conclusion) and the three `moment40_family_r3*` statements
(real-indexed family). **Zero FAIL on 56 kernel-proved statements** remains the soundness check of the
reader; the class-1 kills (every negated conclusion refuted on the models) show the evaluation has content.

**Proposed registration amendment (for Root; the landed GOVERNANCE.md section of #741 governs until
amended):** the coverage row for Math- `formal/` at `19a55750` becomes `63 | 3 PASS | 56 CORROBORATED |
4 UNSUPPORTED`; the sentence "The tool does not cover Mathlib real analysis, binders before the colon or
unbounded quantifiers … the Math- core's 63 statements remain explicitly unsupported" becomes "The tool
samples numeric binders before the colon (ℕ ℤ ℚ ℝ) under decidable hypotheses, evaluates `Real.sqrt`
exactly in ℚ(√d₁, …, √dₖ), and samples measure-theoretic statements on finite discrete models (Ω = Fin n,
rational weights, tables, subsets); a sampled statement is CORROBORATED or FAIL, never PASS. Structural
conclusions, real-indexed families, nested radicals, transcendental functions and unbounded quantifiers
inside a statement remain explicitly unsupported; no registered statement uses `Real.exp`/`log`, so no
interval backend is pending."; the test count becomes 77; and the gate
mapping gains: "For a statement with sampled binders, CORROBORATED at the same source cut as the kernel
receipt is the applicable empirical validation (recorded with seed, sample count and the report's
SHA-256); it is admitted evidence, not an exact check, and is marked as such." The B-007 statute is
unchanged: a `PROOF_INDEX` row needs PASS or, for sampled statements, CORROBORATED — never UNSUPPORTED.

## Proposed gate row and registry entries (for Root; B-003)

**§10 gate matrix, new row:**

| Gate | Command/Check | Pass condition | Blocking |
|---|---|---|---|
| G12 Empirical statement check | `python3 -B -S tools/cbz/statement_fuzz.py --package <formal package> --selftest` for every formal package in scope | every registered target PASS; no UNSUPPORTED; every module's mutation contract `harness_ok` (class 1 all killed, class 2 all kept) and every declared statement control FAIL; exit 0 | ✅ for a target to be CLOSED; a package with UNSUPPORTED targets cannot have them CLOSED |

**`TOOLING.md` entry:** `tools/cbz/statement_fuzz.py` — purpose: independent exact-arithmetic
evaluation of core-Lean arithmetic statements, finite-quantifier enumeration, vacuity/tightness probes;
usage: as above (`--json` writes the full report); self-verification: `tests/test_cbz_statement_fuzz.py`
(35 tests, both modes) and the built-in `--selftest` mutation contract (control, guaranteed-false
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
- CORROBORATED is bounded by the sampler: 64 exact instances (by default) from a pathological pool,
  hypothesis boundaries and seeded random rationals. A statement false only on a thin set the sampler
  does not hit would be corroborated; that is why it is never PASS and why the seed and counts are
  recorded. It still refutes misstatements that are false at 0, ±1, a boundary, or generically.
- Hypotheses are decided by the same interpreter; a hypothesis it cannot read makes the theorem
  UNSUPPORTED rather than letting the conclusion be sampled unconstrained.
