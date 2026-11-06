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
tests. Nothing in this directory certifies those statements. *(6 November: the module was never pushed; as
agreed at [684/6560240334](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6560240334) the
enclosures were implemented in this tool as a follow-on — see "Transcendental enclosure backend" below — and
Claude017's design is credited there.)*

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
σ-algebra, finite measure) — except on a derived measure with a point of infinite mass (`(ENNReal.ofReal 0)⁻¹ • μ`),
where `IsFiniteMeasure` is decided false and integrability is not assumed (a8's N2); `IsProbabilityMeasure ν` is total mass 1; `ν ≪ μ` is "every μ-null point is
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

Tests: 66 → **78** (`FiniteModelTests`, 11: Markov with and without its subset hypothesis, integral
Cauchy–Schwarz with exact square roots and its strict misstatement, set integrals and event masses, the
`weightedLaw` identity and its signed-W refutation, probability/zero/one/≪/a.e.-congruence, set-builder
events and the moment tail with and without the moment bound, `∀ A : Set Ω` enumeration, structural and
family UNSUPPORTED, exact seeded model witnesses, the mutation contract). The 58 adversarial probes still
match; main `formal/` is unchanged.

```sh
python3 -B -S -m unittest tests.test_cbz_statement_fuzz                         # 78 tests OK; same with -O -W error
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

## Census of every Lean declaration in Math- (4 November 2026; `census_math_dfa9b30a_20261104/`)

The tool gained a census mode, `--lean-dir <dir> [--exclude <substring>]…`: every `theorem`/`lemma` in every
`.lean` file under a directory is a target with no manifest, byte-identical files are evaluated once,
definitions and carriers are shared across the group, and UNSUPPORTED reasons are classified. Run over the
16 Lean groups of Math- at `dfa9b30a` (3,782 declarations; raw evidence snapshots excluded): **7 PASS ·
128 CORROBORATED · 0 FAIL · 3,647 UNSUPPORTED** — see [CENSUS.md](census_math_dfa9b30a_20261104/CENSUS.md)
for the per-group table, the reason classes, the forty binder types the interpreter does not model, the
list of decided statements and the reproduce commands. The first run found two reader defects
(structural predicates on arguments the finite model does not carry were taken as true: `Measurable
(sourceA L)` for a function on ℝ, `Integrable … (gaussianReal 0 1)`), fixed with regression tests before
the receipts were taken. Parser repairs that the corpus forced: attribute-prefixed declarations
(`@[simp] theorem …` on one line), Unicode subscript identifiers (`x₀`), nested projections (`p.2.1`),
multi-token ascription types, and a parser exception now fails one declaration instead of its module.

**Review repairs (4 November; Codex's review of #752, findings 4471516437 · 4471516451 · 4471516464; receipts
`*_r4*` and the census re-taken).** Three defects of the reader, each confirmed with a reproduction before
it was fixed, each covered by a regression test (`NameResolutionTests`): (1) a function returned by a
definition bound its parameters at the codomain type (`def f : ℕ → ℝ := fun n => if n - 1 = 0 then 1 else 0`
made `f 0 = 1` FAIL and accepted `f 0 = 0`) — parameters now take the domain types of the declared arrow
type, an explicit binder type wins, and arrow types split at bracket depth 0 only; (2) a group's
definitions were pooled by short name, first module wins (`A.f`/`B.f` collapsed, verdicts depending on file
order) — definitions are kept by Lean-qualified name, `namespace`/`section`/`end`/`open` are tracked as
scopes, a statement resolves a name from its namespace outward and through its `open`s, in its own module
first and then across the group only when exactly one definition (up to identical text) carries the name,
otherwise UNSUPPORTED "ambiguous definition"; (3) the splitter saw declaration keywords at column 0 only —
indented declarations and `open … in` / `set_option … in` prefixes now start declarations, and a
`theorem`/`lemma` keyword the splitter still cannot separate becomes an UNSUPPORTED target "declaration not
reached by the parser", so a census accounts for every declaration. Measured exposure of the landed
receipts: none — no decided statement depended on a collision, a returned function or an unreached
declaration — and the re-run changed **0 of 3,782** census verdicts and **0** package verdicts
(`main_formal_ec6962ff_r4.json`, `math_formal_19a55750_r4.json` equal r3 per target); 695 census targets now
carry their Lean-qualified names, which is what lets every record of the #695 index bind by exact name
(see [CENSUS.md](census_math_dfa9b30a_20261104/CENSUS.md), "Per-record annotation"). The suite is **85
tests** (normal, `-O`, `-W error`); the module-level `defs` of a parsed file are now keyed by qualified name
(`Demo.fact`), and the test fixtures that append a theorem after a closed namespace write the `open … in`
prefix Lean itself would need.

**Declaration-scope repair (6 November; Codex review finding 4472023769).** A selected definition was
looked up in the caller's namespace correctly, but the reader then evaluated its body, pattern clauses and
returned closures in that caller's namespace and `open` set. Thus `A.f` could read `B.x` when called from
`namespace B`, producing both a false FAIL and a false PASS. Commit `55571918ed8772c5c260aa44855af68add8d0a2d`
records each definition's declaration namespace and opens and restores the caller after evaluation; closures
carry the same scope. The regression covers ordinary bodies, returned functions, pattern clauses and
`open Shared in def` with both true and false assertions. The suite is now **86 tests**, passing under normal,
`-O`, and `-W error` modes. The exact Math- `dfa9b30a27ea8f8ee647bb067549fa53c7d06ef6`
census was rerun from a native archive: after normalising only the environment-specific `lean_dir` field,
all 16 JSON reports match the stored receipts exactly, so **0 of 3,782 verdicts changed**. The additive
machine-readable receipt is `declaration_scope_repair_20261106.json`; no historical receipt was rebound.

**Finite-vector backend (6 November; unit `CBZ0-G12-FINVEC-01N-20261106`; receipts `*_r5*` and
[`census_math_dfa9b30a_20261106/`](census_math_dfa9b30a_20261106/CENSUS.md)).** The census's largest reachable
UNSUPPORTED class was binders over finite vectors: `(j : Fin 6 → ℝ)`, `(n : Fin d → ℤ)`, `(i : Fin 6)`,
`(S : Matrix (Fin 2) (Fin 2) ℝ)`, `EuclideanSpace ℝ (Fin n)` and abbreviations of them (`abbrev DM2Jet := Fin 9 → ℝ`) —
428 declarations named one as their first unsupported binder. They are now sampled exactly: structured
vectors (zero, ones, unit vectors, the pool along the diagonal, alternating signs, constants), components at
the boundaries the hypotheses give them (`v 0 ∈ Set.Ioo a b`), random pool / rational entries; indices
0 … n-1; a ℕ binder that is a later vector's dimension from {0, 1, 2, 3, 4} (a `Fin 0` sample is rejected,
never counted). The interpreter reads `![…]` and `!![…]` literals, `∑` / `∏` over a finite index set (the
binder type, a `Finset.range k` / `Finset.univ` domain, or the vectors the index applies to — otherwise
UNSUPPORTED; Mathlib's `r:67` binding, so `∑ i, f i ≤ 1` is `(∑ i, f i) ≤ 1`), `.det` (Laplace expansion,
≤ 6×6), `.trace`, `‖·‖` (the sup norm on the Pi type, the Euclidean norm on `EuclideanSpace`, exact in ℚ(√d)),
`Pi.single` / `EuclideanSpace.single` (the dimension from the context or from the sum's index), `dotProduct` /
`⬝ᵥ`, `•`, pointwise `+ - * /`, the Pi order `v ≤ w`, `v ≠ 0`, `↑i` / `(i : ℕ)`, `fun i => …` where a vector is
expected, definitions with vector / matrix results applied to an index, and implicit dimension parameters
(`def gm8Radius {n : ℕ} (j : Fin n → ℝ)`, `{n}` read off the argument's length). Prop-valued definitions
(`def dm2Negative (H) : Prop := ∀ v, …`) are inlined, so their quantifier over ℝ² is named as the limit.
Refused, fail-closed: arithmetic in `Fin n` (it wraps: `(2 : Fin 3) + 1 = 0`) and a numeral at `Fin n` not
below n; products of vectors; matrix norms; orders on `EuclideanSpace`; `*ᵥ`, `Matrix.of`, `∑'` (tsum).

Soundness run: the 16-group census at `dfa9b30a` re-taken — **17 PASS · 203 CORROBORATED · 0 FAIL ·
3,562 UNSUPPORTED**; 85 verdicts moved, all from UNSUPPORTED (75 → CORROBORATED, 10 → PASS, among them two
exact Taylor-sum identities over `Finset.range` with `Nat.factorial`), **none to FAIL on these kernel-proved
statements**, and no 4 November verdict changed
(`verdict_changes_20261104_to_20261106.json`). Both package receipts (`main_formal_ec6962ff_r5.json`,
`math_formal_19a55750_r5.json`) equal r4 per target; the 58 adversarial probes of #742 still match. The
#695 index annotation re-bound to these receipts reads **34 PASS · 73 CORROBORATED · 260 not applicable**
(the determinant family 13 → 16 CORROBORATED, `smooth` 0 → 1). Tests 86 → **93** (`FiniteVectorTests`:
types and aliases, the determinant-package statements with their misstatements refuted at printed
witnesses — `j = ![0, 0, 0, 0, 0, 0], i = 0`, `n = 1, p = ![1]`, `v = ![0, 0]` — big-operator binding,
norms, literals, `Pi.single`, implicit dimensions, Fin wrapping, the mutation contract on vector statements),
normal, `-O`, `-W error`. One landed expectation changes: `∑ i ∈ Finset.range 3, (i : ℝ) = 3` is now PASS
(the package test keeps its UNSUPPORTED anchor on `Real.exp 0 = 1`), and `(v : Fin 3 → ℝ)` is no longer the
example of an unsampled binder (`Fin 3 → ℝ≥0` is).

**Proposed registration amendment (for Root; the landed GOVERNANCE.md section of #741 governs until
amended):** the coverage row for Math- `formal/` at `19a55750` becomes `63 | 3 PASS | 56 CORROBORATED |
4 UNSUPPORTED`; the sentence "The tool does not cover Mathlib real analysis, binders before the colon or
unbounded quantifiers … the Math- core's 63 statements remain explicitly unsupported" becomes "The tool
samples numeric binders before the colon (ℕ ℤ ℚ ℝ) under decidable hypotheses, evaluates `Real.sqrt`
exactly in ℚ(√d₁, …, √dₖ), and samples measure-theoretic statements on finite discrete models (Ω = Fin n,
rational weights, tables, subsets); a sampled statement is CORROBORATED or FAIL, never PASS. Structural
conclusions, real-indexed families, nested radicals, transcendental functions and unbounded quantifiers
inside a statement remain explicitly unsupported; no registered statement uses `Real.exp`/`log`, so no
interval backend is pending."; the test count becomes 85; and the gate
mapping gains: "For a statement with sampled binders, CORROBORATED at the same source cut as the kernel
receipt is the applicable empirical validation (recorded with seed, sample count and the report's
SHA-256); it is admitted evidence, not an exact check, and is marked as such." The B-007 statute
becomes: a `PROOF_INDEX` row carries `G12: PASS | CORROBORATED | not applicable (<construct>)`, taken from the
registered run or the census at the row's source cut; PASS or CORROBORATED is the applicable empirical
validation, *not applicable* names the construct the interpreter does not model (a structure, a function
on ℝ, an existential, a family) and is never written for a statement the interpreter reaches, and
UNSUPPORTED is never upgraded or left silent. The census shows the CLOSED-track Gaussian-field packages
have no reachable declaration at all; their rows read *not applicable* throughout, and the index should
say so rather than let the addendum rule be read as satisfied.

**Transcendental enclosure backend (6 November; unit `CBZ0-G12-TRANS-01N-20261106`; receipts `*_r6*` and
[`census_math_dfa9b30a_20261106_trans/`](census_math_dfa9b30a_20261106_trans/CENSUS.md)).** The registration of
#741/#756 named `Real.exp` / `Real.log` as the one gap that touched registered-track statements: 49 census
declarations were UNSUPPORTED for a transcendental function, 39 of them in the P15 proofs (`2 < Real.exp 1`,
`197/32 < Real.exp (11/6)`, `Real.log (3 * Real.exp 1 - 2) < 2`, the 29- and 32-digit enclosures of `e` and
`exp x_U` / `exp x_L` in `P15Interpolation`). Claude017 offered an `interval.py` for exactly this on 30 October
([684/6532742483](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6532742483): outward-rounded exact
rational intervals, Mathlib's totalised conventions, `Undecided` instead of a guess); the offer was accepted, the
module was never pushed, and the agreed follow-on ([684/6560240334](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6560240334))
is this backend, which follows that design and adds the exact exponential normal form. They are now read, with no
float anywhere:

- **Exact normal form.** `Real.exp q` for rational q is the monomial e^q; `+ - *`, integer powers, `Real.sinh` /
  `Real.cosh` of rationals and the inverse of a monomial stay in the form Σ c·e^q. Equality of two such values is
  equality of normal forms, because the e^q for distinct rational q are linearly independent over ℚ
  (Lindemann–Weierstrass); so `Real.exp 2 = Real.exp 1 ^ 2`, `Real.exp 1 * Real.exp (-1) = 1`,
  `Real.cosh 1 ^ 2 - Real.sinh 1 ^ 2 = 1` are PASS and `Real.exp 1 = Real.exp 2` is FAIL with no enclosure at
  all. `Real.log (c · e^q)` is `log |c| + q` through the monomial (`Real.log (Real.exp 5) = 5` is PASS exactly).
- **Rigorous enclosures.** Everything else is a closed interval with exact rational endpoints: e from Σ 1/k! with
  the tail bound 1/(N!·N); e^r (|r| ≤ 1/2) from the Taylor series with the Lagrange remainder
  2·|r|^{N+1}/(N+1)!, and e^n by outward-rounded binary powers of the enclosure of e; log from
  q = 2^k·m, log m = 2·atanh((m−1)/(m+1)) with the geometric tail bound z^{2N+3}/((2N+3)(1−z²)); π by Machin's
  formula from the alternating arctan series (the first omitted term bounds the remainder; arguments above 1/2
  are split by the addition formula); sin and cos after reduction modulo 2π — π itself an interval — from their
  Taylor series with the remainder |y|^N/N! (every derivative is bounded by 1), widened to ±1 where a critical
  point may lie in the interval; `Real.sqrt` of an enclosure, and of an irrational algebraic value (a nested
  radical, previously refused), by integer square roots. Every series term and every intermediate is rounded
  **outward** to a bounded number of significant bits (`_round_out`, `Iv.round`, `Iv.pow_rounded`): containment is
  kept, only the width grows, and the endpoints stay short at 1024 bits and for `Real.exp` arguments up to 10⁶ in
  absolute value (e^{10⁶} ≈ 2^{1.44·10⁶} is within the result caps; beyond, Infeasible). Mathlib's totalisations
  are the ones read: `Real.log 0 = 0`, `Real.log (-x) = Real.log x`, `x / 0 = 0`, `Real.sqrt x = 0` for x ≤ 0.
- **Undecided is a verdict class, never a guess.** A comparison the enclosures do not separate raises *Undecided*.
  A closed statement is refined 64 → 128 → 256 → 512 → 1024 bits (the PASS detail records the precision that
  decided it) and is UNSUPPORTED past the cap, the reason naming the operator, the precision and the two
  enclosures. A sampled instance is retried at 256 bits, then skipped and counted (`sampling.undecided`); a
  statement whose undecided instances outnumber the decided ones is UNSUPPORTED, not weakly corroborated. An
  **equality** between two values that are not both exponential polynomials can be refuted by enclosures but
  never confirmed, so `Real.cos Real.pi = -1`, `Real.sin 1 ^ 2 + Real.cos 1 ^ 2 = 1`,
  `Real.arctan 1 = Real.pi / 4` and `Real.log (-3) = Real.log 3` are UNSUPPORTED with that reason, while
  `Real.cos Real.pi = -1 ∨ Real.pi > 3` is PASS: a side of `∧` / `∨` / `→` that is undecided does not decide the
  connective by itself, the other side does when it can (Lean's short-circuit from the left is kept, and an
  *unsupported* construct is never settled away). `min` / `max` are exact when the order is decided and the hull
  of the pointwise extrema when it is not. Outside, fail-closed and named: `Real.rpow`, `Real.arcsin` /
  `arccos` / `Gamma` and every other function the backend does not model.

Soundness evidence. (1) An independent cross-check in development: 1,668 enclosures (`e`, π, log 2, `exp` and
`log` at random rationals, sin / cos / atan at random points and over random intervals, `exp` / `log` / `arctan`
over intervals, exponential polynomials) were compared against mpmath at 320 digits — every one contains its
value and is narrower than 2^{-bits}. This cross-check **found one defect before any receipt was taken**: the
first atanh tail bound was z^{2N+5}/… instead of z^{2N+3}/… (too small by the factor z², so an interval could
miss log q by less than 2^{-bits-1}); it was corrected, and the unit tests now check `_exp_point` / `_log_point`
/ `_pi_enclosure` containment against the standard library's `decimal` at 80 digits and the trigonometric and
arctan series against `math`, plus the outward-rounding invariants (`lo ≤ x ≤ hi`, bounded mantissas) on 200
random rationals. (2) The 16-group census at `dfa9b30a` re-taken — **40 PASS · 233 CORROBORATED · 0 FAIL ·
3,509 UNSUPPORTED**; 53 verdicts moved, all from UNSUPPORTED (23 → PASS, 30 → CORROBORATED), **none to
FAIL on these kernel-proved statements**, and no earlier verdict changed
(`verdict_changes_20261106_finvec_to_trans.json`). Among the gains: every P15 elementary and interpolation
statement (`two_lt_exp_one`, `exp_two_eq`, `exp_neg_three_eq`, `log_three_exp_sub_two_lt_two`,
`exp_eleven_sixths_gt`, `three_exp_sub_two_lt`, `f3_exp_one_gt` / `f3_exp_one_lt` / `f3_exp_xU_ge` /
`f3_exp_xL_le` at 128 bits, `f3_log_lt`, `f3_lt_log`, `f3_hStar_pos`, `log_benchmark_le`), the Side24 elementary
bounds (`ten_lt_exp`, `pow_ten_lt_exp`, `exp_neg_lt`, `big_lt_exp`, `ratio_const_lt_half`), sampled hazard and
cluster inequalities (`neg_log_one_sub_ge`, `cc_affineExp_pos`, `f6_one_sub_exp_neg_mem`,
`rc_half_le_one_sub_exp`, `rc_poisson_tail_two_le_sq`, `li_geom_tail`, `exp_succ_le`, `ib_poly_gauss_le`), and
in the Gaussian-source packages `fixture_scale_ne` (`√2/π ≠ 0`), `sourceC_ne_zero` (`2π/L ≠ 0`), `sourceA_pos`,
`gaussianCompactLowerConstant_pos`, `exp_mul_abs_le`, and the pin identities `pinMultiplier_value` /
`annihilator_pin_values` (`(1 − cos(c·0))² = 0`, exact because cos 0 is the exact 1). The controls refute at
printed witnesses: `Real.exp 1 < 2`, `Real.exp 1 < 271828182845904523536028746620/10^29` (FAIL at 128 bits),
`Real.exp (11/6) < 197/32`, `Real.exp 2 ≠ Real.exp 1 ^ 2`, `Real.pi < 3`, `Real.exp x > 1 + x` at `x = 0`,
`Real.exp x ≤ 1 + x` at `x = 1`. Both package receipts (`main_formal_ec6962ff_r6.json` 59/59 PASS with every
mutant killed, `math_formal_19a55750_r6.json` 3·56·0·4) equal r5 per target and per self-test; of the 58
adversarial probes of #742, 55 match and the three that differ are the intended extensions
(`Real.sqrt (Real.sqrt 2) < 2` PASS, `Real.exp 0 = 1` PASS, `(hx : Real.exp x > 0)` now a readable hypothesis).
The #695 index annotation re-bound to these receipts reads **34 PASS · 75 CORROBORATED · 258 not applicable**.
**Review round (6 November; 01SHCrxv 757/6679600965 PASS, 01DKKe 757/6679615263 PASS; 01SHCrxv 755/6678701025 V1).**
Two same-family soundness reads, each with its own containment harness against mpmath (1,176 and 29-probe runs), found
no defect; their non-blocking items are applied on this branch: the sine critical-point interval for negative k is the
exact `[π_hi·(½+k), π_lo·(½+k)]` (O1); a rational `ExpPoly` hashes as its rational value (N1); a reversed enclosure
raises `Unsupported('reversed enclosure: evaluator defect')` instead of being silently swapped (N2); and the
registration text names Lindemann–Weierstrass as the imported theorem (O3). V1 of the #755 read — a `Fin` index
compared with a ℕ *term* — is confirmed from the pinned sources: at Mathlib `d13f23b7` the ring structure on `Fin n` is
a scoped instance precisely so that `x < n` elaborates as `↑x < n` (`Mathlib/Data/ZMod/Defs.lean`), and the tool now
refuses that comparison under `open Fin.NatCast` / `Fin.CommRing` / `Fin.IntCast`, where the coercion loop exists and
the comparison wraps (zero occurrences in the census corpus). In the same round, `exp`, `log`, `sqrt`, `pi` …
written bare under `open Real` are read as Mathlib's `Real.exp` … (a definition or binder of that name in scope wins, as in
Lean; and when another opened namespace carries the same short name — `open Nat Real` makes `sqrt` an overloaded identifier
that Lean resolves by the expected type, `(sqrt 10 : ℕ)` being `Nat.sqrt 10` — the bare name is refused as ambiguous, the
false PASS 01DKKe found in the readback (757/6680709277 R1) and its second layer, an intrinsically real value ascribed to
ℕ/ℤ, now both UNSUPPORTED); the census receipts in `census_math_dfa9b30a_20261106_trans/` are those of this head — four verdicts moved against the
first run of the day (`productDecay_eq_exp` in three groups and `shifted_gaussian_le`, UNSUPPORTED "unknown function exp" →
CORROBORATED), six more bare-`exp` identities are now named as undecided-by-enclosure rather than unknown, nothing else
changed. Tests 93 → **106** (`TranscendentalTests`: the thirteen P15 statements and their flips; the exact normal form;
Mathlib's totalisations; π and trigonometry; true equalities between enclosed values reported UNSUPPORTED with the
reason; the refinement ladder and the caps; sampled binders with transcendental values, including the
undecided-majority rule; connectives with an undecided side; the mutation contract on transcendental statements —
0 survivors; containment against `decimal` and `math`; outward rounding and message formatting; bare names under `open Real` with the rival-namespace refusal), normal, `-O`,
`-W error`. Landed expectations that change: `Real.exp 1 < 3` and `Real.exp 0 = 1` are PASS (the fail-closed
anchors are now `Real.arcsin 1 < 3`, `Real.exp 1 ^ (1/2 : ℝ) < 2` (rpow) and `Real.cos Real.pi = -1`), a nested
radical is enclosed rather than refused, and `(hx : Real.exp x > 0)` is a readable hypothesis
(`Real.arcsin x > 0` is the unreadable example).

**Registration consequence (for Root; proposed text, the landed §"Empirical statement check" governs until
amended):** the sentence of the amended section "The exp/log gap is narrowed, not withdrawn: 49 census
declarations … remain UNSUPPORTED for a transcendental function" becomes "`Real.exp` of rationals is exact: values of
`Real.exp` at rationals and their sums, products and integer powers are kept in the exponential normal form Σ c·e^q,
whose equalities are decided by the linear independence of the e^q over ℚ for distinct rational q
(Lindemann–Weierstrass — an imported theorem, not re-proved here); `Real.log` / `Real.pi` / trigonometric values and
`Real.exp` of irrational arguments are enclosed by rigorous rational intervals refined to 1024 bits; a comparison the
enclosures do not decide — in particular an equality between two enclosed values — is UNSUPPORTED, never PASS"; the coverage row for the census becomes
`3,782 | 40 PASS | 233 CORROBORATED | 0 FAIL | 3,509 UNSUPPORTED`; the test count becomes 106.

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
