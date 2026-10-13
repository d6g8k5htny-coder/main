# Scope of the proof packet

Scientific effect: **NONE**. Scientific status authority: none. Alignment status of every registered
statement proved here: **PENDING_INDEPENDENT_REVIEW**. Organizational-independence credit: 0.

## In scope

* Theorems whose type is exactly a registered `def … : Prop` of
  `specifications/lean_statements_20261010/REGISTER.json`, applied to the theorem's own parameters,
  with proofs the Lean kernel accepts under `propext`, `Classical.choice` and `Quot.sound` only.
* Helper lemmas and definitions needed by those proofs, `private` or under an `Aux` namespace, never
  counted as inventory and audited for axioms like everything else.
* The gate (`check.py`), its tests, the replay script and the CI workflow that produce a run receipt.

## Out of scope (and refused by the gate where it can see them)

* Editing any statement module or the register. A proof that needs a statement changed, or an extra
  hypothesis to go through, is a **finding** about the statement: report it; do not smuggle the
  premise into the theorem (`theorem foo (h : …) : P` and `theorem foo : P → P` are refused), do not
  narrow the statement (`theorem foo (n : ℕ) : P (n + 1)` is refused).
* Any proof of an informal claim as such. The register's claim-level `formal_progress` stays what the
  register says (`none` or `specified`); no research claim of this programme is `proved`.
* Alignment verdicts (ALIGNED / ALIGNED AT NARROWER SCOPE / MISALIGNED): written only by an
  independent reviewer, never here.
* Transcribed LANDING_CLAIMS dispositions: never re-derived or changed.
* The vault, `legacy/`, `quarantine/` material, and anything under
  `02_LEGACY_Q0_ARCHIVE`: not cited, not used.

## What a kernel-checked theorem here means, per family

| family | the theorems establish | not established |
|---|---|---|
| P15 (`Specifications.P15`, modules P15Elementary, P15Structure, P15Hazard, P15Covers, P15Budget, P15Interpolation) | exact finite arithmetic and finite combinatorics of the registered Props: generator prices, the demand-one refutation data, `ρ* < 6/7` and its certificate, the 816-benchmark counts, palette identities, cover (816 and not 815) and union bound (P11); Proposition P1; both halves of Theorem P2 as the Props state them (the cover criterion (P7) and the hazard bound (P8) for prices `0 ≤ c_v ≤ p_v`) with the local statements (P2)-(P5); the restricted transformed-price budget (T1) with (T4), (T5) and the six-block transformed range; (F9), (F13), the sharpness clause of `ρ*` for the triple block; the hazard interpolation (F5)-(F8) (separate coordinatewise concavity of `t ↦ -log μ_{p(t)}(A)` for a proper downset, the product bound, the `φ` bound, the full-ground price bound and its exactness `iff`), the local bound (F12), Theorem F as `FullTransformedPriceBudget` states it ((F2): for every `p ∈ [0,1]^X`, every `0 ≤ c_v ≤ φ(p_v)` and `K ≥ K_H(d)`, `coverCost ≤ cappedHazard ρ* μ`) and the 20-digit enclosure (F3) of `ρ*`; all for the formal `RealizedData` record, downset, `coverCost` (an `sInf`) and `mu` | the informal P15 claims (P15-PRICE-BOUNDARY, P15-REALIZED-COVERS, FULL-PRICE, PRICE-BUDGET), their dispositions, Theorem F itself: the hazard interpolation (F4)-(F8), the block bound (F12) and the budget (F2), the 20-digit `ρ*` enclosure (F3); attainment of the `sInf`; alignment of `RealizedData`/`IsRealizedFamily`/`downset`/`coverCost`/`mu` with the informal objects (PENDING_INDEPENDENT_REVIEW), any prize statement |
| Side24 (`Specifications.Side24`, modules Side24Elementary, Side24Analysis, Side24Interfaces, Side24Covariance, Side24ConeMoment) | elementary exponential bounds, a term-ratio bound, the D2 cone-moment algebra, the reference-coefficient identity, the elementary cone integral; the summability of the theta family `n ↦ exp(-‖z + 24 n‖²/2)` over `Fin d → ℤ` for every `d` and `z`; the Section 2 derivative bounds for Mathlib's `iteratedFDeriv ℝ q (Kinf d)` at unit tuples and `q ≤ 6` (the pairing bound `exp(-‖x‖²/2) pairingSum q ‖x‖`, `76 ‖x‖⁶ exp(-‖x‖²/2)` at `‖x‖ ≥ 1`, `15` at `0`); the lattice image sum (summable and `≤ 1458 e^{-288}` for `d ≤ 3`); the Section 1 reference blocks in an orthonormal frame (`Cov(G) = I`, `Var(t) = 15`, `Cov(t, G) = (-3, 0, …)`, `pinCov = diagonal (3, 1, …, 1)`, `τ² = 6`) and the pin density product `(2π)^{-d}/√3` for the formal `gaussDensity`; the Section 4 integrand ratio bounds (arithmetic in `epsilon`, `exponentA`, `exponentB`) and the density comparison for positive-definite matrices with `(1 - epsilon) • C0 ≤ C ≤ (1 + epsilon) • C0` in the Loewner order; the reference covariance lower bound: `Cref d (R 0) R - (1/3) • 1` positive semidefinite, its `IsHermitian` half included, for every orthonormal frame `R` and `d = 2` or `3`, as the Prop states it (the proof does not use that hypothesis; only the helper `Aux.cv_psd`, not an inventory theorem, is stated for every `0 < d`); and, for a GIVEN `I : ConeMomentInterface`, `D_1 = 4/3`, the trace moments `5/3`, `25/3`, `√(3/8)` and the Rayleigh law of `x² + y²` as a distribution function, from the interface's law fields alone, and the `m = 2` cone moment `∫ coneIntegrand (s, x, y) ∂I.P = (1/2)(25/3 - 20/3 + 8 - 8 √(3/8))` from the law field `sxy_law` alone (not `A_law`); all as the Props state them, for the formal objects (`E d`, `Kinf`, `latticeVec`, `derivContraction`, `gaussDensity`, `jointCov`, `Cref`, `coneIntegrand`, the interface structures); the reference covariance lower bound and the `m = 2` cone moment (round 5) are first receipted by run 38195810463 (PR head `c6ce0256`, actual tested merge `1320c5b2…` with base `be014794…`; receipt theorem_count 75) | the coefficient enclosure `c_{d,24}` (REVIEWED_SCOPED is transcribed, not derived): nothing here encloses the coefficient, and `CoefficientEnclosure` / `ReferenceComparison` are predicates on `PeriodicCoefficientInterface`, which takes the cone moment as a GIVEN field; `ImageBound`, `CovarianceOrdering`, `EntrywiseCovarianceBound` (no local term-wise iterated differentiation of the lattice sum at Mathlib `d13f23b7`); that `ConeMomentInterface` or `PeriodicCoefficientInterface` has an inhabitant, or that their law fields are the conditional law `A_u \| V_u = 0` of any field (no field is constructed); that any `gradCov`/`pinCov` matrix or `Cref` is the covariance of a random object; symmetry of `Cref d u R` for a general `u` or frame; the odd-block minors and eigenvalues `8 ± sqrt 58` (main's theorems, neither imported nor relied on here); that `D_2` is the parent's cone moment `D_u` of (15.1) for the reference law, and the source's Rayleigh integration route as the source's statements; the source's shell count and closed-form derivative displays, which stay unspecified; alignment of the formal objects with the source's (PENDING_INDEPENDENT_REVIEW) |
| RN (`Specifications.RN`, modules RNProbability, RNCluster) | the elementary probability identities and bounds N1–N5 as the Props state them, for the interface objects they quantify over; Theorem O's obstruction as `RareClusterPoissonObstructionO` states it, for a GIVEN family of pmfs `μ r : ℕ → ℝ` satisfying (H1) `∑ n μ r n ≤ A r³` and (H2) `a r³ ≤ 1 - μ r 0 - μ r 1` on `(0, r0]`: for all small `r`, every Poisson law is at `pmfTV` distance at least `(a/2) r³` from `μ r`, and one (`lam = 0`) is within `A r³`, with `2a ≤ A` derived, not assumed | the Radon–Nikodym uniformity lemma `D3-LEMMA-RN-UNIF` (OPEN), the five validity premises of Theorem D1 v2.2(2) (OPEN), any identification of the interface with the actual field; that (H1) or (H2) hold for the Gaussian window count (the source takes them from [D5] Theorem G_d and [C6] Corollary Theta; neither is revalidated here, and the Prop ties `μ` to no field), the mean-matched statement `lambda = mu`, Theorems C and S, the source's infimum form verbatim (the existential with witness `lam = 0` is what is proved); the 19 remaining RN Props over `PinnedLawInterface` (the D5 window bounds, the dimension lift, Theorem A(2)/A(3), the Palm and Fourier routes, Corollary D, the OPEN witness-collision target), which the interface under-determines (README table) |
| Lifetime, Field | no theorem is inventoried; the round-4 triage (README table) classifies the 31 Props as 24 predicates (vocabulary), 3 interface-blocked (falsified by the constant laws `p ≡ 0` / `p ≡ 1` of `PairLawInterface`) and 4 Lifetime Props blocked on Mathlib developments (density versions, Kac–Rice), not on the interface | everything |

The three lines of each theorem's docstring (`Proves:`, `Method:`, `Does not establish:`) are the
per-theorem version of this table and are required by the gate.

## Labels

* Source mode prints `formalization_status: "proved"`: the label of a committed proof package.
* The run receipt prints `formalization_status: "kernel-checked"`: a label that appears only in a
  receipt, about the inventoried theorems, after a fresh build, `leanchecker` and `#print axioms`.
* `alignment_status: "PENDING_INDEPENDENT_REVIEW"` everywhere.
* `scientific_effect: "NONE"`, `scientific_status_authority: false` everywhere.
* A receipt produced with `--allow-dirty` says `worktree: "UNVERIFIED"` and is not evidence about
  any commit.

## Next step

Independent (not same-provider) alignment review of each registered statement against its Layer 0
anchor, recorded in the review lane; nothing in this packet substitutes for it.
