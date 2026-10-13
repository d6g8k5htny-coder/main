Ordered by how much of each proof already exists in the pinned Mathlib tree (`d13f23b7`; every identifier below
was found by `grep` in `.lake/packages/mathlib/Mathlib/`, paths relative to that directory). The procedural frame
is the same for every step: the proof is written in Math- `formal/` under the lane's admission rules (the statement
packet's checker refuses `theorem`); the register then lists the theorem beneath its claim with the label `proved`;
the claim's own label moves to `proved` only if the theorem proves the claim's display itself, not a sub-identity;
`kernel-checked` is written only by a trusted run receipt; and an independent alignment review is a separate
dimension that none of this produces. No step below changes a Layer 0 status.

**Step 1 — `ρ* < 6/7`, then the 20-digit enclosure of `ρ*`** (`P15.RhoStarBelowSixSevenths`,
`P15.SixSeventhsCertificate`, `P15.RhoStarEnclosure`; the next full-theorem target named in
[docs/RESEARCH_INDEX.md](RESEARCH_INDEX.md)). `rhoStar = 1 / (3 - Real.log (3 * Real.exp 1 - 2))`
(`Specifications/P15.lean:806`), so `ρ* < 6/7 ⇔ h* > 7/6 ⇔ log(3e − 2) < 11/6 ⇔ 3e − 2 < exp(11/6)`: one `Real.log`
and two `Real.exp` values. *Mathlib has*: `Real.exp_one_lt_d9` (Analysis/Complex/ExponentialBounds.lean:38), which
already gives `e < 31967/11760`; `Real.sum_le_exp_of_nonneg` (Analysis/Complex/Exponential.lean:248) for
`197/32 < Σ_{j<6} (11/6)^j/j! ≤ exp(11/6)` (the rational identity `Σ − 197/32 = 26081/933120` is a `decide`
theorem on `main`, `exp_eleven_sixths_taylor_certificate`); `Real.log_lt_log` (Analysis/SpecialFunctions/Log/Basic.lean:155),
`Real.log_exp` :75, `Real.exp_lt_exp` (Exponential.lean:312); `norm_num` for the rationals. *Mathlib lacks*: an
evaluator for `Real.exp` or `Real.log`, so the 20-digit display (F3) must be assembled by hand from the Taylor bounds
`Real.abs_log_sub_add_sum_range_le` (Log/Deriv.lean:217, `|Σ_{i<n} x^(i+1)/(i+1) + log(1 − x)| ≤ |x|^(n+1)/(1 − |x|)`)
and `Real.exp_bound` / `exp_approx_start` / `exp_1_approx_succ_eq` (Exponential.lean:516/597/588), with exact
rationals of about 25 digits; the packaged decimal lemmas `exp_one_near_10` and `log_two_near_10`
(ExponentialBounds.lean:23/71) stop at 10 digits. Effort: small for `6/7`, moderate arithmetic for (F3); the
mathematics is elementary. Outcome if done: `P15.RhoStarBelowSixSevenths` becomes a `proved` statement; the landing
claim `p15-full-price` stays `specified` because Theorem F itself ((F2)) is not touched.

**Step 2 — the finite P15 theorems** (`P15.TriangleNeedsThree`, `ElementaryPaletteFormula`, `TwoCoordGeneratorPrices`,
`LocalRestriction`, `LocalDecomposability`, the `Benchmark816*` family, `MinimalForbiddenCharacterisation`,
`FullBlockCoverIff`, `CoverCostHazardBound`, and the whole of `DemandOneRefutation`). *Fully finite combinatorics*
over `Fintype X`: `Finset.card` (Data/Finset/Card.lean:47; `card_le_card` :66), `Finset.powerset`
(Data/Finset/Powerset.lean:32), `Nat.choose` (Data/Nat/Choose/Basic.lean:50), `Nat.ceil` (`⌈·⌉₊`,
Algebra/Order/Floor/Defs.lean:148, Algebra/Order/Floor/Semiring.lean:178), finite sums and products with the order
lemmas of Algebra/Order/BigOperators/Group/Finset.lean. Statements over the two-coordinate ground set `Fin 2` are
decidable by `decide`; the 816-label benchmark is over `Fin 816`, where enumeration is out of reach and the counting
arguments of the source (P9–P11) must be reproduced with `Finset.card` lemmas. `paletteNumber` and `coverCost` are
`sInf` over ℕ and ℝ (`P15.lean:309`, `:248`), so the lattice lemmas of Data/Nat/Lattice.lean and
Order/ConditionallyCompleteLattice/Basic.lean are needed; confirm each name by `grep` before use. *Plus one log*:
`DemandOneRefutation` needs `2/3 < log 2` (`Real.log_two_gt_d9`, ExponentialBounds.lean:83) and `log(4/3) < 1/3`
(`Real.log_lt_sub_one_of_pos`, Log/Basic.lean:260, at `x = 4/3`); `LogOnePlusBelowIdentity` is the same lemma at
`1 + x`; `LogRatioBound` is `Real.log_le_sub_one_of_pos` (:307) at `1/(1 − p)` with `Real.log_inv` (:143);
`RatioExponentBound` is `pow` monotonicity and `norm_num`. *Mathlib lacks*: nothing essential. Outcome if done:
`p15-price-boundary` (disposition `EXACT_COUNTEREXAMPLE`, transcribed) is the first claim whose own display
(`DemandOneRefutation`) could carry `proved`; the register's next-step column already names it as the smallest
complete research statement in the packet.

**Step 3 — the abstract probability parts of Theorem N** (`RN.HolderCountBoundN1`, `HolderConsequenceN2`,
`ConditionalMeanIdentityN3`, `ExponentialTailLayerCakeN5`, `UniformCounterexampleN4`). *Mathlib has*: Hölder for the
Bochner integral, `MeasureTheory.integral_mul_le_Lp_mul_Lq_of_nonneg` (MeasureTheory/Integral/Bochner/Basic.lean:1225)
with `Real.HolderConjugate` (Basic/Real/ConjExponents.lean:59), and `integral_mul_norm_le_Lp_mul_Lq` :1177, plus
`ENNReal.lintegral_mul_le_Lp_mul_Lq` (MeasureTheory/Integral/MeanInequalities.lean:202); the layer cake
`MeasureTheory.lintegral_eq_lintegral_meas_lt` (MeasureTheory/Integral/Layercake.lean:496),
`lintegral_comp_eq_lintegral_meas_le_mul` :394 and the Bochner form `Integrable.integral_eq_integral_meas_lt` :518;
the conditional measure `ProbabilityTheory.cond` (Probability/ConditionalProbability.lean:76; `cond_apply` :216);
`integral_indicator` (MeasureTheory/Integral/Bochner/Set.lean:172) and `setIntegral_const` :526;
`Real.volume_Icc` (MeasureTheory/Measure/Lebesgue/Basic.lean:86) for `uniformLaw`; the fundamental theorem of
calculus `intervalIntegral.integral_eq_sub_of_hasDerivAt` (MeasureTheory/Integral/IntervalIntegral/FundThmCalculus.lean:1148)
for `∫₀^∞ min(q, A e^{−t/B}) dt`; `Nat.ceil_lt_add_one` (Algebra/Order/Floor/Semiring.lean:357) and
`Real.tendsto_log_atTop` (Log/Basic.lean:351) for (N4); `Real.rpow_le_rpow` (Analysis/SpecialFunctions/Pow/Real.lean:550)
for the exponents of (N2); Markov and Chernoff if wanted, `mul_meas_ge_le_lintegral₀`
(MeasureTheory/Integral/Lebesgue/Markov.lean:52) and `measure_ge_le_exp_mul_mgf` (Probability/Moments/Basic.lean:429).
*Mathlib lacks*: nothing structural; the work is measure-theoretic bookkeeping (indicators, `toReal`, integrability
side conditions). Outcome if done: five `proved` statements, and `rn-count-interface` still at `none`, because
(N6)–(N7) are not stated; stating (N7) with the Kac–Rice numerator as an interface field would make the claim
`specified`, nothing more.

**Step 4 — from the `cRef` definition to an enclosure** (`Side24.cRef`, `Specifications/Side24.lean:96`;
`ConeMomentD2Algebra`, `ElementaryConeIntegral`, `RayleighSquareLaw`, `TraceMoments`, `ConeMomentD1Identity` /
`ConeMomentD2Identity`, `ReferenceCoefficientIdentity`, then `CoefficientEnclosure` and `ReferenceComparison`).
*Mathlib has*: `Real.Gamma` (Analysis/SpecialFunctions/Gamma/Basic.lean:402) with `Gamma_eq_integral` (immediately
after), `Gamma_add_one` :410, `Gamma_one` :416, `Gamma_nat_eq_factorial` :422, `Gamma_pos_of_pos` :443,
`Real.Gamma_one_half_eq` (Analysis/SpecialFunctions/Gaussian/GaussianIntegral.lean:307), `convexOn_log_Gamma`
(Gamma/BohrMollerup.lean:111), `Gamma_two` :325, `Gamma_strictMonoOn_Ici` :353, the reflection formula
`Gamma_mul_Gamma_one_sub` (Gamma/Beta.lean:491); `integral_gaussian` (GaussianIntegral.lean:211) and
`integral_gaussian_Ioi` :290; the Gamma and exponential densities `gammaPDFReal` (Probability/Distributions/Gamma.lean:45),
`exponentialPDFReal` (Probability/Distributions/Exponential.lean:41); `multivariateGaussian`
(Probability/Distributions/Gaussian/Multivariate.lean:166) for the reference law of the 24-jet; `condDistrib`
(Probability/Kernel/CondDistrib.lean:64) in the abstract; `Real.pi_gt_d20` / `pi_lt_d20` (Analysis/Real/Pi/Bounds.lean:190/206);
`Real.sqrt` ordering (Analysis/Real/Sqrt.lean:209/221) and a `norm_num` extension for it (Tactic/NormNum/RealSqrt.lean);
`Real.rpow` (Pow/Real.lean:35, `rpow_natCast` :62); one-variable Jacobi theta summability
`summable_jacobiTheta₂_term_iff` (NumberTheory/ModularForms/JacobiTheta/TwoVariable.lean:123). *Mathlib lacks*:
any numerical bound on `Gamma (7/6)` (log-convexity between `Gamma 1 = Gamma 2 = 1` gives `Gamma (7/6) ≤ 1` and a
crude floor, nothing at the 10^-106 relative precision that `ENCLOSURE.json` records); a Stirling, Lanczos or
Taylor remainder for `Gamma` at real arguments with an explicit error term (Analysis/SpecialFunctions/Stirling.lean
treats `n!` asymptotics); the Gaussian conditional-law formula (Schur complement) needed for `A_u | V_u = 0`; the law
of a sum of squares of independent Gaussians as a theorem; `d`-dimensional lattice theta sums over `Fin d → ℤ`
(`Side24.thetaSum`); and, above all, a proof-producing interval arithmetic over ℝ — Tactic/ offers `norm_num`,
`positivity`, `linarith`, `polyrith`, `bound` and `interval_cases` (ℕ/ℤ only), none of which evaluates `Real.exp`,
`Real.log` or `Real.Gamma`. A proof-producing interval library (dyadic or rational endpoints with proofs, as the Python
`coefficient.py` does without proofs) is a project of its own; until it exists `side24-coefficient` stays
`specified` over `PeriodicCoefficientInterface`, and `ENCLOSURE.json` stays L2 evidence. Intermediate outcomes are
real: `ConeMomentD2Algebra` is `Real.sqrt` algebra and could be the first `proved` statement of this module;
`ElementaryConeIntegral` is one application of the fundamental theorem of calculus.

**Step 5 — the analytic parents** (`Lifetime.ParentUniformSelectionA`, `ParentCompactWindowDensityB`,
`ParentUnrestrictedLeadingDensityC`, `RemainderBoundedR1`, `PlanarElderLowerE2` / `E4` / `E11`; every `RN` Prop over
`PinnedLawInterface`). What a proof would need, in order: (i) a smooth `L`-periodic Gaussian field with covariance
`K_L` as a probability measure on sample paths — Mathlib has `multivariateGaussian` for finite-dimensional
marginals, `IsGaussianProcess` (Probability/Distributions/Gaussian/IsGaussianProcess/Def.lean:30),
`IsProjectiveMeasureFamily` and `IsProjectiveLimit` (MeasureTheory/Constructions/Projective.lean:45/115) and the
Kolmogorov condition `IsKolmogorovProcess` (Probability/Process/Kolmogorov.lean), but its own Brownian-motion file
says the Kolmogorov extension theorem is "not in Mathlib yet" (header of
Probability/BrownianMotion/GaussianProjectiveFamily.lean) and no continuous- or smooth-modification theorem is
found by `grep`; so no inhabitant of `GaussianFieldInterface` can be built from Mathlib alone yet (a finite
trigonometric-sum model would be an inhabitant, but not the source's field); (ii) Kac–Rice: Mathlib has no area or
coarea formula and no critical-point counting formula; its change of variables requires `InjOn f s`
(MeasureTheory/Function/Jacobian.lean:1183/1213); (iii) Morse theory: `IsLocalMax.fderiv_eq_zero`
(Analysis/Calculus/LocalExtr/Basic.lean:191) exists, the Morse lemma and nondegenerate critical-point structure do
not (no grep match for "Morse lemma"); (iv) persistence and the elder rule: nothing; (v) the regression law and Palm
conditioning: `condDistrib` abstractly, no Gaussian regression formulas; (vi) asymptotic notation is available
(`Asymptotics.IsEquivalent`, Analysis/Asymptotics/AsymptoticEquivalent.lean:76). Realistic formal work at this
level is therefore not a proof of any of these Props but (a) inhabitation checks of the interfaces (the Dirac-mass
remark in the `PinnedLawInterface` docstring; a trigonometric Gaussian model for `GaussianFieldInterface`), which
guard against vacuity and are not theorems about the research claims, and (b) the transcriptions the register's
next-step column lists (displays (15.2), (14.1), (11.3), (R19), (N7), Section 6 of Theorem A, the R1–R7
interfaces, the dimension-lift corollaries), each of which raises a `none` row to `specified` and nothing to
`proved`. The claims stay `specified`.

Independent of all five steps, and available now: the alignment review of the 150 declarations against their
anchors by a reviewer of a different provider, family and agent (§8). It is the only action that changes the
alignment dimension, and it is not a step this programme can take.