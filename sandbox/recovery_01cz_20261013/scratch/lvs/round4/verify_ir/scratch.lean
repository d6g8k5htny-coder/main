import Lean
import Specifications.Side24

/-!
# SIDE24 analysis proofs: the Section 4 integrand-ratio bounds (`IntegrandRatioBounds`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` (prefix `ir_`). The route is
pure real arithmetic in `epsilon = 10^(-108)`, `exponentA d` and `exponentB d`; no derivative
or density input enters. For `d = 2` the exponents are `a = 13/6`, `b = 5/2`; for `d = 3` they
are `a = 25/6`, `b = 9/2`; in both cases `1 ≤ a, b ≤ 5`. For any such `a, b`: Bernoulli's
inequality (`one_add_mul_self_le_rpow_one_add` at `s = -epsilon`) gives
`1 - a epsilon ≤ (1 - epsilon)^a` and `1 - b epsilon ≤ (1 - epsilon)^b`; monotonicity of the
real power in the exponent (`Real.rpow_le_rpow_of_exponent_le`, base `1 + epsilon ≥ 1`) gives
`(1 + epsilon)^a, (1 + epsilon)^b ≤ (1 + epsilon)^5`; the two displayed quotients then reduce,
through `le_div_iff₀` and `div_le_iff₀`, to the exact rational inequalities
`(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon` and `(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)`,
which `norm_num` decides after unfolding `epsilon`.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24

namespace Aux

/-! ### (a) Elementary facts about `epsilon = 10^(-108)` -/

/-- `0 < epsilon`. -/
theorem ir_epsilon_pos : (0 : ℝ) < epsilon := by
  unfold epsilon
  positivity

/-- `epsilon < 1/100`: a crude upper bound sufficient for every sign condition below. -/
theorem ir_epsilon_lt : epsilon < (1 / 100 : ℝ) := by
  unfold epsilon
  norm_num

/-- `0 < 1 - epsilon`. -/
theorem ir_one_sub_epsilon_pos : (0 : ℝ) < 1 - epsilon := by
  have := ir_epsilon_lt
  linarith

/-- `1 ≤ 1 + epsilon`. -/
theorem ir_one_le_one_add_epsilon : (1 : ℝ) ≤ 1 + epsilon := by
  have := ir_epsilon_pos
  linarith

/-! ### (b) The two one-sided power estimates -/

/-- For a real exponent `b ≤ 5`, `(1 + epsilon)^b ≤ (1 + epsilon)^5`: monotonicity of the real
power in the exponent for a base `≥ 1` (`Real.rpow_le_rpow_of_exponent_le`), with the natural
power on the right via `Real.rpow_natCast`. -/
theorem ir_rpow_le_pow_five (b : ℝ) (hb : b ≤ 5) :
    (1 + epsilon) ^ b ≤ (1 + epsilon) ^ 5 := by
  have h := Real.rpow_le_rpow_of_exponent_le ir_one_le_one_add_epsilon hb
  rwa [show (5 : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast] at h

/-- Bernoulli's inequality at `s = -epsilon`: for a real exponent `a ≥ 1`,
`1 - a epsilon ≤ (1 - epsilon)^a` (`one_add_mul_self_le_rpow_one_add`). -/
theorem ir_bernoulli (a : ℝ) (ha : 1 ≤ a) : 1 - a * epsilon ≤ (1 - epsilon) ^ a := by
  have hε := ir_epsilon_lt
  have h := one_add_mul_self_le_rpow_one_add (show (-1 : ℝ) ≤ -epsilon by linarith) ha
  rw [show (1 : ℝ) + -epsilon = 1 - epsilon by ring] at h
  linarith

/-! ### (c) The two exact rational inequalities (degree five in `epsilon`) -/

/-- `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon`, exact rational arithmetic. -/
theorem ir_lower_rational : (1 - 13 * epsilon) * (1 + epsilon) ^ 5 ≤ 1 - 5 * epsilon := by
  unfold epsilon
  norm_num

/-- `(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)`, exact rational arithmetic. -/
theorem ir_upper_rational : (1 + epsilon) ^ 5 ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by
  unfold epsilon
  norm_num

/-! ### (d) The ratio bounds for arbitrary exponents in `[1, 5]` -/

/-- Lower bound: for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`,
`1 - 13 epsilon ≤ (1 - epsilon)^a / (1 + epsilon)^b`. -/
theorem ir_lower_bound (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b := by
  have hε := ir_epsilon_pos
  have hε1 := ir_epsilon_lt
  have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
  have hae : a * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right ha5 hε.le
  rw [le_div_iff₀ hpb]
  calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
      ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
        mul_le_mul_of_nonneg_left (ir_rpow_le_pow_five b hb5) (by linarith)
    _ ≤ 1 - 5 * epsilon := ir_lower_rational
    _ ≤ 1 - a * epsilon := by linarith
    _ ≤ (1 - epsilon) ^ a := ir_bernoulli a ha1

/-- Upper bound: for `a ≤ 5` and `1 ≤ b ≤ 5`,
`(1 + epsilon)^a / (1 - epsilon)^b ≤ 1 + 28 epsilon`. -/
theorem ir_upper_bound (a b : ℝ) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon := by
  have hε := ir_epsilon_pos
  have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos ir_one_sub_epsilon_pos _
  have hbe : b * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right hb5 hε.le
  rw [div_le_iff₀ hqb]
  calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := ir_rpow_le_pow_five a ha5
    _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := ir_upper_rational
    _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    _ ≤ (1 + 28 * epsilon) * (1 - epsilon) ^ b :=
        mul_le_mul_of_nonneg_left (ir_bernoulli b hb1) (by linarith)

/-- Both ratio bounds together, for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`. -/
theorem ir_ratio_bounds (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b ∧
      (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon :=
  ⟨ir_lower_bound a b ha1 ha5 hb5, ir_upper_bound a b ha5 hb1 hb5⟩

/-! ### (e) The exponents in dimensions two and three -/

/-- `exponentA 2 = 13/6` (`m = 1`, `n = 1`: `1 + 2/3 + 1/2`). -/
theorem ir_exponentA_two : exponentA 2 = 13 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 2 = 5/2` (`d = 2`, `n = 1`: `2 + 1/2`). -/
theorem ir_exponentB_two : exponentB 2 = 5 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

/-- `exponentA 3 = 25/6` (`m = 2`, `n = 3`: `2 + 2/3 + 3/2`). -/
theorem ir_exponentA_three : exponentA 3 = 25 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 3 = 9/2` (`d = 3`, `n = 3`: `3 + 3/2`). -/
theorem ir_exponentB_three : exponentB 3 = 9 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

end Aux

/-! ## The Section 4 integrand-ratio bounds in dimensions two and three -/

/-- Proves: `UniversalLaw.Spec.Side24.IntegrandRatioBounds` (claim `side24-coefficient`).
Method: case split `d = 2`, `d = 3`, where the exponents evaluate to `a = 13/6, b = 5/2` and
`a = 25/6, b = 9/2` (`Aux.ir_exponentA_two`, `Aux.ir_exponentB_two`, `Aux.ir_exponentA_three`,
`Aux.ir_exponentB_three`); for arbitrary `1 ≤ a, b ≤ 5` both bounds follow
(`Aux.ir_ratio_bounds`) from Bernoulli's inequality `1 - a epsilon ≤ (1 - epsilon)^a`
(`one_add_mul_self_le_rpow_one_add`, `Aux.ir_bernoulli`), exponent monotonicity
`(1 + epsilon)^b ≤ (1 + epsilon)^5` (`Real.rpow_le_rpow_of_exponent_le`,
`Aux.ir_rpow_le_pow_five`), `le_div_iff₀` / `div_le_iff₀`, and the two exact rational
inequalities `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon` and
`(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)` decided by `norm_num`
(`Aux.ir_lower_rational`, `Aux.ir_upper_rational`).
Does not establish: the informal Section 4 display of the source as the source's object (the
alignment of `epsilon`, `exponentA`, `exponentB` and these power quotients with the source's
`epsilon`, `a`, `b` and integrand ratio is PENDING_INDEPENDENT_REVIEW); that the integrand ratio
of the source equals these power quotients (not specified in the statement packet); the
arithmetic `13 ≤ 32`, `28 ≤ 32` and `32*10^(-108) < 10^(-106)` (kernel-checked on main in
`UniversalLaw.Side24.ratio_constant_ordering` and `density_comparison_below_reported`, not
here); `DensityComparison` or any covariance ordering; any enclosure of the coefficient; the
statement module's own `Does not claim:` text (the statement packet is not edited here); the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register,
not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
  intro d hd
  rcases hd with rfl | rfl
  · rw [Aux.ir_exponentA_two, Aux.ir_exponentB_two]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rw [Aux.ir_exponentA_three, Aux.ir_exponentB_three]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end UniversalLaw.Proofs.Side24Analysis

-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====
#check @UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds
#print UniversalLaw.Spec.Side24.IntegrandRatioBounds
#print UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds
#print axioms UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_lt
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_sub_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_le_one_add_epsilon
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_rpow_le_pow_five
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_bernoulli
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_lower_rational
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_upper_rational
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_lower_bound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_upper_bound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_ratio_bounds
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_three
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_three

open Lean Meta Elab Command in
run_cmd do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds | throwError "missing theorem"
  let target := mkConst ``UniversalLaw.Spec.Side24.IntegrandRatioBounds
  logInfo m!"TYPE_PP {ci.type}"
  logInfo (s!"TYPE_EXACT_EXPR_EQ {ci.type == target}")
  logInfo (s!"TYPE_IS_THEOREM {match ci with | .thmInfo _ => true | _ => false}")
  let some pi := env.find? ``UniversalLaw.Spec.Side24.IntegrandRatioBounds | throwError "missing Prop"
  logInfo m!"PROP_VALUE {pi.value?.getD (mkConst `none)}"
  -- every constant this file adds (stage-2 map), with kind
  let mut rows : Array String := #[]
  for (n, c) in env.constants.map₂.toList do
    let kind := match c with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM" | .opaqueInfo _ => "OPAQUE"
      | .quotInfo _ => "QUOT" | .inductInfo _ => "ind" | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (isProp c.type)
    rows := rows.push s!"STAGE2 {kind} prop={isP} unsafe={c.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo m!"STAGE2_TOTAL {rows.size}"

open Lean Meta Elab Command
def auditModules : Array Name := #[]
def auditPrefix : Name := `UniversalLaw
def registeredProps : Array Name := #[`UniversalLaw.Spec.Side24.ThetaSummable, `UniversalLaw.Spec.Side24.IsUnitTuple, `UniversalLaw.Spec.Side24.DerivativePairingBound, `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativeBoundAtZero, `UniversalLaw.Spec.Side24.LatticeImageSumBound, `UniversalLaw.Spec.Side24.SuccessiveTermRatio, `UniversalLaw.Spec.Side24.ExponentialBounds, `UniversalLaw.Spec.Side24.ImageBound, `UniversalLaw.Spec.Side24.CovarianceOrdering, `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound, `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound, `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks, `UniversalLaw.Spec.Side24.DensityComparison, `UniversalLaw.Spec.Side24.IntegrandRatioBounds, `UniversalLaw.Spec.Side24.ReferencePinDensityProduct, `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity, `UniversalLaw.Spec.Side24.NegDefCone, `UniversalLaw.Spec.Side24.ConeMomentD1Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Algebra, `UniversalLaw.Spec.Side24.TraceMoments, `UniversalLaw.Spec.Side24.RayleighSquareLaw, `UniversalLaw.Spec.Side24.ElementaryConeIntegral, `UniversalLaw.Spec.Side24.CoefficientEnclosure, `UniversalLaw.Spec.Side24.ReferenceComparison, `UniversalLaw.Spec.Lifetime.RemainderBoundedR1, `UniversalLaw.Spec.Lifetime.IsUnrestrictedPair, `UniversalLaw.Spec.Lifetime.IsUnrestrictedLeadingCoefficient, `UniversalLaw.Spec.Lifetime.IsLifetimeDensityVersion, `UniversalLaw.Spec.RN.HolderCountBoundN1, `UniversalLaw.Spec.RN.HolderConsequenceN2, `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3, `UniversalLaw.Spec.RN.UniformCounterexampleN4, `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5, `UniversalLaw.Spec.RN.IsCriticalPoint, `UniversalLaw.Spec.RN.HessianHasInertia, `UniversalLaw.Spec.RN.FixedRemoteMeanMeasureA2, `UniversalLaw.Spec.RN.FixedRemoteCountBoundA3, `UniversalLaw.Spec.RN.FixedAnnulusWindowCandidateA2, `UniversalLaw.Spec.P15.IsRealizedFamily, `UniversalLaw.Spec.P15.IsGood, `UniversalLaw.Spec.P15.IsKDecomposable, `UniversalLaw.Spec.P15.IsGeneratorCover, `UniversalLaw.Spec.P15.IsPaletteFeasible, `UniversalLaw.Spec.P15.PaletteNumberAttained, `UniversalLaw.Spec.P15.IsMinimalForbidden, `UniversalLaw.Spec.P15.IsTransversal, `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation, `UniversalLaw.Spec.P15.LocalRestriction, `UniversalLaw.Spec.P15.LocalDecomposability, `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock, `UniversalLaw.Spec.P15.LocalPriceChain, `UniversalLaw.Spec.P15.IndependentBlockHazard, `UniversalLaw.Spec.P15.FullBlockCoverIff, `UniversalLaw.Spec.P15.CoverCostHazardBound, `UniversalLaw.Spec.P15.WholeGroundChromaticNumber, `UniversalLaw.Spec.P15.ElementaryPaletteFormula, `UniversalLaw.Spec.P15.TriangleNeedsThree, `UniversalLaw.Spec.P15.Benchmark816IsRealized, `UniversalLaw.Spec.P15.Benchmark816Palette, `UniversalLaw.Spec.P15.Benchmark816Cover, `UniversalLaw.Spec.P15.Benchmark816WholeGround, `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts, `UniversalLaw.Spec.P15.Benchmark816Price, `UniversalLaw.Spec.P15.Benchmark816UnionBound, `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices, `UniversalLaw.Spec.P15.DemandOneRefutation, `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity, `UniversalLaw.Spec.P15.LogRatioBound, `UniversalLaw.Spec.P15.RatioExponentBound, `UniversalLaw.Spec.P15.RestrictedLocalBudget, `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoTestSufficient, `UniversalLaw.Spec.P15.CapTestAtTwoThirds, `UniversalLaw.Spec.P15.Benchmark816TransformedRange, `UniversalLaw.Spec.P15.Benchmark816RationalPrice, `UniversalLaw.Spec.P15.FullTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoStarEnclosure, `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths, `UniversalLaw.Spec.P15.IsProperDownset, `UniversalLaw.Spec.P15.HazardStarPositive, `UniversalLaw.Spec.P15.CoordinatewiseConcavity, `UniversalLaw.Spec.P15.HazardProductBound, `UniversalLaw.Spec.P15.HazardPhiBound, `UniversalLaw.Spec.P15.FullGroundPriceBound, `UniversalLaw.Spec.P15.PhiAtPStar, `UniversalLaw.Spec.P15.FullGroundBudgetIff, `UniversalLaw.Spec.P15.CapacityHazardIsBinomial, `UniversalLaw.Spec.P15.OddMajorityIdentity, `UniversalLaw.Spec.P15.CapacityWorstCase, `UniversalLaw.Spec.P15.HStarExceedsOne, `UniversalLaw.Spec.P15.LocalPriceBoundFull, `UniversalLaw.Spec.P15.GlobalAssemblyHazard, `UniversalLaw.Spec.P15.SharpnessOfRhoStar, `UniversalLaw.Spec.P15.DemandOneBoundary, `UniversalLaw.Spec.P15.SixSeventhsCertificate, `UniversalLaw.Spec.Field.IsLPeriodic, `UniversalLaw.Spec.Field.IsSmooth, `UniversalLaw.Spec.Field.IsCriticalPoint, `UniversalLaw.Spec.Field.HessianHasInertia, `UniversalLaw.Spec.Field.IsNondegMax, `UniversalLaw.Spec.Field.IsIndexSaddle, `UniversalLaw.Spec.Field.IsCandidatePair, `UniversalLaw.Spec.Field.IsMorseDistinct, `UniversalLaw.Spec.Field.IsFiniteBar, `UniversalLaw.Spec.Field.IsLifetimeDensityVersion, `UniversalLaw.Spec.Lifetime.IsLPeriodic, `UniversalLaw.Spec.Lifetime.IsSmooth, `UniversalLaw.Spec.Lifetime.IsCriticalPoint, `UniversalLaw.Spec.Lifetime.HessianHasInertia, `UniversalLaw.Spec.Lifetime.IsNondegMax, `UniversalLaw.Spec.Lifetime.IsIndexSaddle, `UniversalLaw.Spec.Lifetime.IsCandidatePair, `UniversalLaw.Spec.Lifetime.IsMorseDistinct, `UniversalLaw.Spec.Lifetime.IsFiniteBar, `UniversalLaw.Spec.Lifetime.IsWindowPair, `UniversalLaw.Spec.Lifetime.IsWindowElderPair, `UniversalLaw.Spec.Lifetime.ParentUniformSelectionA, `UniversalLaw.Spec.Lifetime.ParentCompactWindowDensityB, `UniversalLaw.Spec.Lifetime.ParentUnrestrictedLeadingDensityC, `UniversalLaw.Spec.RN.D5dSummedPinNeighbourhoodC2, `UniversalLaw.Spec.RN.AnnulusBridgeAllHeight2, `UniversalLaw.Spec.RN.D5aPuncturedPinDisks, `UniversalLaw.Spec.RN.D5bMicrodisk, `UniversalLaw.Spec.RN.D5eIntermediateShellsI3I4, `UniversalLaw.Spec.RN.WitnessCollisionNearPinPairsTarget, `UniversalLaw.Spec.RN.RemoteCollisionSecondFactorialD, `UniversalLaw.Spec.RN.D5cCollarC1, `UniversalLaw.Spec.RN.D5fGlobalWindowFirstMomentI5, `UniversalLaw.Spec.Lifetime.PlanarElderLowerE2, `UniversalLaw.Spec.Lifetime.PlanarTwoSidedRateE4, `UniversalLaw.Spec.Lifetime.PlanarDensityGapE11, `UniversalLaw.Spec.RN.DimensionLiftPuncturedPinBallP, `UniversalLaw.Spec.RN.DimensionLiftCompactCollarC, `UniversalLaw.Spec.RN.DimensionLiftIntermediateShellsI, `UniversalLaw.Spec.RN.DimensionLiftGlobalWindowG, `UniversalLaw.Spec.RN.FourierCutoffPlanarSecondFactorialF4, `UniversalLaw.Spec.RN.PalmRouteFactorialMomentsQ, `UniversalLaw.Spec.RN.PalmRouteSecondFactorialTheta, `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO]

/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`, an equation lemma `eq_*` of an unfolded statement definition, which may
be generated in several modules and is audited once), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  -- an equation lemma of an imported definition (`foo.eq_1`) is generated on demand in every module
  -- that unfolds it and is one constant of the environment: audit each name once
  let mut seen : NameSet := {}
  let mut unique : Array ConstantInfo := #[]
  for ci in constants do
    unless seen.contains ci.name do
      seen := seen.insert ci.name
      unique := unique.push ci
  constants := unique
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"

