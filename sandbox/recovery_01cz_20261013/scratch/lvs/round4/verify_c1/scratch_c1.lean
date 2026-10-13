import Specifications.Side24

/-!
# SIDE24 interface proofs: the `m = 1` cone moment identity (`ConeMomentD1Identity I`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux` (prefix `c1_`). The route,
for an arbitrary inhabitant `I : ConeMomentInterface` (existence of an inhabitant is neither
asserted nor used): (1) `I.A` is a.e.-measurable, because at this Mathlib revision the image
measure of a non-a.e.-measurable map under a nonzero measure is a Dirac measure
(`Measure.map_of_not_aemeasurable_of_ne_zero`), which charges a singleton, while the Gaussian
`gaussianReal 0 (8/3)` has `NullSingletonClass`; (2) `integral_map` transports the integral of
the indicator integrand along the law `I.A_law` to `gaussianReal 0 (8/3)`, and
`integral_indicator` makes it the set integral of `a ↦ a^2` over `Iio 0`; (3) the symmetry
`gaussianReal_map_neg` (with `neg_zero`) and a second `integral_map` along `x ↦ -x` give
`∫_{Iio 0} a^2 = ∫_{Ioi 0} a^2`; (4) `integral_add_compl` with `Set.compl_Iio` and
`integral_Ici_eq_integral_Ioi` splits the full second moment, which equals the variance `8/3`
by `variance_fun_id_gaussianReal` and `variance_eq_integral`; so the negative half is
`(8/3)/2 = 4/3 = coneMomentD1`. Only the field `A_law` and the instance `isProb` of the
interface are used; the fields `s`, `x`, `y`, `sxy_law` play no role.

Nothing here changes any register label; the disposition of `side24-coefficient`
(REVIEWED_SCOPED) is transcribed in the register, not derived here.
-/

namespace UniversalLaw.Proofs.Side24Interfaces

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

namespace Aux

/-! ### (a) A map with a non-degenerate Gaussian law is a.e.-measurable -/

/-- If `Measure.map A P` is the Gaussian `gaussianReal 0 v` with `v ≠ 0`, then `A` is
a.e.-measurable: otherwise the image measure would be a Dirac measure
(`Measure.map_of_not_aemeasurable_of_ne_zero`), giving a singleton of measure one, whereas the
Gaussian charges no singleton (`nullSingletonClass_gaussianReal`). -/
theorem c1_aemeasurable_of_map_eq_gaussian {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → ℝ) {v : NNReal} (hv : v ≠ 0)
    (h : Measure.map A P = gaussianReal 0 v) : AEMeasurable A P := by
  by_contra hA
  obtain ⟨c, hc⟩ : ∃ c : ℝ, Measure.map A P = Measure.dirac c :=
    ⟨_, Measure.map_of_not_aemeasurable_of_ne_zero hA (IsProbabilityMeasure.ne_zero P)⟩
  rw [hc] at h
  have : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have h1 : Measure.dirac c {c} = gaussianReal 0 v {c} := by rw [h]
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton c), measure_singleton] at h1
  exact one_ne_zero h1

/-! ### (b) Second moments of the centred Gaussian -/

/-- The integrand `a ↦ 1_{Iio 0}(a) a^2` is measurable. -/
theorem c1_measurable_indicator_sq :
    Measurable (fun a : ℝ => Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a) :=
  (measurable_id.pow_const 2).indicator measurableSet_Iio

/-- The second moment of `gaussianReal 0 v` is `v`, from Mathlib's
`variance_fun_id_gaussianReal` rewritten with `variance_eq_integral` (the mean is `0`). -/
theorem c1_second_moment_gaussianReal (v : NNReal) :
    ∫ t, t ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral (by fun_prop)] at h
  simpa using h

/-- `a ↦ a^2` is integrable against `gaussianReal 0 v` (`memLp_id_gaussianReal` at `p = 2`). -/
theorem c1_integrable_sq_gaussianReal (v : NNReal) :
    Integrable (fun a : ℝ => a ^ 2) (gaussianReal 0 v) := by
  have := (memLp_id_gaussianReal (μ := 0) (v := v) 2).integrable_sq
  simpa using this

/-- Symmetry of the centred Gaussian: the second moment over `Iio 0` equals the second moment
over `Ioi 0`, by `gaussianReal_map_neg` and a change of variables `a ↦ -a` (`integral_map`),
using `neg_lt_zero` and `neg_sq` pointwise. -/
theorem c1_setIntegral_sq_Iio_eq_Ioi (v : NNReal) :
    ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) =
      ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) := by
  set γ := gaussianReal 0 v with hγ
  have hneg : γ.map (fun x => -x) = γ := by rw [hγ, gaussianReal_map_neg, neg_zero]
  calc ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂γ
      = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂γ :=
        (integral_indicator measurableSet_Iio).symm
    _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(γ.map (fun x => -x)) := by
        rw [hneg]
    _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (-a) ∂γ :=
        integral_map measurable_neg.aemeasurable c1_measurable_indicator_sq.aestronglyMeasurable
    _ = ∫ a, Set.indicator (Set.Ioi (0 : ℝ)) (fun a => a ^ 2) a ∂γ := by
        congr 1; funext a
        simp only [Set.indicator, Set.mem_Iio, Set.mem_Ioi, neg_lt_zero, neg_sq]
    _ = ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂γ := integral_indicator measurableSet_Ioi

/-- The negative half of the second moment of `gaussianReal 0 v`, `v ≠ 0`, is `v / 2`:
`integral_add_compl` over `Iio 0` and its complement `Ici 0`, `integral_Ici_eq_integral_Ioi`
(the Gaussian charges no singleton), the symmetry lemma, and the full second moment `v`. -/
theorem c1_setIntegral_sq_Iio (v : NNReal) (hv : v ≠ 0) :
    ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) / 2 := by
  have : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have hsplit := integral_add_compl (μ := gaussianReal 0 v) (s := Set.Iio (0 : ℝ))
    measurableSet_Iio (c1_integrable_sq_gaussianReal v)
  rw [Set.compl_Iio, integral_Ici_eq_integral_Ioi, ← c1_setIntegral_sq_Iio_eq_Ioi,
    c1_second_moment_gaussianReal] at hsplit
  linarith

/-! ### (c) Transport along the law of `A` -/

/-- For a map `A` with law `gaussianReal 0 (8/3)` under a probability measure `P`, the integral
of `1_{Iio 0}(A) A^2` equals the set integral of `a ↦ a^2` over `Iio 0` against
`gaussianReal 0 (8/3)`: a.e.-measurability of `A` from its law, then `integral_map` and
`integral_indicator`. Stated for an abstract `(Ω, P, A)` so that the interface's own instance
fields `I.mΩ`, `I.isProb` are supplied explicitly at the point of use. -/
theorem c1_integral_indicator_eq {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → ℝ)
    (hlaw : Measure.map A P = gaussianReal 0 (8 / 3 : NNReal)) :
    ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (A ω) ∂P =
      ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂(gaussianReal 0 (8 / 3 : NNReal)) := by
  have hv : ((8 : NNReal) / 3) ≠ 0 := by norm_num
  have hA : AEMeasurable A P := c1_aemeasurable_of_map_eq_gaussian P A hv hlaw
  have e : ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (A ω) ∂P =
      ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(Measure.map A P) :=
    (integral_map hA c1_measurable_indicator_sq.aestronglyMeasurable).symm
  rw [e, hlaw, integral_indicator measurableSet_Iio]

end Aux

/-! ## The `m = 1` cone moment identity for every inhabitant of the interface -/

/-- Proves: `UniversalLaw.Spec.Side24.ConeMomentD1Identity` (claim `side24-coefficient`).
Method: transport along the law field `I.A_law` (`Aux.c1_integral_indicator_eq`, applied to
`I.Ω`, `I.mΩ`, `I.P`, `I.isProb`, `I.A`; it uses `integral_map` after a.e.-measurability of
`I.A` from `Aux.c1_aemeasurable_of_map_eq_gaussian`, the Dirac-versus-`NullSingletonClass`
argument), then the negative half of the second moment of
`gaussianReal 0 (8/3)` is `(8/3)/2` (`Aux.c1_setIntegral_sq_Iio`, from `integral_add_compl`,
`integral_Ici_eq_integral_Ioi`, the symmetry `gaussianReal_map_neg` and the variance
`variance_fun_id_gaussianReal`), and `push_cast`/`norm_num` give `4/3 = coneMomentD1`.
Does not establish: the informal identity `D_1 = E[A^2 1{A<0}] = 4/3` of the source as the
source's object (the alignment of `ConeMomentInterface`, its field `A_law` and `coneMomentD1`
with the source's `A ~ N(0, 8/3)` and `D_1` is PENDING_INDEPENDENT_REVIEW); that `D_1` is the
parent's cone moment `D_u` of (15.1) for the reference law; that `ConeMomentInterface` has an
inhabitant, or that its laws are the conditional law `A | V = 0` of the reference field; any
enclosure of the coefficient; `ConeMomentD2Identity`, `TraceMoments` or `RayleighSquareLaw`;
the disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the
register, not derived here; nonauthor review or acceptance of
SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem coneMomentD1Identity (I : ConeMomentInterface) :
    UniversalLaw.Spec.Side24.ConeMomentD1Identity I := by
  unfold UniversalLaw.Spec.Side24.ConeMomentD1Identity coneMomentD1
  rw [@Aux.c1_integral_indicator_eq I.Ω I.mΩ I.P I.isProb I.A I.A_law,
    Aux.c1_setIntegral_sq_Iio (8 / 3 : NNReal) (by norm_num)]
  push_cast
  norm_num

end UniversalLaw.Proofs.Side24Interfaces

-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====
#check @UniversalLaw.Proofs.Side24Interfaces.coneMomentD1Identity
#print UniversalLaw.Spec.Side24.ConeMomentD1Identity
#print UniversalLaw.Spec.Side24.coneMomentD1
#print axioms UniversalLaw.Proofs.Side24Interfaces.coneMomentD1Identity
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_aemeasurable_of_map_eq_gaussian
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_measurable_indicator_sq
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_second_moment_gaussianReal
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_integrable_sq_gaussianReal
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_setIntegral_sq_Iio_eq_Ioi
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_setIntegral_sq_Iio
#print axioms UniversalLaw.Proofs.Side24Interfaces.Aux.c1_integral_indicator_eq

open Lean Meta Elab Command

def auditModules : Array Name := #[]
def auditPrefix : Name := `UniversalLaw.Proofs.Side24Interfaces
def registeredProps : Array Name := #[`UniversalLaw.Spec.Field.HessianHasInertia, `UniversalLaw.Spec.Field.IsCandidatePair, `UniversalLaw.Spec.Field.IsCriticalPoint, `UniversalLaw.Spec.Field.IsFiniteBar, `UniversalLaw.Spec.Field.IsIndexSaddle, `UniversalLaw.Spec.Field.IsLPeriodic, `UniversalLaw.Spec.Field.IsLifetimeDensityVersion, `UniversalLaw.Spec.Field.IsMorseDistinct, `UniversalLaw.Spec.Field.IsNondegMax, `UniversalLaw.Spec.Field.IsSmooth, `UniversalLaw.Spec.Lifetime.HessianHasInertia, `UniversalLaw.Spec.Lifetime.IsCandidatePair, `UniversalLaw.Spec.Lifetime.IsCriticalPoint, `UniversalLaw.Spec.Lifetime.IsFiniteBar, `UniversalLaw.Spec.Lifetime.IsIndexSaddle, `UniversalLaw.Spec.Lifetime.IsLPeriodic, `UniversalLaw.Spec.Lifetime.IsLifetimeDensityVersion, `UniversalLaw.Spec.Lifetime.IsMorseDistinct, `UniversalLaw.Spec.Lifetime.IsNondegMax, `UniversalLaw.Spec.Lifetime.IsSmooth, `UniversalLaw.Spec.Lifetime.IsUnrestrictedLeadingCoefficient, `UniversalLaw.Spec.Lifetime.IsUnrestrictedPair, `UniversalLaw.Spec.Lifetime.IsWindowElderPair, `UniversalLaw.Spec.Lifetime.IsWindowPair, `UniversalLaw.Spec.Lifetime.ParentCompactWindowDensityB, `UniversalLaw.Spec.Lifetime.ParentUniformSelectionA, `UniversalLaw.Spec.Lifetime.ParentUnrestrictedLeadingDensityC, `UniversalLaw.Spec.Lifetime.PlanarDensityGapE11, `UniversalLaw.Spec.Lifetime.PlanarElderLowerE2, `UniversalLaw.Spec.Lifetime.PlanarTwoSidedRateE4, `UniversalLaw.Spec.Lifetime.RemainderBoundedR1, `UniversalLaw.Spec.P15.Benchmark816Cover, `UniversalLaw.Spec.P15.Benchmark816IsRealized, `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts, `UniversalLaw.Spec.P15.Benchmark816Palette, `UniversalLaw.Spec.P15.Benchmark816Price, `UniversalLaw.Spec.P15.Benchmark816RationalPrice, `UniversalLaw.Spec.P15.Benchmark816TransformedRange, `UniversalLaw.Spec.P15.Benchmark816UnionBound, `UniversalLaw.Spec.P15.Benchmark816WholeGround, `UniversalLaw.Spec.P15.CapTestAtTwoThirds, `UniversalLaw.Spec.P15.CapacityHazardIsBinomial, `UniversalLaw.Spec.P15.CapacityWorstCase, `UniversalLaw.Spec.P15.CoordinatewiseConcavity, `UniversalLaw.Spec.P15.CoverCostHazardBound, `UniversalLaw.Spec.P15.DemandOneBoundary, `UniversalLaw.Spec.P15.DemandOneRefutation, `UniversalLaw.Spec.P15.ElementaryPaletteFormula, `UniversalLaw.Spec.P15.FullBlockCoverIff, `UniversalLaw.Spec.P15.FullGroundBudgetIff, `UniversalLaw.Spec.P15.FullGroundPriceBound, `UniversalLaw.Spec.P15.FullTransformedPriceBudget, `UniversalLaw.Spec.P15.GlobalAssemblyHazard, `UniversalLaw.Spec.P15.HStarExceedsOne, `UniversalLaw.Spec.P15.HazardPhiBound, `UniversalLaw.Spec.P15.HazardProductBound, `UniversalLaw.Spec.P15.HazardStarPositive, `UniversalLaw.Spec.P15.IndependentBlockHazard, `UniversalLaw.Spec.P15.IsGeneratorCover, `UniversalLaw.Spec.P15.IsGood, `UniversalLaw.Spec.P15.IsKDecomposable, `UniversalLaw.Spec.P15.IsMinimalForbidden, `UniversalLaw.Spec.P15.IsPaletteFeasible, `UniversalLaw.Spec.P15.IsProperDownset, `UniversalLaw.Spec.P15.IsRealizedFamily, `UniversalLaw.Spec.P15.IsTransversal, `UniversalLaw.Spec.P15.LocalDecomposability, `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock, `UniversalLaw.Spec.P15.LocalPriceBoundFull, `UniversalLaw.Spec.P15.LocalPriceChain, `UniversalLaw.Spec.P15.LocalRestriction, `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity, `UniversalLaw.Spec.P15.LogRatioBound, `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation, `UniversalLaw.Spec.P15.OddMajorityIdentity, `UniversalLaw.Spec.P15.PaletteNumberAttained, `UniversalLaw.Spec.P15.PhiAtPStar, `UniversalLaw.Spec.P15.RatioExponentBound, `UniversalLaw.Spec.P15.RestrictedLocalBudget, `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths, `UniversalLaw.Spec.P15.RhoStarEnclosure, `UniversalLaw.Spec.P15.RhoTestSufficient, `UniversalLaw.Spec.P15.SharpnessOfRhoStar, `UniversalLaw.Spec.P15.SixSeventhsCertificate, `UniversalLaw.Spec.P15.TriangleNeedsThree, `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices, `UniversalLaw.Spec.P15.WholeGroundChromaticNumber, `UniversalLaw.Spec.RN.AnnulusBridgeAllHeight2, `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3, `UniversalLaw.Spec.RN.D5aPuncturedPinDisks, `UniversalLaw.Spec.RN.D5bMicrodisk, `UniversalLaw.Spec.RN.D5cCollarC1, `UniversalLaw.Spec.RN.D5dSummedPinNeighbourhoodC2, `UniversalLaw.Spec.RN.D5eIntermediateShellsI3I4, `UniversalLaw.Spec.RN.D5fGlobalWindowFirstMomentI5, `UniversalLaw.Spec.RN.DimensionLiftCompactCollarC, `UniversalLaw.Spec.RN.DimensionLiftGlobalWindowG, `UniversalLaw.Spec.RN.DimensionLiftIntermediateShellsI, `UniversalLaw.Spec.RN.DimensionLiftPuncturedPinBallP, `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5, `UniversalLaw.Spec.RN.FixedAnnulusWindowCandidateA2, `UniversalLaw.Spec.RN.FixedRemoteCountBoundA3, `UniversalLaw.Spec.RN.FixedRemoteMeanMeasureA2, `UniversalLaw.Spec.RN.FourierCutoffPlanarSecondFactorialF4, `UniversalLaw.Spec.RN.HessianHasInertia, `UniversalLaw.Spec.RN.HolderConsequenceN2, `UniversalLaw.Spec.RN.HolderCountBoundN1, `UniversalLaw.Spec.RN.IsCriticalPoint, `UniversalLaw.Spec.RN.PalmRouteFactorialMomentsQ, `UniversalLaw.Spec.RN.PalmRouteSecondFactorialTheta, `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO, `UniversalLaw.Spec.RN.RemoteCollisionSecondFactorialD, `UniversalLaw.Spec.RN.UniformCounterexampleN4, `UniversalLaw.Spec.RN.WitnessCollisionNearPinPairsTarget, `UniversalLaw.Spec.Side24.CoefficientEnclosure, `UniversalLaw.Spec.Side24.ConeMomentD1Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Algebra, `UniversalLaw.Spec.Side24.ConeMomentD2Identity, `UniversalLaw.Spec.Side24.CovarianceOrdering, `UniversalLaw.Spec.Side24.DensityComparison, `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativeBoundAtZero, `UniversalLaw.Spec.Side24.DerivativePairingBound, `UniversalLaw.Spec.Side24.ElementaryConeIntegral, `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound, `UniversalLaw.Spec.Side24.ExponentialBounds, `UniversalLaw.Spec.Side24.ImageBound, `UniversalLaw.Spec.Side24.IntegrandRatioBounds, `UniversalLaw.Spec.Side24.IsUnitTuple, `UniversalLaw.Spec.Side24.LatticeImageSumBound, `UniversalLaw.Spec.Side24.NegDefCone, `UniversalLaw.Spec.Side24.RayleighSquareLaw, `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity, `UniversalLaw.Spec.Side24.ReferenceComparison, `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound, `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks, `UniversalLaw.Spec.Side24.ReferencePinDensityProduct, `UniversalLaw.Spec.Side24.SuccessiveTermRatio, `UniversalLaw.Spec.Side24.ThetaSummable, `UniversalLaw.Spec.Side24.TraceMoments]

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

#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut rows : Array String := #[]
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp ci.type)
    rows := rows.push s!"STAGE2 {kind} prop={isP} unsafe={ci.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"STAGE2 TOTAL {rows.size}"
