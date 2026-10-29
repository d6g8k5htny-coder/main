import SourceEndpointParameters

/-!
# Canonical physical endpoint law at every prescribed null pin

The physical and adjusted endpoint vectors differ by a literal pin-dependent
linear correction. Both canonical kernels use the same actual residual law.
The correction vanishes at the prescribed vector, proving equality there as
literal measures. This argument does not substitute an arbitrary conditional
law version at a zero-probability fiber. See `SOURCE_PHYSICAL_ENDPOINT_KERNEL.md`.
-/

noncomputable section

open I4SmoothCarrier I4SourceEndpointRegression I4SourceEndpointParameters ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators Matrix

set_option autoImplicit false

namespace I4SourcePhysicalEndpointKernel

/-- The literal correction from adjusted entries to physical normalized endpoint entries. -/
def sourcePhysicalCorrection (r : ℝ) : (Fin 6 → ℝ) →L[ℝ] (Fin 6 → ℝ) where
  toFun x := ![x 3/r, x 3/r, x 4/r, x 4/r, 0, 0]
  map_add' x y := by ext i; fin_cases i <;> simp <;> ring
  map_smul' c x := by ext i; fin_cases i <;> simp <;> ring
  cont := by fun_prop

def sourcePhysicalRegressionMap (L r : ℝ) : (Fin 6 → ℝ) →L[ℝ] (Fin 6 → ℝ) :=
  sourceRegressionMapAt L r + sourcePhysicalCorrection r

/-- A pointwise identity for the actual source observations, at every separation. -/
theorem sourceNormalizedEndpoints_eq_adjusted (L r : ℝ) (ω : SmoothSample 2 L) :
    sourceNormalizedEndpoints L r ω =
      sourceAdjustedEndpoints L r ω + sourcePhysicalCorrection r (sourceConfluentPinsAt L r ω) := by
  ext i
  fin_cases i <;> simp [sourceNormalizedEndpoints, sourceAdjustedEndpoints, sourcePhysicalCorrection]
  all_goals ring

/-- The physical canonical kernel retains exactly the same actual adjusted residual law. -/
def sourcePhysicalEndpointKernel (L r : ℝ) : Kernel (Fin 6 → ℝ) (Fin 6 → ℝ) :=
  regressionKernel ((smoothLaw 2 L).map (sourceEndpointResidual L r))
    (sourcePhysicalRegressionMap L r) 0 0

theorem sourcePhysicalEndpointKernel_markov {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    IsMarkovKernel (sourcePhysicalEndpointKernel L r) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  unfold sourcePhysicalEndpointKernel
  infer_instance

theorem sourcePhysicalEndpointKernel_measurable (L r : ℝ) :
    Measurable (sourcePhysicalEndpointKernel L r) := Kernel.measurable _

/-- Its all-pin pushforward is literal, rather than an almost-everywhere kernel identity. -/
theorem sourcePhysicalEndpointKernel_apply {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ) :
    sourcePhysicalEndpointKernel L r x =
      ((smoothLaw 2 L).map (sourceEndpointResidual L r)).map
        (fun v => v + sourceRegressionMapAt L r x + sourcePhysicalCorrection r x) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  simpa only [sourcePhysicalEndpointKernel, sourcePhysicalRegressionMap, sub_zero, add_zero,
    _root_.add_apply, add_assoc] using
    regressionKernel_apply ((smoothLaw 2 L).map (sourceEndpointResidual L r))
      (sourcePhysicalRegressionMap L r) 0 0 x

/-- The physical observations have this canonical conditional law, derived from the actual residual. -/
theorem sourcePhysicalEndpointKernel_hasCondDistrib {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasCondDistrib (sourceNormalizedEndpoints L r) (sourceConfluentPinsAt L r)
      (sourcePhysicalEndpointKernel L r) (smoothLaw 2 L) := by
  have hXV := sourcePins_adjusted_hasGaussianLaw hL r
  have hXR : HasGaussianLaw (fun ω =>
      (sourceConfluentPinsAt L r ω, sourceEndpointResidual L r ω)) (smoothLaw 2 L) := by
    unfold sourceEndpointResidual
    simpa only [sub_zero] using regression_residual_jointGaussian hXV (sourceRegressionMapAt L r) 0 0
  have hzero : ∀ i j, covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω => sourceEndpointResidual L r ω j) (smoothLaw 2 L) = 0 := by
    unfold sourceEndpointResidual
    simpa only [sub_zero] using regression_residual_zero_covariance hXV
      (sourceRegressionMapAt L r) 0 0 (sourceRegression_normal_equations L r hG)
  have hcond := gaussianRegression_hasCondDistrib hXR hzero (sourcePhysicalRegressionMap L r) 0 0
  unfold sourcePhysicalEndpointKernel
  convert hcond using 1
  funext ω
  rw [sourceNormalizedEndpoints_eq_adjusted]
  simp only [add_zero, sub_zero, sourceEndpointResidual, sourcePhysicalRegressionMap,
    _root_.add_apply]
  abel

/-- The explicit physical correction is zero at every prescribed pin, for arbitrary height. -/
theorem sourcePhysicalCorrection_prescribed (r b : ℝ) :
    sourcePhysicalCorrection r (sourcePrescribedPins r b) = 0 := by
  ext i
  fin_cases i <;> simp [sourcePhysicalCorrection, sourcePrescribedPins]

/-- Equality at the prescribed null pin is equality of the two literal canonical measures. -/
theorem sourcePhysicalEndpointKernel_at_prescribed {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    sourcePhysicalEndpointKernel L r (sourcePrescribedPins r b) =
      sourceEndpointKernel L r (sourcePrescribedPins r b) := by
  rw [sourcePhysicalEndpointKernel_apply hL, sourceEndpointKernel_apply hL,
    sourcePhysicalCorrection_prescribed]
  simp only [add_zero]

/-- The physical kernel has the extra pin-dependent correction in its mean away from the prescribed fiber. -/
theorem sourcePhysicalEndpointKernel_integral {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ) :
    (∫ v, v ∂sourcePhysicalEndpointKernel L r x) =
      sourceRegressionMapAt L r x + sourcePhysicalCorrection r x := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hR := sourceEndpointResidual_hasGaussianLaw hL r
  rw [sourcePhysicalEndpointKernel_apply hL r x,
    integral_map (by fun_prop) (by fun_prop), integral_map hR.aemeasurable (by fun_prop)]
  rw [integral_add (f := fun ω => sourceEndpointResidual L r ω + sourceRegressionMapAt L r x)
      (g := fun _ => sourcePhysicalCorrection r x)
      (hR.integrable.add (integrable_const _)) (integrable_const _),
    integral_add hR.integrable (integrable_const _), sourceEndpointResidual_integral_zero hL r]
  simp

/-- All-point physical kernel specification; no residual hypothesis is assumed. -/
theorem sourcePhysicalEndpointKernel_spec {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    IsMarkovKernel (sourcePhysicalEndpointKernel L r) ∧
      Measurable (sourcePhysicalEndpointKernel L r) ∧
      (∀ x, sourcePhysicalEndpointKernel L r x =
        ((smoothLaw 2 L).map (sourceEndpointResidual L r)).map
          (fun v => v + sourceRegressionMapAt L r x + sourcePhysicalCorrection r x)) ∧
      HasCondDistrib (sourceNormalizedEndpoints L r) (sourceConfluentPinsAt L r)
        (sourcePhysicalEndpointKernel L r) (smoothLaw 2 L) :=
  ⟨sourcePhysicalEndpointKernel_markov hL r, sourcePhysicalEndpointKernel_measurable L r,
    sourcePhysicalEndpointKernel_apply hL r, sourcePhysicalEndpointKernel_hasCondDistrib hL r hG⟩

end I4SourcePhysicalEndpointKernel
