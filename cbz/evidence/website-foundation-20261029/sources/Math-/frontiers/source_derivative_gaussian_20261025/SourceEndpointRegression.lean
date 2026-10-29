import SourceConfluentPins
import ResearchFormalCoreR1.GaussianRegressionKernel
import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# Canonical regression kernel for actual adjusted endpoint observations

The definitions are families in every real separation, so limits can be stated
without dependent proof arguments. Their values at zero are not claimed to be
contact limits. For nonzero period the actual pin/adjusted-endpoint pair is a
linear image of the proved twelve-observation Gaussian vector.

The only current algebraic premise in the regression theorem is invertibility
of the actual pin covariance. The residual law and the conditional-law conclusion
are derived. Values of the explicit kernel are specified at every pin vector.
See `SOURCE_ENDPOINT_REGRESSION.md` for the frozen source contract.
-/

noncomputable section

open I4SmoothCarrier I4SourceConfluentPins ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators Matrix

set_option autoImplicit false

namespace I4SourceEndpointRegression

/-- Actual confluent observations as a nondependent family in r. -/
def sourceConfluentPinsAt (L r : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  confluentTransform r (rawPins L r ω)

@[simp] theorem sourceConfluentPinsAt_eq (L r : ℝ) (hr : r ≠ 0) :
    sourceConfluentPinsAt L r = confluentPins L r hr := rfl

/-- LM006 order `(xxM,xxS,xyM,xyS,yyM,yyS)`, with the confluent xx/xy pins subtracted. -/
def sourceAdjustedEndpoints (L r : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  let x := sourceConfluentPinsAt L r ω
  let v := rawPinsWithHessians L r ω
  ![(v 6 - x 3)/r, (v 9 - x 3)/r, (v 7 - x 4)/r, (v 10 - x 4)/r, v 8, v 11]

/-- The actual pin covariance G. -/
def sourcePinCovarianceAt (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => covariance (fun ω => sourceConfluentPinsAt L r ω i)
    (fun ω => sourceConfluentPinsAt L r ω j) (smoothLaw 2 L)

/-- The actual adjusted-endpoint/pin cross covariance C, with endpoint rows. -/
def sourceCrossCovarianceAt (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => covariance (fun ω => sourceAdjustedEndpoints L r ω i)
    (fun ω => sourceConfluentPinsAt L r ω j) (smoothLaw 2 L)

/-- The actual adjusted endpoint covariance, without an assumed limiting model. -/
def sourceEndpointCovarianceAt (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => covariance (fun ω => sourceAdjustedEndpoints L r ω i)
    (fun ω => sourceAdjustedEndpoints L r ω j) (smoothLaw 2 L)

def sourceRegressionMatrixAt (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  sourceCrossCovarianceAt L r * (sourcePinCovarianceAt L r)⁻¹

def sourceRegressionMapAt (L r : ℝ) : (Fin 6 → ℝ) →L[ℝ] (Fin 6 → ℝ) :=
  (Matrix.toLin' (sourceRegressionMatrixAt L r)).toContinuousLinearMap

/-- Deterministic twelve-observation map: no observation law is built into its definition. -/
def endpointObservationMap (r : ℝ) :
    (Fin 12 → ℝ) →L[ℝ] ((Fin 6 → ℝ) × (Fin 6 → ℝ)) where
  toFun v :=
    let x := confluentTransform r (fun i : Fin 6 => v (Fin.castAdd 6 i))
    (x, ![(v 6 - x 3)/r, (v 9 - x 3)/r, (v 7 - x 4)/r, (v 10 - x 4)/r, v 8, v 11])
  map_add' v w := by
    ext i <;> fin_cases i <;> simp [confluentTransform] <;> ring
  map_smul' c v := by
    ext i <;> fin_cases i <;> simp [confluentTransform] <;> ring
  cont := by
    unfold confluentTransform
    fun_prop

lemma endpointObservationMap_actual (L r : ℝ) (ω : SmoothSample 2 L) :
    endpointObservationMap r (rawPinsWithHessians L r ω) =
      (sourceConfluentPinsAt L r ω, sourceAdjustedEndpoints L r ω) := by
  have hp : (fun i : Fin 6 => rawPinsWithHessians L r ω (Fin.castAdd 6 i)) =
      rawPins L r ω := by
    ext i
    fin_cases i <;> rfl
  change (confluentTransform r _, _) = _
  simp only [hp]
  rfl

/-- The actual joint Gaussian law is a proved image of the actual twelve derivative observations. -/
theorem sourcePins_adjusted_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    HasGaussianLaw (fun ω => (sourceConfluentPinsAt L r ω, sourceAdjustedEndpoints L r ω))
      (smoothLaw 2 L) := by
  have hG := (rawPinsWithHessians_hasGaussianLaw hL r).map_fun (endpointObservationMap r)
  apply hG.congr
  filter_upwards with ω
  exact endpointObservationMap_actual L r ω

@[simp] theorem sourceRegressionMapAt_apply (L r : ℝ) (x : Fin 6 → ℝ) :
    sourceRegressionMapAt L r x = sourceRegressionMatrixAt L r *ᵥ x := rfl

@[simp] theorem sourceRegressionMapAt_single (L r : ℝ) (l j : Fin 6) :
    sourceRegressionMapAt L r (Pi.single l 1) j = sourceRegressionMatrixAt L r j l := by
  simp [sourceRegressionMapAt]

/-- The matrix inverse derives the regression normal equations for these actual observations. -/
theorem sourceRegression_normal_equations (L r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    ∀ i j, covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω => sourceAdjustedEndpoints L r ω j) (smoothLaw 2 L) =
      ∑ l, sourceRegressionMapAt L r (Pi.single l 1) j *
        covariance (fun ω => sourceConfluentPinsAt L r ω i)
          (fun ω => sourceConfluentPinsAt L r ω l) (smoothLaw 2 L) := by
  let _ : Invertible (sourcePinCovarianceAt L r) := hG.invertible
  have hprod : sourceCrossCovarianceAt L r * (sourcePinCovarianceAt L r)⁻¹ *
      sourcePinCovarianceAt L r = sourceCrossCovarianceAt L r :=
    Matrix.inv_mul_cancel_right_of_invertible _ _
  intro i j
  calc
    covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω => sourceAdjustedEndpoints L r ω j) (smoothLaw 2 L) =
        sourceCrossCovarianceAt L r j i := covariance_comm _ _
    _ = ∑ l, sourceRegressionMatrixAt L r j l * sourcePinCovarianceAt L r l i :=
      (congrArg (fun M => M j i) hprod).symm
    _ = ∑ l, sourceRegressionMapAt L r (Pi.single l 1) j *
        covariance (fun ω => sourceConfluentPinsAt L r ω i)
          (fun ω => sourceConfluentPinsAt L r ω l) (smoothLaw 2 L) := by
      apply Finset.sum_congr rfl
      intro l _
      rw [sourceRegressionMapAt_single]
      congr 1
      exact covariance_comm _ _

/-- The actual affine regression residual, with no assumed residual law. -/
def sourceEndpointResidual (L r : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  sourceAdjustedEndpoints L r ω - sourceRegressionMapAt L r (sourceConfluentPinsAt L r ω)

/-- The canonical actual endpoint kernel, specified at every pin value. -/
def sourceEndpointKernel (L r : ℝ) : Kernel (Fin 6 → ℝ) (Fin 6 → ℝ) :=
  regressionKernel ((smoothLaw 2 L).map (sourceEndpointResidual L r))
    (sourceRegressionMapAt L r) 0 0

theorem sourceEndpointKernel_markov {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    IsMarkovKernel (sourceEndpointKernel L r) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  unfold sourceEndpointKernel
  infer_instance

theorem sourceEndpointKernel_measurable (L r : ℝ) : Measurable (sourceEndpointKernel L r) :=
  Kernel.measurable _

/-- The all-pin pushforward identity includes every probability-zero prescribed pin vector. -/
theorem sourceEndpointKernel_apply {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ) :
    sourceEndpointKernel L r x =
      ((smoothLaw 2 L).map (sourceEndpointResidual L r)).map
        (fun v => v + sourceRegressionMapAt L r x) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  simpa [sourceEndpointKernel] using
    regressionKernel_apply ((smoothLaw 2 L).map (sourceEndpointResidual L r))
      (sourceRegressionMapAt L r) 0 0 x

/-- The actual target's conditional law, with Gaussian residual hypotheses derived rather than assumed.
Current pin-covariance invertibility is the sole algebraic premise. -/
theorem sourceEndpointKernel_hasCondDistrib {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasCondDistrib (sourceAdjustedEndpoints L r) (sourceConfluentPinsAt L r)
      (sourceEndpointKernel L r) (smoothLaw 2 L) := by
  have hcond := gaussianRegression_observed_hasCondDistrib
    (sourcePins_adjusted_hasGaussianLaw hL r) (sourceRegressionMapAt L r) 0 0
    (sourceRegression_normal_equations L r hG)
  unfold sourceEndpointKernel sourceEndpointResidual
  simpa only [sub_zero] using hcond

/-- Combined canonical law for the actual pins and adjusted endpoint vector. -/
theorem sourceEndpointKernel_spec {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    IsMarkovKernel (sourceEndpointKernel L r) ∧
      Measurable (sourceEndpointKernel L r) ∧
      (∀ x, sourceEndpointKernel L r x =
        ((smoothLaw 2 L).map (sourceEndpointResidual L r)).map
          (fun v => v + sourceRegressionMapAt L r x)) ∧
      HasCondDistrib (sourceAdjustedEndpoints L r) (sourceConfluentPinsAt L r)
        (sourceEndpointKernel L r) (smoothLaw 2 L) :=
  ⟨sourceEndpointKernel_markov hL r, sourceEndpointKernel_measurable L r,
    sourceEndpointKernel_apply hL r, sourceEndpointKernel_hasCondDistrib hL r hG⟩

/-- The literal physical normalized endpoint vector in LM006 order. -/
def sourceNormalizedEndpoints (L r : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  let v := rawPinsWithHessians L r ω
  ![v 6/r, v 9/r, v 7/r, v 10/r, v 8, v 11]

def sourcePrescribedPins (r b : ℝ) : Fin 6 → ℝ := ![b-r^3/12,0,0,0,0,2]

/-- On the two zero xx/xy pin coordinates, adjusted entries are exactly physical normalized entries. -/
theorem sourceAdjustedEndpoints_on_pin_fiber (L r : ℝ) (ω : SmoothSample 2 L)
    (h3 : sourceConfluentPinsAt L r ω 3 = 0) (h4 : sourceConfluentPinsAt L r ω 4 = 0) :
    sourceAdjustedEndpoints L r ω = sourceNormalizedEndpoints L r ω := by
  simp [sourceAdjustedEndpoints, sourceNormalizedEndpoints, h3, h4]

/-- Exact deterministic reconstruction at the prescribed pin fiber, without choosing an arbitrary RCD. -/
theorem sourceAdjustedEndpoints_on_prescribed (L r b : ℝ) (ω : SmoothSample 2 L)
    (hp : sourceConfluentPinsAt L r ω = sourcePrescribedPins r b) :
    sourceAdjustedEndpoints L r ω = sourceNormalizedEndpoints L r ω := by
  apply sourceAdjustedEndpoints_on_pin_fiber
  · simpa [sourcePrescribedPins] using congrFun hp 3
  · simpa [sourcePrescribedPins] using congrFun hp 4

end I4SourceEndpointRegression
