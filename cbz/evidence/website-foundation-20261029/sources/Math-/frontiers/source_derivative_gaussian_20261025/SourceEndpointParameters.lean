import SourceEndpointRegression
import Mathlib.Probability.Distributions.Gaussian.Multivariate

/-!
# Actual canonical endpoint-kernel Gaussian parameters

Unconditional means are derived from the literal differentiated source series.
The residual covariance is derived from bilinearity and the actual regression
normal equations. Its Schur matrix is positive semidefinite because it is an
actual Gaussian covariance; no positivity premise is added.

At every pin the canonical kernel, transported from the Pi space to Euclidean
space, equals the literal `multivariateGaussian` with mean `C G⁻¹ x` and Schur
covariance. Singular residual covariance is retained. Current pin covariance
invertibility is explicit pending the separate covariance-limit join.
See `SOURCE_ENDPOINT_PARAMETERS.md` for the frozen source contract.
-/

noncomputable section

open I4Foundation I4Source I4SourceGaussian I4SmoothCarrier I4SourceDerivativeGaussian
open I4SourceConfluentPins I4SourceEndpointRegression ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Matrix WithLp
open scoped BigOperators Matrix RealInnerProductSpace

set_option autoImplicit false

namespace I4SourceEndpointParameters

theorem smoothSourceField_derivative_integral_zero (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k : ℕ) (x : Fin d → ℝ) (u : Fin k → (Fin d → ℝ)) :
    (∫ ω, iteratedFDeriv ℝ k (smoothSourceField d L ω) x u ∂smoothLaw d L) = 0 := by
  simp_rw [smoothSourceField_derivative_series d hL]
  have hm := integral_map (μ := smoothLaw d L) (sampleInclusion_measurable d L).aemeasurable
    (scalarSeries_measurable (derivativeCoefficient d L k x u)).aestronglyMeasurable
  rw [sampleInclusion_map d hL] at hm
  rw [← hm]
  exact scalarSeries_integral_zero _ (derivativeCoefficient_summable d hL k x u)

theorem rawPinsWithHessians_integral_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i : Fin 12) :
    (∫ ω, rawPinsWithHessians L r ω i ∂smoothLaw 2 L) = 0 := by
  fin_cases i
  all_goals dsimp [rawPinsWithHessians, rawPins]
  all_goals first
    | exact smoothSourceField_derivative_integral_zero 2 hL 2 _ _
    | simpa only [iteratedFDeriv_zero_apply] using
        smoothSourceField_derivative_integral_zero 2 hL 0 _ ![]
    | simpa [iteratedFDeriv_one_apply] using
        smoothSourceField_derivative_integral_zero 2 hL 1 _ ![directionX]
    | simpa [iteratedFDeriv_one_apply] using
        smoothSourceField_derivative_integral_zero 2 hL 1 _ ![directionY]

theorem rawPinsWithHessians_integral_vector_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    (∫ ω, rawPinsWithHessians L r ω ∂smoothLaw 2 L) = 0 := by
  ext i
  have hh := (ContinuousLinearMap.proj i : (Fin 12 → ℝ) →L[ℝ] ℝ).integral_comp_comm
    (rawPinsWithHessians_hasGaussianLaw hL r).integrable
  change (∫ ω, rawPinsWithHessians L r ω i ∂smoothLaw 2 L) =
    (∫ ω, rawPinsWithHessians L r ω ∂smoothLaw 2 L) i at hh
  change (∫ ω, rawPinsWithHessians L r ω ∂smoothLaw 2 L) i = 0
  rw [← hh]
  exact rawPinsWithHessians_integral_zero hL r i

theorem sourcePins_adjusted_integral_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    (∫ ω, (sourceConfluentPinsAt L r ω, sourceAdjustedEndpoints L r ω) ∂smoothLaw 2 L) = 0 := by
  have hh := (endpointObservationMap r).integral_comp_comm
    (rawPinsWithHessians_hasGaussianLaw hL r).integrable
  simp_rw [endpointObservationMap_actual] at hh
  rw [rawPinsWithHessians_integral_vector_zero hL r, map_zero] at hh
  exact hh

theorem sourceConfluentPinsAt_integral_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    (∫ ω, sourceConfluentPinsAt L r ω ∂smoothLaw 2 L) = 0 := by
  have hh := congrArg Prod.fst (sourcePins_adjusted_integral_zero hL r)
  rw [fst_integral (sourcePins_adjusted_hasGaussianLaw hL r).integrable] at hh
  exact hh

theorem sourceAdjustedEndpoints_integral_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    (∫ ω, sourceAdjustedEndpoints L r ω ∂smoothLaw 2 L) = 0 := by
  have hh := congrArg Prod.snd (sourcePins_adjusted_integral_zero hL r)
  rw [snd_integral (sourcePins_adjusted_hasGaussianLaw hL r).integrable] at hh
  exact hh

theorem sourceEndpointResidual_integral_zero {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    (∫ ω, sourceEndpointResidual L r ω ∂smoothLaw 2 L) = 0 := by
  have hG := sourcePins_adjusted_hasGaussianLaw hL r
  unfold sourceEndpointResidual
  rw [integral_sub hG.snd.integrable (hG.fst.map_fun (sourceRegressionMapAt L r)).integrable,
    (sourceRegressionMapAt L r).integral_comp_comm hG.fst.integrable,
    sourceAdjustedEndpoints_integral_zero hL r, sourceConfluentPinsAt_integral_zero hL r,
    map_zero, sub_zero]

theorem sourceEndpointResidual_covariance {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (i j : Fin 6) :
    covariance (fun ω => sourceEndpointResidual L r ω i)
      (fun ω => sourceEndpointResidual L r ω j) (smoothLaw 2 L) =
      sourceEndpointCovarianceAt L r i j -
        (sourceCrossCovarianceAt L r * (sourcePinCovarianceAt L r)⁻¹ *
          (sourceCrossCovarianceAt L r)ᵀ) i j := by
  let X := sourceConfluentPinsAt L r
  let V := sourceAdjustedEndpoints L r
  let B := sourceRegressionMapAt L r
  let R := sourceEndpointResidual L r
  let P := smoothLaw 2 L
  have hXV := sourcePins_adjusted_hasGaussianLaw hL r
  have : IsProbabilityMeasure P := hXV.isProbabilityMeasure
  have hXi (l : Fin 6) : MemLp (fun ω => X ω l) 2 P := (hXV.fst.eval l).memLp_two
  have hVi (l : Fin 6) : MemLp (fun ω => V ω l) 2 P := (hXV.snd.eval l).memLp_two
  have hBi (l : Fin 6) : MemLp (fun ω => B (X ω) l) 2 P :=
    ((hXV.fst.map_fun B).eval l).memLp_two
  have hRi (l : Fin 6) : MemLp (fun ω => R ω l) 2 P := by
    exact (hVi l).sub (hBi l)
  have hXR (l t : Fin 6) :
      covariance (fun ω => X ω l) (fun ω => R ω t) P = 0 := by
    simpa only [sub_zero, X, R, P, sourceEndpointResidual] using
      regression_residual_zero_covariance hXV B 0 0
        (sourceRegression_normal_equations L r hG) l t
  have hBR (l t : Fin 6) :
      covariance (fun ω => B (X ω) l) (fun ω => R ω t) P = 0 := by
    have heq : (fun ω => B (X ω) l) =
        (fun ω => ∑ a, B (Pi.single a 1) l * X ω a) := by
      funext ω
      exact regression_map_coordinate B (X ω) l
    rw [heq, covariance_fun_sum_left (fun a => (hXi a).const_mul _) (hRi t)]
    simp_rw [covariance_const_mul_left, hXR, mul_zero]
    exact Finset.sum_const_zero
  have hBV (l t : Fin 6) :
      covariance (fun ω => B (X ω) l) (fun ω => V ω t) P =
        (sourceRegressionMatrixAt L r * (sourceCrossCovarianceAt L r)ᵀ) l t := by
    have heq : (fun ω => B (X ω) l) =
        (fun ω => ∑ a, B (Pi.single a 1) l * X ω a) := by
      funext ω
      exact regression_map_coordinate B (X ω) l
    rw [heq, covariance_fun_sum_left (fun a => (hXi a).const_mul _) (hVi t)]
    simp_rw [covariance_const_mul_left]
    change (∑ a, sourceRegressionMapAt L r (Pi.single a 1) l *
      covariance (fun ω => X ω a) (fun ω => V ω t) P) = _
    simp only [sourceRegressionMapAt_single, Matrix.mul_apply, Matrix.transpose_apply]
    apply Finset.sum_congr rfl
    intro a _
    congr 1
    exact covariance_comm _ _
  change covariance (fun ω => R ω i) (fun ω => V ω j - B (X ω) j) P = _
  rw [covariance_fun_sub_right (hRi i) (hVi j) (hBi j),
    covariance_comm (fun ω => R ω i) (fun ω => B (X ω) j), hBR j i, sub_zero]
  change covariance (fun ω => V ω i - B (X ω) i) (fun ω => V ω j) P = _
  rw [covariance_fun_sub_left (hVi i) (hBi i) (hVi j), hBV i j]
  rfl

lemma finiteGaussian_law_parameters {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n : ℕ} {Z : Ω → Fin n → ℝ}
    (hZ : HasGaussianLaw Z P) (m : Fin n → ℝ) (hm : (∫ ω, Z ω ∂P) = m)
    (S : Matrix (Fin n) (Fin n) ℝ)
    (hcov : ∀ i j, covariance (fun ω => Z ω i) (fun ω => Z ω j) P = S i j) :
    S.PosSemidef ∧ P.map (fun ω => toLp 2 (Z ω)) = multivariateGaussian (toLp 2 m) S := by
  have : IsProbabilityMeasure P := hZ.isProbabilityMeasure
  let e : (Fin n → ℝ) ≃L[ℝ] EuclideanSpace ℝ (Fin n) :=
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm
  have hE : HasGaussianLaw (fun ω => toLp 2 (Z ω)) P := hZ.map_equiv_fun e
  have : IsGaussian (P.map (fun ω => toLp 2 (Z ω))) := hE.isGaussian_map
  have hZi (i : Fin n) : MemLp (fun ω => Z ω i) 2 P := (hZ.eval i).memLp_two
  have hS : S.PosSemidef := by
    apply Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
    · ext i j
      simp only [Matrix.conjTranspose_apply, star_trivial]
      rw [← hcov i j, ← hcov j i]
      exact covariance_comm _ _
    · intro x
      have hp := covarianceBilin_self_nonneg
        (μ := P.map (fun ω => toLp 2 (Z ω))) (toLp 2 x)
      rw [covarianceBilin_apply_pi hZi] at hp
      simpa [hcov, dotProduct, Matrix.mulVec, Finset.mul_sum, mul_assoc, mul_left_comm,
        mul_comm] using hp
  refine ⟨hS, ?_⟩
  apply IsGaussian.ext
  · simp only [id_eq]
    rw [integral_id_multivariateGaussian]
    rw [integral_map hE.aemeasurable (by fun_prop)]
    change (∫ ω, e (Z ω) ∂P) = e m
    rw [e.integral_comp_comm, hm]
  · rw [← ContinuousLinearMap.toBilinForm_inj]
    refine LinearMap.BilinForm.ext_basis (EuclideanSpace.basisFun (Fin n) ℝ).toBasis fun i j => ?_
    simp only [ContinuousLinearMap.toBilinForm_apply]
    change covarianceBilin (P.map (fun ω => toLp 2 (Z ω)))
        (EuclideanSpace.basisFun (Fin n) ℝ i) (EuclideanSpace.basisFun (Fin n) ℝ j) =
      covarianceBilin (multivariateGaussian (toLp 2 m) S)
        (EuclideanSpace.basisFun (Fin n) ℝ i) (EuclideanSpace.basisFun (Fin n) ℝ j)
    rw [covarianceBilin_apply_basisFun hZi, covarianceBilin_multivariateGaussian hS]
    simpa using hcov i j

/-- The actual adjusted-endpoint Schur covariance. -/
def sourceEndpointSchurAt (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  sourceEndpointCovarianceAt L r -
    sourceCrossCovarianceAt L r * (sourcePinCovarianceAt L r)⁻¹ * (sourceCrossCovarianceAt L r)ᵀ

/-- Gaussianity is inherited from the actual jointly Gaussian pin/endpoint vector. -/
theorem sourceEndpointResidual_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    HasGaussianLaw (sourceEndpointResidual L r) (smoothLaw 2 L) := by
  have hR := (regression_residual_jointGaussian (sourcePins_adjusted_hasGaussianLaw hL r)
    (sourceRegressionMapAt L r) 0 0).snd
  unfold sourceEndpointResidual
  simpa only [sub_zero] using hR

/-- A literal posterior sample at a prescribed pin. -/
def sourceEndpointPosterior (L r : ℝ) (x : Fin 6 → ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  sourceEndpointResidual L r ω + sourceRegressionMapAt L r x

theorem sourceEndpointPosterior_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) : HasGaussianLaw (sourceEndpointPosterior L r x) (smoothLaw 2 L) := by
  have hR := sourceEndpointResidual_hasGaussianLaw hL r
  refine ⟨hR.aemeasurable.add aemeasurable_const, ?_⟩
  have : IsGaussian ((smoothLaw 2 L).map (sourceEndpointResidual L r)) := hR.isGaussian_map
  convert (inferInstance : IsGaussian (((smoothLaw 2 L).map (sourceEndpointResidual L r)).map
    (fun v => v + sourceRegressionMapAt L r x))) using 1
  rw [AEMeasurable.map_map_of_aemeasurable (by fun_prop) hR.aemeasurable]
  rfl

theorem sourceEndpointPosterior_integral {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) : (∫ ω, sourceEndpointPosterior L r x ω ∂smoothLaw 2 L) =
      sourceRegressionMapAt L r x := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  unfold sourceEndpointPosterior
  rw [integral_add (sourceEndpointResidual_hasGaussianLaw hL r).integrable (integrable_const _),
    sourceEndpointResidual_integral_zero hL r, integral_const, zero_add]
  simp

theorem sourceEndpointPosterior_covariance {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) (i j : Fin 6) :
    covariance (fun ω => sourceEndpointPosterior L r x ω i)
      (fun ω => sourceEndpointPosterior L r x ω j) (smoothLaw 2 L) =
      sourceEndpointSchurAt L r i j := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hR := sourceEndpointResidual_hasGaussianLaw hL r
  change covariance (fun ω => sourceEndpointResidual L r ω i + sourceRegressionMapAt L r x i)
    (fun ω => sourceEndpointResidual L r ω j + sourceRegressionMapAt L r x j) _ = _
  rw [covariance_add_const_left (hR.eval i).integrable,
    covariance_add_const_right (hR.eval j).integrable]
  exact sourceEndpointResidual_covariance hL r hG i j

/-- The canonical all-pin law is the literal posterior sampling map. -/
theorem sourceEndpointKernel_eq_map_posterior {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) : sourceEndpointKernel L r x =
      (smoothLaw 2 L).map (sourceEndpointPosterior L r x) := by
  rw [sourceEndpointKernel_apply hL r x,
    AEMeasurable.map_map_of_aemeasurable (by fun_prop)
      (sourceEndpointResidual_hasGaussianLaw hL r).aemeasurable]
  rfl

theorem sourceEndpointKernel_isGaussian {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) : IsGaussian (sourceEndpointKernel L r x) := by
  rw [sourceEndpointKernel_eq_map_posterior hL r x]
  exact (sourceEndpointPosterior_hasGaussianLaw hL r x).isGaussian_map

/-- Positive semidefiniteness follows from the actual residual covariance, including singular cases. -/
theorem sourceEndpointSchurAt_posSemidef {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) : (sourceEndpointSchurAt L r).PosSemidef := by
  exact (finiteGaussian_law_parameters (sourceEndpointPosterior_hasGaussianLaw hL r 0)
    (sourceRegressionMapAt L r 0) (sourceEndpointPosterior_integral hL r 0)
    (sourceEndpointSchurAt L r) (sourceEndpointPosterior_covariance hL r hG 0)).1

/-- Exact literal multivariate Gaussian law at every prescribed pin, with no residual PSD premise. -/
theorem sourceEndpointKernel_eq_multivariateGaussian {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) :
    (sourceEndpointKernel L r x).map (toLp 2) =
      multivariateGaussian (toLp 2 (sourceRegressionMapAt L r x)) (sourceEndpointSchurAt L r) := by
  rw [sourceEndpointKernel_eq_map_posterior hL r x,
    AEMeasurable.map_map_of_aemeasurable (by fun_prop)
      (sourceEndpointPosterior_hasGaussianLaw hL r x).aemeasurable]
  exact (finiteGaussian_law_parameters (sourceEndpointPosterior_hasGaussianLaw hL r x)
    (sourceRegressionMapAt L r x) (sourceEndpointPosterior_integral hL r x)
    (sourceEndpointSchurAt L r) (sourceEndpointPosterior_covariance hL r hG x)).2

/-- The original Pi-space kernel mean is the actual regression mean. -/
theorem sourceEndpointKernel_integral {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) : (∫ v, v ∂sourceEndpointKernel L r x) = sourceRegressionMapAt L r x := by
  rw [sourceEndpointKernel_eq_map_posterior hL r x,
    integral_map (sourceEndpointPosterior_hasGaussianLaw hL r x).aemeasurable (by fun_prop)]
  exact sourceEndpointPosterior_integral hL r x

/-- The original Pi-space kernel covariance is the derived actual Schur covariance. -/
theorem sourceEndpointKernel_covariance {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) (i j : Fin 6) :
    covariance (fun v => v i) (fun v => v j) (sourceEndpointKernel L r x) =
      sourceEndpointSchurAt L r i j := by
  rw [sourceEndpointKernel_eq_map_posterior hL r x,
    covariance_map_fun (measurable_pi_apply i).aestronglyMeasurable
      (measurable_pi_apply j).aestronglyMeasurable
      (sourceEndpointPosterior_hasGaussianLaw hL r x).aemeasurable]
  exact sourceEndpointPosterior_covariance hL r hG x i j

end I4SourceEndpointParameters
