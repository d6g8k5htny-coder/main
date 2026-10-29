import SourceDerivativeGaussian
import Mathlib.Analysis.Normed.Group.FunctionSeries

/-! Continuity of the literal derivative covariance via its actual infinite
Fourier pairing. The summable bounds hold globally in both evaluation points.
No confluent difference quotient or conditional covariance limit is inferred. -/

noncomputable section
open I4Foundation I4Source I4Smooth I4SourceGaussian I4SmoothCarrier I4SourceDerivativeGaussian
open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace I4SourceDerivativeContinuity

/-- A global, point-independent bound for a fixed directional derivative mode. -/
def derivativeMajorant (d : ℕ) (L : ℝ) (k : ℕ) (u : Fin k → (Fin d → ℝ))
    (m : SourceMode d) : ℝ :=
  orderModeBound k d L m * ∏ j, ‖u j‖

theorem derivativeMajorant_nonneg (d : ℕ) (L : ℝ) (k : ℕ)
    (u : Fin k → (Fin d → ℝ)) (m : SourceMode d) :
    0 ≤ derivativeMajorant d L k u m :=
  mul_nonneg (orderModeBound_nonneg k d L m)
    (Finset.prod_nonneg (fun _ _ => norm_nonneg _))

theorem derivativeMajorant_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin d → ℝ)) : Summable (derivativeMajorant d L k u) :=
  (orderModeBound_summable k d hL).mul_right _

theorem derivativeCoefficient_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin d → ℝ)) (m : SourceMode d) (x : Fin d → ℝ) :
    ‖derivativeCoefficient d L k x u m‖ ≤ derivativeMajorant d L k u m := by
  exact ((iteratedFDeriv ℝ k (modeCoefficient d L m) x).le_opNorm u).trans
    (mul_le_mul_of_nonneg_right (modeCoefficient_iteratedFDeriv_order_bound d hL m k x)
      (Finset.prod_nonneg (fun _ _ => norm_nonneg _)))

theorem derivativeCoefficient_continuous (d : ℕ) (L : ℝ) (k : ℕ)
    (u : Fin k → (Fin d → ℝ)) (m : SourceMode d) :
    Continuous (fun x => derivativeCoefficient d L k x u m) := by
  have hf : ContDiff ℝ k (modeCoefficient d L m) := by
    cases m with
    | none =>
      change ContDiff ℝ k (fun _ : Fin d → ℝ => Real.sqrt (1 / I4Weights.sourceZ d L))
      fun_prop
    | some p =>
      rcases p with ⟨q, b⟩
      change ContDiff ℝ k (fun x : Fin d → ℝ =>
        Real.sqrt (2 * I4Weights.sourceWeight d L q / I4Weights.sourceZ d L) *
          if b then Real.sin (phase d L q x) else Real.cos (phase d L q x))
      cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;>
        unfold phase <;> fun_prop
  exact (continuous_eval_const u).comp
    (hf.continuous_iteratedFDeriv le_rfl)

private theorem derivativeMajorant_product_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k l : ℕ) (u : Fin k → (Fin d → ℝ)) (v : Fin l → (Fin d → ℝ)) :
    Summable (fun m => derivativeMajorant d L k u m * derivativeMajorant d L l v m) := by
  have h := summable_mul_of_summable_norm
    (derivativeMajorant_summable d hL k u).norm
    (derivativeMajorant_summable d hL l v).norm
  simpa only [Function.comp_def] using h.comp_injective
    (show Function.Injective (fun m : SourceMode d => (m,m)) from
      fun _ _ h => congrArg Prod.fst h)

/-- The covariance of two actual fixed-order directional derivatives is
continuous in their two evaluation points, including coincident points. -/
theorem smoothSourceField_derivative_covariance_continuous (d : ℕ) {L : ℝ}
    (hL : L ≠ 0) (k l : ℕ) (u : Fin k → (Fin d → ℝ))
    (v : Fin l → (Fin d → ℝ)) :
    Continuous (fun xy : (Fin d → ℝ) × (Fin d → ℝ) =>
      cov[fun ω => iteratedFDeriv ℝ k (smoothSourceField d L ω) xy.1 u,
        fun ω => iteratedFDeriv ℝ l (smoothSourceField d L ω) xy.2 v; smoothLaw d L]) := by
  simp_rw [smoothSourceField_derivative_covariance d hL]
  apply continuous_tsum
  · intro m
    exact ((derivativeCoefficient_continuous d L k u m).comp continuous_fst).mul
      ((derivativeCoefficient_continuous d L l v m).comp continuous_snd)
  · exact derivativeMajorant_product_summable d hL k l u v
  · intro m xy
    rw [norm_mul]
    exact mul_le_mul (derivativeCoefficient_bound d hL k u m xy.1)
      (derivativeCoefficient_bound d hL l v m xy.2) (norm_nonneg _)
      (derivativeMajorant_nonneg d L k u m)

/-- Every actual directional derivative is centered under the unchanged source
law. Noncentered conditional means are not identified with these means. -/
theorem smoothSourceField_derivative_integral_zero (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k : ℕ) (x : Fin d → ℝ) (u : Fin k → (Fin d → ℝ)) :
    (∫ ω, iteratedFDeriv ℝ k (smoothSourceField d L ω) x u ∂smoothLaw d L) = 0 := by
  simp_rw [smoothSourceField_derivative_series d hL]
  have h := scalarSeries_integral_zero (derivativeCoefficient d L k x u)
    (derivativeCoefficient_summable d hL k x u)
  change (∫ ω, scalarSeries (derivativeCoefficient d L k x u) ω ∂sourceLaw d) = 0 at h
  rw [← sampleInclusion_map d hL,
    integral_map (sampleInclusion_measurable d L).aemeasurable
      (scalarSeries_measurable _).aestronglyMeasurable] at h
  exact h

end I4SourceDerivativeContinuity
