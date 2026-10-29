import SourceFourierSeries
import Mathlib.Analysis.Calculus.SmoothSeries
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-! Actual same-sample C4 of the canonical normalized Fourier lift.
Covariance identification, Gaussian field/jet laws and pin regression are separate. -/

noncomputable section
open I4Foundation I4Weights I4Source MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace I4C4

def phaseCLM (d : ℕ) (L : ℝ) (k : Fin d → ℤ) : (Fin d → ℝ) →L[ℝ] ℝ :=
  ∑ i, (sourceC L * (k i : ℝ)) • ContinuousLinearMap.proj i

theorem phaseCLM_apply (d : ℕ) (L : ℝ) (k : Fin d → ℤ) (x : Fin d → ℝ) :
    phaseCLM d L k x = phase d L k x := by
  simp [phaseCLM, phase]

theorem phaseCLM_norm_le (d : ℕ) (L : ℝ) (k : Fin d → ℤ) :
    ‖phaseCLM d L k‖ ≤ ∑ i, |sourceC L * (k i : ℝ)| := by
  apply ContinuousLinearMap.opNorm_le_bound _
    (Finset.sum_nonneg (fun i _ => abs_nonneg _))
  intro x
  rw [phaseCLM_apply]
  calc
    ‖phase d L k x‖ ≤ ∑ i, ‖sourceC L * (k i : ℝ) * x i‖ := norm_sum_le _ _
    _ ≤ ∑ i, |sourceC L * (k i : ℝ)| * ‖x‖ := by
      apply Finset.sum_le_sum
      intro i _
      rw [norm_mul, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_left (norm_le_pi_norm x i) (abs_nonneg _)
    _ = (∑ i, |sourceC L * (k i : ℝ)|) * ‖x‖ := by rw [Finset.sum_mul]

theorem modeCoefficient_iteratedFDeriv_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) :
    ‖iteratedFDeriv ℝ j (modeCoefficient d L m) x‖ ≤ modeMajorant d L m := by
  classical
  cases m with
  | none =>
    cases j with
    | zero =>
      simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using
        modeCoefficient_bound d hL none x
    | succ j =>
      change ‖iteratedFDeriv ℝ (j + 1)
        (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L)) x‖ ≤ Real.sqrt (1 / sourceZ d L)
      rw [iteratedFDeriv_succ_const]
      simp only [Pi.zero_apply, norm_zero]
      exact Real.sqrt_nonneg _
  | some p =>
    rcases p with ⟨k, b⟩
    let A := Real.sqrt (2 * sourceWeight d L k / sourceZ d L)
    let P := ∏ i, (1 + |sourceC L * (k.1 i : ℝ)|)
    let g : ℝ → ℝ := if b then Real.sin else Real.cos
    let ell := phaseCLM d L k
    have hg : ContDiff ℝ j g := by
      cases b <;> simp only [g, Bool.false_eq_true, ite_false, ite_true] <;> fun_prop
    have hgc : ContDiff ℝ j (g ∘ ell) := hg.comp_continuousLinearMap
    have hgn : ‖iteratedFDeriv ℝ j g (ell x)‖ ≤ 1 := by
      cases b with
      | false =>
        simpa only [g, Bool.false_eq_true, ite_false,
          norm_iteratedFDeriv_eq_norm_iteratedDeriv, Real.norm_eq_abs] using
          Real.abs_iteratedDeriv_cos_le_one j (ell x)
      | true =>
        simpa only [g, ite_true, norm_iteratedFDeriv_eq_norm_iteratedDeriv,
          Real.norm_eq_abs] using Real.abs_iteratedDeriv_sin_le_one j (ell x)
    have hP : 1 ≤ P := by
      apply Finset.one_le_prod₀
      intro i _
      linarith [abs_nonneg (sourceC L * (k.1 i : ℝ))]
    have hl : ‖ell‖ ≤ P := by
      calc
        ‖ell‖ ≤ ∑ i, |sourceC L * (k.1 i : ℝ)| := phaseCLM_norm_le d L k
        _ ≤ 1 + ∑ i, |sourceC L * (k.1 i : ℝ)| := by linarith
        _ ≤ P := frequency_l1_le_product d _ (fun i => abs_nonneg _)
    have hpow : ‖ell‖ ^ j ≤ P ^ 4 := by
      calc
        ‖ell‖ ^ j ≤ P ^ j := by gcongr
        _ ≤ P ^ 4 := pow_le_pow_right₀ hP hj
    have hder : ‖iteratedFDeriv ℝ j (g ∘ ell) x‖ ≤ ‖ell‖ ^ j := by
      rw [ell.iteratedFDeriv_comp_right hg x le_rfl]
      calc
        _ ≤ ‖iteratedFDeriv ℝ j g (ell x)‖ * ∏ _i : Fin j, ‖ell‖ :=
          (iteratedFDeriv ℝ j g (ell x)).norm_compContinuousLinearMap_le (fun _ => ell)
        _ ≤ 1 * ∏ _i : Fin j, ‖ell‖ :=
          mul_le_mul_of_nonneg_right hgn (Finset.prod_nonneg (fun _ _ => norm_nonneg _))
        _ = ‖ell‖ ^ j := by simp
    have heq : modeCoefficient d L (some (k, b)) = A • (g ∘ ell) := by
      funext y
      cases b <;> simp [modeCoefficient, A, g, Function.comp_apply, Pi.smul_apply,
        smul_eq_mul, ell, phaseCLM_apply]
    rw [heq, iteratedFDeriv_const_smul_apply hgc.contDiffAt, norm_smul,
      Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg _)]
    calc
      _ ≤ A * P ^ 4 := mul_le_mul_of_nonneg_left (hder.trans hpow) (Real.sqrt_nonneg _)
      _ = modeMajorant d L (some (k, b)) := by
        rw [modeMajorant, pairedBound_eq_amplitude d hL k]
        simp only [A, P, Finset.prod_pow]

theorem sourceField_contDiff_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ContDiff ℝ 4 (sourceField d L ω) := by
  have hc (m : SourceMode d) : ContDiff ℝ 4 (modeCoefficient d L m) := by
    cases m with
    | none =>
      change ContDiff ℝ 4 (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L))
      fun_prop
    | some p =>
      rcases p with ⟨k, b⟩
      change ContDiff ℝ 4 (fun x : Fin d → ℝ =>
        Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
          (if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)))
      cases b <;> simp [phase] <;> fun_prop
  filter_upwards [sourceMajorant_ae d hL] with ω hω
  change ContDiff ℝ 4 (fun x => ∑' m, modeCoefficient d L m x * ω m)
  apply contDiff_tsum (v := fun _ m => modeMajorant d L m * |ω m|)
  · intro m
    exact (hc m).mul contDiff_const
  · intro j hj
    exact hω
  · intro j m x hj
    have hj' : j ≤ 4 := by exact_mod_cast hj
    have hcj : ContDiff ℝ j (modeCoefficient d L m) :=
      (hc m).of_le (by exact_mod_cast hj')
    have heq : (fun y => modeCoefficient d L m y * ω m) = ω m • modeCoefficient d L m := by
      funext y
      simp [smul_eq_mul, mul_comm]
    rw [heq, iteratedFDeriv_const_smul_apply hcj.contDiffAt, norm_smul, Real.norm_eq_abs]
    simpa only [mul_comm] using mul_le_mul_of_nonneg_left
      (modeCoefficient_iteratedFDeriv_bound d hL m j hj' x) (abs_nonneg (ω m))

theorem sourceField_contDiff_affine_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ∀ (T : (Fin d → ℝ) →L[ℝ] (Fin d → ℝ)) (a : Fin d → ℝ),
      ContDiff ℝ 4 (fun x => sourceField d L ω (a + T x)) := by
  filter_upwards [sourceField_contDiff_ae d hL] with ω hω
  intro T a
  exact hω.comp (contDiff_const.add T.contDiff)

end I4C4
