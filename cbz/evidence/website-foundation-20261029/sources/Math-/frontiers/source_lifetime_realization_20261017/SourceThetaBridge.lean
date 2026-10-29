import Lifetime
import SourceKernel
import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation

/-! Exact all-dimensional image/spectral bridge for the unchanged Lifetime target.
No probabilistic interface or theta identity is assumed. -/

noncomputable section
open scoped BigOperators
open I4Weights I4Source

namespace I4ThetaBridge

open UniversalLaw.Spec.Lifetime

theorem latticeVec_apply (d : ℕ) (L : ℝ) (n : Fin d → ℤ) (j : Fin d) :
    latticeVec d L n j = L * (n j : ℝ) := by
  change (EuclideanSpace.projₗ j) (∑ i, EuclideanSpace.single i (L * (n i : ℝ))) = _
  rw [map_sum]
  simp

theorem theta_summand_eq_product (d : ℕ) (L : ℝ) (z : E d) (n : Fin d → ℤ) :
    Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2) =
      ∏ j, Real.exp (-((z j + L * (n j : ℝ)) ^ 2) / 2) := by
  rw [EuclideanSpace.real_norm_sq_eq, ← Real.exp_sum]
  congr 1
  simp only [PiLp.add_apply, latticeVec_apply]
  rw [← Finset.sum_neg_distrib, Finset.sum_div]

theorem shifted_gaussian_le (L t : ℝ) (n : ℤ) :
    Real.exp (-((t + L * (n : ℝ)) ^ 2) / 2) ≤
      Real.exp (t ^ 2 / 2) * decay (L ^ 2 / 4) n := by
  unfold decay
  rw [← Real.exp_add, Real.exp_le_exp]
  nlinarith [sq_nonneg (t + L * (n : ℝ) / 2)]

theorem shifted_gaussian_summable {L : ℝ} (hL : 0 < L) (t : ℝ) :
    Summable (fun n : ℤ => Real.exp (-((t + L * (n : ℝ)) ^ 2) / 2)) := by
  have hd : Summable (decay (L ^ 2 / 4)) := by
    simpa using summable_decay_pow (show 0 < L ^ 2 / 4 by positivity) 0
  apply Summable.of_norm_bounded (hd.mul_left (Real.exp (t ^ 2 / 2)))
  intro n
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact shifted_gaussian_le L t n

theorem theta_summable (d : ℕ) {L : ℝ} (hL : 0 < L) (z : E d) :
    Summable (fun n : Fin d → ℤ =>
      Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)) := by
  have hd : Summable (decay (L ^ 2 / 4)) := by
    simpa using summable_decay_pow (show 0 < L ^ 2 / 4 by positivity) 0
  have hp := finite_product_summable hd (fun n => (Real.exp_pos _).le) d
  apply Summable.of_norm_bounded
    (hp.mul_left (Real.exp (∑ j : Fin d, (z j) ^ 2 / 2)))
  intro n
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), theta_summand_eq_product]
  calc
    _ ≤ ∏ j : Fin d, (Real.exp ((z j) ^ 2 / 2) * decay (L ^ 2 / 4) (n j)) :=
      Finset.prod_le_prod₀ (fun j _ => (Real.exp_pos _).le)
        (fun j _ => shifted_gaussian_le L (z j) (n j))
    _ = _ := by rw [Finset.prod_mul_distrib, ← Real.exp_sum]

theorem theta_zero_pos (d : ℕ) {L : ℝ} (hL : 0 < L) :
    0 < thetaSum d L 0 := by
  apply (theta_summable d hL 0).tsum_pos (fun _ => (Real.exp_pos _).le) (fun _ => 0)
  exact Real.exp_pos _

private theorem summable_nonneg_fin_product
    (d : ℕ) (f : Fin d → ℤ → ℝ)
    (hf : ∀ j, Summable (f j)) (hn : ∀ j n, 0 ≤ f j n) :
    Summable (fun k : Fin d → ℤ => ∏ j, f j (k j)) := by
  classical
  induction d with
  | zero => exact (hasSum_fintype _).summable
  | succ d ih =>
    have ht : Summable (fun k : Fin d → ℤ => ∏ j : Fin d, f j.succ (k j)) :=
      ih (fun j => f j.succ) (fun j => hf j.succ) (fun j => hn j.succ)
    have hp : Summable (fun p : ℤ × (Fin d → ℤ) =>
        f 0 p.1 * ∏ j : Fin d, f j.succ (p.2 j)) :=
      Summable.mul_of_nonneg (ι := ℤ) (ι' := Fin d → ℤ) (f := f 0)
        (g := fun k : Fin d → ℤ => ∏ j : Fin d, f j.succ (k j)) (hf 0) ht (hn 0)
        (fun k => Finset.prod_nonneg (fun j _ => hn j.succ (k j)))
    have heq :
        ((fun k : Fin (d + 1) → ℤ => ∏ j, f j (k j)) ∘
          (Fin.consEquiv (fun _ : Fin (d + 1) => ℤ))) =
        (fun p : ℤ × (Fin d → ℤ) => f 0 p.1 * ∏ j : Fin d, f j.succ (p.2 j)) := by
      funext p
      simp only [Function.comp_apply, Fin.consEquiv_apply, Fin.prod_univ_succ,
        Fin.cons_zero, Fin.cons_succ]
    apply (Fin.consEquiv (fun _ : Fin (d + 1) => ℤ)).summable_iff.mp
    rw [heq]
    exact hp

private theorem summable_norm_fin_product
    (d : ℕ) (f : Fin d → ℤ → ℂ)
    (hf : ∀ j, Summable (fun n : ℤ => ‖f j n‖)) :
    Summable (fun k : Fin d → ℤ => ‖∏ j, f j (k j)‖) := by
  simpa only [norm_prod] using summable_nonneg_fin_product d
    (fun j n => ‖f j n‖) hf (fun _ _ => norm_nonneg _)

private theorem tsum_fin_product
    (d : ℕ) (f : Fin d → ℤ → ℂ)
    (hf : ∀ j, Summable (fun n : ℤ => ‖f j n‖)) :
    (∑' k : Fin d → ℤ, ∏ j, f j (k j)) = ∏ j, ∑' n : ℤ, f j n := by
  classical
  induction d with
  | zero => simp
  | succ d ih =>
    rw [← (Fin.consEquiv (fun _ : Fin (d + 1) => ℤ)).tsum_eq]
    simp only [Fin.consEquiv_apply, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [← tsum_mul_tsum_of_summable_norm (hf 0)
      (summable_norm_fin_product d (fun j => f j.succ) (fun j => hf j.succ))]
    rw [ih (fun j => f j.succ) (fun j => hf j.succ)]

private def poissonScale (L : ℝ) : ℝ :=
  1 / (L ^ 2 / (2 * Real.pi)) ^ (1 / 2 : ℝ)

private theorem poissonScale_eq {L : ℝ} (hL : 0 < L) :
    poissonScale L = Real.sqrt (2 * Real.pi) / L := by
  unfold poissonScale
  rw [← Real.sqrt_eq_rpow, Real.sqrt_div (sq_nonneg L),
    Real.sqrt_sq_eq_abs, abs_of_pos hL]
  field_simp

private def spectral1Term (L t : ℝ) (k : ℤ) : ℂ :=
  (decay (2 * sourceA L) k : ℂ) *
    Complex.exp (((sourceC L * (k : ℝ) * t : ℝ) : ℂ) * Complex.I)

private theorem spectral1Term_norm (L t : ℝ) (k : ℤ) :
    ‖spectral1Term L t k‖ = decay (2 * sourceA L) k := by
  have hp : 0 < decay (2 * sourceA L) k := Real.exp_pos _
  rw [spectral1Term, norm_mul, Complex.norm_exp_ofReal_mul_I,
    Complex.norm_real, Real.norm_eq_abs, abs_of_pos hp, mul_one]

private theorem spectral1Term_summable_norm {L : ℝ} (hL : 0 < L) (t : ℝ) :
    Summable (fun k => ‖spectral1Term L t k‖) := by
  simp_rw [spectral1Term_norm]
  simpa using summable_decay_pow
    (show 0 < 2 * sourceA L from mul_pos (by norm_num) (sourceA_pos hL.ne')) 0

private theorem poisson_one_dimension {L : ℝ} (hL : 0 < L) (t : ℝ) :
    ((∑' n : ℤ, Real.exp (-((t + L * (n : ℝ)) ^ 2) / 2) : ℝ) : ℂ) =
      (poissonScale L : ℂ) * ∑' k : ℤ, spectral1Term L t k := by
  have ha0 : 0 < L ^ 2 / (2 * Real.pi) := by positivity
  set a : ℂ := ((L ^ 2 / (2 * Real.pi) : ℝ) : ℂ) with ha_def
  set b : ℂ := ((-(t * L / (2 * Real.pi)) : ℝ) : ℂ) with hb_def
  have ha : 0 < a.re := by rw [ha_def, Complex.ofReal_re]; exact ha0
  have h := Complex.tsum_exp_neg_quadratic ha b
  have hlterm (n : ℤ) :
      Complex.exp (-((Real.pi : ℝ) : ℂ) * a * (n : ℂ) ^ 2 +
        2 * ((Real.pi : ℝ) : ℂ) * b * (n : ℂ)) =
      ((Real.exp (t ^ 2 / 2) * Real.exp (-((t + L * (n : ℝ)) ^ 2) / 2) : ℝ) : ℂ) := by
    rw [Complex.ofReal_mul, Complex.ofReal_exp, Complex.ofReal_exp, ← Complex.exp_add]
    congr 1
    rw [ha_def, hb_def]
    push_cast
    field_simp
    ring
  have hrterm (n : ℤ) :
      Complex.exp (-((Real.pi : ℝ) : ℂ) / a * ((n : ℂ) + Complex.I * b) ^ 2) =
      ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) * spectral1Term L t n := by
    unfold spectral1Term decay sourceA sourceC
    rw [Complex.ofReal_exp, Complex.ofReal_exp, ← Complex.exp_add, ← Complex.exp_add]
    congr 1
    have hLc : (L : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hL.ne'
    have hπc : ((Real.pi : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
    rw [ha_def, hb_def]
    push_cast
    field_simp
    linear_combination (-(L : ℂ) ^ 2 * (t : ℂ) ^ 2) * Complex.I_sq
  have hcp : (1 : ℂ) / a ^ (1 / 2 : ℂ) = (poissonScale L : ℂ) := by
    unfold poissonScale
    rw [Complex.ofReal_div, Complex.ofReal_one, Complex.ofReal_cpow ha0.le, ha_def]
    push_cast
    rfl
  have hl : (∑' n : ℤ, Complex.exp (-((Real.pi : ℝ) : ℂ) * a * (n : ℂ) ^ 2 +
        2 * ((Real.pi : ℝ) : ℂ) * b * (n : ℂ))) =
      ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) *
        ((∑' n : ℤ, Real.exp (-((t + L * (n : ℝ)) ^ 2) / 2) : ℝ) : ℂ) := by
    rw [tsum_congr hlterm, ← Complex.ofReal_tsum, tsum_mul_left, Complex.ofReal_mul]
  have hr : (∑' n : ℤ,
      Complex.exp (-((Real.pi : ℝ) : ℂ) / a * ((n : ℂ) + Complex.I * b) ^ 2)) =
      ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) * ∑' n : ℤ, spectral1Term L t n := by
    rw [tsum_congr hrterm, tsum_mul_left]
  rw [hl, hr, hcp] at h
  have hE : ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.exp_pos _).ne'
  apply mul_left_cancel₀ hE
  calc
    _ = _ := h
    _ = _ := by ring

private theorem spectral1Term_prod (d : ℕ) (L : ℝ) (x : Fin d → ℝ)
    (k : Fin d → ℤ) :
    (∏ j, spectral1Term L (x j) (k j)) =
      (sourceWeight d L k : ℂ) * Complex.exp ((phase d L k x : ℂ) * Complex.I) := by
  unfold spectral1Term sourceWeight productDecay
  rw [Finset.prod_mul_distrib, ← Complex.ofReal_prod, ← Complex.exp_sum]
  congr 1
  congr 1
  rw [← Finset.sum_mul]
  simp [phase]

private theorem complex_spectral_summable (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x : Fin d → ℝ) :
    Summable (fun k : Fin d → ℤ => (sourceWeight d L k : ℂ) *
      Complex.exp ((phase d L k x : ℂ) * Complex.I)) := by
  apply Summable.of_norm_bounded (sourceWeight_summable d hL.ne')
  intro k
  have hp : 0 < sourceWeight d L k := by
    rw [sourceWeight_eq_physical]
    exact Real.exp_pos _
  rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hp,
    Complex.norm_exp_ofReal_mul_I, mul_one]

private theorem thetaSum_complex (d : ℕ) {L : ℝ} (hL : 0 < L) (z : E d) :
    (thetaSum d L z : ℂ) = (poissonScale L : ℂ) ^ d *
      ∑' k : Fin d → ℤ, (sourceWeight d L k : ℂ) *
        Complex.exp ((phase d L k (fun i => z i) : ℂ) * Complex.I) := by
  have hn (j : Fin d) :
      Summable (fun n : ℤ => ‖(Real.exp (-((z j + L * (n : ℝ)) ^ 2) / 2) : ℂ)‖) := by
    simpa only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
      using shifted_gaussian_summable hL (z j)
  have ht : (thetaSum d L z : ℂ) = ∑' n : Fin d → ℤ,
      ∏ j, (Real.exp (-((z j + L * (n j : ℝ)) ^ 2) / 2) : ℂ) := by
    rw [thetaSum, Complex.ofReal_tsum]
    apply tsum_congr
    intro n
    rw [theta_summand_eq_product, Complex.ofReal_prod]
  rw [ht, tsum_fin_product d _ hn]
  simp_rw [← Complex.ofReal_tsum, poisson_one_dimension hL]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  congr 1
  rw [← tsum_fin_product d (fun j => spectral1Term L (z j))
    (fun j => spectral1Term_summable_norm hL (z j))]
  apply tsum_congr
  intro k
  exact spectral1Term_prod d L (fun i => z i) k

theorem thetaSum_eq_scaled_cos (d : ℕ) {L : ℝ} (hL : 0 < L) (z : E d) :
    thetaSum d L z = (Real.sqrt (2 * Real.pi) / L) ^ d *
      ∑' k : Fin d → ℤ, sourceWeight d L k *
        Real.cos (phase d L k (fun i => z i)) := by
  have h := congrArg Complex.re (thetaSum_complex d hL z)
  rw [Complex.ofReal_re, ← Complex.ofReal_pow, Complex.re_ofReal_mul,
    Complex.re_tsum (complex_spectral_summable d hL (fun i => z i))] at h
  simp only [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re] at h
  rw [poissonScale_eq hL] at h
  exact h

theorem thetaSum_zero_eq (d : ℕ) {L : ℝ} (hL : 0 < L) :
    thetaSum d L 0 = (Real.sqrt (2 * Real.pi) / L) ^ d * sourceZ d L := by
  simpa [phase, sourceZ] using thetaSum_eq_scaled_cos d hL (0 : E d)

theorem sourceSpectralKernel_eq_periodicCovariance
    (d : ℕ) {L : ℝ} (hL : 0 < L) (z : E d) :
    I4Kernel.sourceSpectralKernel d L (fun i => z i) = periodicCovariance d L z := by
  have hA : (Real.sqrt (2 * Real.pi) / L) ^ d ≠ 0 := by positivity
  rw [periodicCovariance, thetaSum_eq_scaled_cos d hL z, thetaSum_zero_eq d hL]
  rw [mul_div_mul_left _ _ hA]
  unfold I4Kernel.sourceSpectralKernel
  rw [← tsum_div_const]
  apply tsum_congr
  intro k
  ring

end I4ThetaBridge
