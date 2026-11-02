import SourceCovariance
import W3Scratch.Side24Poisson

noncomputable section
open I4Weights I4Source I4Kernel I4SourceCovariance MeasureTheory ProbabilityTheory
open scoped BigOperators
namespace I4Spatial

/-- Literal Euclidean coordinate square sum, not the Pi sup norm. -/
def sourceImageSum (d : ℕ) (L : ℝ) (z : Fin d → ℝ) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(∑ i, (z i + L * (n i : ℝ)) ^ 2) / 2)

def sourceSpatialKernel (d : ℕ) (L : ℝ) (z : Fin d → ℝ) : ℝ :=
  sourceImageSum d L z / sourceImageSum d L 0

set_option maxHeartbeats 800000 in
private theorem tensor_tsum {R : Type*} [NormedCommRing R] [CompleteSpace R]
    (d : ℕ) (f : Fin d → ℤ → R) (hf : ∀ i, Summable (fun n => ‖f i n‖)) :
    Summable (fun k : Fin d → ℤ => ‖∏ i, f i (k i)‖) ∧
      (∑' k : Fin d → ℤ, ∏ i, f i (k i)) = ∏ i, ∑' n : ℤ, f i n := by
  classical
  induction d with
  | zero =>
    constructor
    · exact (hasSum_fintype _).summable
    · simp [tsum_fintype]
  | succ d ih =>
    obtain ⟨hn, ht⟩ := ih (fun i => f i.succ) (fun i => hf i.succ)
    let e := Fin.consEquiv (fun _ : Fin (d + 1) => ℤ)
    have hb : Summable (fun p : ℤ × (Fin d → ℤ) =>
        ‖f 0 p.1‖ * ‖∏ i, f i.succ (p.2 i)‖) :=
      (hf 0).mul_of_nonneg hn (fun _ => norm_nonneg _) (fun _ => norm_nonneg _)
    have hp : Summable (fun p : ℤ × (Fin d → ℤ) =>
        ‖f 0 p.1 * ∏ i, f i.succ (p.2 i)‖) :=
      Summable.of_nonneg_of_le (fun p => norm_nonneg _)
        (fun p => norm_mul_le (f 0 p.1) (∏ i, f i.succ (p.2 i))) hb
    have hs : Summable (fun k : Fin (d + 1) → ℤ => ‖∏ i, f i (k i)‖) := by
      apply e.summable_iff.mp
      change Summable (fun p : ℤ × (Fin d → ℤ) => ‖∏ i, f i ((e p) i)‖)
      simpa only [e, Function.comp_apply, Fin.consEquiv_apply, Fin.prod_univ_succ,
        Fin.cons_zero, Fin.cons_succ] using hp
    have h0 : Summable (f 0) := (hf 0).of_norm
    have htail : Summable (fun k : Fin d → ℤ => ∏ i, f i.succ (k i)) := hn.of_norm
    have hpair : Summable (fun p : ℤ × (Fin d → ℤ) =>
        f 0 p.1 * ∏ i, f i.succ (p.2 i)) := hp.of_norm
    refine ⟨hs, ?_⟩
    calc
      (∑' k : Fin (d + 1) → ℤ, ∏ i, f i (k i)) =
          ∑' p : ℤ × (Fin d → ℤ), f 0 p.1 * ∏ i, f i.succ (p.2 i) := by
        rw [← e.tsum_eq (fun k => ∏ i, f i (k i))]
        simp only [e, Fin.consEquiv_apply, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]
      _ = (∑' n : ℤ, f 0 n) * ∑' k : Fin d → ℤ, ∏ i, f i.succ (k i) :=
        (h0.tsum_mul_tsum htail hpair).symm
      _ = _ := by rw [ht, Fin.prod_univ_succ]

private theorem shifted_summable {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    Summable (fun n : ℤ => Real.exp (-(t + L * (n : ℝ)) ^ 2 / 2)) := by
  simpa only [Function.iterate_zero, id_eq, W3Scratch.Side24.gauss, neg_div] using
    W3Scratch.Side24.summable_thetaD_terms hL 0 t

private theorem image_term_prod (d : ℕ) (L : ℝ) (z : Fin d → ℝ) (n : Fin d → ℤ) :
    Real.exp (-(∑ i, (z i + L * (n i : ℝ)) ^ 2) / 2) =
      ∏ i, Real.exp (-(z i + L * (n i : ℝ)) ^ 2 / 2) := by
  rw [← Real.exp_sum]
  congr 1
  rw [← Finset.sum_div, Finset.sum_neg_distrib]

private theorem image_eq_prod (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    sourceImageSum d L z = ∏ i, W3Scratch.Side24.theta L (z i) := by
  unfold sourceImageSum
  simp_rw [image_term_prod]
  exact (tensor_tsum d (fun i n => Real.exp (-(z i + L * (n : ℝ)) ^ 2 / 2))
    (fun i => (shifted_summable hL (z i)).norm)).2

private theorem weight_eq_decay {L : ℝ} (hL : L ≠ 0) (k : ℤ) :
    W3Scratch.Side24.weight L k = decay (2 * sourceA L) k := by
  unfold W3Scratch.Side24.weight W3Scratch.Side24.atom decay sourceA
  congr 1
  field_simp

private theorem weight_prod (d : ℕ) {L : ℝ} (hL : L ≠ 0) (k : Fin d → ℤ) :
    (∏ i, W3Scratch.Side24.weight L (k i)) = sourceWeight d L k := by
  unfold sourceWeight productDecay
  simp_rw [weight_eq_decay hL]

private def complexFactor (L t : ℝ) (n : ℤ) : ℂ :=
  (W3Scratch.Side24.weight L n : ℂ) *
    Complex.exp (((W3Scratch.Side24.atom L n * t : ℝ) : ℂ) * Complex.I)

private theorem complexFactor_norm (L t : ℝ) (n : ℤ) :
    ‖complexFactor L t n‖ = W3Scratch.Side24.weight L n := by
  rw [complexFactor, norm_mul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos (W3Scratch.Side24.weight_pos L n), Complex.norm_exp_ofReal_mul_I, mul_one]

private theorem complexFactor_summable_norm {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    Summable (fun n : ℤ => ‖complexFactor L t n‖) := by
  simp_rw [complexFactor_norm]
  simpa only [pow_zero, mul_one] using W3Scratch.Side24.summable_weight_mul_atom_pow hL 0

private theorem complexFactor_tsum {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    (∑' n : ℤ, complexFactor L t n) =
      ((∑' n : ℤ, W3Scratch.Side24.weight L n *
        Real.cos (W3Scratch.Side24.atom L n * t) : ℝ) : ℂ) := by
  have hs := (complexFactor_summable_norm hL t).of_norm
  have ho (n : ℤ) : W3Scratch.Side24.weight L (-n) *
      Real.sin (W3Scratch.Side24.atom L (-n) * t) =
      -(W3Scratch.Side24.weight L n * Real.sin (W3Scratch.Side24.atom L n * t)) := by
    have ha : W3Scratch.Side24.atom L (-n) = -W3Scratch.Side24.atom L n := by
      simp only [W3Scratch.Side24.atom, Int.cast_neg, mul_neg, neg_div]
    rw [W3Scratch.Side24.weight, ha, neg_sq, neg_mul, Real.sin_neg, mul_neg]
    rfl
  have hz : (∑' n : ℤ, W3Scratch.Side24.weight L n *
      Real.sin (W3Scratch.Side24.atom L n * t)) = 0 := by
    have hh := tsum_comp_neg (fun n : ℤ => W3Scratch.Side24.weight L n *
      Real.sin (W3Scratch.Side24.atom L n * t))
    simp_rw [ho] at hh
    rw [tsum_neg] at hh
    linarith
  apply Complex.ext
  · rw [Complex.re_tsum hs, Complex.ofReal_re]
    apply tsum_congr
    intro n
    exact Complex.re_ofReal_mul _ _ |>.trans (congrArg
      (fun r => W3Scratch.Side24.weight L n * r) (Complex.exp_ofReal_mul_I_re _))
  · rw [Complex.im_tsum hs, Complex.ofReal_im]
    simpa only [complexFactor, Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_im] using hz

private theorem complex_product_term (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (z : Fin d → ℝ) (k : Fin d → ℤ) :
    (∏ i, complexFactor L (z i) (k i)) =
      (sourceWeight d L k : ℂ) * Complex.exp (((phase d L k z : ℝ) : ℂ) * Complex.I) := by
  classical
  unfold complexFactor
  rw [Finset.prod_mul_distrib]
  have hw : (∏ i, (W3Scratch.Side24.weight L (k i) : ℂ)) = (sourceWeight d L k : ℂ) := by
    exact_mod_cast weight_prod d hL k
  rw [hw, ← Complex.exp_sum, ← Finset.sum_mul]
  congr 2
  have hp : (∑ i, W3Scratch.Side24.atom L (k i) * z i) = phase d L k z := by
    unfold W3Scratch.Side24.atom phase sourceC
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [← hp]
  (push_cast; rfl)

private theorem rawFourier_eq_prod (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    (∑' k : Fin d → ℤ, sourceWeight d L k * Real.cos (phase d L k z)) =
      ∏ i, ∑' n : ℤ, W3Scratch.Side24.weight L n *
        Real.cos (W3Scratch.Side24.atom L n * z i) := by
  classical
  obtain ⟨hn, ht⟩ := tensor_tsum d (fun i n => complexFactor L (z i) n)
    (fun i => complexFactor_summable_norm hL (z i))
  have hs : Summable (fun k : Fin d → ℤ =>
      (sourceWeight d L k : ℂ) * Complex.exp (((phase d L k z : ℝ) : ℂ) * Complex.I)) := by
    simpa only [complex_product_term d hL z] using hn.of_norm
  have he : (∑' k : Fin d → ℤ, (sourceWeight d L k : ℂ) *
      Complex.exp (((phase d L k z : ℝ) : ℂ) * Complex.I)) =
      ((∏ i, ∑' n : ℤ, W3Scratch.Side24.weight L n *
        Real.cos (W3Scratch.Side24.atom L n * z i) : ℝ) : ℂ) := by
    calc
      _ = ∑' k : Fin d → ℤ, ∏ i, complexFactor L (z i) (k i) :=
        tsum_congr (fun k => (complex_product_term d hL z k).symm)
      _ = ∏ i, ∑' n : ℤ, complexFactor L (z i) n := ht
      _ = _ := by simp_rw [complexFactor_tsum hL]; push_cast; rfl
  have hr := congrArg Complex.re he
  rw [Complex.re_tsum hs, Complex.ofReal_re] at hr
  simpa only [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re] using hr

private theorem image_eq_prefactor_raw (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    sourceImageSum d L z = (1 / (L ^ 2 / (2 * Real.pi)) ^ (1 / 2 : ℝ)) ^ d *
      ∑' k : Fin d → ℤ, sourceWeight d L k * Real.cos (phase d L k z) := by
  rw [image_eq_prod d hL z]
  simp_rw [W3Scratch.Side24.theta_eq_cos_series hL]
  rw [Finset.prod_mul_distrib, ← rawFourier_eq_prod d hL z]
  simp

theorem image_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    Summable (fun n : Fin d → ℤ =>
      Real.exp (-(∑ i, (z i + L * (n i : ℝ)) ^ 2) / 2)) := by
  simp_rw [image_term_prod]
  exact (tensor_tsum d (fun i n => Real.exp (-(z i + L * (n : ℝ)) ^ 2 / 2))
    (fun i => (shifted_summable hL (z i)).norm)).1.of_norm

theorem image_zero_pos (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    0 < sourceImageSum d L 0 := by
  exact (image_summable d hL 0).tsum_pos (fun n => (Real.exp_pos _).le) 0 (Real.exp_pos _)

theorem spectralKernel_eq_spatialKernel (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    sourceSpectralKernel d L z = sourceSpatialKernel d L z := by
  have hc : (1 / (L ^ 2 / (2 * Real.pi)) ^ (1 / 2 : ℝ)) ^ d ≠ 0 := by
    apply pow_ne_zero
    apply one_div_ne_zero
    exact (Real.rpow_pos_of_pos (by positivity) _).ne'
  have hz : sourceImageSum d L 0 =
      (1 / (L ^ 2 / (2 * Real.pi)) ^ (1 / 2 : ℝ)) ^ d * sourceZ d L := by
    rw [image_eq_prefactor_raw d hL 0]
    simp only [phase, Pi.zero_apply, mul_zero, Finset.sum_const_zero, Real.cos_zero, mul_one]
    rfl
  unfold sourceSpatialKernel sourceSpectralKernel
  rw [image_eq_prefactor_raw d hL z, hz, mul_div_mul_left _ _ hc]
  rw [← tsum_div_const]
  apply tsum_congr
  intro k
  ring

theorem sourceField_covariance_spatial (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x y : Fin d → ℝ) :
    cov[fun ω => sourceField d L ω x, fun ω => sourceField d L ω y; sourceLaw d] =
      sourceSpatialKernel d L (x - y) := by
  rw [sourceField_covariance d hL x y, spectralKernel_eq_spatialKernel d hL]

end I4Spatial

