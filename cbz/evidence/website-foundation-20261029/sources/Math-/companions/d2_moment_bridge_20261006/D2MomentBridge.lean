import ResearchFormalCoreR1.D2Schur
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-!
# Actual-law d2 moment bridge

Dylan Roy — delegated AI work. Author: OpenAI / GPT-6 Astra Pro.
This companion uses actual integrals, not assigned scalar moments. It does not
construct a periodic spectral law, a Gaussian field, or a Palm conditioning.
Second/fourth/sixth integrability and positive atom masses remain explicit.
-/

noncomputable section
namespace D2MomentBridge

open MeasureTheory ResearchFormalCoreR1
set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

def moment (μ : Measure Ω) (X : Ω → ℝ) (n : ℕ) : ℝ := ∫ w, X w ^ n ∂μ

def cubicResidual (c x : ℝ) : ℝ := x ^ 3 - c * x

theorem residual_sq_expand (c x : ℝ) :
    cubicResidual c x ^ 2 = x ^ 6 - (2 * c) * x ^ 4 + c ^ 2 * x ^ 2 := by
  unfold cubicResidual
  ring

theorem residual_sq_integrable (μ : Measure Ω) (X : Ω → ℝ) (c : ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ) :
    Integrable (fun w => cubicResidual c (X w) ^ 2) μ := by
  simp_rw [residual_sq_expand]
  exact (h6.sub (h4.const_mul (2 * c))).add (h2.const_mul (c ^ 2))

/-- The exact residual identity; integrability prevents Bochner totalization. -/
theorem residual_integral_eq_delta (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (hm2 : moment μ X 2 ≠ 0) :
    (∫ w, cubicResidual (moment μ X 4 / moment μ X 2) (X w) ^ 2 ∂μ) =
      d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) := by
  simp_rw [residual_sq_expand]
  have hs : Integrable
      (fun w => X w ^ 6 - (2 * (moment μ X 4 / moment μ X 2)) * X w ^ 4) μ :=
    h6.sub (h4.const_mul (2 * (moment μ X 4 / moment μ X 2)))
  have hl : Integrable
      (fun w => (moment μ X 4 / moment μ X 2) ^ 2 * X w ^ 2) μ :=
    h2.const_mul ((moment μ X 4 / moment μ X 2) ^ 2)
  rw [integral_add hs hl,
    integral_sub h6 (h4.const_mul (2 * (moment μ X 4 / moment μ X 2))),
    integral_const_mul, integral_const_mul]
  change moment μ X 6 - (2 * (moment μ X 4 / moment μ X 2)) * moment μ X 4 +
    (moment μ X 4 / moment μ X 2) ^ 2 * moment μ X 2 =
      d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6)
  unfold d2Delta
  field_simp [hm2]
  ring

theorem delta_nonneg (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (hm2 : moment μ X 2 ≠ 0) :
    0 ≤ d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) := by
  rw [← residual_integral_eq_delta μ X h2 h4 h6 hm2]
  exact integral_nonneg (fun w => sq_nonneg _)

theorem delta_eq_zero_iff_residual (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (hm2 : moment μ X 2 ≠ 0) :
    d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) = 0 ↔
      ∀ᵐ w ∂μ, cubicResidual (moment μ X 4 / moment μ X 2) (X w) = 0 := by
  rw [← residual_integral_eq_delta μ X h2 h4 h6 hm2]
  have hz := integral_eq_zero_iff_of_nonneg_ae
    (Filter.Eventually.of_forall (fun w => sq_nonneg
      (cubicResidual (moment μ X 4 / moment μ X 2) (X w))))
    (residual_sq_integrable μ X _ h2 h4 h6)
  constructor
  · intro h
    filter_upwards [hz.mp h] with w hw
    exact sq_eq_zero_iff.mp hw
  · intro h
    apply hz.mpr
    filter_upwards [h] with w hw
    change cubicResidual (moment μ X 4 / moment μ X 2) (X w) ^ 2 = 0
    exact sq_eq_zero_iff.mpr hw

theorem cubicResidual_eq_zero_iff (c x : ℝ) :
    cubicResidual c x = 0 ↔ x = 0 ∨ x ^ 2 = c := by
  have h : cubicResidual c x = x * (x ^ 2 - c) := by
    unfold cubicResidual
    ring
  rw [h, mul_eq_zero, sub_eq_zero]

/-- Zero may carry positive mass: the nonzero support, not all |X|, has one radius. -/
theorem delta_eq_zero_iff_support (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (hm2 : moment μ X 2 ≠ 0) :
    d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) = 0 ↔
      ∀ᵐ w ∂μ, X w = 0 ∨ X w ^ 2 = moment μ X 4 / moment μ X 2 := by
  rw [delta_eq_zero_iff_residual μ X h2 h4 h6 hm2]
  simp_rw [cubicResidual_eq_zero_iff]

theorem delta_pos_iff_not_support (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (hm2 : moment μ X 2 ≠ 0) :
    0 < d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) ↔
      ¬ (∀ᵐ w ∂μ, X w = 0 ∨ X w ^ 2 = moment μ X 4 / moment μ X 2) := by
  have hn := delta_nonneg μ X h2 h4 h6 hm2
  have he := delta_eq_zero_iff_support μ X h2 h4 h6 hm2
  constructor
  · intro hp hs
    exact (ne_of_gt hp) (he.mpr hs)
  · intro hs
    by_contra hp
    exact hs (he.mp (le_antisymm (le_of_not_gt hp) hn))

/-- Positive mass turns an a.e. value property into a property of that atom. -/
theorem property_of_ae_of_atom (μ : Measure Ω) (X : Ω → ℝ) (p : ℝ → Prop)
    (h : ∀ᵐ w ∂μ, p (X w)) (a : ℝ) (ha : μ {w | X w = a} ≠ 0) : p a := by
  by_contra hp
  apply ha
  have hz : μ {w | ¬ p (X w)} = 0 := ae_iff.mp h
  refine measure_mono_null ?_ hz
  intro w hw
  change X w = a at hw
  change ¬ p (X w)
  simpa only [hw] using hp

theorem moment_two_pos_of_atom (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (a : ℝ) (ha0 : a ≠ 0) (ha : μ {w | X w = a} ≠ 0) : 0 < moment μ X 2 := by
  have hn : 0 ≤ moment μ X 2 := integral_nonneg (fun w => sq_nonneg _)
  by_contra hp
  have he : moment μ X 2 = 0 := le_antisymm (le_of_not_gt hp) hn
  have hz := (integral_eq_zero_iff_of_nonneg_ae
    (Filter.Eventually.of_forall (fun w => sq_nonneg (X w))) h2).mp he
  have hz' : ∀ᵐ w ∂μ, X w ^ 2 = 0 := hz
  have ha2 := property_of_ae_of_atom μ X (fun x => x ^ 2 = 0) hz' a ha
  exact ha0 (sq_eq_zero_iff.mp ha2)

theorem delta_pos_of_two_atoms (μ : Measure Ω) (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (a b : ℝ) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hab : a ^ 2 ≠ b ^ 2)
    (ha : μ {w | X w = a} ≠ 0) (hb : μ {w | X w = b} ≠ 0) :
    0 < d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) := by
  have hm2 := moment_two_pos_of_atom μ X h2 a ha0 ha
  apply (delta_pos_iff_not_support μ X h2 h4 h6 (ne_of_gt hm2)).mpr
  intro hs
  have hsa := property_of_ae_of_atom μ X
    (fun x => x = 0 ∨ x ^ 2 = moment μ X 4 / moment μ X 2) hs a ha
  have hsb := property_of_ae_of_atom μ X
    (fun x => x = 0 ∨ x ^ 2 = moment μ X 4 / moment μ X 2) hs b hb
  exact hab ((hsa.resolve_left ha0).trans (hsb.resolve_left hb0).symm)

theorem square_gap_integrable (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) :
    Integrable (fun w => (X w ^ 2 - moment μ X 2) ^ 2) μ := by
  have hf : (fun w => (X w ^ 2 - moment μ X 2) ^ 2) =
      (fun w => X w ^ 4 - (2 * moment μ X 2) * X w ^ 2 + (moment μ X 2) ^ 2) := by
    funext w
    ring
  rw [hf]
  exact (h4.sub (h2.const_mul _)).add (integrable_const _)

/-- Probability normalization is essential for the unscaled m4-m2^2 gap. -/
theorem square_gap_integral (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) :
    (∫ w, (X w ^ 2 - moment μ X 2) ^ 2 ∂μ) = moment μ X 4 - (moment μ X 2) ^ 2 := by
  have hf : (fun w => (X w ^ 2 - moment μ X 2) ^ 2) =
      (fun w => X w ^ 4 - (2 * moment μ X 2) * X w ^ 2 + (moment μ X 2) ^ 2) := by
    funext w
    ring
  have hs : Integrable
      (fun w => X w ^ 4 - (2 * moment μ X 2) * X w ^ 2) μ :=
    h4.sub (h2.const_mul (2 * moment μ X 2))
  have hc : Integrable (fun _ : Ω => (moment μ X 2) ^ 2) μ := integrable_const _
  rw [hf, integral_add hs hc,
    integral_sub h4 (h2.const_mul (2 * moment μ X 2)), integral_const_mul]
  simp only [integral_const, probReal_univ, one_smul]
  change moment μ X 4 - (2 * moment μ X 2) * moment μ X 2 + (moment μ X 2) ^ 2 = _
  ring

theorem fourth_gt_second_sq_of_two_atoms (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (a b : ℝ) (hab : a ^ 2 ≠ b ^ 2)
    (ha : μ {w | X w = a} ≠ 0) (hb : μ {w | X w = b} ≠ 0) :
    (moment μ X 2) ^ 2 < moment μ X 4 := by
  have hn : 0 ≤ moment μ X 4 - (moment μ X 2) ^ 2 := by
    rw [← square_gap_integral μ X h2 h4]
    exact integral_nonneg (fun w => sq_nonneg _)
  by_contra hp
  have he : moment μ X 4 - (moment μ X 2) ^ 2 = 0 := by linarith
  rw [← square_gap_integral μ X h2 h4] at he
  have hz := (integral_eq_zero_iff_of_nonneg_ae
    (Filter.Eventually.of_forall (fun w => sq_nonneg (X w ^ 2 - moment μ X 2)))
    (square_gap_integrable μ X h2 h4)).mp he
  have hs : ∀ᵐ w ∂μ, X w ^ 2 = moment μ X 2 := by
    filter_upwards [hz] with w hw
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hw)
  exact hab ((property_of_ae_of_atom μ X (fun x => x ^ 2 = moment μ X 2) hs a ha).trans
    (property_of_ae_of_atom μ X (fun x => x ^ 2 = moment μ X 2) hs b hb).symm)

/-- The actual-law sufficient assumptions now supply all strict moment premises. -/
theorem tau_pos_of_two_atoms (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ)
    (a b q : ℝ) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hab : a ^ 2 ≠ b ^ 2)
    (ha : μ {w | X w = a} ≠ 0) (hb : μ {w | X w = b} ≠ 0)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) :
    0 < d2Tau (moment μ X 2) (moment μ X 4) (moment μ X 6) q :=
  d2_tau_pos _ _ _ q (moment_two_pos_of_atom μ X h2 a ha0 ha)
    (fourth_gt_second_sq_of_two_atoms μ X h2 h4 a b hab ha hb)
    (delta_pos_of_two_atoms μ X h2 h4 h6 a b ha0 hb0 hab ha hb) hq0 hq1

/-- Moments of the zero/+-1 countermodel; the actual finite law is tested separately. -/
theorem zero_atom_counterexample :
    (1 / 2 : ℝ) - (1 / 2) ^ 2 = 1 / 4 ∧ d2Delta (1 / 2) (1 / 2) (1 / 2) = 0 := by
  norm_num [d2Delta]

end D2MomentBridge
