import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.Positivity

/-!
# Gaussian reference positivity and an endpoint-box weight floor

The literal continuous normalized endpoint weight is from LM006 stability,
sections 1–2. These components do not construct the conditioned field law,
its endpoint convergence, a positive box mass, or the full normalizer floor.
See ../SOURCE_GAUSSIAN_NORMALIZER.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

set_option autoImplicit false

/-- The six coordinates are aM,aS,bM,bS,cM,cS. For physical positive radius,
the source identifies this continuous expression with its typed W_r/r². -/
noncomputable def gn_normalizedWeight (r : ℝ) (v : Fin 6 → ℝ) : ℝ :=
  max (max (-v 0) 0 * max (-v 4) 0 - r * (v 2)^2) 0 *
    max (r * (v 3)^2 - v 1 * v 5) 0

/-- The contact negative-curvature reference has an integrable, strictly
positive squared negative part, with arbitrary mean and nonzero variance. -/
theorem gn_gaussian_reference_integrable_pos (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    Integrable (fun q : ℝ => (max (-q) 0)^2) (gaussianReal m v) ∧
      0 < ∫ q : ℝ, (max (-q) 0)^2 ∂gaussianReal m v := by
  have hsq : Integrable (fun q : ℝ => q^2) (gaussianReal m v) := by
    simpa using (memLp_id_gaussianReal (μ := m) (v := v) 2).integrable_sq
  have hint : Integrable (fun q : ℝ => (max (-q) 0)^2) (gaussianReal m v) := by
    apply hsq.mono' (by fun_prop)
    filter_upwards [] with q
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    rcases le_total 0 q with hq | hq
    · rw [max_eq_right (neg_nonpos.mpr hq)]
      simpa using sq_nonneg q
    · rw [max_eq_left (neg_nonneg.mpr hq)]
      nlinarith
  have hsupp : Function.support (fun q : ℝ => (max (-q) 0)^2) = Set.Iio 0 := by
    ext q
    simp only [Function.mem_support, Set.mem_Iio]
    constructor
    · intro h
      by_contra hn
      have hq : 0 ≤ q := le_of_not_gt hn
      apply h
      simp [max_eq_right (neg_nonpos.mpr hq)]
    · intro hq
      have hp : 0 < (max (-q) 0)^2 := sq_pos_of_pos (lt_max_of_lt_left (neg_pos.mpr hq))
      exact hp.ne'
  have hmass : 0 < gaussianReal m v (Set.Iio 0) := by
    apply pos_iff_ne_zero.mpr
    intro hzero
    have hvol := (gaussianReal_absolutelyContinuous' m hv) hzero
    simp at hvol
  refine ⟨hint, ?_⟩
  apply (integral_pos_iff_support_of_nonneg (fun q => sq_nonneg _) hint).mpr
  simpa only [hsupp] using hmass

/-- An explicit open endpoint box has a uniform positive normalized weight.
This is a pointwise component, with no claimed probability mass for the box. -/
theorem gn_open_box_floor
    (r K : ℝ) (v : Fin 6 → ℝ)
    (hr : 0 ≤ r) (hK : 0 < K) (hsmall : r * K^2 ≤ 1/4)
    (haM : v 0 ∈ Set.Ioo (-5/4) (-3/4))
    (haS : v 1 ∈ Set.Ioo (3/4) (5/4))
    (hbM : |v 2| < K) (_hbS : |v 3| < K)
    (hcM : v 4 ∈ Set.Ioo (-2) (-1))
    (hcS : v 5 ∈ Set.Ioo (-2) (-1)) :
    (3 : ℝ)/8 ≤ gn_normalizedWeight r v := by
  have ha0 : 3/4 ≤ -v 0 := by linarith [haM.2]
  have hc4 : 1 ≤ -v 4 := by linarith [hcM.2]
  have ha1 : 3/4 ≤ v 1 := haS.1.le
  have hc5 : 1 ≤ -v 5 := by linarith [hcS.2]
  have hAM : max (-v 0) 0 = -v 0 := max_eq_left (by linarith)
  have hCM : max (-v 4) 0 = -v 4 := max_eq_left (by linarith)
  have hb2 : (v 2)^2 ≤ K^2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg (v 2)) hK.le).2 hbM.le
  have hrb : r * (v 2)^2 ≤ 1/4 :=
    (mul_le_mul_of_nonneg_left hb2 hr).trans hsmall
  have hprodM : (3 : ℝ)/4 ≤ (-v 0) * (-v 4) := by
    have h := mul_le_mul ha0 hc4 (by norm_num : (0 : ℝ) ≤ 1) (by linarith : 0 ≤ -v 0)
    norm_num at h ⊢
    exact h
  have hprodS : (3 : ℝ)/4 ≤ -(v 1 * v 5) := by
    have h := mul_le_mul ha1 hc5 (by norm_num : (0 : ℝ) ≤ 1) (by linarith : 0 ≤ v 1)
    norm_num at h
    nlinarith
  have hF : (1 : ℝ)/2 ≤ max (max (-v 0) 0 * max (-v 4) 0 - r * (v 2)^2) 0 := by
    rw [hAM, hCM]
    exact (show (1 : ℝ)/2 ≤ (-v 0) * (-v 4) - r * (v 2)^2 by linarith).trans
      (le_max_left _ _)
  have hG : (3 : ℝ)/4 ≤ max (r * (v 3)^2 - v 1 * v 5) 0 := by
    have hp : 0 ≤ r * (v 3)^2 := mul_nonneg hr (sq_nonneg _)
    exact (show (3 : ℝ)/4 ≤ r * (v 3)^2 - v 1 * v 5 by linarith).trans
      (le_max_left _ _)
  have h := mul_le_mul hF hG (by norm_num : (0 : ℝ) ≤ 3/4)
    (le_max_right _ _)
  norm_num at h
  exact h

end ResearchFormalCoreR1
