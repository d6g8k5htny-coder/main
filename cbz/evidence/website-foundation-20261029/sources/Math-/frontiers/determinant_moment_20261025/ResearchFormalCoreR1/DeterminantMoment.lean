import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Topology.Instances.Matrix
import Mathlib.MeasureTheory.Order.Group.Lattice
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.Push

namespace ResearchFormalCoreR1

set_option autoImplicit false

open MeasureTheory
open scoped BigOperators

abbrev DM2Jet := Fin 9 → ℝ

def dm2Radius (j : DM2Jet) : ℝ := 1 + ∑ i, |j i|

noncomputable def dm2HM (r : ℝ) (j : DM2Jet) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![-r + r^2*j 4/12, -j 1*r/2 + r^2*j 5/12;
     -j 1*r/2 + r^2*j 5/12, j 0-r*j 2/2+r^2*j 6/8]

noncomputable def dm2HS (r : ℝ) (j : DM2Jet) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![r + r^2*j 4/12, j 1*r/2 + r^2*j 5/12;
     j 1*r/2 + r^2*j 5/12, j 0+r*j 2/2+r^2*j 6/8]

/-- The source's negative-definite condition, stated as the quadratic form. -/
def dm2Negative (H : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ v : Fin 2 → ℝ, v ≠ 0 → ∑ i, ∑ k, v i * H i k * v k < 0

noncomputable def dm2Weight (r : ℝ) (j : DM2Jet) : ℝ := by
  classical
  exact if dm2Negative (dm2HM r j) ∧ (dm2HS r j).det < 0 then
    |(dm2HM r j).det * (dm2HS r j).det| else 0

private lemma dm2_scale (r x : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1) :
    |r*x| ≤ |x| := by
  rw [abs_mul, abs_of_nonneg hr]
  nlinarith [abs_nonneg x]

private lemma dm2_factors (r A B C R : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hR : 0 ≤ R) (hA : |A| ≤ R) (hB : |B| ≤ R) (hC : |C| ≤ R) :
    |A*C-r*B^2| ≤ 2*R^2 := by
  have hAC : |A*C| ≤ R^2 := by
    rw [abs_mul, pow_two]
    exact mul_le_mul hA hC (abs_nonneg _) hR
  have hBB : B^2 ≤ R^2 := by nlinarith [sq_abs B, abs_nonneg B]
  have hrBB : |r*B^2| ≤ R^2 := by
    rw [abs_of_nonneg (mul_nonneg hr (sq_nonneg B))]
    nlinarith [sq_nonneg B]
  exact (abs_sub _ _).trans (by linarith)

/-- Exact untyped determinant domination; no jet-coordinate or type assumptions. -/
theorem dm2_untyped (r : ℝ) (j : DM2Jet) (hr : 0 < r) (hr1 : r ≤ 1) :
    ((dm2HM r j).det * (dm2HS r j).det)^2 ≤ 16*r^4*dm2Radius j^8 := by
  let R := dm2Radius j
  have hR : 0 ≤ R := by dsimp [R, dm2Radius]; positivity
  have hsum : (∑ i, |j i|) = |j 0|+|j 1|+|j 2|+|j 3|+|j 4|+|j 5|+|j 6|+|j 7|+|j 8| := by
    simp [Fin.sum_univ_succ]; ring
  have hbudget : 1+|j 0|+|j 1|+|j 2|+|j 4|+|j 5|+|j 6| ≤ R := by
    dsimp [R, dm2Radius]; rw [hsum]
    linarith [abs_nonneg (j 3), abs_nonneg (j 7), abs_nonneg (j 8)]
  have hscale (i : Fin 9) : r*|j i| ≤ |j i| := by nlinarith [abs_nonneg (j i)]
  have hscale2 (i : Fin 9) : r^2*|j i| ≤ |j i| := by
    exact mul_le_of_le_one_left (abs_nonneg _) (by nlinarith)
  have haM : |-1+r*j 4/12| ≤ R := by
    calc
      _ ≤ 1+|j 4| := by
        have := abs_add_le (-1 : ℝ) (r*j 4/12)
        rw [abs_div] at this
        norm_num [abs_of_pos hr] at this
        linarith [hscale 4, abs_nonneg (j 4)]
      _ ≤ R := by linarith [abs_nonneg (j 0), abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 5), abs_nonneg (j 6)]
  have haS : |1+r*j 4/12| ≤ R := by
    have := abs_add_le (1 : ℝ) (r*j 4/12)
    rw [abs_div] at this
    norm_num [abs_of_pos hr] at this
    linarith [hscale 4, abs_nonneg (j 0), abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 4), abs_nonneg (j 5), abs_nonneg (j 6)]
  have hbM : |-j 1/2+r*j 5/12| ≤ R := by
    have := abs_add_le (-j 1/2) (r*j 5/12)
    simp only [abs_div, abs_neg] at this
    norm_num [abs_of_pos hr] at this
    linarith [hscale 5, abs_nonneg (j 0), abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 4), abs_nonneg (j 5), abs_nonneg (j 6)]
  have hbS : |j 1/2+r*j 5/12| ≤ R := by
    have := abs_add_le (j 1/2) (r*j 5/12)
    simp only [abs_div] at this
    norm_num [abs_of_pos hr] at this
    linarith [hscale 5, abs_nonneg (j 0), abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 4), abs_nonneg (j 5), abs_nonneg (j 6)]
  have hcM : |j 0-r*j 2/2+r^2*j 6/8| ≤ R := by
    have h1 := abs_sub (j 0) (r*j 2/2)
    have h2 := abs_add_le (j 0-r*j 2/2) (r^2*j 6/8)
    simp only [abs_div] at h1 h2
    norm_num [abs_of_pos hr] at h1 h2
    linarith [hscale 2, hscale2 6, abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 4), abs_nonneg (j 5), abs_nonneg (j 6)]
  have hcS : |j 0+r*j 2/2+r^2*j 6/8| ≤ R := by
    have h1 := abs_add_le (j 0) (r*j 2/2)
    have h2 := abs_add_le (j 0+r*j 2/2) (r^2*j 6/8)
    simp only [abs_div] at h1 h2
    norm_num [abs_of_pos hr] at h1 h2
    linarith [hscale 2, hscale2 6, abs_nonneg (j 1), abs_nonneg (j 2), abs_nonneg (j 4), abs_nonneg (j 5), abs_nonneg (j 6)]
  have hM := dm2_factors r _ _ _ R hr.le hr1 hR haM hbM hcM
  have hS := dm2_factors r _ _ _ R hr.le hr1 hR haS hbS hcS
  have hprod : |((-1+r*j 4/12)*(j 0-r*j 2/2+r^2*j 6/8)-r*(-j 1/2+r*j 5/12)^2) *
      ((1+r*j 4/12)*(j 0+r*j 2/2+r^2*j 6/8)-r*(j 1/2+r*j 5/12)^2)| ≤ (2*R^2)^2 := by
    rw [abs_mul]
    calc
      _ ≤ (2*R^2)*(2*R^2) := mul_le_mul hM hS (abs_nonneg _) (by positivity)
      _ = (2*R^2)^2 := (pow_two _).symm
  have hsquare : (((-1+r*j 4/12)*(j 0-r*j 2/2+r^2*j 6/8)-r*(-j 1/2+r*j 5/12)^2) *
      ((1+r*j 4/12)*(j 0+r*j 2/2+r^2*j 6/8)-r*(j 1/2+r*j 5/12)^2))^2 ≤ ((2*R^2)^2)^2 := by
    simpa only [sq_abs] using
      (sq_le_sq₀ (abs_nonneg _) (sq_nonneg (2*R^2))).2 hprod
  have hid : ((dm2HM r j).det*(dm2HS r j).det)^2 = r^4 *
      (((-1+r*j 4/12)*(j 0-r*j 2/2+r^2*j 6/8)-r*(-j 1/2+r*j 5/12)^2) *
      ((1+r*j 4/12)*(j 0+r*j 2/2+r^2*j 6/8)-r*(j 1/2+r*j 5/12)^2))^2 := by
    simp [dm2HM, dm2HS, Matrix.det_fin_two]; ring
  rw [hid]
  calc
    _ ≤ r^4*((2*R^2)^2)^2 := mul_le_mul_of_nonneg_left hsquare (by positivity)
    _ = 16*r^4*dm2Radius j^8 := by dsimp [R]; ring

/-- P02-LM-002 section 2, with the literal quadratic-form type indicator. -/
theorem dm2_pointwise (r : ℝ) (j : DM2Jet) (hr : 0 < r) (hr1 : r ≤ 1) :
    dm2Weight r j ^ 2 ≤ 16*r^4*dm2Radius j^8 := by
  classical
  unfold dm2Weight
  split_ifs
  · simpa only [sq_abs] using dm2_untyped r j hr hr1
  · simpa only [zero_pow (by decide : (2 : ℕ) ≠ 0)] using
      (show 0 ≤ 16*r^4*dm2Radius j^8 by positivity)

/-- Two-dimensional Sylvester criterion for the literal quadratic-form definition. -/
theorem dm2_negative_iff (a b c : ℝ) :
    dm2Negative !![a,b;b,c] ↔ a < 0 ∧ 0 < a*c-b^2 := by
  constructor
  · intro h
    have ha := h ![1,0] (by simp)
    simp [Fin.sum_univ_two] at ha
    have hv : (![(-b),a] : Fin 2 → ℝ) ≠ 0 := by
      intro heq
      have := congrFun heq 1
      simp at this
      linarith
    have hd := h ![-b,a] hv
    simp [Fin.sum_univ_two] at hd
    refine ⟨ha, ?_⟩
    by_contra hn
    have hh : 0 ≤ a*(a*c-b^2) := mul_nonneg_of_nonpos_of_nonpos ha.le (le_of_not_gt hn)
    nlinarith
  · rintro ⟨ha,hd⟩ v hv
    have hvxy : v 0 ≠ 0 ∨ v 1 ≠ 0 := by
      by_contra hn
      push Not at hn
      apply hv
      ext i
      fin_cases i <;> simp_all
    have hq : 0 < (a*v 0+b*v 1)^2+(a*c-b^2)*(v 1)^2 := by
      rcases eq_or_ne (v 1) 0 with hy|hy
      · have hx : v 0 ≠ 0 := hvxy.resolve_right (not_ne_iff.mpr hy)
        simpa [hy] using sq_pos_of_ne_zero (mul_ne_zero ha.ne hx)
      · exact add_pos_of_nonneg_of_pos (sq_nonneg _) (mul_pos hd (sq_pos_of_ne_zero hy))
    simp [Fin.sum_univ_two]
    nlinarith

/-- Measurability of the exact typed weight, with no assumed type-null boundary. -/
theorem dm2_weight_measurable (r : ℝ) : Measurable (dm2Weight r) := by
  classical
  have hm : Continuous (fun j : DM2Jet => (dm2HM r j).det) := by
    simp only [dm2HM, Matrix.det_fin_two]
    fun_prop
  have hs : Continuous (fun j : DM2Jet => (dm2HS r j).det) := by
    simp only [dm2HS, Matrix.det_fin_two]
    fun_prop
  have htype : MeasurableSet {j : DM2Jet | dm2Negative (dm2HM r j) ∧ (dm2HS r j).det < 0} := by
    simp only [dm2HM, dm2_negative_iff]
    apply MeasurableSet.inter
    · apply MeasurableSet.inter
      · have h : Continuous (fun j : DM2Jet => -r+r^2*j 4/12) := by fun_prop
        exact (isOpen_lt h continuous_const).measurableSet
      · have h : Continuous (fun j : DM2Jet => (-r+r^2*j 4/12)*(j 0-r*j 2/2+r^2*j 6/8)-(-j 1*r/2+r^2*j 5/12)^2) := by fun_prop
        exact (isOpen_lt continuous_const h).measurableSet
    · exact (isOpen_lt hs continuous_const).measurableSet
  change Measurable (fun j : DM2Jet => if dm2Negative (dm2HM r j) ∧ (dm2HS r j).det < 0 then |(dm2HM r j).det*(dm2HS r j).det| else 0)
  exact Measurable.ite htype (hm.measurable.mul hs.measurable).abs measurable_const

/-- P02-LM-002's expectation step: eighth-moment control supplies integrability
and the second moment, rather than presupposing that second moment. -/
theorem dm2_moment {Ω : Type*} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    (J : Ω → DM2Jet) (hJ : Measurable J) (r M8 : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (h8 : MeasureTheory.Integrable (fun x => dm2Radius (J x)^8) μ)
    (hM8 : (∫ x, dm2Radius (J x)^8 ∂μ) ≤ M8) :
    MeasureTheory.Integrable (fun x => dm2Weight r (J x)^2) μ ∧
      (∫ x, dm2Weight r (J x)^2 ∂μ) ≤ 16*M8*r^4 := by
  have hmajor : MeasureTheory.Integrable (fun x => 16*r^4*dm2Radius (J x)^8) μ :=
    h8.const_mul _
  have hmeas : MeasureTheory.AEStronglyMeasurable (fun x => dm2Weight r (J x)^2) μ :=
    (((dm2_weight_measurable r).comp hJ).pow_const 2).aestronglyMeasurable
  have hi : MeasureTheory.Integrable (fun x => dm2Weight r (J x)^2) μ :=
    hmajor.mono' hmeas (Filter.Eventually.of_forall (fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact dm2_pointwise r (J x) hr hr1))
  refine ⟨hi, ?_⟩
  calc
    _ ≤ ∫ x, 16*r^4*dm2Radius (J x)^8 ∂μ :=
      MeasureTheory.integral_mono hi hmajor (fun x => dm2_pointwise r (J x) hr hr1)
    _ = 16*r^4*(∫ x, dm2Radius (J x)^8 ∂μ) := MeasureTheory.integral_const_mul _ _
    _ ≤ 16*r^4*M8 := mul_le_mul_of_nonneg_left hM8 (by positivity)
    _ = 16*M8*r^4 := by ring

end ResearchFormalCoreR1


