import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Algebra.Order.Ring.Basic
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Scalar eighth moments for noncentered Gaussian laws. The finite standard
Gaussian moment is proved integrable; it is not an assumed model bound. -/

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory
open scoped NNReal

set_option autoImplicit false

noncomputable def gm8C8 : ℝ := ∫ z : ℝ, |z| ^ 8 ∂gaussianReal 0 1

theorem gm8_standard_integrable :
    Integrable (fun z : ℝ => |z| ^ 8) (gaussianReal 0 1) := by
  simpa only [id_eq, Real.norm_eq_abs] using
    (memLp_id_gaussianReal' (μ := 0) (v := 1) 8 (by norm_num)).integrable_norm_pow
      (by norm_num : (8 : ℕ) ≠ 0)

theorem gm8_standard_nonneg : 0 ≤ gm8C8 := by
  exact integral_nonneg (fun z => pow_nonneg (abs_nonneg z) 8)

private theorem gm8_integrable (m : ℝ) (v : ℝ≥0) :
    Integrable (fun x : ℝ => |x| ^ 8) (gaussianReal m v) := by
  simpa only [id_eq, Real.norm_eq_abs] using
    (memLp_id_gaussianReal' (μ := m) (v := v) 8 (by norm_num)).integrable_norm_pow
      (by norm_num : (8 : ℕ) ≠ 0)

private theorem gm8_sqrt_pow (V : ℝ) (hV : 0 ≤ V) :
    (Real.sqrt V) ^ 8 = V ^ 4 := by
  calc
    _ = ((Real.sqrt V) ^ 2) ^ 4 := by ring
    _ = V ^ 4 := by rw [Real.sq_sqrt hV]

private theorem gm8_affine_map (m : ℝ) (v : ℝ≥0) :
    (gaussianReal 0 1).map (fun z : ℝ => m + Real.sqrt (v : ℝ) * z) =
      gaussianReal m v := by
  calc
    _ = ((gaussianReal 0 1).map (fun z : ℝ => Real.sqrt (v : ℝ) * z)).map
        (fun z : ℝ => m + z) := by
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      rfl
    _ = gaussianReal m v := by
      rw [gaussianReal_map_const_mul, gaussianReal_map_const_add]
      simp only [mul_zero, zero_add, mul_one]
      congr 1
      apply NNReal.eq
      exact Real.sq_sqrt v.property

private theorem gm8_affine_bound (m B V : ℝ) (v : ℝ≥0) (hB : 0 ≤ B) (hV : 0 ≤ V)
    (hm : |m| ≤ B) (hv : (v : ℝ) ≤ V) (z : ℝ) :
    |m + Real.sqrt (v : ℝ) * z| ^ 8 ≤ 128 * (B ^ 8 + V ^ 4 * |z| ^ 8) := by
  have habs : |m + Real.sqrt (v : ℝ) * z| ≤ B + Real.sqrt V * |z| := by
    calc
      _ ≤ |m| + |Real.sqrt (v : ℝ) * z| := abs_add_le _ _
      _ = |m| + Real.sqrt (v : ℝ) * |z| := by
        rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
      _ ≤ B + Real.sqrt V * |z| :=
        add_le_add hm (mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt hv) (abs_nonneg z))
  calc
    _ ≤ (B + Real.sqrt V * |z|) ^ 8 := pow_le_pow_left₀ (abs_nonneg _) habs 8
    _ ≤ 128 * (B ^ 8 + (Real.sqrt V * |z|) ^ 8) := by
      convert add_pow_le hB (mul_nonneg (Real.sqrt_nonneg V) (abs_nonneg z)) 8 using 1
      norm_num
    _ = 128 * (B ^ 8 + V ^ 4 * |z| ^ 8) := by
      rw [mul_pow, gm8_sqrt_pow V hV]

/-- Bounded means and variances give a uniform eighth-moment bound, including
zero variance and nonzero means. No eighth-moment hypothesis is assumed. -/
theorem gm8_scalar_integrable_bound (m B V : ℝ) (v : ℝ≥0)
    (hB : 0 ≤ B) (hV : 0 ≤ V) (hm : |m| ≤ B) (hv : (v : ℝ) ≤ V) :
    Integrable (fun x : ℝ => |x| ^ 8) (gaussianReal m v) ∧
      (∫ x : ℝ, |x| ^ 8 ∂gaussianReal m v) ≤
        128 * (B ^ 8 + V ^ 4 * gm8C8) := by
  have hi := gm8_integrable m v
  have hmajor : Integrable (fun z : ℝ => 128 * (B ^ 8 + V ^ 4 * |z| ^ 8))
      (gaussianReal 0 1) :=
    ((integrable_const (B ^ 8)).add (gm8_standard_integrable.const_mul (V ^ 4))).const_mul 128
  have hiaff : Integrable (fun z : ℝ => |m + Real.sqrt (v : ℝ) * z| ^ 8)
      (gaussianReal 0 1) := by
    rw [← gm8_affine_map m v] at hi
    exact (integrable_map_measure (by fun_prop) (by fun_prop)).mp hi
  refine ⟨hi, ?_⟩
  calc
    _ = ∫ z : ℝ, |m + Real.sqrt (v : ℝ) * z| ^ 8 ∂gaussianReal 0 1 := by
      rw [← gm8_affine_map m v, integral_map (by fun_prop) (by fun_prop)]
    _ ≤ ∫ z : ℝ, 128 * (B ^ 8 + V ^ 4 * |z| ^ 8) ∂gaussianReal 0 1 :=
      integral_mono hiaff hmajor (gm8_affine_bound m B V v hB hV hm hv)
    _ = 128 * (B ^ 8 + V ^ 4 * gm8C8) := by
      rw [integral_const_mul, integral_add (integrable_const _)
        (gm8_standard_integrable.const_mul _), integral_const_mul]
      simp only [integral_const, probReal_univ, smul_eq_mul, one_mul, gm8C8]

end ResearchFormalCoreR1
