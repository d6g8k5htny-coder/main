import ResearchFormalCoreR1.DeterminantMoment
import ResearchFormalCoreR1.GaussianScalarMoments
import Mathlib.Probability.HasLaw
import Mathlib.Algebra.Order.Chebyshev

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

set_option autoImplicit false

/-- Radius for a finite vector, including its constant one. -/
def gm8Radius {n : ℕ} (j : Fin n → ℝ) : ℝ := 1 + ∑ i, |j i|

/-- One finite-dimensional Gaussian radius moment constant. -/
noncomputable def gm8FiniteM8 (n : ℕ) (B V : ℝ) : ℝ :=
  ((n+1 : ℕ) : ℝ)^7 * (1 + (n : ℝ)*128*(B^8 + V^4*gm8C8))

theorem gm8FiniteM8_nonneg (n : ℕ) (B V : ℝ) : 0 ≤ gm8FiniteM8 n B V := by
  unfold gm8FiniteM8
  have hC := gm8_standard_nonneg
  positivity

/-- The source's exact fixed nine-coordinate constant. -/
noncomputable def gm8M8 (B V : ℝ) : ℝ :=
  (10 : ℝ)^7 * (1 + 9*128*(B^8 + V^4*gm8C8))

theorem gm8M8_nonneg (B V : ℝ) : 0 ≤ gm8M8 B V := by
  simpa [gm8M8, gm8FiniteM8] using gm8FiniteM8_nonneg 9 B V

theorem gm8_finite_radius_power_bound (n : ℕ) (j : Fin n → ℝ) :
    gm8Radius j ^ 8 ≤ ((n+1 : ℕ) : ℝ)^7 * (1 + ∑ i, |j i|^8) := by
  let f : Fin (n+1) → ℝ := Fin.cases 1 (fun i : Fin n => |j i|)
  have h := pow_sum_le_card_mul_sum_pow (s := Finset.univ) (f := f)
    (fun i _ => by
      refine Fin.cases ?_ (fun k => ?_) i
      · norm_num [f]
      · exact abs_nonneg (j k)) 7
  simpa [f, gm8Radius, Fin.sum_univ_succ] using h

theorem gm8_finite_integrable_bound (n : ℕ) {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (J : Ω → Fin n → ℝ)
    (hJ : Measurable J) (m : Fin n → ℝ) (v : Fin n → ℝ≥0)
    (B V : ℝ) (hB : 0 ≤ B) (hV : 0 ≤ V)
    (hmean : ∀ i, |m i| ≤ B) (hvar : ∀ i, (v i : ℝ) ≤ V)
    (hlaw : ∀ i, HasLaw (fun x => J x i) (gaussianReal (m i) (v i)) μ) :
    Integrable (fun x => gm8Radius (J x)^8) μ ∧
      (∫ x, gm8Radius (J x)^8 ∂μ) ≤ gm8FiniteM8 n B V := by
  have hi (i : Fin n) : Integrable (fun x => |J x i|^8) μ := by
    exact (hlaw i).integrable_comp
      (gm8_scalar_integrable_bound (m i) B V (v i) hB hV (hmean i) (hvar i)).1
  have hsum : Integrable (fun x => ∑ i, |J x i|^8) μ :=
    integrable_finsetSum _ (fun i _ => hi i)
  have hmajor : Integrable (fun x => ((n+1 : ℕ) : ℝ)^7*(1+∑ i, |J x i|^8)) μ :=
    ((integrable_const 1).add hsum).const_mul _
  have hc : Continuous (gm8Radius (n := n)) := by unfold gm8Radius; fun_prop
  have hmeas : AEStronglyMeasurable (fun x => gm8Radius (J x)^8) μ :=
    ((hc.measurable.comp hJ).pow_const 8).aestronglyMeasurable
  have hR : Integrable (fun x => gm8Radius (J x)^8) μ :=
    hmajor.mono' hmeas (Filter.Eventually.of_forall (fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg (by unfold gm8Radius; positivity) 8)]
      exact gm8_finite_radius_power_bound n (J x)))
  have hcoord (i : Fin n) :
      (∫ x, |J x i|^8 ∂μ) ≤ 128*(B^8+V^4*gm8C8) := by
    have heq := (hlaw i).integral_comp
      (show AEStronglyMeasurable (fun y : ℝ => |y|^8) (gaussianReal (m i) (v i)) by fun_prop)
    exact heq.trans_le (gm8_scalar_integrable_bound (m i) B V (v i) hB hV
      (hmean i) (hvar i)).2
  refine ⟨hR, ?_⟩
  calc
    _ ≤ ∫ x, ((n+1 : ℕ) : ℝ)^7*(1+∑ i, |J x i|^8) ∂μ :=
      integral_mono hR hmajor (fun x => gm8_finite_radius_power_bound n (J x))
    _ = ((n+1 : ℕ) : ℝ)^7*(1+∑ i, ∫ x, |J x i|^8 ∂μ) := by
      rw [integral_const_mul, integral_add (integrable_const _) hsum,
        integral_finsetSum _ (fun i _ => hi i)]
      simp
    _ ≤ ((n+1 : ℕ) : ℝ)^7*(1+∑ _i : Fin n, 128*(B^8+V^4*gm8C8)) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      exact add_le_add le_rfl (Finset.sum_le_sum (fun i _ => hcoord i))
    _ = gm8FiniteM8 n B V := by simp [gm8FiniteM8, mul_assoc]

/-- The exact DM2 radius is the nine-dimensional specialization. -/
theorem gm8_radius_power_bound (j : DM2Jet) :
    dm2Radius j ^ 8 ≤ (10 : ℝ)^7 * (1 + ∑ i, |j i|^8) := by
  simpa [gm8Radius, dm2Radius] using gm8_finite_radius_power_bound 9 j

/-- The nine-jet Gaussian marginal moment interface, with no independence or
moment conclusion assumed. -/
theorem gm8_jet_integrable_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (J : Ω → DM2Jet)
    (hJ : Measurable J) (m : Fin 9 → ℝ) (v : Fin 9 → ℝ≥0)
    (B V : ℝ) (hB : 0 ≤ B) (hV : 0 ≤ V)
    (hmean : ∀ i, |m i| ≤ B) (hvar : ∀ i, (v i : ℝ) ≤ V)
    (hlaw : ∀ i, HasLaw (fun x => J x i) (gaussianReal (m i) (v i)) μ) :
    Integrable (fun x => dm2Radius (J x)^8) μ ∧
      (∫ x, dm2Radius (J x)^8 ∂μ) ≤ gm8M8 B V := by
  simpa [gm8Radius, dm2Radius, gm8M8, gm8FiniteM8] using
    gm8_finite_integrable_bound 9 μ J hJ m v B V hB hV hmean hvar hlaw

/-- The same B,V give one determinant second-moment coefficient throughout
0<r<=r0<=1. No radius or determinant moment is an input hypothesis. -/
theorem gm8_family_weight_moment {Ω : Type*} [MeasurableSpace Ω]
    (μ : ℝ → Measure Ω) (J : ℝ → Ω → DM2Jet)
    (m : ℝ → Fin 9 → ℝ) (v : ℝ → Fin 9 → ℝ≥0)
    (B V r0 : ℝ) (hB : 0 ≤ B) (hV : 0 ≤ V) (hr01 : r0 ≤ 1)
    (hprob : ∀ r, 0 < r → r ≤ r0 → IsProbabilityMeasure (μ r))
    (hJ : ∀ r, 0 < r → r ≤ r0 → Measurable (J r))
    (hmean : ∀ r, 0 < r → r ≤ r0 → ∀ i, |m r i| ≤ B)
    (hvar : ∀ r, 0 < r → r ≤ r0 → ∀ i, (v r i : ℝ) ≤ V)
    (hlaw : ∀ r, 0 < r → r ≤ r0 → ∀ i,
      HasLaw (fun x => J r x i) (gaussianReal (m r i) (v r i)) (μ r)) :
    0 ≤ 16*gm8M8 B V ∧ ∀ r, 0 < r → r ≤ r0 →
      Integrable (fun x => dm2Weight r (J r x)^2) (μ r) ∧
        (∫ x, dm2Weight r (J r x)^2 ∂μ r) ≤ (16*gm8M8 B V)*r^4 := by
  refine ⟨mul_nonneg (by norm_num) (gm8M8_nonneg B V), ?_⟩
  intro r hr hr0
  have : IsProbabilityMeasure (μ r) := hprob r hr hr0
  obtain ⟨h8, hbound⟩ := gm8_jet_integrable_bound (μ r) (J r) (hJ r hr hr0)
    (m r) (v r) B V hB hV (hmean r hr hr0) (hvar r hr hr0) (hlaw r hr hr0)
  exact dm2_moment (μ r) (J r) (hJ r hr hr0) r (gm8M8 B V) hr (hr0.trans hr01)
    h8 hbound

end ResearchFormalCoreR1
