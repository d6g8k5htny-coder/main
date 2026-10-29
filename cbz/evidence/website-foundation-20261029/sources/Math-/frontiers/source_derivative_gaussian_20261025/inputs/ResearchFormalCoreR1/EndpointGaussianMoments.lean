import ResearchFormalCoreR1.GaussianJetMoments

/-! Gaussian moments for the six actual normalized endpoint entries from
GP-DER-047 section 13. Coordinates are (fxx(M)/r,fxy(M)/r,fyy(M),
fxx(S)/r,fxy(S)/r,fyy(S)); no polynomial midpoint-jet identity is assumed. -/

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

set_option autoImplicit false

def egm_HM (r : ℝ) (j : Fin 6 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![r * j 0, r * j 1; r * j 1, j 2]

def egm_HS (r : ℝ) (j : Fin 6 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![r * j 3, r * j 4; r * j 4, j 5]

noncomputable def egm_weight (r : ℝ) (j : Fin 6 → ℝ) : ℝ := by
  classical
  exact if dm2Negative (egm_HM r j) ∧ (egm_HS r j).det < 0 then
    |(egm_HM r j).det * (egm_HS r j).det| else 0

/-- The maximum condition is the original strict quadratic-form condition;
no null-boundary assumption is needed for measurable typing. -/
theorem egm_weight_measurable (r : ℝ) : Measurable (egm_weight r) := by
  classical
  have hm : Continuous (fun j : Fin 6 → ℝ => (egm_HM r j).det) := by
    simp only [egm_HM, Matrix.det_fin_two]
    fun_prop
  have hs : Continuous (fun j : Fin 6 → ℝ => (egm_HS r j).det) := by
    simp only [egm_HS, Matrix.det_fin_two]
    fun_prop
  have htype : MeasurableSet {j : Fin 6 → ℝ |
      dm2Negative (egm_HM r j) ∧ (egm_HS r j).det < 0} := by
    simp only [egm_HM, dm2_negative_iff]
    apply MeasurableSet.inter
    · apply MeasurableSet.inter
      · have h : Continuous (fun j : Fin 6 → ℝ => r * j 0) := by fun_prop
        exact (isOpen_lt h continuous_const).measurableSet
      · have h : Continuous (fun j : Fin 6 → ℝ =>
            (r * j 0) * j 2 - (r * j 1) ^ 2) := by fun_prop
        exact (isOpen_lt continuous_const h).measurableSet
    · exact (isOpen_lt hs continuous_const).measurableSet
  change Measurable (fun j => if dm2Negative (egm_HM r j) ∧ (egm_HS r j).det < 0
    then |(egm_HM r j).det * (egm_HS r j).det| else 0)
  exact Measurable.ite htype (hm.measurable.mul hs.measurable).abs measurable_const

private theorem egm_coordinate_bound (j : Fin 6 → ℝ) (i : Fin 6) :
    |j i| ≤ gm8Radius j := by
  have hsum : |j i| ≤ ∑ k, |j k| :=
    Finset.single_le_sum (fun k _ => abs_nonneg (j k)) (Finset.mem_univ i)
  unfold gm8Radius
  linarith

private theorem egm_factor_bound (r a b c R : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hR : 0 ≤ R) (ha : |a| ≤ R) (hb : |b| ≤ R) (hc : |c| ≤ R) :
    |a * c - r * b ^ 2| ≤ 2 * R ^ 2 := by
  have hac : |a * c| ≤ R ^ 2 := by
    rw [abs_mul, pow_two]
    exact mul_le_mul ha hc (abs_nonneg c) hR
  have hbb : b ^ 2 ≤ R ^ 2 := by nlinarith [sq_abs b, abs_nonneg b]
  have hrbb : |r * b ^ 2| ≤ R ^ 2 := by
    rw [abs_of_nonneg (mul_nonneg hr (sq_nonneg b))]
    exact (mul_le_of_le_one_left (sq_nonneg b) hr1).trans hbb
  exact (abs_sub _ _).trans (by linarith)

private theorem egm_untyped (r : ℝ) (j : Fin 6 → ℝ) (hr : 0 < r) (hr1 : r ≤ 1) :
    ((egm_HM r j).det * (egm_HS r j).det) ^ 2 ≤
      16 * r ^ 4 * gm8Radius j ^ 8 := by
  let R := gm8Radius j
  let dM := j 0 * j 2 - r * j 1 ^ 2
  let dS := j 3 * j 5 - r * j 4 ^ 2
  have hR : 0 ≤ R := by dsimp [R, gm8Radius]; positivity
  have hM : |dM| ≤ 2 * R ^ 2 :=
    egm_factor_bound r _ _ _ R hr.le hr1 hR
      (egm_coordinate_bound j 0) (egm_coordinate_bound j 1) (egm_coordinate_bound j 2)
  have hS : |dS| ≤ 2 * R ^ 2 :=
    egm_factor_bound r _ _ _ R hr.le hr1 hR
      (egm_coordinate_bound j 3) (egm_coordinate_bound j 4) (egm_coordinate_bound j 5)
  have hprod : |dM * dS| ≤ (2 * R ^ 2) ^ 2 := by
    rw [abs_mul, pow_two]
    exact mul_le_mul hM hS (abs_nonneg _) (by positivity)
  have hsq : (dM * dS) ^ 2 ≤ ((2 * R ^ 2) ^ 2) ^ 2 := by
    simpa only [sq_abs] using
      (sq_le_sq₀ (abs_nonneg (dM * dS)) (sq_nonneg (2 * R ^ 2))).2 hprod
  have hdet : ((egm_HM r j).det * (egm_HS r j).det) ^ 2 =
      r ^ 4 * (dM * dS) ^ 2 := by
    simp [egm_HM, egm_HS, Matrix.det_fin_two, dM, dS]
    ring
  calc
    _ = r ^ 4 * (dM * dS) ^ 2 := hdet
    _ ≤ r ^ 4 * ((2 * R ^ 2) ^ 2) ^ 2 :=
      mul_le_mul_of_nonneg_left hsq (by positivity)
    _ = 16 * r ^ 4 * gm8Radius j ^ 8 := by dsimp [R]; ring

/-- Exact endpoint determinant domination in the actual normalized six-vector.
The type indicators can only reduce the nonnegative determinant product. -/
theorem egm_pointwise (r : ℝ) (j : Fin 6 → ℝ) (hr : 0 < r) (hr1 : r ≤ 1) :
    egm_weight r j ^ 2 ≤ 16 * r ^ 4 * gm8Radius j ^ 8 := by
  classical
  unfold egm_weight
  split_ifs
  · simpa only [sq_abs] using egm_untyped r j hr hr1
  · simpa only [zero_pow (by decide : (2 : ℕ) ≠ 0)] using
      (show 0 ≤ 16 * r ^ 4 * gm8Radius j ^ 8 by positivity)

/-- Genuine Gaussian marginal laws of the actual normalized endpoint vector
supply both integrability and its determinant second moment. The derivative-map,
conditioning law and uniform parameter bounds remain explicit model inputs. -/
theorem egm_gaussian_moment {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (J : Ω → Fin 6 → ℝ)
    (hJ : Measurable J) (m : Fin 6 → ℝ) (v : Fin 6 → ℝ≥0)
    (B V r : ℝ) (hB : 0 ≤ B) (hV : 0 ≤ V) (hr : 0 < r) (hr1 : r ≤ 1)
    (hmean : ∀ i, |m i| ≤ B) (hvar : ∀ i, (v i : ℝ) ≤ V)
    (hlaw : ∀ i, HasLaw (fun x => J x i) (gaussianReal (m i) (v i)) μ) :
    Integrable (fun x => egm_weight r (J x) ^ 2) μ ∧
      (∫ x, egm_weight r (J x) ^ 2 ∂μ) ≤ 16 * gm8FiniteM8 6 B V * r ^ 4 := by
  obtain ⟨h8, hbound⟩ := gm8_finite_integrable_bound 6 μ J hJ m v B V hB hV
    hmean hvar hlaw
  have hmajor : Integrable (fun x => 16 * r ^ 4 * gm8Radius (J x) ^ 8) μ :=
    h8.const_mul _
  have hmeas : AEStronglyMeasurable (fun x => egm_weight r (J x) ^ 2) μ :=
    (((egm_weight_measurable r).comp hJ).pow_const 2).aestronglyMeasurable
  have hi : Integrable (fun x => egm_weight r (J x) ^ 2) μ :=
    hmajor.mono' hmeas (Filter.Eventually.of_forall (fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact egm_pointwise r (J x) hr hr1))
  refine ⟨hi, ?_⟩
  calc
    _ ≤ ∫ x, 16 * r ^ 4 * gm8Radius (J x) ^ 8 ∂μ :=
      integral_mono hi hmajor (fun x => egm_pointwise r (J x) hr hr1)
    _ = 16 * r ^ 4 * (∫ x, gm8Radius (J x) ^ 8 ∂μ) := integral_const_mul _ _
    _ ≤ 16 * r ^ 4 * gm8FiniteM8 6 B V :=
      mul_le_mul_of_nonneg_left hbound (by positivity)
    _ = 16 * gm8FiniteM8 6 B V * r ^ 4 := by ring

end ResearchFormalCoreR1
