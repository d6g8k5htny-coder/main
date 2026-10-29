import ResearchFormalCoreR1.WeightedLaw

/-!
# Same-law stability of normalized nonnegative weights

An actual L1 error controls the normalizer and every measurable-event probability.
The reference lower bound and the concrete L1 estimate remain premises. Both
weights use the same underlying measure. See ../WEIGHT_PERTURBATION.md.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Integrability, not a totalized divergent integral, justifies subtraction. -/
theorem weightPerturbation_integral
    (μ : Measure Ω) (W V : Ω → ℝ) (hW : Integrable W μ) (hV : Integrable V μ) :
    |(∫ x, W x ∂μ) - (∫ x, V x ∂μ)| ≤ ∫ x, |W x - V x| ∂μ := by
  rw [← integral_sub hW hV]
  exact abs_integral_le_integral_abs

/-- Restricting to any set cannot increase the integral absolute error. The
probability formula below separately requires measurable events. -/
theorem weightPerturbation_setIntegral
    (μ : Measure Ω) (W V : Ω → ℝ) (A : Set Ω)
    (hW : Integrable W μ) (hV : Integrable V μ) :
    |(∫ x in A, W x ∂μ) - (∫ x in A, V x ∂μ)| ≤ ∫ x, |W x - V x| ∂μ := by
  exact (weightPerturbation_integral (μ.restrict A) W V
    hW.integrableOn hV.integrableOn).trans
    (setIntegral_le_integral (hW.sub hV).abs
      (Filter.Eventually.of_forall (fun x => abs_nonneg (W x - V x))))

/-- A reference normalizer lower bound loses at most the supplied L1 budget. -/
theorem weightPerturbation_normalizer_lower
    (μ : Measure Ω) (W V : Ω → ℝ) (hW : Integrable W μ) (hV : Integrable V μ)
    (c δ : ℝ) (hreference : c ≤ ∫ x, V x ∂μ)
    (herror : (∫ x, |W x - V x| ∂μ) ≤ δ) :
    c - δ ≤ ∫ x, W x ∂μ := by
  have h := (abs_le.mp ((weightPerturbation_integral μ W V hW hV).trans herror)).1
  linarith

/-- A nonnegative event numerator lies between zero and its own normalizer. -/
theorem weightPerturbation_event_integral_bounds
    (μ : Measure Ω) (W : Ω → ℝ) (A : Set Ω)
    (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W) :
    0 ≤ (∫ x in A, W x ∂μ) ∧ (∫ x in A, W x ∂μ) ≤ ∫ x, W x ∂μ := by
  exact ⟨integral_nonneg_of_ae (ae_restrict_of_ae hW0),
    setIntegral_le_integral hW hW0⟩

/-- Scalar ratio stability with the reference denominator t. The factor 2 is
convenient, not claimed optimal. No sign premise on b is needed for this lemma. -/
theorem weightPerturbation_quotient_bound
    (a b z t δ : ℝ) (hz : 0 < z) (ht : 0 < t)
    (ha : 0 ≤ a) (haz : a ≤ z) (hab : |a - b| ≤ δ) (hzt : |z - t| ≤ δ) :
    |a / z - b / t| ≤ 2 * δ / t := by
  have hfrac0 : 0 ≤ a / z := div_nonneg ha hz.le
  have hfrac1 : a / z ≤ 1 := (div_le_one hz).2 haz
  have hgap : |t - z| ≤ δ := by
    rw [abs_sub_comm]
    exact hzt
  have hprod : |a / z| * |t - z| ≤ δ := by
    rw [abs_of_nonneg hfrac0]
    calc
      (a / z) * |t - z| ≤ 1 * |t - z| :=
        mul_le_mul_of_nonneg_right hfrac1 (abs_nonneg _)
      _ ≤ δ := by simpa only [one_mul] using hgap
  have heq : a / z - b / t = ((a - b) + (a / z) * (t - z)) / t := by
    field_simp [ne_of_gt hz, ne_of_gt ht] <;> ring
  calc
    |a / z - b / t| = |(a - b) + (a / z) * (t - z)| / t := by
      rw [heq, abs_div, abs_of_pos ht]
    _ ≤ (|a - b| + |(a / z) * (t - z)|) / t :=
      div_le_div_of_nonneg_right (abs_add_le _ _) ht.le
    _ ≤ (δ + δ) / t := by
      apply div_le_div_of_nonneg_right _ ht.le
      rw [abs_mul]
      exact add_le_add hab hprod
    _ = 2 * δ / t := by ring

/-- Under one common measure, a small L1 weight error transfers a positive
reference normalizer to W and controls all measurable-event probabilities.
The ambient measure need not be finite or a probability measure. -/
theorem weightedLaw_event_perturbation
    (μ : Measure Ω) (W V : Ω → ℝ) (hW : Integrable W μ) (hV : Integrable V μ)
    (hW0 : 0 ≤ᵐ[μ] W) (hV0 : 0 ≤ᵐ[μ] V)
    (c δ : ℝ) (hc : 0 < c) (hreference : c ≤ ∫ x, V x ∂μ)
    (herror : (∫ x, |W x - V x| ∂μ) ≤ δ) (hsmall : δ ≤ c / 2) :
    c / 2 ≤ (∫ x, W x ∂μ) ∧
      IsProbabilityMeasure (weightedLaw μ W) ∧
      IsProbabilityMeasure (weightedLaw μ V) ∧
      ∀ A : Set Ω, MeasurableSet A →
        |(weightedLaw μ W A).toReal - (weightedLaw μ V A).toReal| ≤ 2 * δ / c := by
  have hL1nonneg : 0 ≤ ∫ x, |W x - V x| ∂μ :=
    integral_nonneg (fun x => abs_nonneg (W x - V x))
  have hδ0 : 0 ≤ δ := hL1nonneg.trans herror
  have hlower := weightPerturbation_normalizer_lower μ W V hW hV c δ hreference herror
  have hz : 0 < ∫ x, W x ∂μ := by linarith
  have ht : 0 < ∫ x, V x ∂μ := lt_of_lt_of_le hc hreference
  refine ⟨by linarith, weightedLaw_isProbabilityMeasure μ W hW hW0 hz,
    weightedLaw_isProbabilityMeasure μ V hV hV0 ht, ?_⟩
  intro A hA
  rw [weightedLaw_event_real μ W A hW hW0 hA hz,
    weightedLaw_event_real μ V A hV hV0 hA ht]
  have hb := weightPerturbation_event_integral_bounds μ W A hW hW0
  exact (weightPerturbation_quotient_bound _ _ _ _ δ hz ht hb.1 hb.2
    ((weightPerturbation_setIntegral μ W V A hW hV).trans herror)
    ((weightPerturbation_integral μ W V hW hV).trans herror)).trans
      (div_le_div_of_nonneg_left (mul_nonneg (by norm_num) hδ0) hc hreference)

/-- The natural r² normalizer and L1-error scales cancel. No r ≤ 1 restriction
is needed. Uniform application requires uniform c and the displayed budgets. -/
theorem weightedLaw_event_perturbation_r2
    (μ : Measure Ω) (W V : Ω → ℝ) (hW : Integrable W μ) (hV : Integrable V μ)
    (hW0 : 0 ≤ᵐ[μ] W) (hV0 : 0 ≤ᵐ[μ] V)
    (r c η : ℝ) (hr : 0 < r) (hc : 0 < c)
    (hreference : c * r ^ 2 ≤ ∫ x, V x ∂μ)
    (herror : (∫ x, |W x - V x| ∂μ) ≤ η * r ^ 2) (hsmall : η ≤ c / 2) :
    (c * r ^ 2) / 2 ≤ (∫ x, W x ∂μ) ∧
      IsProbabilityMeasure (weightedLaw μ W) ∧
      IsProbabilityMeasure (weightedLaw μ V) ∧
      ∀ A : Set Ω, MeasurableSet A →
        |(weightedLaw μ W A).toReal - (weightedLaw μ V A).toReal| ≤ 2 * η / c := by
  have hs : η * r ^ 2 ≤ (c * r ^ 2) / 2 := by
    calc
      η * r ^ 2 ≤ (c / 2) * r ^ 2 :=
        mul_le_mul_of_nonneg_right hsmall (sq_nonneg r)
      _ = (c * r ^ 2) / 2 := by ring
  obtain ⟨hlower, hpW, hpV, hA⟩ := weightedLaw_event_perturbation μ W V hW hV hW0 hV0
    (c * r ^ 2) (η * r ^ 2) (mul_pos hc (sq_pos_of_pos hr)) hreference herror hs
  refine ⟨hlower, hpW, hpV, ?_⟩
  intro A hmeas
  have heq : 2 * (η * r ^ 2) / (c * r ^ 2) = 2 * η / c := by
    field_simp [ne_of_gt hr, ne_of_gt hc]
  simpa only [heq] using hA A hmeas

end ResearchFormalCoreR1
