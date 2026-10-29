import ResearchFormalCoreR1.MeasureBridge

/-!
# A normalized nonnegative weighted probability law

This supplies the probability-law construction deliberately excluded from
MeasureBridge. The concrete field, conditioning, moment, normalizer and tail
estimates are not constructed. See ../WEIGHTED_LAW.md for exact scope.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

/-- A measure for any W; it is a probability measure only under the hypotheses
proved below. `ofReal` clips negative values, so a.e. nonnegativity is essential
when identifying its density integral with the signed Bochner integral. -/
noncomputable def weightedLaw (μ : Measure Ω) (W : Ω → ℝ) : Measure Ω :=
  (ENNReal.ofReal (∫ x, W x ∂μ))⁻¹ •
    μ.withDensity (fun x => ENNReal.ofReal (W x))

/-- Measurable-event mass before taking real values. Genuine integrability and
nonnegativity identify the density integral with the Bochner set integral. -/
theorem weightedLaw_apply
    (μ : Measure Ω) (W : Ω → ℝ) (A : Set Ω)
    (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A) :
    weightedLaw μ W A = (ENNReal.ofReal (∫ x, W x ∂μ))⁻¹ *
      ENNReal.ofReal (∫ x in A, W x ∂μ) := by
  simp only [weightedLaw, Measure.smul_apply, smul_eq_mul, withDensity_apply _ hA]
  rw [← ofReal_integral_eq_lintegral_ofReal hW.integrableOn (ae_restrict_of_ae hW0)]

/-- Total mass one follows from a positive finite normalizer. This theorem does
not assume that the measure it constructs is already a probability measure. -/
theorem weightedLaw_isProbabilityMeasure
    (μ : Measure Ω) (W : Ω → ℝ)
    (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W)
    (hz : 0 < ∫ x, W x ∂μ) : IsProbabilityMeasure (weightedLaw μ W) := by
  constructor
  rw [weightedLaw_apply μ W Set.univ hW hW0 MeasurableSet.univ]
  simp only [Measure.restrict_univ]
  exact ENNReal.inv_mul_cancel (ne_of_gt (ENNReal.ofReal_pos.mpr hz))
    ENNReal.ofReal_ne_top

/-- The real event probability is the same-law integral ratio. The positive
normalizer is retained explicitly for its intended probability interpretation. -/
theorem weightedLaw_event_real
    (μ : Measure Ω) (W : Ω → ℝ) (A : Set Ω)
    (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (hz : 0 < ∫ x, W x ∂μ) :
    (weightedLaw μ W A).toReal = (∫ x in A, W x ∂μ) / (∫ x, W x ∂μ) := by
  rw [weightedLaw_apply μ W A hW hW0 hA, ENNReal.toReal_mul,
    ENNReal.toReal_inv, ENNReal.toReal_ofReal hz.le,
    ENNReal.toReal_ofReal (integral_nonneg_of_ae (ae_restrict_of_ae hW0))]
  ring

/-- A null event under the original law is also null under the weighted law.
No independence or strict positivity of W is required for absolute continuity. -/
theorem weightedLaw_absolutelyContinuous
    (μ : Measure Ω) (W : Ω → ℝ) : weightedLaw μ W ≪ μ := by
  intro A hA
  have hzero := (withDensity_absolutelyContinuous μ (fun x => ENNReal.ofReal (W x))) hA
  simp only [weightedLaw, Measure.smul_apply, smul_eq_mul, hzero, mul_zero]

/-- Changing W on a null set changes neither its normalizer nor the law. -/
theorem weightedLaw_congr_ae
    (μ : Measure Ω) (W V : Ω → ℝ) (h : W =ᵐ[μ] V) :
    weightedLaw μ W = weightedLaw μ V := by
  have hd : (fun x => ENNReal.ofReal (W x)) =ᵐ[μ]
      (fun x => ENNReal.ofReal (V x)) :=
    h.mono (fun _ hx => congrArg ENNReal.ofReal hx)
  unfold weightedLaw
  rw [integral_congr_ae h, withDensity_congr_ae hd]

/-- The zero weight gives the zero measure, not a probability law. This exposes
why normalization positivity cannot be removed from the construction theorem. -/
theorem weightedLaw_zero (μ : Measure Ω) :
    weightedLaw μ (fun _ : Ω => (0 : ℝ)) = 0 := by
  simp [weightedLaw]

/-- A unit weight on a probability space recovers the original probability law. -/
theorem weightedLaw_one (μ : Measure Ω) [IsProbabilityMeasure μ] :
    weightedLaw μ (fun _ : Ω => (1 : ℝ)) = μ := by
  simp [weightedLaw]

/-- P02-LM-008 for the actual constructed measure, including its probability
property. The concrete moment and lower-normalizer estimates remain premises. -/
theorem p02_lm008_probability_transfer
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (r cW cZ : ℝ) (hr : 0 < r) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ) :
    IsProbabilityMeasure (weightedLaw μ W) ∧
      (weightedLaw μ W A).toReal ≤ (Real.sqrt cW / cZ) * Real.sqrt ((μ A).toReal) := by
  have hi : Integrable W μ := hW.integrable (by norm_num)
  have hz : 0 < ∫ x, W x ∂μ :=
    lt_of_lt_of_le (mul_pos hcZ (sq_pos_of_pos hr)) hlower
  refine ⟨weightedLaw_isProbabilityMeasure μ W hi hW0 hz, ?_⟩
  rw [weightedLaw_event_real μ W A hi hW0 hA hz]
  exact p02_lm008_measure_transfer μ W A hW hW0 hA r cW cZ hr hcW hcZ hsecond hlower

/-- The supplied eighth-order event bound transfers to the constructed law's
cubic bound. Neither the event bound nor uniformity in r is proved here. -/
theorem p02_lm009_probability_transfer_r8_to_r3
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (r cW cZ K : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (hcW : 0 ≤ cW) (hcZ : 0 < cZ) (hK : 0 ≤ K)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ)
    (htail : (μ A).toReal ≤ K * r ^ 8) :
    IsProbabilityMeasure (weightedLaw μ W) ∧
      (weightedLaw μ W A).toReal ≤ ((Real.sqrt cW / cZ) * Real.sqrt K) * r ^ 3 := by
  have hi : Integrable W μ := hW.integrable (by norm_num)
  have hz : 0 < ∫ x, W x ∂μ :=
    lt_of_lt_of_le (mul_pos hcZ (sq_pos_of_pos hr)) hlower
  refine ⟨weightedLaw_isProbabilityMeasure μ W hi hW0 hz, ?_⟩
  rw [weightedLaw_event_real μ W A hi hW0 hA hz]
  exact p02_lm009_measure_transfer_r8_to_r3 μ W A hW hW0 hA r cW cZ K
    hr hr1 hcW hcZ hK hsecond hlower htail

end ResearchFormalCoreR1
