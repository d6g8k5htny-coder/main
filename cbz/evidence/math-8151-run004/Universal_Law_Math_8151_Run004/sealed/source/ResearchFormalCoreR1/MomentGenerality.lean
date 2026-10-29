import ResearchFormalCoreR1.MomentTail

/-!
# Finite-measure and interval-only moment-tail companions

Executable adaptation by OpenAI / GPT-6 Astra Pro of the separately attributed
Grok Bot agent 6 and agent 7 proposals in Math-#277/#278. Their underlying model
providers are not inferred from their role labels. See ../MOMENT_GENERALITY.md.
The original 36 theorem statements and proofs are unchanged.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The fortieth-moment event bound for any finite measure. The event mass is
not normalized by the total mass; both sides scale with the ambient measure. -/
theorem p02_lm009_moment40_tail_finite
    (μ : Measure Ω) [IsFiniteMeasure μ] (R : Ω → ℝ)
    (r ε M : ℝ) (hr : 0 < r) (hε : 0 < ε)
    (hRi : Integrable (fun x => R x ^ 40) μ)
    (hMbound : (∫ x, R x ^ 40 ∂μ) ≤ M) :
    (μ (badJetEvent R r ε)).toReal ≤ (M / ε ^ 8) * r ^ 8 := by
  have hnonneg : 0 ≤ᵐ[μ] (fun x => R x ^ 40) := by
    apply Filter.Eventually.of_forall
    intro x
    positivity
  have ha : 0 < (ε / r) ^ 8 := pow_pos (div_pos hε hr) 8
  calc
    (μ (badJetEvent R r ε)).toReal ≤ (∫ x, R x ^ 40 ∂μ) / (ε / r) ^ 8 :=
      p02_lm009_markov_event μ (fun x => R x ^ 40) (badJetEvent R r ε)
        hRi hnonneg ((ε / r) ^ 8) ha (p02_lm009_badJet_subset R r ε hr hε)
    _ ≤ M / (ε / r) ^ 8 := div_le_div_of_nonneg_right hMbound ha.le
    _ = (M / ε ^ 8) * r ^ 8 := by
      field_simp [ne_of_gt hr, ne_of_gt hε]

/-- The original probability-space conclusion specializes from the finite
measure companion. This does not replace or edit the original declaration. -/
theorem p02_lm009_moment40_tail_of_finite
    (μ : Measure Ω) [IsProbabilityMeasure μ] (R : Ω → ℝ)
    (r ε M : ℝ) (hr : 0 < r) (hε : 0 < ε)
    (hRi : Integrable (fun x => R x ^ 40) μ)
    (hMbound : (∫ x, R x ^ 40 ∂μ) ≤ M) :
    (μ (badJetEvent R r ε)).toReal ≤ (M / ε ^ 8) * r ^ 8 :=
  p02_lm009_moment40_tail_finite μ R r ε M hr hε hRi hMbound

/-- One fixed cubic constant, with probability and all model bounds required
only on 0 < r ≤ r0. No probability assumption is made outside that interval. -/
theorem p02_lm009_moment40_family_r3_interval
    (μ : ℝ → Measure Ω)
    (W R : ℝ → Ω → ℝ) (r0 ε M cW cZ : ℝ)
    (hr0 : 0 < r0 ∧ r0 ≤ 1) (hε : 0 < ε)
    (hM : 0 ≤ M) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hdata : ∀ r, 0 < r → r ≤ r0 →
      IsProbabilityMeasure (μ r) ∧
      MemLp (W r) 2 (μ r) ∧ (0 ≤ᵐ[μ r] W r) ∧ Measurable (R r) ∧
      Integrable (fun x => R r x ^ 40) (μ r) ∧
      (∫ x, R r x ^ 40 ∂(μ r)) ≤ M ∧
      (∫ x, W r x ^ 2 ∂(μ r)) ≤ cW * r ^ 4 ∧
      cZ * r ^ 2 ≤ ∫ x, W r x ∂(μ r)) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ r, 0 < r → r ≤ r0 →
      IsProbabilityMeasure (weightedLaw (μ r) (W r)) ∧
      (weightedLaw (μ r) (W r) (badJetEvent (R r) r ε)).toReal ≤ C * r ^ 3 := by
  refine ⟨(Real.sqrt cW / cZ) * Real.sqrt (M / ε ^ 8), by positivity, ?_⟩
  intro r hr hrr0
  obtain ⟨hProb, hW, hW0, hR, hRi, hMb, hsecond, hlower⟩ := hdata r hr hrr0
  letI : IsProbabilityMeasure (μ r) := hProb
  exact p02_lm009_moment40_weighted_r3 (μ r) (W r) (R r) hW hW0 hR
    r ε M cW cZ hr (hrr0.trans hr0.2) hε hM hcW hcZ hRi hMb hsecond hlower

/-- A globally supplied probability family is a specialization of the interval
version, so the previous interface is retained without changing its source. -/
theorem p02_lm009_moment40_family_r3_of_interval
    (μ : ℝ → Measure Ω) [∀ r, IsProbabilityMeasure (μ r)]
    (W R : ℝ → Ω → ℝ) (r0 ε M cW cZ : ℝ)
    (hr0 : 0 < r0 ∧ r0 ≤ 1) (hε : 0 < ε)
    (hM : 0 ≤ M) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hdata : ∀ r, 0 < r → r ≤ r0 →
      MemLp (W r) 2 (μ r) ∧ (0 ≤ᵐ[μ r] W r) ∧ Measurable (R r) ∧
      Integrable (fun x => R r x ^ 40) (μ r) ∧
      (∫ x, R r x ^ 40 ∂(μ r)) ≤ M ∧
      (∫ x, W r x ^ 2 ∂(μ r)) ≤ cW * r ^ 4 ∧
      cZ * r ^ 2 ≤ ∫ x, W r x ∂(μ r)) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ r, 0 < r → r ≤ r0 →
      IsProbabilityMeasure (weightedLaw (μ r) (W r)) ∧
      (weightedLaw (μ r) (W r) (badJetEvent (R r) r ε)).toReal ≤ C * r ^ 3 :=
  p02_lm009_moment40_family_r3_interval μ W R r0 ε M cW cZ hr0 hε hM hcW hcZ
    (fun r hr hrr0 => ⟨inferInstance, hdata r hr hrr0⟩)

end ResearchFormalCoreR1
