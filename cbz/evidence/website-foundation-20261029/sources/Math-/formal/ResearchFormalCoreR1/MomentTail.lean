import ResearchFormalCoreR1.WeightedLaw

/-!
# From a genuine fortieth moment to a weighted bad-event bound

This module supplies Markov's measure-theoretic step. It does not prove a
fortieth-moment estimate for a Gaussian jet, or a concrete Palm normalizer.
All law/weight/jet objects and uniform family premises remain explicit.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The exact strict threshold event used by the fifth-power jet ledger. -/
def badJetEvent (R : Ω → ℝ) (r ε : ℝ) : Set Ω :=
  {x | ε < r * R x ^ 5}

/-- Markov for any event contained in a positive superlevel set, on a finite
measure. Integrability prevents totalized divergent integrals from being used. -/
theorem p02_lm009_markov_event
    (μ : Measure Ω) [IsFiniteMeasure μ] (Y : Ω → ℝ) (A : Set Ω)
    (hY : Integrable Y μ) (hY0 : 0 ≤ᵐ[μ] Y)
    (a : ℝ) (ha : 0 < a) (hsub : A ⊆ {x | a ≤ Y x}) :
    (μ A).toReal ≤ (∫ x, Y x ∂μ) / a := by
  have hm : (μ A).toReal ≤ (μ {x | a ≤ Y x}).toReal :=
    ENNReal.toReal_mono (by finiteness) (measure_mono hsub)
  have hmarkov := mul_meas_ge_le_integral_of_nonneg hY0 hY a
  apply (le_div_iff₀ ha).2
  calc
    (μ A).toReal * a = a * (μ A).toReal := by ring
    _ ≤ a * (μ {x | a ≤ Y x}).toReal := mul_le_mul_of_nonneg_left hm ha.le
    _ ≤ ∫ x, Y x ∂μ := by simpa only [measureReal_def] using hmarkov

/-- Actual measurability of the bad event from measurability of the jet norm. -/
theorem p02_lm009_badJet_measurable
    (R : Ω → ℝ) (hR : Measurable R) (r ε : ℝ) :
    MeasurableSet (badJetEvent R r ε) := by
  unfold badJetEvent
  exact measurableSet_lt measurable_const (by fun_prop)

/-- The event itself implies the needed positive fifth-power threshold, even
for signed R; no global sign assumption on R is necessary. -/
theorem p02_lm009_badJet_subset
    (R : Ω → ℝ) (r ε : ℝ) (hr : 0 < r) (hε : 0 < ε) :
    badJetEvent R r ε ⊆ {x | (ε / r) ^ 8 ≤ R x ^ 40} := by
  intro x hx
  change ε < r * R x ^ 5 at hx
  change (ε / r) ^ 8 ≤ R x ^ 40
  have ht := p02_lm009_bad_event_threshold r (R x) ε hr hx
  have hp := pow_le_pow_left₀ (le_of_lt (div_pos hε hr)) ht.le 8
  simpa only [p02_lm009_power40_identity] using hp

/-- A supplied integrable fortieth-moment bound gives the actual eighth-order
event tail, with the exact epsilon denominator and no independence hypothesis. -/
theorem p02_lm009_moment40_tail
    (μ : Measure Ω) [IsProbabilityMeasure μ] (R : Ω → ℝ)
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

/-- The constructed weighted probability law satisfies the stronger fourth-
order tail. The model's moment and positive lower-normalizer bounds are premises. -/
theorem p02_lm009_moment40_weighted_r4
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W R : Ω → ℝ)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hR : Measurable R)
    (r ε M cW cZ : ℝ) (hr : 0 < r) (hε : 0 < ε)
    (hM : 0 ≤ M) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hRi : Integrable (fun x => R x ^ 40) μ)
    (hMbound : (∫ x, R x ^ 40 ∂μ) ≤ M)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ) :
    IsProbabilityMeasure (weightedLaw μ W) ∧
      (weightedLaw μ W (badJetEvent R r ε)).toReal ≤
        ((Real.sqrt cW / cZ) * Real.sqrt (M / ε ^ 8)) * r ^ 4 := by
  have hA := p02_lm009_badJet_measurable R hR r ε
  have ht := p02_lm009_moment40_tail μ R r ε M hr hε hRi hMbound
  have hK : 0 ≤ M / ε ^ 8 := by positivity
  have hs : Real.sqrt ((μ (badJetEvent R r ε)).toReal) ≤
      Real.sqrt (M / ε ^ 8) * r ^ 4 := by
    calc
      _ ≤ Real.sqrt ((M / ε ^ 8) * r ^ 8) := Real.sqrt_le_sqrt ht
      _ = _ := by
        rw [Real.sqrt_mul hK]
        rw [show r ^ 8 = (r ^ 4) ^ 2 by ring, Real.sqrt_sq (pow_nonneg hr.le 4)]
  obtain ⟨hp, hb⟩ := p02_lm008_probability_transfer μ W (badJetEvent R r ε)
    hW hW0 hA r cW cZ hr hcW hcZ hsecond hlower
  refine ⟨hp, hb.trans ?_⟩
  calc
    _ ≤ (Real.sqrt cW / cZ) * (Real.sqrt (M / ε ^ 8) * r ^ 4) :=
      mul_le_mul_of_nonneg_left hs (by positivity)
    _ = _ := by ring

/-- Cubic consequence on the explicit domain 0 < r ≤ 1. -/
theorem p02_lm009_moment40_weighted_r3
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W R : Ω → ℝ)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hR : Measurable R)
    (r ε M cW cZ : ℝ) (hr : 0 < r) (hr1 : r ≤ 1) (hε : 0 < ε)
    (hM : 0 ≤ M) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hRi : Integrable (fun x => R x ^ 40) μ)
    (hMbound : (∫ x, R x ^ 40 ∂μ) ≤ M)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ) :
    IsProbabilityMeasure (weightedLaw μ W) ∧
      (weightedLaw μ W (badJetEvent R r ε)).toReal ≤
        ((Real.sqrt cW / cZ) * Real.sqrt (M / ε ^ 8)) * r ^ 3 := by
  obtain ⟨hp, hb⟩ := p02_lm009_moment40_weighted_r4 μ W R hW hW0 hR
    r ε M cW cZ hr hε hM hcW hcZ hRi hMbound hsecond hlower
  exact ⟨hp, p02_lm009_palm_r4_to_r3 _ _ r (by positivity) hr.le hr1 hb⟩

/-- A uniform family conclusion only when all law-specific bounds are supplied
uniformly. This proves the quantifier assembly, not the existence of model bounds. -/
theorem p02_lm009_moment40_family_r3
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
      (weightedLaw (μ r) (W r) (badJetEvent (R r) r ε)).toReal ≤ C * r ^ 3 := by
  refine ⟨(Real.sqrt cW / cZ) * Real.sqrt (M / ε ^ 8), by positivity, ?_⟩
  intro r hr hrr0
  obtain ⟨hW, hW0, hR, hRi, hMb, hsecond, hlower⟩ := hdata r hr hrr0
  exact p02_lm009_moment40_weighted_r3 (μ r) (W r) (R r) hW hW0 hR
    r ε M cW cZ hr (hrr0.trans hr0.2) hε hM hcW hcZ hRi hMb hsecond hlower

end ResearchFormalCoreR1
