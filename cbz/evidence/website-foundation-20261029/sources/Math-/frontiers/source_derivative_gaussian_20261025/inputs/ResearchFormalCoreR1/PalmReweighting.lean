import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Measure.WithDensity
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# A canonical weighted probability and its square-root event bound

This constructs the literal measure with density `W / integral W`, proves its
event formula, and applies integral Cauchy–Schwarz. The determinant second
moment and normalizer estimates cancel their exact radius powers. Identifying
this measure with a geometric Kac–Rice Palm law remains an application join.
See ../SOURCE_PALM_REWEIGHTING.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

/-- The literal normalized nonnegative real density, defined for every input. -/
noncomputable def prwMeasure {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) : Measure Ω :=
  μ.withDensity (fun ω => ENNReal.ofReal (W ω / ∫ x, W x ∂μ))

/-- On a finite measure, a measurable square-integrable real weight is integrable. -/
theorem prw_integrable_of_integrable_sq {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (W : Ω → ℝ) (hW : Measurable W)
    (hsq : Integrable (fun ω => W ω ^ 2) μ) : Integrable W μ := by
  have hmajor : Integrable (fun ω => 1 + W ω ^ 2) μ :=
    (integrable_const 1).add hsq
  apply hmajor.mono' hW.aestronglyMeasurable
  filter_upwards [] with ω
  rw [Real.norm_eq_abs]
  nlinarith [sq_nonneg (|W ω| - 1 / 2), sq_abs (W ω)]

/-- The actual event mass is the normalized set integral, in extended reals. -/
theorem prw_measure_apply {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) (hW : Integrable W μ)
    (hW0 : ∀ ω, 0 ≤ W ω) (hZ : 0 < ∫ x, W x ∂μ)
    (E : Set Ω) (hE : MeasurableSet E) :
    prwMeasure μ W E = ENNReal.ofReal ((∫ x in E, W x ∂μ) / ∫ x, W x ∂μ) := by
  unfold prwMeasure
  rw [withDensity_apply _ hE]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (hW.integrableOn.div_const _) (Filter.Eventually.of_forall (fun ω =>
      div_nonneg (hW0 ω) hZ.le))]
  rw [integral_div]

/-- Positive finite normalizer makes the literal density a probability law. -/
theorem prw_isProbabilityMeasure {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) (hW : Integrable W μ)
    (hW0 : ∀ ω, 0 ≤ W ω) (hZ : 0 < ∫ x, W x ∂μ) :
    IsProbabilityMeasure (prwMeasure μ W) := by
  constructor
  rw [prw_measure_apply μ W hW hW0 hZ Set.univ MeasurableSet.univ]
  simp [hZ.ne']

/-- The event identity is proved for the constructed law, not assumed. -/
theorem prw_measure_real {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) (hW : Integrable W μ)
    (hW0 : ∀ ω, 0 ≤ W ω) (hZ : 0 < ∫ x, W x ∂μ)
    (E : Set Ω) (hE : MeasurableSet E) :
    (prwMeasure μ W).real E = (∫ x in E, W x ∂μ) / ∫ x, W x ∂μ := by
  rw [measureReal_def, prw_measure_apply μ W hW hW0 hZ E hE,
    ENNReal.toReal_ofReal (div_nonneg (integral_nonneg hW0) hZ.le)]

/-- Integral Cauchy–Schwarz against the event indicator, including null events. -/
theorem prw_numerator_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (W : Ω → ℝ) (hW : Measurable W)
    (hW0 : ∀ ω, 0 ≤ W ω) (hsq : Integrable (fun ω => W ω ^ 2) μ)
    (E : Set Ω) (hE : MeasurableSet E) :
    (∫ x in E, W x ∂μ) ≤
      Real.sqrt (∫ x, W x ^ 2 ∂μ) * Real.sqrt (μ.real E) := by
  classical
  let I : Ω → ℝ := E.indicator (fun _ => 1)
  have hI0 : ∀ ω, 0 ≤ I ω := by
    intro ω
    simp only [I, Set.indicator_apply]
    split_ifs <;> norm_num
  have hWLp : MemLp W (ENNReal.ofReal 2) μ := by
    simpa using (memLp_two_iff_integrable_sq hW.aestronglyMeasurable).mpr hsq
  have hILp : MemLp I (ENNReal.ofReal 2) μ :=
    (memLp_const (1 : ℝ)).indicator hE
  have hCS := integral_mul_le_Lp_mul_Lq_of_nonneg Real.HolderConjugate.two_two
    (Filter.Eventually.of_forall hW0) (Filter.Eventually.of_forall hI0) hWLp hILp
  have hleft : (fun ω => W ω * I ω) = E.indicator W := by
    funext ω
    by_cases hω : ω ∈ E <;> simp [I, hω]
  have hright : (fun ω => I ω ^ (2 : ℝ)) = E.indicator (fun _ => (1 : ℝ)) := by
    funext ω
    by_cases hω : ω ∈ E <;> simp [I, hω]
  have hIint : (∫ ω, I ω ^ (2 : ℝ) ∂μ) = μ.real E := by
    rw [hright]
    change (∫ ω, E.indicator (1 : Ω → ℝ) ω ∂μ) = μ.real E
    exact integral_indicator_one hE
  rw [hleft, integral_indicator hE, hIint] at hCS
  simpa only [Real.rpow_two, ← Real.sqrt_eq_rpow] using hCS

/-- The exact `r^4` second moment and `r^2` normalizer cancel in the literal
weighted probability. No event probability is assumed positive. -/
theorem prw_square_root_event_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (hW : Measurable W)
    (hW0 : ∀ ω, 0 ≤ W ω) (hsq : Integrable (fun ω => W ω ^ 2) μ)
    (C c r : ℝ) (hC : 0 ≤ C) (hc : 0 < c) (hr : 0 < r)
    (hmoment : (∫ x, W x ^ 2 ∂μ) ≤ C * r ^ 4)
    (hfloor : c * r ^ 2 ≤ ∫ x, W x ∂μ)
    (E : Set Ω) (hE : MeasurableSet E) :
    IsProbabilityMeasure (prwMeasure μ W) ∧
      (prwMeasure μ W).real E ≤ (Real.sqrt C / c) * Real.sqrt (μ.real E) := by
  have hZ : 0 < ∫ x, W x ∂μ :=
    (mul_pos hc (sq_pos_of_pos hr)).trans_le hfloor
  have hW1 : Integrable W μ := prw_integrable_of_integrable_sq μ W hW hsq
  refine ⟨prw_isProbabilityMeasure μ W hW1 hW0 hZ, ?_⟩
  rw [prw_measure_real μ W hW1 hW0 hZ E hE]
  have hnum := prw_numerator_bound μ W hW hW0 hsq E hE
  have hsqrt : Real.sqrt (∫ x, W x ^ 2 ∂μ) ≤ Real.sqrt C * r ^ 2 := by
    calc
      _ ≤ Real.sqrt (C * r ^ 4) := Real.sqrt_le_sqrt hmoment
      _ = Real.sqrt C * r ^ 2 := by
        rw [Real.sqrt_mul hC]
        congr 1
        have heq : r ^ 4 = (r ^ 2) ^ 2 := by ring
        rw [heq, Real.sqrt_sq (sq_nonneg r)]
  calc
    _ ≤ (Real.sqrt C * r ^ 2 * Real.sqrt (μ.real E)) / (c * r ^ 2) := by
      apply div_le_div₀ (by positivity)
        (hnum.trans (mul_le_mul_of_nonneg_right hsqrt (Real.sqrt_nonneg _)))
        (mul_pos hc (sq_pos_of_pos hr)) hfloor
    _ = (Real.sqrt C / c) * Real.sqrt (μ.real E) := by
      field_simp [hr.ne']

end ResearchFormalCoreR1
