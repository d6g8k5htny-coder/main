import ResearchFormalCoreR1.ProbabilityCompanionsV2

/-!
# Measure-theoretic companions to P02-LM-008

These declarations use one measure and one weight throughout. They supply the
Cauchy-Schwarz step rather than assuming its numerator estimate. The concrete
second-moment and lower-normalizer estimates remain hypotheses. No Gaussian
field, typed Palm measure, uniform family of constants, or parent theorem is
constructed here. See ../MEASURE_BRIDGE.md for the exact source correspondence.
-/

namespace ResearchFormalCoreR1

open MeasureTheory
open scoped ENNReal

set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Actual Bochner-integral Cauchy-Schwarz for nonnegative square-integrable
functions, obtained from the pinned mathlib Holder theorem with exponents 2,2. -/
theorem p02_lm008_integral_cs
    (μ : Measure Ω) (f g : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ)
    (hf0 : 0 ≤ᵐ[μ] f) (hg0 : 0 ≤ᵐ[μ] g) :
    (∫ x, f x * g x ∂μ) ≤
      Real.sqrt (∫ x, f x ^ 2 ∂μ) * Real.sqrt (∫ x, g x ^ 2 ∂μ) := by
  have h := integral_mul_le_Lp_mul_Lq_of_nonneg
    Real.HolderConjugate.two_two hf0 hg0
    (by simpa using hf) (by simpa using hg)
  simpa only [Real.rpow_two, ← Real.sqrt_eq_rpow] using h

/-- The event may depend arbitrarily on W. Its indicator's second moment is
its probability under this same μ; no independence assumption is used. -/
theorem p02_lm008_event_cs
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A) :
    (∫ x in A, W x ∂μ) ≤
      Real.sqrt (∫ x, W x ^ 2 ∂μ) * Real.sqrt ((μ A).toReal) := by
  classical
  have hI : MemLp (A.indicator (fun _ : Ω => (1 : ℝ))) 2 μ :=
    (memLp_const (1 : ℝ)).indicator hA
  have hI0 : 0 ≤ᵐ[μ] A.indicator (fun _ : Ω => (1 : ℝ)) := by
    apply Filter.Eventually.of_forall
    intro x
    by_cases hx : x ∈ A <;> simp [hx]
  have hcs := p02_lm008_integral_cs μ W
    (A.indicator (fun _ : Ω => (1 : ℝ))) hW hI hW0 hI0
  have hprod : (fun x => W x * A.indicator (fun _ : Ω => (1 : ℝ)) x) =
      A.indicator W := by
    funext x
    by_cases hx : x ∈ A <;> simp [hx]
  have hsquare : (fun x => (A.indicator (fun _ : Ω => (1 : ℝ)) x) ^ 2) =
      A.indicator (fun _ : Ω => (1 : ℝ)) := by
    funext x
    by_cases hx : x ∈ A <;> simp [hx]
  rw [hprod, integral_indicator hA, hsquare, integral_indicator hA] at hcs
  simpa [MeasureTheory.measureReal_def] using hcs

/-- The r^4 second-moment premise supplies the previously assumed r^2
numerator bound. This theorem does not establish that premise for a model. -/
theorem p02_lm008_event_numerator
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (r cW : ℝ) (hcW : 0 ≤ cW)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4) :
    (∫ x in A, W x ∂μ) ≤ Real.sqrt cW * r ^ 2 * Real.sqrt ((μ A).toReal) := by
  have hsqrt : Real.sqrt (∫ x, W x ^ 2 ∂μ) ≤ Real.sqrt cW * r ^ 2 := by
    calc
      Real.sqrt (∫ x, W x ^ 2 ∂μ) ≤ Real.sqrt (cW * r ^ 4) :=
        Real.sqrt_le_sqrt hsecond
      _ = Real.sqrt cW * r ^ 2 := by
        rw [Real.sqrt_mul hcW]
        rw [show r ^ 4 = (r ^ 2) ^ 2 by ring, Real.sqrt_sq (sq_nonneg r)]
  exact (p02_lm008_event_cs μ W A hW hW0 hA).trans
    (mul_le_mul_of_nonneg_right hsqrt (Real.sqrt_nonneg _))

/-- Same-law integral-ratio transfer. Positivity of the actual normalizer
is derived from its explicit lower bound, never from an upper bound. -/
theorem p02_lm008_measure_transfer
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (r cW cZ : ℝ) (hr : 0 < r) (hcW : 0 ≤ cW) (hcZ : 0 < cZ)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ) :
    (∫ x in A, W x ∂μ) / (∫ x, W x ∂μ) ≤
      (Real.sqrt cW / cZ) * Real.sqrt ((μ A).toReal) := by
  have hzpos : 0 < ∫ x, W x ∂μ :=
    lt_of_lt_of_le (mul_pos hcZ (sq_pos_of_pos hr)) hlower
  exact p02_lm008_quotient_bound
    (∫ x in A, W x ∂μ) (∫ x, W x ∂μ) r cW cZ ((μ A).toReal)
    hr hcW hcZ ENNReal.toReal_nonneg hzpos hlower
    (p02_lm008_event_numerator μ W A hW hW0 hA r cW hcW hsecond)

/-- An explicitly supplied eighth-order event bound transfers to a cubic
bound on 0 < r ≤ 1. Uniformity in r and the input tail remain unproved here. -/
theorem p02_lm009_measure_transfer_r8_to_r3
    (μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)
    (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)
    (r cW cZ K : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (hcW : 0 ≤ cW) (hcZ : 0 < cZ) (hK : 0 ≤ K)
    (hsecond : (∫ x, W x ^ 2 ∂μ) ≤ cW * r ^ 4)
    (hlower : cZ * r ^ 2 ≤ ∫ x, W x ∂μ)
    (htail : (μ A).toReal ≤ K * r ^ 8) :
    (∫ x in A, W x ∂μ) / (∫ x, W x ∂μ) ≤
      ((Real.sqrt cW / cZ) * Real.sqrt K) * r ^ 3 := by
  have hcoef : 0 ≤ Real.sqrt cW / cZ :=
    div_nonneg (Real.sqrt_nonneg _) hcZ.le
  have hsqrt : Real.sqrt ((μ A).toReal) ≤ Real.sqrt K * r ^ 4 := by
    calc
      Real.sqrt ((μ A).toReal) ≤ Real.sqrt (K * r ^ 8) := Real.sqrt_le_sqrt htail
      _ = Real.sqrt K * r ^ 4 := by
        rw [Real.sqrt_mul hK]
        rw [show r ^ 8 = (r ^ 4) ^ 2 by ring, Real.sqrt_sq (pow_nonneg hr.le 4)]
  have hfourth : (∫ x in A, W x ∂μ) / (∫ x, W x ∂μ) ≤
      ((Real.sqrt cW / cZ) * Real.sqrt K) * r ^ 4 := by
    calc
      (∫ x in A, W x ∂μ) / (∫ x, W x ∂μ) ≤
          (Real.sqrt cW / cZ) * Real.sqrt ((μ A).toReal) :=
        p02_lm008_measure_transfer μ W A hW hW0 hA r cW cZ hr hcW hcZ hsecond hlower
      _ ≤ (Real.sqrt cW / cZ) * (Real.sqrt K * r ^ 4) :=
        mul_le_mul_of_nonneg_left hsqrt hcoef
      _ = ((Real.sqrt cW / cZ) * Real.sqrt K) * r ^ 4 := by ring
  exact p02_lm009_palm_r4_to_r3 _ _ r
    (mul_nonneg hcoef (Real.sqrt_nonneg _)) hr.le hr1 hfourth

/-- Scalar data realized by W=4 on an event of probability 1/4 and W=0
elsewhere. Cauchy-Schwarz is sharp; omitting the event square root is false.
The realization is checked separately by exact finite-model Python tests. -/
theorem p02_lm008_sqrt_counterexample :
    (1 : ℝ) = Real.sqrt 4 * Real.sqrt (1 / 4 : ℝ) ∧
      ¬ (1 : ℝ) ≤ Real.sqrt 4 * (1 / 4 : ℝ) := by
  norm_num

/-- Scalar data realized by W=1 on a one-point probability space, r=1,
cW=1, cZ=2. The upper bound on z holds but the proposed ratio bound fails.
This does not assert that the required lower bound holds. -/
theorem p02_lm008_upper_normalizer_counterexample :
    (1 : ℝ) ≤ 2 * 1 ^ 2 ∧
      (1 : ℝ) / 1 > (Real.sqrt 1 / 2) * Real.sqrt 1 := by
  norm_num

end ResearchFormalCoreR1
