import Mathlib

namespace ResearchFormalCoreR1

set_option autoImplicit false

/-- P02-LM-008 algebraic companion: exact cancellation of the `r²` factor
    after the Cauchy–Schwarz numerator bound and the normalizer lower bound
    have been established in the surrounding probability theorem. -/
theorem p02_lm008_r2_cancel
    (r cW cZ q : ℝ) (hr : r ≠ 0) (hcZ : cZ ≠ 0) :
    (Real.sqrt cW * r ^ 2 * Real.sqrt q) / (cZ * r ^ 2) =
      (Real.sqrt cW / cZ) * Real.sqrt q := by
  field_simp [hr, hcZ]
  ring

/-- P02-LM-008 cross-multiplied transfer companion.  This theorem starts
    from an already supplied Cauchy–Schwarz numerator estimate and an exact
    normalizer lower bound.  It does not formalize the measure-theoretic
    Cauchy–Schwarz step itself. -/
theorem p02_lm008_cross_multiplied
    (n z r cW cZ q : ℝ)
    (hr : 0 < r) (hcW : 0 ≤ cW) (hcZ : 0 < cZ) (hq : 0 ≤ q)
    (hz : cZ * r ^ 2 ≤ z)
    (hn : n ≤ Real.sqrt cW * r ^ 2 * Real.sqrt q) :
    n * (cZ * r ^ 2) ≤
      (Real.sqrt cW * r ^ 2 * Real.sqrt q) * z := by
  have hden : 0 ≤ cZ * r ^ 2 :=
    mul_nonneg (le_of_lt hcZ) (sq_nonneg r)
  have hnum : 0 ≤ Real.sqrt cW * r ^ 2 * Real.sqrt q := by
    positivity
  calc
    n * (cZ * r ^ 2) ≤
        (Real.sqrt cW * r ^ 2 * Real.sqrt q) * (cZ * r ^ 2) := by
      exact mul_le_mul_of_nonneg_right hn hden
    _ ≤ (Real.sqrt cW * r ^ 2 * Real.sqrt q) * z := by
      exact mul_le_mul_of_nonneg_left hz hnum

/-- P02-LM-008 quotient companion.  With positive denominators, the
    cross-multiplied inequality and exact `r²` cancellation yield the Palm
    square-root transfer coefficient. -/
theorem p02_lm008_quotient_bound
    (n z r cW cZ q : ℝ)
    (hr : 0 < r) (hcW : 0 ≤ cW) (hcZ : 0 < cZ) (hq : 0 ≤ q)
    (hzpos : 0 < z) (hz : cZ * r ^ 2 ≤ z)
    (hn : n ≤ Real.sqrt cW * r ^ 2 * Real.sqrt q) :
    n / z ≤ (Real.sqrt cW / cZ) * Real.sqrt q := by
  have hscaled : 0 < cZ * r ^ 2 :=
    mul_pos hcZ (sq_pos_of_pos hr)
  have hcross :=
    p02_lm008_cross_multiplied n z r cW cZ q hr hcW hcZ hq hz hn
  rw [← p02_lm008_r2_cancel r cW cZ q (ne_of_gt hr) (ne_of_gt hcZ)]
  exact (div_le_div_iff₀ hzpos hscaled).2 hcross

/-- P02-LM-009 deterministic threshold companion.  Failure of
    `r * R^5 ≤ ε` implies the fifth-power threshold used before Markov's
    inequality. -/
theorem p02_lm009_bad_event_threshold
    (r R ε : ℝ) (hr : 0 < r) (hbad : ε < r * R ^ 5) :
    ε / r < R ^ 5 := by
  exact (div_lt_iff₀ hr).2 (by simpa [mul_comm] using hbad)

/-- The fixed exponent identity used when the 40th moment is selected in
    the P02-LM-009 application. -/
theorem p02_lm009_power40_identity (R : ℝ) :
    (R ^ 5) ^ 8 = R ^ 40 := by
  ring

/-- In the final P0.2 power ledger, an `O(r^4)` Palm tail is also `O(r^3)`
    on `0 ≤ r ≤ 1`. -/
theorem p02_lm009_r4_le_r3
    (r : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    r ^ 4 ≤ r ^ 3 := by
  have h : 0 ≤ r ^ 3 * (1 - r) :=
    mul_nonneg (pow_nonneg hr0 3) (sub_nonneg.mpr hr1)
  nlinarith

/-- Conditional rate companion: once the Palm square-root transfer has
    produced a uniform `C r^4` bound, the cubic target follows on `r ≤ 1`. -/
theorem p02_lm009_palm_r4_to_r3
    (p C r : ℝ) (hC : 0 ≤ C) (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (hp : p ≤ C * r ^ 4) :
    p ≤ C * r ^ 3 := by
  calc
    p ≤ C * r ^ 4 := hp
    _ ≤ C * r ^ 3 := by
      exact mul_le_mul_of_nonneg_left (p02_lm009_r4_le_r3 r hr0 hr1) hC

end ResearchFormalCoreR1
