import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

noncomputable section
open scoped BigOperators
namespace I4TwoPinAlgebra

def phase (d : ℕ) (c : ℝ) (k : Fin d → ℤ) (x : Fin d → ℝ) : ℝ :=
  c * ∑ j, (k j : ℝ) * x j

def cosSymbol (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ) (k : Fin d → ℤ) : ℝ :=
  A * Real.cos (phase d c k x) - phase d c k B * Real.sin (phase d c k x) +
    C * Real.cos (phase d c k y) - phase d c k D * Real.sin (phase d c k y)

def sinSymbol (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ) (k : Fin d → ℤ) : ℝ :=
  A * Real.sin (phase d c k x) + phase d c k B * Real.cos (phase d c k x) +
    C * Real.sin (phase d c k y) + phase d c k D * Real.cos (phase d c k y)

private def unit (d : ℕ) (i : Fin d) : Fin d → ℤ := Pi.single i 1

private theorem phase_unit (d : ℕ) (c : ℝ) (i : Fin d) (z : Fin d → ℝ) :
    phase d c (unit d i) z = c * z i := by
  classical
  simp [phase, unit, Pi.single_apply]

private theorem phase_add (d : ℕ) (c : ℝ) (k l : Fin d → ℤ) (z : Fin d → ℝ) :
    phase d c (k + l) z = phase d c k z + phase d c l z := by
  simp only [phase, Pi.add_apply, Int.cast_add, add_mul, Finset.sum_add_distrib, mul_add]

private theorem phase_sub (d : ℕ) (c : ℝ) (k l : Fin d → ℤ) (z : Fin d → ℝ) :
    phase d c (k - l) z = phase d c k z - phase d c l z := by
  simp only [phase, Pi.sub_apply, Int.cast_sub, sub_mul, Finset.sum_sub_distrib, mul_sub]

private def sineShift (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ) (k : Fin d → ℤ) (t : ℝ) : ℝ :=
  A * Real.sin (phase d c k x - t) + phase d c k B * Real.cos (phase d c k x - t) +
    C * Real.sin (phase d c k y - t) + phase d c k D * Real.cos (phase d c k y - t)

private theorem sineShift_zero (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ)
    (hcos : ∀ k, cosSymbol d c x y A C B D k = 0)
    (hsin : ∀ k, sinSymbol d c x y A C B D k = 0) (k : Fin d → ℤ) (t : ℝ) :
    sineShift d c x y A C B D k t = 0 := by
  calc
    _ = sinSymbol d c x y A C B D k * Real.cos t -
        cosSymbol d c x y A C B D k * Real.sin t := by
      simp only [sineShift, sinSymbol, cosSymbol, Real.sin_sub, Real.cos_sub]
      ring
    _ = 0 := by rw [hcos k, hsin k]; ring

private theorem cosineShift_zero (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ)
    (hcos : ∀ k, cosSymbol d c x y A C B D k = 0)
    (hsin : ∀ k, sinSymbol d c x y A C B D k = 0) (k : Fin d → ℤ) (t : ℝ) :
    A * Real.cos (phase d c k x - t) - phase d c k B * Real.sin (phase d c k x - t) +
      C * Real.cos (phase d c k y - t) - phase d c k D * Real.sin (phase d c k y - t) = 0 := by
  calc
    _ = cosSymbol d c x y A C B D k * Real.cos t +
        sinSymbol d c x y A C B D k * Real.sin t := by
      simp only [sinSymbol, cosSymbol, Real.sin_sub, Real.cos_sub]
      ring
    _ = 0 := by rw [hcos k, hsin k]; ring

/-- The product test `(1-cos(c*(t_i-y_i)))*sin(c*(t_j-x_j))`
has zero jets at y and isolates B_j at x. Its three sine frequencies are
e_j and e_j plus or minus e_i; the two shifted sums below are its exact product-to-sum expansion. -/
private theorem left_gradient_zero (d : ℕ) (c : ℝ) (hc : c ≠ 0)
    (x y : Fin d → ℝ) (A C : ℝ) (B D : Fin d → ℝ) (i : Fin d)
    (hsep : Real.cos (c * (x i - y i)) ≠ 1)
    (hcos : ∀ k, cosSymbol d c x y A C B D k = 0)
    (hsin : ∀ k, sinSymbol d c x y A C B D k = 0) : B = 0 := by
  classical
  funext j
  have h0 := sineShift_zero d c x y A C B D hcos hsin (unit d j) (c * x j)
  have hp := sineShift_zero d c x y A C B D hcos hsin
    (unit d j + unit d i) (c * x j + c * y i)
  have hm := sineShift_zero d c x y A C B D hcos hsin
    (unit d j - unit d i) (c * x j - c * y i)
  simp only [sineShift, phase_add, phase_sub, phase_unit] at h0 hp hm
  have hxy : c * y j - c * x j = c * (y j - x j) := by ring
  have hpx : c * x j + c * x i - (c * x j + c * y i) = c * (x i - y i) := by ring
  have hpy : c * y j + c * y i - (c * x j + c * y i) = c * (y j - x j) := by ring
  have hmx : c * x j - c * x i - (c * x j - c * y i) = -(c * (x i - y i)) := by ring
  have hmy : c * y j - c * y i - (c * x j - c * y i) = c * (y j - x j) := by ring
  simp only [sub_self, Real.sin_zero, Real.cos_zero, mul_zero, mul_one, zero_add,
    hxy] at h0
  rw [hpx, hpy] at hp
  rw [hmx, hmy, Real.sin_neg, Real.cos_neg] at hm
  have hz : c * B j * (1 - Real.cos (c * (x i - y i))) = 0 := by
    linear_combination h0 - (hp + hm) / 2
  have hfactor : 1 - Real.cos (c * (x i - y i)) ≠ 0 := by
    intro h
    apply hsep
    linarith
  have hb : c * B j = 0 := (mul_eq_zero.mp hz).resolve_right hfactor
  exact (mul_eq_zero.mp hb).resolve_left hc

theorem trigJet_separation (d : ℕ) (c : ℝ) (hc : c ≠ 0)
    (x y : Fin d → ℝ) (A C : ℝ) (B D : Fin d → ℝ)
    (hsep : ∃ i, Real.cos (c * (x i - y i)) ≠ 1)
    (hcos : ∀ k : Fin d → ℤ, cosSymbol d c x y A C B D k = 0)
    (hsin : ∀ k : Fin d → ℤ, sinSymbol d c x y A C B D k = 0) :
    A = 0 ∧ C = 0 ∧ B = 0 ∧ D = 0 := by
  classical
  rcases hsep with ⟨i, hi⟩
  have hB := left_gradient_zero d c hc x y A C B D i hi hcos hsin
  have hi' : Real.cos (c * (y i - x i)) ≠ 1 := by
    have he : c * (y i - x i) = -(c * (x i - y i)) := by ring
    simpa only [he, Real.cos_neg] using hi
  have hc' (k : Fin d → ℤ) : cosSymbol d c y x C A D B k = 0 := by
    calc
      _ = cosSymbol d c x y A C B D k := by unfold cosSymbol; ring
      _ = 0 := hcos k
  have hs' (k : Fin d → ℤ) : sinSymbol d c y x C A D B k = 0 := by
    calc
      _ = sinSymbol d c x y A C B D k := by unfold sinSymbol; ring
      _ = 0 := hsin k
  have hD := left_gradient_zero d c hc y x C A D B i hi' hc' hs'
  have hzero := hcos (0 : Fin d → ℤ)
  have hshift := cosineShift_zero d c x y A C B D hcos hsin (unit d i) (c * y i)
  have hd : c * x i - c * y i = c * (x i - y i) := by ring
  simp only [hB, hD, cosSymbol, phase, Pi.zero_apply, Int.cast_zero, zero_mul,
    Finset.sum_const_zero, mul_zero, Real.cos_zero, Real.sin_zero, sub_zero, mul_one] at hzero
  simp only [phase_unit] at hshift
  simp only [hB, hD, Pi.zero_apply, mul_zero, sub_self,
    Real.cos_zero, mul_one, hd] at hshift
  have hz : A * (Real.cos (c * (x i - y i)) - 1) = 0 := by
    linear_combination hshift - hzero
  have hA : A = 0 := (mul_eq_zero.mp hz).resolve_right (sub_ne_zero.mpr hi)
  have hC : C = 0 := by simpa only [hA, zero_add] using hzero
  exact ⟨hA, hC, hB, hD⟩

end I4TwoPinAlgebra

