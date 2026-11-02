import SourceJetLaw

set_option autoImplicit false
set_option maxRecDepth 3000
set_option maxHeartbeats 800000

namespace SourceContactRank

theorem contact_phase_elimination (s co R S : ℝ)
    (hunit : s ^ 2 + co ^ 2 = 1)
    (hcos : R * co - S * s = 0)
    (hsin : R * s + S * co = 0) : R = 0 ∧ S = 0 := by
  constructor
  · calc
      R = R * (s ^ 2 + co ^ 2) := by rw [hunit]; ring
      _ = (R * co - S * s) * co + (R * s + S * co) * s := by ring
      _ = 0 := by rw [hcos, hsin]; ring
  · calc
      S = S * (s ^ 2 + co ^ 2) := by rw [hunit]; ring
      _ = (R * s + S * co) * co - (R * co - S * s) * s := by ring
      _ = 0 := by rw [hcos, hsin]; ring

theorem contact_polynomial_separation {d : ℕ} (c : ℝ) (hc : c ≠ 0)
    (u V W : Fin d → ℝ) (hu : u ≠ 0) (A C : ℝ) (hA : A = 0)
    (hS1 : ∀ i, c * V i - C * (c * u i) ^ 3 = 0)
    (hS2 : ∀ i, 2 * c * V i - C * (2 * c * u i) ^ 3 = 0)
    (hR1 : ∀ i, A - (c * u i) * (c * W i) = 0)
    (hR2 : ∀ i j, i ≠ j → A - (c * (u i + u j)) * (c * (W i + W j)) = 0) :
    C = 0 ∧ V = 0 ∧ W = 0 := by
  classical
  have hex : ∃ i, u i ≠ 0 := by
    by_contra h
    apply hu
    funext i
    exact not_not.mp ((not_exists.mp h) i)
  obtain ⟨i, hi⟩ := hex
  have hcu : c * u i ≠ 0 := mul_ne_zero hc hi
  have hCprod : C * (c * u i) ^ 3 = 0 := by
    nlinarith [hS1 i, hS2 i]
  have hC : C = 0 := (mul_eq_zero.mp hCprod).resolve_right (pow_ne_zero 3 hcu)
  have hV : V = 0 := by
    funext j
    have h := hS1 j
    rw [hC, zero_mul, sub_zero] at h
    exact (mul_eq_zero.mp h).resolve_left hc
  have hdiag : ∀ j, u j * W j = 0 := by
    intro j
    have h : c ^ 2 * (u j * W j) = 0 := by nlinarith [hR1 j]
    exact (mul_eq_zero.mp h).resolve_left (pow_ne_zero 2 hc)
  have hWi : W i = 0 := (mul_eq_zero.mp (hdiag i)).resolve_left hi
  have hW : W = 0 := by
    funext j
    by_cases hij : i = j
    · simpa [← hij] using hWi
    · have hcross : c ^ 2 * (u i * W j + u j * W i) = 0 := by
        nlinarith [hR1 i, hR1 j, hR2 i j hij]
      have hcross' := (mul_eq_zero.mp hcross).resolve_left (pow_ne_zero 2 hc)
      rw [hWi, mul_zero, add_zero] at hcross'
      exact (mul_eq_zero.mp hcross').resolve_left hi
  exact ⟨hC, hV, hW⟩

end SourceContactRank

