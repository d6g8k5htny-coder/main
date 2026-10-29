import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Algebraic d=2 Schur and positivity companion

Dylan Roy — delegated AI work. Actual author: OpenAI / GPT-6 Astra Pro.
Source: Math-#302 NOTE section 2 and additive identity comment 6005199919.
These are real algebraic expressions with explicit hypotheses, not a
construction of the Gaussian field or a proof of the coefficient integral.
-/

namespace ResearchFormalCoreR1

noncomputable def d2Alpha (m2 m4 q : ℝ) : ℝ :=
  m4 - 2 * (m4 - 3 * m2 ^ 2) * q

noncomputable def d2Gamma (m2 m4 q : ℝ) : ℝ :=
  m2 ^ 2 + 2 * (m4 - 3 * m2 ^ 2) * q

noncomputable def d2DetV (m2 m4 q : ℝ) : ℝ :=
  m2 ^ 2 * m4 + (m4 - 3 * m2 ^ 2) * (m4 + m2 ^ 2) * q

noncomputable def d2Numerator (m2 m4 : ℝ) : ℝ :=
  m2 ^ 2 * (m4 ^ 2 - m2 ^ 4)

noncomputable def d2Schur (m2 m4 q : ℝ) : ℝ :=
  d2Alpha m2 m4 q -
    (d2Gamma m2 m4 q ^ 3 +
      (2 * d2Gamma m2 m4 q + d2Alpha m2 m4 q) *
        (m4 - 3 * m2 ^ 2) ^ 2 * q * (1 - 4 * q)) / d2DetV m2 m4 q

noncomputable def d2Delta (m2 m4 m6 : ℝ) : ℝ := m6 - m4 ^ 2 / m2

noncomputable def d2Tau (m2 m4 m6 q : ℝ) : ℝ :=
  (1 - 4 * q) * d2Delta m2 m4 m6 +
    4 * q * ((d2Delta m2 m4 m6 + 9 * m2 * (m4 - m2 ^ 2)) / 4)

/-- Exact polynomial cancellation; it does not divide by the pin determinant. -/
theorem d2_schur_numerator (m2 m4 q : ℝ) :
    d2Alpha m2 m4 q * d2DetV m2 m4 q - d2Gamma m2 m4 q ^ 3 -
      (2 * d2Gamma m2 m4 q + d2Alpha m2 m4 q) *
        (m4 - 3 * m2 ^ 2) ^ 2 * q * (1 - 4 * q) = d2Numerator m2 m4 := by
  unfold d2Alpha d2Gamma d2DetV d2Numerator
  ring

theorem d2_det_axis (m2 m4 : ℝ) : d2DetV m2 m4 0 = m2 ^ 2 * m4 := by
  simp [d2DetV]

theorem d2_det_diagonal (m2 m4 : ℝ) :
    d2DetV m2 m4 (1 / 4) = (m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4 := by
  unfold d2DetV
  ring

theorem d2_det_interpolation (m2 m4 q : ℝ) :
    d2DetV m2 m4 q = (1 - 4 * q) * (m2 ^ 2 * m4) +
      4 * q * ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) := by
  unfold d2DetV
  ring

/-- The nonzero hypothesis is essential even with Lean's total division. -/
theorem d2_schur_ratio (m2 m4 q : ℝ) (hD : d2DetV m2 m4 q ≠ 0) :
    d2Schur m2 m4 q = d2Numerator m2 m4 / d2DetV m2 m4 q := by
  have h := d2_schur_numerator m2 m4 q
  unfold d2Schur
  apply (eq_div_iff hD).2
  rw [sub_mul, div_mul_cancel₀ _ hD]
  linarith

theorem d2_numerator_pos (m2 m4 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) :
    0 < d2Numerator m2 m4 := by
  have hx : 0 < m2 ^ 2 := sq_pos_of_pos hm2
  have hv : 0 < m4 - m2 ^ 2 := sub_pos.mpr hm4
  have hs : 0 < m4 + m2 ^ 2 := by linarith
  have he : d2Numerator m2 m4 = m2 ^ 2 * (m4 - m2 ^ 2) * (m4 + m2 ^ 2) := by
    unfold d2Numerator
    ring
  rw [he]
  exact mul_pos (mul_pos hx hv) hs

theorem d2_det_pos (m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) : 0 < d2DetV m2 m4 q := by
  have hx : 0 < m2 ^ 2 := sq_pos_of_pos hm2
  have hm4pos : 0 < m4 := lt_trans hx hm4
  have ha : 0 < m2 ^ 2 * m4 := mul_pos hx hm4pos
  have hv : 0 < m4 - m2 ^ 2 := sub_pos.mpr hm4
  have hs : 0 < m4 + 3 * m2 ^ 2 := by positivity
  have hb : 0 < (m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4 := by positivity
  rw [d2_det_interpolation]
  by_cases hq : q = 0
  · simpa [hq] using ha
  · have hqp : 0 < q := lt_of_le_of_ne hq0 (Ne.symm hq)
    have hw : 0 ≤ 1 - 4 * q := by linarith
    exact add_pos_of_nonneg_of_pos (mul_nonneg hw ha.le) (mul_pos (by positivity) hb)

theorem d2_schur_pos (m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) : 0 < d2Schur m2 m4 q := by
  have hD := d2_det_pos m2 m4 q hm2 hm4 hq0 hq1
  rw [d2_schur_ratio m2 m4 q (ne_of_gt hD)]
  exact div_pos (d2_numerator_pos m2 m4 hm2 hm4) hD

theorem d2_det_le_endpoint_max (m2 m4 q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) :
    d2DetV m2 m4 q ≤
      max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) := by
  have hw : 0 ≤ 1 - 4 * q := by linarith
  have hv : 0 ≤ 4 * q := by positivity
  rw [d2_det_interpolation]
  calc
    _ ≤ (1 - 4 * q) * max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) +
        4 * q * max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) :=
      add_le_add (mul_le_mul_of_nonneg_left (le_max_left _ _) hw)
        (mul_le_mul_of_nonneg_left (le_max_right _ _) hv)
    _ = _ := by ring

/-- A uniform positive lower bound on q in [0,1/4], with moment premises explicit. -/
theorem d2_schur_endpoint_lower (m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) :
    d2Numerator m2 m4 /
        max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) ≤
      d2Schur m2 m4 q := by
  have hD := d2_det_pos m2 m4 q hm2 hm4 hq0 hq1
  have hbound := d2_det_le_endpoint_max m2 m4 q hq0 hq1
  have hM := lt_of_lt_of_le hD hbound
  rw [d2_schur_ratio m2 m4 q (ne_of_gt hD)]
  exact (div_le_div_iff₀ hM hD).2
    (mul_le_mul_of_nonneg_left hbound (d2_numerator_pos m2 m4 hm2 hm4).le)

theorem d2_tau_cumulant_eq (m2 m4 m6 q : ℝ) (hm2 : m2 ≠ 0) :
    6 * m2 ^ 3 + 9 * m2 * (m4 - 3 * m2 ^ 2) * (1 - 2 * q) +
      (m6 - 15 * m4 * m2 + 30 * m2 ^ 3 - (m4 - 3 * m2 ^ 2) ^ 2 / m2) *
        (1 - 3 * q) = d2Tau m2 m4 m6 q := by
  unfold d2Tau d2Delta
  field_simp [hm2]
  <;> ring

theorem d2_tau_axis (m2 m4 m6 : ℝ) : d2Tau m2 m4 m6 0 = d2Delta m2 m4 m6 := by
  simp [d2Tau]

theorem d2_tau_diagonal (m2 m4 m6 : ℝ) :
    d2Tau m2 m4 m6 (1 / 4) =
      (d2Delta m2 m4 m6 + 9 * m2 * (m4 - m2 ^ 2)) / 4 := by
  unfold d2Tau
  ring

theorem d2_tau_pos (m2 m4 m6 q : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4)
    (hDelta : 0 < d2Delta m2 m4 m6) (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4) :
    0 < d2Tau m2 m4 m6 q := by
  have hv : 0 < m4 - m2 ^ 2 := sub_pos.mpr hm4
  have hb : 0 < (d2Delta m2 m4 m6 + 9 * m2 * (m4 - m2 ^ 2)) / 4 := by positivity
  unfold d2Tau
  by_cases hq : q = 0
  · simpa [hq] using hDelta
  · have hqp : 0 < q := lt_of_le_of_ne hq0 (Ne.symm hq)
    have hw : 0 ≤ 1 - 4 * q := by linarith
    exact add_pos_of_nonneg_of_pos (mul_nonneg hw hDelta.le) (mul_pos (by positivity) hb)

/-- Gaussian-reference algebra, for all real q, not a probabilistic identification. -/
theorem d2_gaussian_control (q : ℝ) :
    d2DetV 1 3 q = 3 ∧ d2Schur 1 3 q = 8 / 3 ∧ d2Tau 1 3 15 q = 6 := by
  norm_num [d2DetV, d2Schur, d2Alpha, d2Gamma, d2Tau, d2Delta]
  <;> ring

/-- At a singular moment boundary the unguarded quotient identity is false. -/
theorem d2_degenerate_control :
    d2DetV 1 1 (1 / 4) = 0 ∧ d2Schur 1 1 (1 / 4) = 2 ∧
      d2Numerator 1 1 / d2DetV 1 1 (1 / 4) = 0 ∧ d2Tau 1 1 1 0 = 0 := by
  norm_num [d2DetV, d2Schur, d2Alpha, d2Gamma, d2Numerator, d2Tau, d2Delta]

end ResearchFormalCoreR1
