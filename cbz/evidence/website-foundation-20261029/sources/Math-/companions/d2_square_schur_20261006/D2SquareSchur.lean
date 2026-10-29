import ResearchFormalCoreR1.D2Schur

/-!
Four real scalar statements from #328 sections 3 and 5. This module does not
construct a measure, a field, or a conditional covariance. All strict moment
and direction assumptions remain in the exact-type Contract.
-/
namespace D2SquareSchur

open ResearchFormalCoreR1
set_option autoImplicit false

theorem weighted_square_identity (u v a b c : ℝ) (huv : u + v ≠ 0) :
    u * (a-c)^2 + v * (b-c)^2 =
      u*v/(u+v)*(a-b)^2 + (u+v)*(c-(u*a+v*b)/(u+v))^2 := by
  field_simp [huv]
  ring

theorem weighted_square_lower (u v a b c : ℝ) (hu : 0 < u) (hv : 0 < v) :
    u*v/(u+v)*(a-b)^2 ≤ u*(a-c)^2 + v*(b-c)^2 := by
  rw [weighted_square_identity u v a b c (ne_of_gt (add_pos hu hv))]
  exact le_add_of_nonneg_right (mul_nonneg (add_pos hu hv).le (sq_nonneg _))

/-- Endpoint upper control of the denominator, not lower-bound substitution. -/
theorem schur_no_upper (m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2^2 < m4)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1/4) :
    min (m4-m2^2) (2*m2^2) ≤ d2Schur m2 m4 q := by
  let x : ℝ := m2^2
  let g : ℝ := m4-x
  let A : ℝ := x*m4
  let B : ℝ := g*(m4+3*x)/4
  let N : ℝ := d2Numerator m2 m4
  let h : ℝ := min g (2*x)
  have hx : 0 < x := sq_pos_of_pos hm2
  have hg : 0 < g := sub_pos.mpr hm4
  have hm4pos : 0 < m4 := lt_trans hx hm4
  have hA : 0 < A := mul_pos hx hm4pos
  have hB : 0 < B := by
    dsimp [B]
    positivity
  have hNA : N-g*A = x^2*g := by
    dsimp [N, g, A, x, d2Numerator]
    ring
  have hNB : N-(2*x)*B = x*g^2/2 := by
    dsimp [N, g, B, x, d2Numerator]
    ring
  have hga : g*A ≤ N := by
    have hp := mul_nonneg (sq_nonneg x) hg.le
    linarith [hNA]
  have h2xb : (2*x)*B ≤ N := by
    have hp : 0 ≤ x*g^2/2 := div_nonneg (mul_nonneg hx.le (sq_nonneg g)) (by norm_num)
    linarith [hNB]
  have hmulA : h*A ≤ N :=
    le_trans (mul_le_mul_of_nonneg_right (min_le_left g (2*x)) hA.le) hga
  have hmulB : h*B ≤ N :=
    le_trans (mul_le_mul_of_nonneg_right (min_le_right g (2*x)) hB.le) h2xb
  have hM : 0 < max A B := lt_of_lt_of_le hA (le_max_left _ _)
  have hmax : h*max A B ≤ N := by
    rcases le_total A B with hab | hba
    · simpa only [max_eq_right hab] using hmulB
    · simpa only [max_eq_left hba] using hmulA
  have hquot : h ≤ N/max A B := (le_div_iff₀ hM).2 hmax
  exact le_trans hquot (d2_schur_endpoint_lower m2 m4 q hm2 hm4 hq0 hq1)

theorem schur_from_lower_inputs (m2 m4 q s0 g0 : ℝ) (hs0 : 0 < s0) (hg0 : 0 < g0)
    (hm2 : s0 ≤ m2) (hgap : g0 ≤ m4-m2^2) (hq0 : 0 ≤ q) (hq1 : q ≤ 1/4) :
    min g0 (2*s0^2) ≤ d2Schur m2 m4 q := by
  have hm2pos : 0 < m2 := lt_of_lt_of_le hs0 hm2
  have hm4 : m2^2 < m4 := by linarith
  have hsquare : s0^2 ≤ m2^2 := by
    have hp := mul_nonneg (sub_nonneg.mpr hm2) (add_nonneg hm2pos.le hs0.le)
    nlinarith
  have hmin : min g0 (2*s0^2) ≤ min (m4-m2^2) (2*m2^2) :=
    min_le_min hgap (by linarith)
  exact le_trans hmin (schur_no_upper m2 m4 q hm2pos hm4 hq0 hq1)

end D2SquareSchur
