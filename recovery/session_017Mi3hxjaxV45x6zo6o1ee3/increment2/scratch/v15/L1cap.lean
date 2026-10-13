import CapFirstExit

open Set

namespace CapFirstExit

section L1

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem hermite_gap_le {g G : ℝ → ℝ} {a c m : ℝ} (hac : a < c)
    (hg : ∀ x ∈ Icc a c, HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc a c) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc a c) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G a = 0) (hGc : G c = 0) :
    g a - g c ≤ m * (c - a) ^ 3 / 12 := by
  have hpoly : ∀ x, HasDerivAt (fun t => (c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
      ((x - a) * (c - x)) x := by
    intro x
    have h1 : HasDerivAt (fun t => t - a) 1 x := (hasDerivAt_id x).sub_const a
    have h2 := ((h1.pow 2).const_mul (c - a)).div_const 2
    have h3 := (h1.pow 3).div_const 3
    convert h2.sub h3 using 1
    push_cast
    ring
  set k : ℝ → ℝ := fun t => g t + m / 2 * ((c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
    with hk
  have hkd : ∀ x ∈ Icc a c, HasDerivAt k (G x + m / 2 * ((x - a) * (c - x))) x :=
    fun x hx => (hg x hx).add ((hpoly x).const_mul (m / 2))
  have hmono : MonotoneOn k (Icc a c) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a c)
      (fun x hx => (hkd x hx).continuousAt.continuousWithinAt)
      (fun x hx => (hkd x (interior_subset hx)).hasDerivWithinAt) (fun x hx => ?_)
    have hx := interior_subset hx
    have hb := two_node_bound hcv hcc (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac
      hGa hGc hx
    have hP : (x - a) * (x - c) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith [hx.1]) (by linarith [hx.2])
    rw [abs_of_nonpos hP] at hb
    have := (abs_le.mp hb).1
    nlinarith
  have hkac := hmono (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
  simp only [hk] at hkac
  have e0 : m / 2 * ((c - a) * (a - a) ^ 2 / 2 - (a - a) ^ 3 / 3) = 0 := by ring
  have e1 : m / 2 * ((c - a) * (c - a) ^ 2 / 2 - (c - a) ^ 3 / 3) = m * (c - a) ^ 3 / 12 := by
    ring
  linarith

theorem hermite_m_ge_two {g G : ℝ → ℝ} {r m : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G (-r / 2) = 0) (hGc : G (r / 2) = 0) (hgap : g (-r / 2) - g (r / 2) = r ^ 3 / 6) :
    2 ≤ m := by
  have h := hermite_gap_le (by linarith) hg hcv hcc hGa hGc
  have hcr : r / 2 - -r / 2 = r := by ring
  rw [hgap, hcr] at h
  have hr3 : 0 < r ^ 3 := by positivity
  nlinarith

end L1

section Normalized

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem ridge_capHyp_normalized {f : ℝ × E → ℝ} {h : ℝ → E} {F G0 : ℝ → ℝ}
    {w : ℝ → E →L[ℝ] ℝ} {r m δ : ℝ} (hr : 0 < r)
    (hg0 : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (G0 x) x)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t - m / 2 * t ^ 2))
    (hG0a : G0 (-r / 2) = 0) (hG0c : G0 (r / 2) = 0)
    (hδ : r * m * (8 * m - 5) < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hha : h (-r / 2) = 0) (hhc : h (r / 2) = 0)
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) = r ^ 3 / 6) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  have hm : 2 ≤ m := hermite_m_ge_two hr hg0 hG0cv hG0cc hG0a hG0c (by simpa [hha, hhc] using hnorm)
  exact ridge_capHyp_sp hr hm hδ hh hg hFa hFc hconv hconc hball hcrit hha hhc hw hwcv hwcc
    hnorm.le

end Normalized

section NormalizedToy

theorem spToy_normalized :
    CapHyp spToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (spToyF (-1 / 2, 0)) (spToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → spToyF x < spToyF (1 / 2, 0) := by
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => spToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 (0 : ℝ)).const_mul 12).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext y
      simp only [spToyF]
    · simp
  have hgx : ∀ x : ℝ, HasDerivAt (fun x => spToyF (x, 0)) (x ^ 2 - 1 / 4) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [spToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
    ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
  refine ridge_capHyp_normalized (E := ℝ) (f := spToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (G0 := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0) (r := 1)
    (m := 2) (δ := 24) one_pos (fun x _ => hgx x) ?_ ?_ (by norm_num) (by norm_num)
    (by norm_num) continuousOn_const (fun x _ => hgx x) (by norm_num) (by norm_num) ?_ ?_
    (fun x _ => by simp) (fun x _ => hslice x) rfl rfl (fun x _ => hslice x) ?_ ?_ ?_
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    ring
  · refine (hq.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [spToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · simp only [spToyF]
    norm_num

end NormalizedToy

end CapFirstExit
