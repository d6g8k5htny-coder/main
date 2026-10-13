import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

section L5

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Adding the affine function `(m/2)((t - a)(t - c) - t²)` preserves convexity. -/
theorem convexOn_add_node {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2)) :
    ConvexOn ℝ D (fun t => u t + m / 2 * ((t - a) * (t - c))) := by
  refine ⟨hcv.1, fun x hx y hy p q hp hq hpq => ?_⟩
  have h := hcv.2 hx hy hp hq hpq
  simp only [smul_eq_mul] at h ⊢
  have key : (u (p * x + q * y) + m / 2 * ((p * x + q * y - a) * (p * x + q * y - c))) -
      (p * (u x + m / 2 * ((x - a) * (x - c))) + q * (u y + m / 2 * ((y - a) * (y - c)))) =
      (u (p * x + q * y) + m / 2 * (p * x + q * y) ^ 2) -
      (p * (u x + m / 2 * x ^ 2) + q * (u y + m / 2 * y ^ 2)) := by
    have hq' : q = 1 - p := by linarith
    subst hq'
    ring
  linarith

theorem two_node_bound {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ D (fun t => u t - m / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : u a = 0) (hc : u c = 0) {x : ℝ}
    (hx : x ∈ D) : |u x| ≤ m / 2 * |(x - a) * (x - c)| := by
  have hψ := convexOn_add_node (a := a) (c := c) hcv
  have hcv' : ConvexOn ℝ D (fun t => -u t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hχ := convexOn_add_node (a := a) (c := c) hcv'
  have hψa : u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hψc : u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  have hχa : -u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hχc : -u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  rcases le_or_gt x a with hxa | hxa
  · have h1 := convex_zeros_left hψ hcD hac hψa hψc hx hxa
    have h2 := convex_zeros_left hχ hcD hac hχa hχc hx hxa
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg_of_nonpos_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  rcases le_or_gt c x with hcx | hcx
  · have h1 := convex_zeros_right hψ haD hac hψa hψc hx hcx
    have h2 := convex_zeros_right hχ haD hac hχa hχc hx hcx
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  · have h1 := convex_zeros_between hψ haD hcD hac.le hψa hψc ⟨hxa.le, hcx.le⟩
    have h2 := convex_zeros_between hχ haD hcD hac.le hχa hχc ⟨hxa.le, hcx.le⟩
    have hP : (x - a) * (x - c) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonpos hP, abs_le]
    constructor <;> linarith

theorem slice_gradient_bound {w : ℝ → E →L[ℝ] ℝ} {D : Set ℝ} {a c m : ℝ} (hm : 0 ≤ m)
    (hcv : ∀ v : E, ConvexOn ℝ D (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hcc : ∀ v : E, ConcaveOn ℝ D (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : w a = 0) (hc : w c = 0) {x : ℝ}
    (hx : x ∈ D) : ‖w x‖ ≤ m / 2 * |(x - a) * (x - c)| := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have h := two_node_bound (u := fun t => w t v) (m := m * ‖v‖) (hcv v) (hcc v) haD hcD hac
    (by simp [ha]) (by simp [hc]) hx
  rw [Real.norm_eq_abs]
  calc |w x v| ≤ m * ‖v‖ / 2 * |(x - a) * (x - c)| := h
    _ = m / 2 * |(x - a) * (x - c)| * ‖v‖ := by ring

end L5

section Capstone

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem ridge_capHyp_sp {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {w : ℝ → E →L[ℝ] ℝ}
    {r m δ : ℝ} (hr : 0 < r) (hm : 2 ≤ m) (hδ : r * m * (8 * m - 5) < δ)
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
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) ≤ r ^ 3 / 6) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  have hm0 : 0 ≤ m := by linarith
  have hδ0 : 0 < δ := by
    have : 0 < r * m * (8 * m - 5) := by
      have : 0 < 8 * m - 5 := by linarith
      positivity
    linarith
  have ha : -r / 2 ∈ Icc (-2 * r) (r / 2) := ⟨by linarith, by linarith⟩
  have hc : r / 2 ∈ Icc (-2 * r) (r / 2) := ⟨by linarith, le_rfl⟩
  have hwa : w (-r / 2) = 0 := (hw _ ha).unique (by simpa [hha] using hcrit _ ha)
  have hwc : w (r / 2) = 0 := (hw _ hc).unique (by simpa [hhc] using hcrit _ hc)
  have hω : ∀ x ∈ Icc (-2 * r) (r / 2), ‖w x‖ ≤ 2 * m * r ^ 2 := by
    intro x hx
    have h1 := slice_gradient_bound hm0 hwcv hwcc ha hc (by linarith) hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 4 * r ^ 2 := by
      rw [abs_le]
      constructor <;> nlinarith [hx.1, hx.2]
    calc ‖w x‖ ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
      _ ≤ m / 2 * (4 * r ^ 2) := by gcongr
      _ = 2 * m * r ^ 2 := by ring
  obtain ⟨hωr, hcurv⟩ := sp_curved_side_constants hr hm hδ (by positivity) le_rfl
  refine ridge_capHyp_of_slices hr hδ0 hh hg hFa hFc hconv hconc hball hcrit hw hω hωr ?_ ?_
  · calc 2 * δ * (f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2))) ≤ 2 * δ * (r ^ 3 / 6) := by
          gcongr
      _ < _ := hcurv
  · have : r ^ 3 / 6 < 9 * r ^ 3 / 32 := by
      have : 0 < r ^ 3 := by positivity
      linarith
    linarith

end Capstone

section SpToy

noncomputable def spToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - 12 * p.2 ^ 2

theorem spToy_capHyp :
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
  refine ridge_capHyp_sp (E := ℝ) (f := spToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0) (r := 1) (m := 2) (δ := 24) one_pos le_rfl
    (by norm_num) continuousOn_const ?_ ?_ ?_ ?_ ?_ ?_ (fun x _ => hslice x) rfl rfl
    (fun x _ => hslice x) ?_ ?_ ?_
  · intro x _
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [spToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  · norm_num
  · norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [spToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro x _
    simp
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

end SpToy

end CapFirstExit
