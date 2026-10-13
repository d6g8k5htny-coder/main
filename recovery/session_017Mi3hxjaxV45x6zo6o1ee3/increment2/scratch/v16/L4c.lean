import CapFirstExit

open Set

namespace CapFirstExit

section L4

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem strongConcaveOn_of_hessian {K : Set E} {φ : E → ℝ} {Dφ : E → E →L[ℝ] ℝ}
    {H : E → E →L[ℝ] E →L[ℝ] ℝ} {δ : ℝ} (hK : Convex ℝ K)
    (hD : ∀ y ∈ K, HasFDerivAt φ (Dφ y) y) (hH : ∀ y ∈ K, HasFDerivAt Dφ (H y) y)
    (hneg : ∀ y ∈ K, ∀ v : E, H y v v ≤ -δ * ‖v‖ ^ 2) : StrongConcaveOn K δ φ := by
  refine ⟨hK, fun x hx y hy a b ha hb hab => ?_⟩
  set v := y - x with hv
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1, x + t • v ∈ K := fun t ht => hK.add_smul_sub_mem hx hy ht
  set ψ : ℝ → ℝ := fun t => φ (x + t • v) + δ / 2 * ‖v‖ ^ 2 * t ^ 2 with hψ
  set ψ' : ℝ → ℝ := fun t => Dφ (x + t • v) v + δ * ‖v‖ ^ 2 * t with hψ'
  set ψ'' : ℝ → ℝ := fun t => H (x + t • v) v v + δ * ‖v‖ ^ 2 with hψ''
  have hline : ∀ t : ℝ, HasDerivAt (fun s : ℝ => x + s • v) v t := fun t => by
    simpa using ((hasDerivAt_id t).smul_const v).const_add x
  have hd1 : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt ψ (ψ' t) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => φ (x + s • v)) (Dφ (x + t • v) v) t :=
      (hD _ (hmem t ht)).comp_hasDerivAt t (hline t)
    have h2 : HasDerivAt (fun s : ℝ => δ / 2 * ‖v‖ ^ 2 * s ^ 2) (δ * ‖v‖ ^ 2 * t) t := by
      convert (hasDerivAt_pow 2 t).const_mul (δ / 2 * ‖v‖ ^ 2) using 1
      push_cast
      ring
    exact h1.add h2
  have hd2 : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt ψ' (ψ'' t) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => Dφ (x + s • v)) (H (x + t • v) v) t :=
      (hH _ (hmem t ht)).comp_hasDerivAt t (hline t)
    have h1' : HasDerivAt (fun s => Dφ (x + s • v) v) (H (x + t • v) v v) t := by
      simpa using h1.clm_apply (hasDerivAt_const t v)
    have h2 : HasDerivAt (fun s : ℝ => δ * ‖v‖ ^ 2 * s) (δ * ‖v‖ ^ 2) t := by
      simpa using (hasDerivAt_id t).const_mul (δ * ‖v‖ ^ 2)
    exact h1'.add h2
  have hcc : ConcaveOn ℝ (Icc (0 : ℝ) 1) ψ := by
    refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc 0 1)
      (fun t ht => (hd1 t ht).continuousAt.continuousWithinAt)
      (fun t ht => (hd1 t (interior_subset ht)).hasDerivWithinAt)
      (fun t ht => (hd2 t (interior_subset ht)).hasDerivWithinAt) (fun t ht => ?_)
    have := hneg _ (hmem t (interior_subset ht)) v
    simp only [hψ'']
    linarith
  have key := hcc.2 (left_mem_Icc.mpr zero_le_one) (right_mem_Icc.mpr zero_le_one) ha hb hab
  simp only [hψ, smul_eq_mul, mul_zero, zero_smul, add_zero, mul_one, one_smul, zero_add] at key
  have hab' : a = 1 - b := by linarith
  have hpt : a • x + b • y = x + b • v := by
    rw [hab', hv, smul_sub, sub_smul, one_smul]; abel
  have hn : ‖x - y‖ = ‖v‖ := by rw [hv, norm_sub_rev]
  rw [hpt, hn, smul_eq_mul, smul_eq_mul]
  have hxv : x + v = y := by rw [hv]; abel
  rw [hxv] at key
  show a * φ x + b * φ y + a * b * (δ / 2 * ‖v‖ ^ 2) ≤ φ (x + b • v)
  rw [hab'] at key ⊢
  linear_combination key

theorem transverse_hessian_le {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r)
    (hm : 0 ≤ m)
    (hlip : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      H p v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ v
  have hy' : ‖y‖ ≤ 2 * r := by simpa using hy
  have hdx : |x - -r / 2| ≤ 3 * r / 2 := by
    rw [abs_le]; constructor <;> linarith [hx.1, hx.2]
  have hinc : ‖H (x, y) - H (-r / 2, 0)‖ ≤ 5 * r * m := by
    have := hlip (x, y) ⟨hx, hy⟩
    calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ m * (|x - -r / 2| + ‖y‖) := this
      _ ≤ m * (3 * r / 2 + 2 * r) := by gcongr
      _ ≤ 5 * r * m := by nlinarith
  have hsplit : H (x, y) v v = H (-r / 2, 0) v v + (H (x, y) - H (-r / 2, 0)) v v := by
    simp
  have hb := (H (x, y) - H (-r / 2, 0)).le_opNorm₂ v v
  have hb' : (H (x, y) - H (-r / 2, 0)) v v ≤ 5 * r * m * ‖v‖ ^ 2 := by
    have h1 := (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using hb)
    calc (H (x, y) - H (-r / 2, 0)) v v ≤ ‖H (x, y) - H (-r / 2, 0)‖ * ‖v‖ * ‖v‖ := h1
      _ ≤ 5 * r * m * ‖v‖ * ‖v‖ := by gcongr
      _ = 5 * r * m * ‖v‖ ^ 2 := by ring
  rw [hsplit]
  have := hlam v
  nlinarith

theorem slices_strongConcave {f : ℝ × E → ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r) (hm : 0 ≤ m)
    (hD : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hlip : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) (lam - 5 * r * m) (fun y => f (x, y)) := by
  intro x hx
  have hbd := transverse_hessian_le hr hm hlip hlam
  exact strongConcaveOn_of_hessian (Dφ := fun y => Dφ (x, y)) (H := fun y => H (x, y))
    (convex_closedBall _ _) (fun y hy => hD (x, y) ⟨hx, hy⟩) (fun y hy => hH (x, y) ⟨hx, hy⟩)
    (fun y hy v => hbd (x, y) ⟨hx, hy⟩ v)

end L4

section H1

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem ridge_capHyp_H1 {f : ℝ × E → ℝ} {h : ℝ → E} {F G0 : ℝ → ℝ}
    {w : ℝ → E →L[ℝ] ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ} {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ}
    {r m lam : ℝ} (hr : 0 < r)
    (hg0 : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (G0 x) x)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t - m / 2 * t ^ 2))
    (hG0a : G0 (-r / 2) = 0) (hG0c : G0 (r / 2) = 0)
    (hD : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hlip : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
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
  have hm : 2 ≤ m :=
    hermite_m_ge_two hr hg0 hG0cv hG0cc hG0a hG0c (by simpa [hha, hhc] using hnorm)
  have hconc := slices_strongConcave hr (by linarith) hD hH hlip hlam
  have hδ : r * m * (8 * m - 5) < lam - 5 * r * m := by nlinarith
  exact ridge_capHyp_normalized hr hg0 hG0cv hG0cc hG0a hG0c hδ hh hg hFa hFc hconv hconc hball
    hcrit hha hhc hw hwcv hwcc hnorm

end H1

section H1Toy

noncomputable def h1ToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2

theorem h1Toy_capHyp :
    CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → h1ToyF x < h1ToyF (1 / 2, 0) := by
  have hslope : ∀ x y : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y))
      (ContinuousLinearMap.mul ℝ ℝ (-40 * y)) y := by
    intro x y
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 y).const_mul 20).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext t
      simp only [h1ToyF]
    · simp
      ring
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    simpa using hslope x 0
  have hhess : ∀ y : ℝ, HasFDerivAt (fun y : ℝ => ContinuousLinearMap.mul ℝ ℝ (-40 * y))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) y := by
    intro y
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := y) using 1
    funext t
    ext
    simp
  have hgx : ∀ x : ℝ, HasDerivAt (fun x => h1ToyF (x, 0)) (x ^ 2 - 1 / 4) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [h1ToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
    ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
  refine ridge_capHyp_H1 (E := ℝ) (f := h1ToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (G0 := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0)
    (Dφ := fun p => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (r := 1) (m := 2) (lam := 40)
    one_pos (fun x _ => hgx x) ?_ ?_ (by norm_num) (by norm_num)
    (fun p _ => hslope p.1 p.2) (fun p _ => hhess p.2) ?_ ?_ (by norm_num)
    continuousOn_const (fun x _ => hgx x) (by norm_num) (by norm_num) ?_
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
  · intro p hp
    have hp2 : 0 ≤ |p.1 - -1 / 2| + ‖p.2‖ := by positivity
    simp only [sub_self, ContinuousLinearMap.opNorm_zero]
    linarith
  · intro v
    simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul,
      Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg v]
  · refine (hq.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
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
  · simp only [h1ToyF]
    norm_num

end H1Toy

end CapFirstExit
