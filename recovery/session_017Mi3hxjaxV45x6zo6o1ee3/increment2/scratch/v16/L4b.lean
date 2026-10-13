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

end CapFirstExit
