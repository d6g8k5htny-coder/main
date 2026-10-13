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

end L4

end CapFirstExit
