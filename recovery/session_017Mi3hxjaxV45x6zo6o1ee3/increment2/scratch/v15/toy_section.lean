
/-! ### The ridge toy through the slice route -/

/-- The ridge toy's joint derivative along the ridge `y = 0` is `D x = (x² - 1/4) · dx`. Its
transverse part vanishes (the ridge equation), and `D x (1, 0) = x² - 1/4 = F x`. -/
theorem ridgeToy_joint (x : ℝ) :
    HasFDerivAt ridgeToyF ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (x, 0) ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ) =
        0 ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = x ^ 2 - 1 / 4 := by
  refine ⟨?_, by ext; simp, by simp⟩
  have hu : HasDerivAt (fun t : ℝ => t ^ 3 / 3 - t / 4) (x ^ 2 - 1 / 4) x := by
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext t; simp
    · push_cast; ring
  have hv : HasDerivAt (fun t : ℝ => t ^ 2) 0 (0 : ℝ) := by
    simpa using hasDerivAt_pow 2 (0 : ℝ)
  have h1 := hu.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_fst (𝕜 := ℝ))
  have h2 := hv.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_snd (𝕜 := ℝ))
  convert h1.sub h2 using 1
  · funext p; simp [ridgeToyF]
  · simp

/-- **The slice route is jointly satisfiable.** The ridge toy (`r = 1`, `h = 0`) meets every
input of `ridge_inputs_of_joint` and `ridge_capHyp_of_slices`, with `δ = 2` (each slice is
`-y²` plus a constant), `w = 0`, `ω = 0` and gap `1/6 < 2δ⁻¹ (2δ)² / 2`; the combined theorems
re-derive (C1)–(C3) and (C2⁺). -/
theorem ridgeToy_slices :
    CapHyp ridgeToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (ridgeToyF (-1 / 2, 0)) (ridgeToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → ridgeToyF x < ridgeToyF (1 / 2, 0) := by
  obtain ⟨hh, hg, hcrit⟩ := ridge_inputs_of_joint (E := ℝ) (f := ridgeToyF)
    (h := fun _ => (0 : ℝ)) (h' := fun _ => (0 : ℝ))
    (D := fun x => (x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (r := 1) one_pos
    (fun x _ => (ridgeToy_joint x).1) (fun x _ => hasDerivAt_const x (0 : ℝ))
    (fun x _ => (ridgeToy_joint x).2.1)
  refine ridge_capHyp_of_slices (E := ℝ) (f := ridgeToyF) (h := fun _ => (0 : ℝ))
    (w := fun _ => 0) (r := 1) (δ := 2) (ω := 0) one_pos two_pos hh hg ?_ ?_ ?_ ?_ ?_
    hcrit (fun x hx => hcrit x hx) ?_ ?_ ?_ ?_
  · simp only [(ridgeToy_joint _).2.2]
    norm_num
  · simp only [(ridgeToy_joint _).2.2]
    norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [(ridgeToy_joint _).2.2, smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [ridgeToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro x _
    simp
  · intro x _
    simp
  · norm_num
  · simp only [ridgeToyF]
    norm_num
  · simp only [ridgeToyF]
    norm_num
