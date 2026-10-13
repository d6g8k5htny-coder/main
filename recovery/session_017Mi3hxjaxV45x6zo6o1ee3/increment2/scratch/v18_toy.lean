/-- **`ridge_capHyp_source` is not vacuous.** `h1ToyF` meets every source hypothesis on the full
domain `[-5/2, 5/2] × B̄(0, 2)`: joint derivative `(x² - 1/4, -40y)`, transverse gradient `-40y`
with derivative `-40 · snd`, constant transverse Hessian `-40`, `m = 2`, `λ = 40 > 32`, pins
`(∓1/2, 0)` and gap `1/6`. The theorem then gives the ridge, and for `h = 0`, which meets L12
(`F = x² - 1/4`), the cap with `M = (-1/2, 0)`. -/
theorem h1Toy_source :
    (∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-5 * 1 / 2 : ℝ) (5 * 1 / 2), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
      hasFDerivAt_fst
    have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      hasFDerivAt_snd
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3)) (HasFDerivAt.const_mul hf (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp
        (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hH : ∀ p : ℝ × ℝ, HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
    intro p
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
    funext y
    ext
    simp
  have hsrc := ridge_capHyp_source (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (lam := 40) one_pos
    (fun p _ => hDf p) (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun x _ => by simp only [ContinuousLinearMap.opNorm_zero]; norm_num)
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => by simp only [ContinuousLinearMap.opNorm_zero]; positivity)
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num) ?_ ?_ ?_ ?_ (by simp only [h1ToyF]; norm_num)
  · obtain ⟨hex, hcap⟩ := hsrc
    have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
      intro x
      have := slice_hasFDerivAt (hDf (x, 0))
      convert this using 1
      ext
      simp
    have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
      ((hq.subset (subset_univ _) (convex_Icc _ _)).congr fun x _ => by
        simp only [smul_eq_mul]
        simp
        ring)
    exact ⟨hex, hC⟩
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    simp
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    simp
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp
    ring
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp
    ring

