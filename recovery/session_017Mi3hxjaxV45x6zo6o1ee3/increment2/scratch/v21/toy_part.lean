/-- **`ridge_capHyp_D` is not vacuous.** `h1ToyF` meets every hypothesis of `ridge_capHyp_D` with
`ε₀ = 1/2`. `T3` is constant, so `T4 = 0` and `K = 0`, and `|f_xxx| = 2 = m` on the pin segment.
The theorem chooses the collar width itself. It gives the ridge on `(−2 − ε, 2 + ε)` for some
`ε > 0` and the cap with `M = (−1/2, 0)`, from hypotheses on `D` and qualitative collar data only. -/
theorem h1Toy_D :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
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
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4) ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hD3 : ∀ p : ℝ × ℝ, ∀ a b : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ =>
      ((((2 * q.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) a) b)
      ((2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
    intro p a b
    have := (HasFDerivAt.const_mul (hf p) (2 * a.1 * b.1)).add_const (-40 * a.2 * b.2)
    convert this using 1
    funext q
    simp
    ring
  have hT : ∀ p : ℝ × ℝ, ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : ℝ),
      |((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
          s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
            ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0)| ≤
        2 * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
    intro p s₁ s₂ s₃ y₁ y₂ y₃
    have h0 : ((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
        s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
          ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = 0 := by
      simp
      ring
    rw [h0, abs_zero]
    have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
    have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
      (by positivity)
    linarith
  have hsrc := ridge_capHyp_D (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (D2f := fun p => ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
        (ContinuousLinearMap.fst ℝ ℝ ℝ) +
      ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ))
    (T3 := fun _ a b => (2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
    (T4 := fun _ _ _ _ => 0) (Φ4 := fun _ => 0) (n := 0)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (K := 0) (lam := 40) (ε₀ := 1 / 2) one_pos
    (by norm_num) le_rfl
    (fun p _ => hDf p) (fun p _ => hD2 p) (fun p _ a b => hD3 p a b)
    (fun p _ a b c => hasDerivAt_const p.1 _) (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by simp)
    (fun p _ => hT p) (fun x _ => by simp) (fun p _ => hasFDerivAt_const _ p) (fun p _ => by simp)
    (fun p _ => by simp) (by norm_num)
    (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num)
    (by simp only [h1ToyF]; norm_num)
  obtain ⟨ε, hε, -, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩
