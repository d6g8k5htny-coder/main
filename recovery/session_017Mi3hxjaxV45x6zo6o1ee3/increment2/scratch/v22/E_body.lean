theorem ridge_capHyp_E [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r m n lam : ℝ} (hr : 0 < r) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |iteratedFDeriv ℝ 3 f p ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] -
          s₁ * s₂ * s₃ * iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)]| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 f (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ m)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 f p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ n * ‖v‖)
    (hH2 : r * n ≤ 1 / 20)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 8 * r * m ^ 2 < lam)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    ∃ ε, 0 < ε ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  obtain ⟨ε₀, hε₀, hcU⟩ := collar_subset_open hr hU hDU
  have hcpt : IsCompact (Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r)) :=
    isCompact_Icc.prod (isCompact_closedBall _ _)
  -- regularity of the derivatives on `U`
  have hf1 : ContDiffOn ℝ 3 (fderiv ℝ f) U := hf.fderiv_of_isOpen hU (by norm_num)
  have hf2 : ContDiffOn ℝ 2 (fderiv ℝ (fderiv ℝ f)) U := hf1.fderiv_of_isOpen hU (by norm_num)
  have dDf : ∀ p ∈ U, HasFDerivAt f (fderiv ℝ f p) p := fun p hp =>
    ((hf.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have dD2 : ∀ p ∈ U, HasFDerivAt (fderiv ℝ f) (fderiv ℝ (fderiv ℝ f) p) p := fun p hp =>
    ((hf1.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have dD3 : ∀ p ∈ U, HasFDerivAt (fderiv ℝ (fderiv ℝ f)) (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) p) p :=
    fun p hp => ((hf2.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have sD2 : ∀ p ∈ U, HasStrictFDerivAt (fderiv ℝ f) (fderiv ℝ (fderiv ℝ f) p) p := fun p hp =>
    (hf1.contDiffAt (hU.mem_nhds hp)).hasStrictFDerivAt (by norm_num)
  have dI2 : ∀ p ∈ U, HasFDerivAt (iteratedFDeriv ℝ 2 f) (fderiv ℝ (iteratedFDeriv ℝ 2 f) p) p :=
    fun p hp => ((hf.contDiffAt (hU.mem_nhds hp)).differentiableAt_iteratedFDeriv
      (by norm_num)).hasFDerivAt
  have dI3 : ∀ p ∈ U, HasFDerivAt (iteratedFDeriv ℝ 3 f) (fderiv ℝ (iteratedFDeriv ℝ 3 f) p) p :=
    fun p hp => ((hf.contDiffAt (hU.mem_nhds hp)).differentiableAt_iteratedFDeriv
      (by norm_num)).hasFDerivAt
  have hc4 : ContinuousOn (iteratedFDeriv ℝ 4 f) U :=
    (hf.continuousOn_iteratedFDerivWithin le_rfl hU.uniqueDiffOn).congr
      (fun x hx => (iteratedFDerivWithin_of_isOpen 4 hU hx).symm)
  -- the bound `K` on `D⁴f` over the compact collar
  obtain ⟨C, hC⟩ := hcpt.exists_bound_of_continuousOn (hc4.mono hcU)
  have hD2 : ∀ (p : ℝ × E) (a b : ℝ × E),
      fderiv ℝ (fderiv ℝ f) p a b = iteratedFDeriv ℝ 2 f p ![a, b] := by
    intro p a b; rw [iteratedFDeriv_two_apply]; rfl
  have he : ‖((1 : ℝ), (0 : E))‖ = 1 := by simp [Prod.norm_def]
  have hpath : ∀ (x : ℝ) (y : E), HasDerivAt (fun t : ℝ => ((t, y) : ℝ × E)) ((1 : ℝ), (0 : E)) x :=
    fun x y => (hasDerivAt_id x).prodMk (hasDerivAt_const x y)
  set Ψ : ((ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] ℝ) :=
    ContinuousLinearMap.precomp ℝ (ContinuousLinearMap.inr ℝ ℝ E) with hΨ
  set Ξ : ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] E →L[ℝ] ℝ) :=
    (ContinuousLinearMap.postcomp E Ψ).comp
      (ContinuousLinearMap.precomp ((ℝ × E) →L[ℝ] ℝ) (ContinuousLinearMap.inr ℝ ℝ E)) with hΞ
  have hn0 : 0 ≤ n := (abs_nonneg _).trans (hn4x (0, 0) ⟨⟨by linarith, by linarith⟩, by
    simp only [Metric.mem_closedBall, dist_self]; linarith⟩)
  have key := ridge_capHyp_D (E := E) (f := f) (Df := fderiv ℝ f) (D2f := fderiv ℝ (fderiv ℝ f))
    (T3 := fun p a b => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]).comp
      (fderiv ℝ (iteratedFDeriv ℝ 2 f) p))
    (T4 := fun p a b c => iteratedFDeriv ℝ 4 f p ![(1, 0), c, a, b])
    (Φ4 := fun p => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ
      ![(1, 0), (1, 0), (1, 0)]).comp (fderiv ℝ (iteratedFDeriv ℝ 3 f) p)) (n := n)
    (DG := fun p => Ψ.comp (fderiv ℝ (fderiv ℝ f) p))
    (H := fun p => Ξ (fderiv ℝ (fderiv ℝ f) p))
    (Hx := fun x => Ξ (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) (x, 0) (1, 0)))
    (Hr := fun p s => Ξ (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) (p.1, s • p.2) (0, p.2)))
    (r := r) (m := m) (K := max C 0) (lam := lam) (ε₀ := ε₀) hr hε₀ (le_max_right _ _)
    (fun p hp => dDf p (hcU hp)) (fun p hp => dD2 p (hcU hp))
    (fun p hp a b => by
      have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]).hasFDerivAt.comp
        p (dI2 p (hcU hp))
      have hfun : (fun q => fderiv ℝ (fderiv ℝ f) q a b) =
          (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]) ∘
            iteratedFDeriv ℝ 2 f := funext fun q => hD2 q a b
      rw [hfun]; exact h)
    (fun p hp a b c => by
      have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ ![c, a, b]).hasFDerivAt.comp_hasDerivAt
        p.1 ((dI3 (p.1, p.2) (hcU hp)).comp_hasDerivAt p.1 (hpath p.1 p.2))
      exact h)
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => by
      have hB : ‖(iteratedFDeriv ℝ 4 f p).curryLeft ((1 : ℝ), (0 : E))‖ ≤ max C 0 := by
        refine ((iteratedFDeriv ℝ 4 f p).curryLeft.le_opNorm _).trans ?_
        rw [ContinuousMultilinearMap.curryLeft_norm, he, mul_one]
        exact (hC p hp).trans (le_max_left _ _)
      have h := block_of_norm hB s₃ s₁ s₂ y₃ y₁ y₂
      show |iteratedFDeriv ℝ 4 f p ![(1, 0), (s₃, y₃), (s₁, y₁), (s₂, y₂)] -
        s₁ * s₂ * s₃ * iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ _
      rw [show s₁ * s₂ * s₃ = s₃ * s₁ * s₂ by ring]
      refine h.trans (le_of_eq ?_)
      ring)
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => by
      have h := hT p hp s₃ s₁ s₂ y₃ y₁ y₂
      show |iteratedFDeriv ℝ 3 f p ![(s₃, y₃), (s₁, y₁), (s₂, y₂)] -
        s₁ * s₂ * s₃ * iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)]| ≤ _
      rw [show s₁ * s₂ * s₃ = s₃ * s₁ * s₂ by ring]
      refine h.trans (le_of_eq ?_)
      ring)
    (fun x hx => hTe x hx)
    (fun p hp => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ
      ![(1, 0), (1, 0), (1, 0)]).hasFDerivAt.comp p (dI3 p (hDU hp)))
    (fun p hp => hn4x p hp)
    (fun p hp => ContinuousLinearMap.opNorm_le_bound _ hn0 fun v => by
      rw [Real.norm_eq_abs]; exact hn4y p hp v)
    hH2
    (fun p hp => Ψ.hasStrictFDerivAt.comp p (sD2 p (hcU hp)))
    (fun p hp => Ψ.hasFDerivAt.comp p.2 ((dD2 (p.1, p.2) (hcU hp)).comp p.2
      (hasFDerivAt_prodMk_right p.1 p.2)))
    (fun x hx => by
      have hx' : ((x, 0) : ℝ × E) ∈ U := hcU ⟨hx, by
        simp only [Metric.mem_closedBall, dist_self]; linarith⟩
      have hΞ : HasFDerivAt Ξ Ξ (fderiv ℝ (fderiv ℝ f) (x, 0)) :=
        ContinuousLinearMap.hasFDerivAt (E := (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)
          (F := E →L[ℝ] E →L[ℝ] ℝ) _
      exact HasFDerivAt.comp_hasDerivAt (l := Ξ)
        (f := fun t => fderiv ℝ (fderiv ℝ f) (t, 0)) x hΞ (HasFDerivAt.comp_hasDerivAt
          (l := fderiv ℝ (fderiv ℝ f)) (f := fun t : ℝ => ((t, (0 : E)) : ℝ × E)) x (dD3 (x, 0) hx')
          (hpath x 0)))
    (fun p hp s hs => by
      have hmem : ((p.1, s • p.2) : ℝ × E) ∈ U := hcU ⟨hp.1, by
        have := hp.2
        rw [Metric.mem_closedBall, dist_zero_right] at this ⊢
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hs.1]
        nlinarith [hs.2, norm_nonneg p.2]⟩
      have hp' : HasDerivAt (fun s : ℝ => ((p.1, s • p.2) : ℝ × E)) ((0 : ℝ), p.2) s := by
        have := (hasDerivAt_const s p.1).prodMk ((hasDerivAt_id s).smul_const p.2)
        simpa using this
      have hΞ : HasFDerivAt Ξ Ξ (fderiv ℝ (fderiv ℝ f) (p.1, s • p.2)) :=
        ContinuousLinearMap.hasFDerivAt (E := (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)
          (F := E →L[ℝ] E →L[ℝ] ℝ) _
      exact HasFDerivAt.comp_hasDerivAt (l := Ξ)
        (f := fun s => fderiv ℝ (fderiv ℝ f) (p.1, s • p.2)) s hΞ (HasFDerivAt.comp_hasDerivAt
          (l := fderiv ℝ (fderiv ℝ f)) (f := fun s : ℝ => ((p.1, s • p.2) : ℝ × E)) s (dD3 _ hmem) hp'))
    (fun v => by
      have h := hlam v
      rw [← hD2] at h
      exact h)
    hH1 hM hS hnorm
  obtain ⟨ε, hε, -, hrest⟩ := key
  exact ⟨ε, hε, hrest⟩

