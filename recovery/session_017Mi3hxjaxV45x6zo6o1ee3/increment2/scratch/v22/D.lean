import CapFirstExit

open Set unitInterval Filter Topology

namespace CapFirstExit

section Slice

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem block_of_norm {B : ContinuousMultilinearMap ℝ (fun _ : Fin 3 => ℝ × E) ℝ} {K : ℝ}
    (hB : ‖B‖ ≤ K) (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E) :
    |B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![(1, 0), (1, 0), (1, 0)]| ≤
      K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  have hK : 0 ≤ K := (norm_nonneg _).trans hB
  have hb : ∀ a b c : ℝ × E, |B ![a, b, c]| ≤ K * ‖a‖ * ‖b‖ * ‖c‖ := by
    intro a b c
    have h := B.le_opNorm ![a, b, c]
    simp only [Fin.prod_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons] at h
    rw [← Real.norm_eq_abs]
    calc ‖B ![a, b, c]‖ ≤ ‖B‖ * (‖a‖ * ‖b‖ * ‖c‖) := h
      _ ≤ K * (‖a‖ * ‖b‖ * ‖c‖) := by gcongr
      _ = K * ‖a‖ * ‖b‖ * ‖c‖ := by ring
  have h1 : ∀ x x' b c : ℝ × E, B ![x, b, c] - B ![x', b, c] = B ![x - x', b, c] := by
    intro x x' b c
    have hu : ∀ z, Function.update ![x, b, c] 0 z = ![z, b, c] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![x, b, c] 0 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  have h2 : ∀ a x x' c : ℝ × E, B ![a, x, c] - B ![a, x', c] = B ![a, x - x', c] := by
    intro a x x' c
    have hu : ∀ z, Function.update ![a, x, c] 1 z = ![a, z, c] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![a, x, c] 1 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  have h3 : ∀ a b x x' : ℝ × E, B ![a, b, x] - B ![a, b, x'] = B ![a, b, x - x'] := by
    intro a b x x'
    have hu : ∀ z, Function.update ![a, b, x] 2 z = ![a, b, z] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![a, b, x] 2 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  set e : ℝ × E := ((1 : ℝ), (0 : E)) with he_def
  have hsm : B ![s₁ • e, s₂ • e, s₃ • e] = s₁ * s₂ * s₃ * B ![e, e, e] := by
    have := B.map_smul_univ ![s₁, s₂, s₃] ![e, e, e]
    have hv : (fun i => (![s₁, s₂, s₃] : Fin 3 → ℝ) i • (![e, e, e] : Fin 3 → ℝ × E) i) =
        ![s₁ • e, s₂ • e, s₃ • e] := by
      funext i; fin_cases i <;> simp
    rw [hv, Fin.prod_univ_three] at this
    simpa [smul_eq_mul, mul_assoc] using this
  have hd : ∀ (s : ℝ) (y : E), ((s, y) : ℝ × E) - s • e = ((0 : ℝ), y) := by
    intro s y; ext <;> simp [he_def]
  have hn : ∀ (s : ℝ) (y : E), ‖((s, y) : ℝ × E)‖ ≤ |s| + ‖y‖ := by
    intro s y
    rw [Prod.norm_def, Real.norm_eq_abs]
    exact max_le_add_of_nonneg (abs_nonneg _) (norm_nonneg _)
  have hn0 : ∀ y : E, ‖((0 : ℝ), y)‖ = ‖y‖ := by intro y; simp [Prod.norm_def]
  have hne : ∀ s : ℝ, ‖s • e‖ = |s| := by
    intro s; rw [norm_smul, Real.norm_eq_abs]; simp [he_def, Prod.norm_def]
  have tel : B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![e, e, e] =
      B ![((0 : ℝ), y₁), (s₂, y₂), (s₃, y₃)] + B ![s₁ • e, ((0 : ℝ), y₂), (s₃, y₃)] +
        B ![s₁ • e, s₂ • e, ((0 : ℝ), y₃)] := by
    rw [← hsm, ← hd s₁ y₁, ← hd s₂ y₂, ← hd s₃ y₃, ← h1, ← h2, ← h3]
    ring
  rw [tel]
  have k1 : |B ![((0 : ℝ), y₁), (s₂, y₂), (s₃, y₃)]| ≤
      K * ‖y₁‖ * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) := by
    refine (hb _ _ _).trans ?_
    rw [hn0]
    gcongr
    · exact hn _ _
    · exact hn _ _
  have k2 : |B ![s₁ • e, ((0 : ℝ), y₂), (s₃, y₃)]| ≤ K * |s₁| * ‖y₂‖ * (|s₃| + ‖y₃‖) := by
    refine (hb _ _ _).trans ?_
    rw [hn0, hne]
    gcongr
    exact hn _ _
  have k3 : |B ![s₁ • e, s₂ • e, ((0 : ℝ), y₃)]| ≤ K * |s₁| * |s₂| * ‖y₃‖ := by
    refine (hb _ _ _).trans ?_
    rw [hn0, hne, hne]
  calc _ ≤ _ := abs_add_three _ _ _
    _ ≤ K * ‖y₁‖ * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) + K * |s₁| * ‖y₂‖ * (|s₃| + ‖y₃‖) +
        K * |s₁| * |s₂| * ‖y₃‖ := by gcongr
    _ = _ := by ring

omit [NormedSpace ℝ E] in
theorem collar_subset_open [ProperSpace E] {U : Set (ℝ × E)} {r : ℝ} (hr : 0 < r) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U) :
    ∃ ε₀ > 0, Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U := by
  obtain ⟨δ, hδ, hsub⟩ :=
    (isCompact_Icc.prod (isCompact_closedBall (0 : E) (2 * r))).exists_cthickening_subset_open hU hDU
  refine ⟨δ, hδ, ?_⟩
  rintro ⟨x, y⟩ ⟨hx, hy⟩
  apply hsub
  obtain ⟨x₀, hx₀, hxx₀⟩ : ∃ x₀ ∈ Icc (-2 * r) (2 * r), |x - x₀| ≤ δ := by
    by_cases h1 : x ≤ -2 * r
    · refine ⟨-2 * r, ⟨le_rfl, ?_⟩, ?_⟩
      · linarith [hx.1]
      · rw [abs_le]; constructor <;> linarith [hx.1]
    · by_cases h2 : 2 * r ≤ x
      · refine ⟨2 * r, ⟨?_, le_rfl⟩, ?_⟩
        · linarith
        · rw [abs_le]; constructor <;> linarith [hx.2]
      · exact ⟨x, ⟨by linarith, by linarith⟩, by simp [hδ.le]⟩
  refine Metric.mem_cthickening_of_dist_le (x, y) (x₀, y) δ _ ⟨hx₀, hy⟩ ?_
  rw [Prod.dist_eq, dist_self, Real.dist_eq]
  exact max_le hxx₀ hδ.le

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

end Slice

theorem h1Toy_E :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
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
  have hcd : ContDiff ℝ 4 h1ToyF := by unfold h1ToyF; fun_prop
  have hfd : fderiv ℝ h1ToyF = fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ := funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ),
      iteratedFDeriv ℝ 2 h1ToyF y m = 2 * y.1 * (m 0).1 * (m 1).1 - 40 * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ),
      iteratedFDeriv ℝ 3 h1ToyF x m = 2 * (m 0).1 * (m 1).1 * (m 2).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 h1ToyF y (Fin.tail m))
        ((2 * (m 1).1 * (m 2).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have := ((hf x).const_mul (2 * (m 1).1 * (m 2).1)).sub_const (40 * (m 1).2 * (m 2).2)
      convert this using 1
      funext y; rw [h2]; simp [Fin.tail]; ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ), iteratedFDeriv ℝ 4 h1ToyF x m = 0 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : (fun y => iteratedFDeriv ℝ 3 h1ToyF y (Fin.tail m)) =
        fun _ => 2 * (m 1).1 * (m 2).1 * (m 3).1 := by
      funext y; rw [h3]; simp [Fin.tail]
    rw [hg]
    simp
  have hsrc := ridge_capHyp_E (E := ℝ) (f := h1ToyF) (U := univ) (r := 1) (m := 2) (n := 0)
    (lam := 40) one_pos isOpen_univ (subset_univ _) hcd.contDiffOn
    (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by
      rw [h3, h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons]
      have h0 : 2 * s₁ * s₂ * s₃ - s₁ * s₂ * s₃ * (2 * 1 * 1 * 1) = 0 := by ring
      rw [h0, abs_zero]
      exact mul_nonneg (by norm_num) (block_nonneg s₁ s₂ s₃ y₁ y₂ y₃))
    (fun x _ => by rw [h3]; simp)
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (by norm_num)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [h1ToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩

end CapFirstExit
