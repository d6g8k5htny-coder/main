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

