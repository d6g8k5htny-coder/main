import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

section PartialBlock

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

theorem iteratedFDeriv_three_swap₁₂ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![b, a, c] := by
  have h2 : ContDiffAt ℝ 2 (fderiv ℝ f) p := hf.fderiv_right (by norm_num)
  have hs : IsSymmSndFDerivAt ℝ (fderiv ℝ f) p := h2.isSymmSndFDerivAt (by simp)
  have hr : ∀ x y : F, iteratedFDeriv ℝ 3 f p ![x, y, c] =
      iteratedFDeriv ℝ 2 (fderiv ℝ f) p ![x, y] c := by
    intro x y
    rw [iteratedFDeriv_succ_apply_right]
    have e : Fin.init ![x, y, c] = ![x, y] := by ext i; fin_cases i <;> rfl
    rw [e]
    rfl
  rw [hr, hr, IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hs)]

theorem iteratedFDeriv_three_swap₂₃ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![a, c, b] := by
  -- `D²f` is symmetric near `p`, so its derivative at `p` is symmetric in the last two slots.
  have hev : (fun q => iteratedFDeriv ℝ 2 f q ![b, c]) =ᶠ[𝓝 p]
      (fun q => iteratedFDeriv ℝ 2 f q ![c, b]) := by
    filter_upwards [hf.eventually (by simp)] with q hq
    have hq2 : ContDiffAt ℝ 2 f q := hq.of_le (by norm_num)
    exact IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hq2.isSymmSndFDerivAt (by simp))
  have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ 2 f) p :=
    (hf.iteratedFDeriv_right (i := 2) (m := 1) (by norm_num)).differentiableAt (by norm_num)
  have key : ∀ v : Fin 2 → F, fderiv ℝ (fun q => iteratedFDeriv ℝ 2 f q v) p a =
      fderiv ℝ (iteratedFDeriv ℝ 2 f) p a v := by
    intro v
    have := ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v).hasFDerivAt.comp p
      hd.hasFDerivAt).fderiv
    rw [show (fun q => iteratedFDeriv ℝ 2 f q v) =
      (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v) ∘ iteratedFDeriv ℝ 2 f from rfl,
      this]
    rfl
  rw [iteratedFDeriv_succ_apply_left, iteratedFDeriv_succ_apply_left]
  have t1 : Fin.tail ![a, b, c] = ![b, c] := by ext i; fin_cases i <;> rfl
  have t2 : Fin.tail ![a, c, b] = ![c, b] := by ext i; fin_cases i <;> rfl
  simp only [Matrix.cons_val_zero, t1, t2]
  rw [← key, ← key, hev.fderiv_eq]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem block_of_partial {B : ContinuousMultilinearMap ℝ (fun _ : Fin 3 => ℝ × E) ℝ} {m : ℝ}
    (hs₁₂ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![b, a, c])
    (hs₂₃ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![a, c, b])
    (k₁ : ∀ y : E, |B ![((0 : ℝ), y), (1, 0), (1, 0)]| ≤ m * ‖y‖)
    (k₂ : ∀ y₁ y₂ : E, |B ![((0 : ℝ), y₁), (0, y₂), (1, 0)]| ≤ m * ‖y₁‖ * ‖y₂‖)
    (k₃ : ∀ y₁ y₂ y₃ : E, |B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E) :
    |B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![(1, 0), (1, 0), (1, 0)]| ≤
      m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  have hu0 : ∀ x b c z : ℝ × E, Function.update ![x, b, c] 0 z = ![z, b, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu1 : ∀ a x c z : ℝ × E, Function.update ![a, x, c] 1 z = ![a, z, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu2 : ∀ a b x z : ℝ × E, Function.update ![a, b, x] 2 z = ![a, b, z] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have a₁ : ∀ x x' b c : ℝ × E, B ![x + x', b, c] = B ![x, b, c] + B ![x', b, c] := by
    intro x x' b c
    have := B.map_update_add ![x, b, c] 0 x x'
    rwa [hu0, hu0, hu0] at this
  have a₂ : ∀ a x x' c : ℝ × E, B ![a, x + x', c] = B ![a, x, c] + B ![a, x', c] := by
    intro a x x' c
    have := B.map_update_add ![a, x, c] 1 x x'
    rwa [hu1, hu1, hu1] at this
  have a₃ : ∀ a b x x' : ℝ × E, B ![a, b, x + x'] = B ![a, b, x] + B ![a, b, x'] := by
    intro a b x x'
    have := B.map_update_add ![a, b, x] 2 x x'
    rwa [hu2, hu2, hu2] at this
  have m₁ : ∀ (t : ℝ) (x b c : ℝ × E), B ![t • x, b, c] = t * B ![x, b, c] := by
    intro t x b c
    have := B.map_update_smul ![x, b, c] 0 t x
    rwa [hu0, hu0, smul_eq_mul] at this
  have m₂ : ∀ (t : ℝ) (a x c : ℝ × E), B ![a, t • x, c] = t * B ![a, x, c] := by
    intro t a x c
    have := B.map_update_smul ![a, x, c] 1 t x
    rwa [hu1, hu1, smul_eq_mul] at this
  have m₃ : ∀ (t : ℝ) (a b x : ℝ × E), B ![a, b, t • x] = t * B ![a, b, x] := by
    intro t a b x
    have := B.map_update_smul ![a, b, x] 2 t x
    rwa [hu2, hu2, smul_eq_mul] at this
  set e : ℝ × E := ((1 : ℝ), (0 : E)) with he_def
  have hsp : ∀ (s : ℝ) (y : E), ((s, y) : ℝ × E) = s • e + ((0 : ℝ), y) := by
    intro s y; ext <;> simp [he_def]
  set P₁ := |s₁| + ‖y₁‖
  set P₂ := |s₂| + ‖y₂‖
  set P₃ := |s₃| + ‖y₃‖
  -- the multilinear expansion
  have expand : B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![e, e, e] =
      (s₂ * s₃ * B ![((0 : ℝ), y₁), e, e] + s₂ * B ![((0 : ℝ), y₁), e, (0, y₃)] +
        s₃ * B ![((0 : ℝ), y₁), (0, y₂), e] + B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]) +
      (s₁ * s₃ * B ![e, ((0 : ℝ), y₂), e] + s₁ * B ![e, ((0 : ℝ), y₂), (0, y₃)]) +
      s₁ * s₂ * B ![e, e, ((0 : ℝ), y₃)] := by
    rw [hsp s₁ y₁, hsp s₂ y₂, hsp s₃ y₃]
    simp only [a₁, a₂, a₃, m₁, m₂, m₃]
    ring
  rw [he_def] at expand ⊢
  rw [expand]
  -- the seven partial blocks, brought to the canonical order by symmetry
  have b₁ := k₁ y₁
  have b₂ : |B ![((0 : ℝ), y₁), (1, 0), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₃‖ := by
    rw [hs₂₃]; exact k₂ y₁ y₃
  have b₃ := k₂ y₁ y₂
  have b₄ := k₃ y₁ y₂ y₃
  have b₅ : |B ![(1, 0), ((0 : ℝ), y₂), (1, 0)]| ≤ m * ‖y₂‖ := by
    rw [hs₁₂]; exact k₁ y₂
  have b₆ : |B ![(1, 0), ((0 : ℝ), y₂), (0, y₃)]| ≤ m * ‖y₂‖ * ‖y₃‖ := by
    rw [hs₁₂, hs₂₃]; exact k₂ y₂ y₃
  have b₇ : |B ![(1, 0), (1, 0), ((0 : ℝ), y₃)]| ≤ m * ‖y₃‖ := by
    rw [hs₂₃, hs₁₂]; exact k₁ y₃
  have hP : m * (P₁ * P₂ * P₃ - |s₁| * |s₂| * |s₃|) =
      (|s₂| * |s₃| * (m * ‖y₁‖) + |s₂| * (m * ‖y₁‖ * ‖y₃‖) + |s₃| * (m * ‖y₁‖ * ‖y₂‖) +
        m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖) +
      (|s₁| * |s₃| * (m * ‖y₂‖) + |s₁| * (m * ‖y₂‖ * ‖y₃‖)) + |s₁| * |s₂| * (m * ‖y₃‖) := by
    simp only [P₁, P₂, P₃]; ring
  rw [hP]
  refine (abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
    ((abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
      ((abs_add_le _ _).trans (add_le_add ?_ ?_)) ?_)) ?_)) ((abs_add_le _ _).trans
      (add_le_add ?_ ?_)))) ?_)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₁ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₂ (abs_nonneg _)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₃ (abs_nonneg _)
  · exact b₄
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₅ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₆ (abs_nonneg _)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₇ (by positivity)

theorem ridge_capHyp_blocks [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 f (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 f p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = k * r ^ 3) :
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
  refine ridge_capHyp_K hr hk hU hDU hf (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => ?_) h30 h40 h41 hH2 hlam
    hH1 hM hS hnorm
  have hp3 : ContDiffAt ℝ 3 f p := (hf.contDiffAt (hU.mem_nhds (hDU hp))).of_le (by norm_num)
  exact block_of_partial (iteratedFDeriv_three_swap₁₂ hp3) (iteratedFDeriv_three_swap₂₃ hp3)
    (h31 p hp) (h32 p hp) (h33 p hp) s₁ s₂ s₃ y₁ y₂ y₃

end PartialBlock

/-- A toy with a nonzero mixed block: `∂_x D_y² f = 2`. -/
noncomputable def mixedToyF (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2 + p.1 * p.2 ^ 2

theorem mixedToy_blocks :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => mixedToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp mixedToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (mixedToyF (-1 / 2, 0)) (mixedToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt mixedToyF
      ((p.1 ^ 2 - 1 / 4 + p.2 ^ 2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20))
      (HasFDerivAt.mul (hf p) (HasFDerivAt.pow (hs p) 2))
    convert h1 using 1
    · funext q
      simp only [mixedToyF, Pi.sub_apply, Pi.add_apply, Pi.mul_apply]
      ring
    · ext <;> simp
      ring
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + q.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2 + 2 * q.1 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((2 * p.2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (-40 + 2 * p.1) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + q.2 ^ 2)
        ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
      have := ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)).add (HasFDerivAt.pow (hs p) 2)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2 + 2 * q.1 * q.2)
        ((2 * p.2) • ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 + 2 * p.1) • ContinuousLinearMap.snd ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.const_mul (hs p) (-40)).add
        ((HasFDerivAt.const_mul (hf p) 2).mul (hs p))
      convert this using 1
      ext <;> simp
      ring
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hcd : ContDiff ℝ 4 mixedToyF := by unfold mixedToyF; fun_prop
  have hfd : fderiv ℝ mixedToyF = fun p => (p.1 ^ 2 - 1 / 4 + p.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ), iteratedFDeriv ℝ 2 mixedToyF y m =
      2 * y.1 * (m 0).1 * (m 1).1 + 2 * y.2 * ((m 0).1 * (m 1).2 + (m 0).2 * (m 1).1) +
        (-40 + 2 * y.1) * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ), iteratedFDeriv ℝ 3 mixedToyF x m =
      2 * (m 0).1 * (m 1).1 * (m 2).1 + 2 * (m 0).2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1) +
        2 * (m 0).1 * (m 1).2 * (m 2).2 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 mixedToyF y (Fin.tail m))
        ((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) • ContinuousLinearMap.snd ℝ ℝ ℝ) x := by
      have := (((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) •
            ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt (x := x)).add_const (-40 * (m 1).2 * (m 2).2)
      convert this using 1
      funext y; rw [h2]; simp [Fin.tail]; ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ), iteratedFDeriv ℝ 4 mixedToyF x m = 0 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hc : (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m)) =
        fun _ => iteratedFDeriv ℝ 3 mixedToyF x (Fin.tail m) := by
      funext y; rw [h3, h3]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m))
        (0 : ℝ × ℝ →L[ℝ] ℝ) x := by
      rw [hc]; exact hasFDerivAt_const _ _
    rw [hg.fderiv]
    simp
  have hsrc := ridge_capHyp_blocks (E := ℝ) (f := mixedToyF) (U := univ) (r := 1) (k := 1 / 6)
    (M3 := 2) (M4 := 0) (lam := 41) one_pos (by norm_num) isOpen_univ (subset_univ _)
    hcd.contDiffOn
    (fun x _ => by rw [h3]; simp)
    (fun p _ y => by rw [h3]; simp)
    (fun p _ y₁ y₂ => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, Real.norm_eq_abs]
      rw [show (2 * (0 : ℝ) * 0 * 1 + 2 * y₁ * (0 * 0 + y₂ * 1) + 2 * 0 * y₂ * 0) = 2 * (y₁ * y₂) by
        ring, abs_mul, abs_mul]
      norm_num [mul_assoc])
    (fun p _ y₁ y₂ y₃ => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, Real.norm_eq_abs]
      rw [show (2 * (0 : ℝ) * 0 * 0 + 2 * y₁ * (0 * y₃ + y₂ * 0) + 2 * 0 * y₂ * y₃) = 0 by ring,
        abs_zero]
      positivity)
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [mixedToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => mixedToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩

#print axioms ridge_capHyp_blocks
#print axioms mixedToy_blocks

end CapFirstExit
