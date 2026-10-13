section Ridge

/-- A convex function that vanishes at `a ≤ c` is `≤ 0` between them. -/
theorem convex_zeros_between {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a ≤ c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ}
    (hx : x ∈ Icc a c) : φ x ≤ 0 := by
  have h := hφ.le_on_segment haD hcD (by rw [segment_eq_Icc hac]; exact hx)
  rwa [ha, hc, max_self] at h

/-- A convex function that vanishes at `a < c` is `≥ 0` to the right of `c`. -/
theorem convex_zeros_right {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (haD : a ∈ D) (hac : a < c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ} (hxD : x ∈ D)
    (hx : c ≤ x) : 0 ≤ φ x := by
  rcases eq_or_lt_of_le hx with rfl | hlt
  · rw [hc]
  · have h := hφ.slope_mono_adjacent haD hxD hac hlt
    rw [ha, hc, sub_self, zero_div, sub_zero] at h
    rw [le_div_iff₀ (sub_pos.mpr hlt), zero_mul] at h
    exact h

/-- A convex function that vanishes at `a < c` is `≥ 0` to the left of `a`. -/
theorem convex_zeros_left {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (hcD : c ∈ D) (hac : a < c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ} (hxD : x ∈ D)
    (hx : x ≤ a) : 0 ≤ φ x := by
  rcases eq_or_lt_of_le hx with rfl | hlt
  · rw [ha]
  · have h := hφ.slope_mono_adjacent hxD hcD hlt hac
    rw [ha, hc, sub_self, zero_div, zero_sub] at h
    rw [div_le_iff₀ (sub_pos.mpr hlt), zero_mul] at h
    linarith

/-- **The ridge profile (L13, L16, L17, L20).** On `D = [-2r, 2r]`, let `g' = F`, with `F`
vanishing at the pins `∓r/2` and `F - (x + r/2)(x - r/2)/8` convex. Then:
- `g ≤ g(-r/2)` on `[-2r, r/2]`;
- `g ≥ g(r/2)` on `[-r/2, 2r]`;
- `g(-2r) ≤ g(-r/2) - 9r³/32`;
- `g(2r) ≥ g(r/2) + 9r³/32`. -/
theorem ridge_profile {g F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt g (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8)) :
    (∀ x ∈ Icc (-2 * r) (r / 2), g x ≤ g (-r / 2)) ∧
      (∀ x ∈ Icc (-r / 2) (2 * r), g (r / 2) ≤ g x) ∧
      g (-2 * r) ≤ g (-r / 2) - 9 * r ^ 3 / 32 ∧ g (r / 2) + 9 * r ^ 3 / 32 ≤ g (2 * r) := by
  have hac : -r / 2 < r / 2 := by linarith
  have haD : -r / 2 ∈ Icc (-2 * r) (2 * r) := ⟨by linarith, by linarith⟩
  have hcD : r / 2 ∈ Icc (-2 * r) (2 * r) := ⟨by linarith, by linarith⟩
  have hLD : Icc (-2 * r) (-r / 2) ⊆ Icc (-2 * r) (2 * r) := Icc_subset_Icc le_rfl (by linarith)
  have hMD : Icc (-r / 2) (r / 2) ⊆ Icc (-2 * r) (2 * r) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hRD : Icc (r / 2) (2 * r) ⊆ Icc (-2 * r) (2 * r) := Icc_subset_Icc (by linarith) le_rfl
  set φ : ℝ → ℝ := fun x => F x - (x + r / 2) * (x - r / 2) / 8 with hφdef
  have hφa : φ (-r / 2) = 0 := by simp only [hφdef, hFa]; ring
  have hφc : φ (r / 2) = 0 := by simp only [hφdef, hFc]; ring
  have hgc : ContinuousOn g (Icc (-2 * r) (2 * r)) :=
    fun x hx => (hg x hx).continuousAt.continuousWithinAt
  -- L13: the sign pattern of `F`
  have hFl : ∀ x ∈ Icc (-2 * r) (-r / 2), 0 ≤ F x := by
    intro x hx
    have h1 := convex_zeros_left hconv hcD hac hφa hφc (hLD hx) hx.2
    have h2 : 0 ≤ (x + r / 2) * (x - r / 2) / 8 := by
      have : 0 ≤ (x + r / 2) * (x - r / 2) :=
        mul_nonneg_of_nonpos_of_nonpos (by linarith [hx.2]) (by linarith [hx.2])
      positivity
    simp only [hφdef] at h1
    linarith
  have hFm : ∀ x ∈ Icc (-r / 2) (r / 2), F x ≤ 0 := by
    intro x hx
    have h1 := convex_zeros_between hconv haD hcD hac.le hφa hφc hx
    have h2 : (x + r / 2) * (x - r / 2) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith [hx.1]) (by linarith [hx.2])
    simp only [hφdef] at h1
    linarith
  have hFr : ∀ x ∈ Icc (r / 2) (2 * r), 0 ≤ F x := by
    intro x hx
    have h1 := convex_zeros_right hconv haD hac hφa hφc (hRD hx) hx.1
    have h2 : 0 ≤ (x + r / 2) * (x - r / 2) := mul_nonneg (by linarith [hx.1]) (by linarith [hx.1])
    simp only [hφdef] at h1
    linarith
  -- monotonicity of `g` on the three pieces
  have hmonoL : MonotoneOn g (Icc (-2 * r) (-r / 2)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hgc.mono hLD)
      (fun x hx => (hg x (hLD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFl x (interior_subset hx))
  have hantiM : AntitoneOn g (Icc (-r / 2) (r / 2)) :=
    antitoneOn_of_hasDerivWithinAt_nonpos (convex_Icc _ _) (hgc.mono hMD)
      (fun x hx => (hg x (hMD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFm x (interior_subset hx))
  have hmonoR : MonotoneOn g (Icc (r / 2) (2 * r)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hgc.mono hRD)
      (fun x hx => (hg x (hRD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFr x (interior_subset hx))
  -- L17: compare `g` with the cubic `Q`, whose derivative is `(x + r/2)(x - r/2)/8`
  set Q : ℝ → ℝ := fun x => x ^ 3 / 24 - r ^ 2 * x / 32 with hQdef
  have hQ : ∀ x, HasDerivAt Q ((x + r / 2) * (x - r / 2) / 8) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 24).sub ((hasDerivAt_id x).const_mul (r ^ 2 / 32))
    convert h1 using 1
    · funext y
      simp only [hQdef, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => g x - Q x) (φ x) x :=
    fun x hx => (hg x hx).sub (hQ x)
  have hGc : ContinuousOn (fun x => g x - Q x) (Icc (-2 * r) (2 * r)) :=
    fun x hx => (hG x hx).continuousAt.continuousWithinAt
  have hGL : MonotoneOn (fun x => g x - Q x) (Icc (-2 * r) (-r / 2)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hGc.mono hLD)
      (fun x hx => (hG x (hLD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => convex_zeros_left hconv hcD hac hφa hφc (hLD (interior_subset hx))
        (interior_subset hx).2)
  have hGR : MonotoneOn (fun x => g x - Q x) (Icc (r / 2) (2 * r)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hGc.mono hRD)
      (fun x hx => (hG x (hRD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => convex_zeros_right hconv haD hac hφa hφc (hRD (interior_subset hx))
        (interior_subset hx).1)
  have hLa : (-r / 2 : ℝ) ∈ Icc (-2 * r) (-r / 2) := ⟨by linarith, le_rfl⟩
  have hL2 : (-2 * r : ℝ) ∈ Icc (-2 * r) (-r / 2) := ⟨le_rfl, by linarith⟩
  have hRc : (r / 2 : ℝ) ∈ Icc (r / 2) (2 * r) := ⟨le_rfl, by linarith⟩
  have hR2 : (2 * r : ℝ) ∈ Icc (r / 2) (2 * r) := ⟨by linarith, le_rfl⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro x hx
    rcases le_or_gt x (-r / 2) with h | h
    · exact hmonoL ⟨hx.1, h⟩ hLa h
    · exact hantiM ⟨le_rfl, hac.le⟩ ⟨h.le, hx.2⟩ h.le
  · intro x hx
    rcases le_or_gt x (r / 2) with h | h
    · exact hantiM ⟨hx.1, h⟩ ⟨hac.le, le_rfl⟩ h
    · exact hmonoR hRc ⟨h.le, hx.2⟩ h.le
  · have h := hGL hL2 hLa (by linarith)
    simp only [hQdef] at h
    nlinarith [h]
  · have h := hGR hRc hR2 (by linarith)
    simp only [hQdef] at h
    nlinarith [h]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **The cap hypotheses from the ridge (L16, L17, L19, L20).** Under the analytic inputs listed
above, `f` satisfies (C1)–(C3) on `C = [-2r, r/2] × B̄(0, 2r)`, with `M = (-r/2, h(-r/2))`,
`z = (2r, h(2r))`, `b = g(-r/2)` and `s = g(r/2)`. Every input is local: `g' = F` and the convexity
hold on `[-2r, 2r]`, and `h` is continuous on `[-r/2, 2r]`. -/
theorem ridge_capHyp {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), f p ≤ f (p.1, h p.1))
    (hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p ≤ f (r / 2, h (r / 2)))
    (hM : ‖h (-r / 2)‖ ≤ 2 * r)
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
      (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) := by
  obtain ⟨hceil, hfloor, hleft, hright⟩ := ridge_profile hr hg hFa hFc hconv
  have hcyl : frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) =
      ({-2 * r, r / 2} : Set ℝ) ×ˢ Metric.closedBall (0 : E) (2 * r) ∪
        Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r) :=
    frontier_cylinder (by linarith) (by positivity)
  set xt : I → ℝ := fun t => -r / 2 + (t : ℝ) * (2 * r - -r / 2) with hxt
  have hxc : Continuous xt := continuous_const.add (continuous_subtype_val.mul continuous_const)
  have hx0 : xt 0 = -r / 2 := by simp [hxt]
  have hx1 : xt 1 = 2 * r := by simp [hxt]
  have hxmem : ∀ t, xt t ∈ Icc (-r / 2) (2 * r) := by
    intro t
    have h0 := t.2.1
    have h1 := t.2.2
    simp only [hxt]
    constructor <;> nlinarith
  let γ : Path ((-r / 2 : ℝ), h (-r / 2)) (2 * r, h (2 * r)) :=
    { toFun := fun t => (xt t, h (xt t))
      continuous_toFun := hxc.prodMk (hh.comp_continuous hxc hxmem)
      source' := by simp only [hx0]
      target' := by simp only [hx1] }
  refine ⟨⟨⟨by linarith, by linarith⟩, mem_closedBall_zero_iff.mpr hM⟩, ?_, ?_, ?_, ⟨γ, ?_⟩⟩
  · intro p hp
    exact (hT p hp).trans (hceil p.1 hp.1)
  · intro p hp
    rw [hcyl] at hp
    rcases hp with ⟨hp1, hp2⟩ | hp
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
        refine ⟨?_, hp2⟩
        rcases hp1 with h1 | h1
        · rw [h1]; exact ⟨le_refl _, by linarith⟩
        · rw [h1]; exact ⟨by linarith, le_refl _⟩
      have h0 := hT p hpC
      rcases hp1 with h1 | h1
      · rw [h1] at h0
        linarith
      · rw [h1] at h0
        exact h0
    · exact hside p hp
  · linarith
  · intro t
    exact hfloor (xt t) (hxmem t)

/-- **(C2⁺) from the ridge.** If, in addition, `h x` is the unique transverse maximizer on the
cap and the curved side is strictly below `g c` (L7, L18), then `S = (r/2, h(r/2))` is the only
frontier point at height `s`. -/
theorem ridge_saddle_strict {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), f p ≤ f (p.1, h p.1))
    (hTs : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), p.2 ≠ h p.1 →
      f p < f (p.1, h p.1))
    (hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p < f (r / 2, h (r / 2)))
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
      x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  obtain ⟨-, -, hleft, -⟩ := ridge_profile hr hg hFa hFc hconv
  intro p hp hne
  rw [frontier_cylinder (by linarith) (by positivity)] at hp
  rcases hp with ⟨hp1, hp2⟩ | hp
  · rcases hp1 with h1 | h1
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
        ⟨by rw [h1]; exact ⟨le_refl _, by linarith⟩, hp2⟩
      have h0 := hT p hpC
      rw [h1] at h0
      linarith
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
        ⟨by rw [h1]; exact ⟨by linarith, le_refl _⟩, hp2⟩
      have hy : p.2 ≠ h p.1 := by
        intro hy
        apply hne
        rw [h1] at hy
        exact Prod.ext h1 hy
      have h0 := hTs p hpC hy
      rw [h1] at h0
      exact h0
  · exact hside p hp

end Ridge
