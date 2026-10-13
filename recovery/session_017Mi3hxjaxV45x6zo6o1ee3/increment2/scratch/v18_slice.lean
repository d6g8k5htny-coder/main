
/-! ### The full domain (v1.8)

SP proves L4 on all of `D`, where the increment from `M` is at most `9r/2 < 5r`. The lemmas below
take the source data for `x ∈ X = [-5r/2, 5r/2]`, where `|x + r/2| ≤ 3r` keeps the increment at
most `5r`. So every slice over the open set `(-5r/2, 5r/2) ⊇ [-2r, 2r]` is `(λ - 5rm)`-strongly
concave, which is what the L7 lemmas need, and `ridge_capHyp_source` composes L7 with the cap. -/

/-- **L3 on the full domain.** As `hessian_increment`, with the `x`-range `[-5r/2, 5r/2]`. -/
theorem hessian_increment_D {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m : ℝ} (hr : 0 < r)
    (hHx : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖) :
    ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖) := by
  rintro ⟨x, y⟩ hp
  have hx := hp.1
  have h1 : ‖H (x, (1 : ℝ) • y) - H (x, (0 : ℝ) • y)‖ ≤ m * ‖y‖ * ‖(1 : ℝ) - 0‖ :=
    (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun s : ℝ => H (x, s • y)) (fun s hs => (hHr (x, y) hp s hs).hasDerivWithinAt)
      (fun s hs => hHrb (x, y) hp s hs) (left_mem_Icc.mpr zero_le_one)
      (right_mem_Icc.mpr zero_le_one)
  have ha : -r / 2 ∈ Icc (-5 * r / 2) (5 * r / 2) := ⟨by linarith, by linarith⟩
  have h2 : ‖H (x, 0) - H (-r / 2, 0)‖ ≤ m * ‖x - -r / 2‖ :=
    (convex_Icc (-5 * r / 2) (5 * r / 2)).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun t => H (t, 0)) (fun t ht => (hHx t ht).hasDerivWithinAt) hHxb ha hx
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at h1
  rw [Real.norm_eq_abs] at h2
  calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ ‖H (x, y) - H (x, 0)‖ + ‖H (x, 0) - H (-r / 2, 0)‖ :=
        norm_sub_le_norm_sub_add_norm_sub (H (x, y)) (H (x, 0)) (H (-r / 2, 0))
    _ ≤ m * ‖y‖ + m * |x - -r / 2| := add_le_add h1 h2
    _ = m * (|(x, y).1 - -r / 2| + ‖(x, y).2‖) := by simp only; ring

/-- **L4's Hessian bound on the full domain.** As `transverse_hessian_le`, for
`x ∈ [-5r/2, 5r/2]`: there `|x + r/2| ≤ 3r`, so the increment is still at most `5r`. -/
theorem transverse_hessian_le_D {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r)
    (hm : 0 ≤ m)
    (hlip : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      H p v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ v
  have hy' : ‖y‖ ≤ 2 * r := by simpa using hy
  have hdx : |x - -r / 2| ≤ 3 * r := by
    rw [abs_le]; constructor <;> linarith [hx.1, hx.2]
  have hinc : ‖H (x, y) - H (-r / 2, 0)‖ ≤ 5 * r * m := by
    have := hlip (x, y) ⟨hx, hy⟩
    calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ m * (|x - -r / 2| + ‖y‖) := this
      _ ≤ m * (3 * r + 2 * r) := by gcongr
      _ = 5 * r * m := by ring
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

/-- **L4 on the full domain.** Every slice `y ↦ f (x, y)`, `x ∈ [-5r/2, 5r/2]`, is
`(λ - 5rm)`-strongly concave on `B̄(0, 2r)`. -/
theorem slices_strongConcave_D {f : ℝ × E → ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r) (hm : 0 ≤ m)
    (hD : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hlip : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) (lam - 5 * r * m) (fun y => f (x, y)) := by
  intro x hx
  have hbd := transverse_hessian_le_D hr hm hlip hlam
  exact strongConcaveOn_of_hessian (Dφ := fun y => Dφ (x, y)) (H := fun y => H (x, y))
    (convex_closedBall _ _) (fun y hy => hD (x, y) ⟨hx, hy⟩) (fun y hy => hH (x, y) ⟨hx, hy⟩)
    (fun y hy v => hbd (x, y) ⟨hx, hy⟩ v)

/-- **The cap from SP's source data (L1–L5, L7, L8, L11, L18 composed).** Assume, for
`x ∈ [-5r/2, 5r/2]` and `y ∈ B̄(0, 2r)`: a joint derivative `Df` of `f`, strictly differentiable in
its transverse part `∂_y f = Df ∘ inr` with derivative `DG`; the transverse Hessian `H` with the
third-derivative bounds `m` of L3; (H1) `8rm² < λ` with `H(M) ≤ -λ`; the critical pins
`Df(∓r/2, 0) = 0`; the weak bounds `|∂_x³ f(·, 0)| ≤ m` on `[-r/2, r/2]` (L1) and
`|∂_x² ∂_y f(·, 0)| ≤ m` on `[-5r/2, 5r/2]` (L5); and the gap `f(M) - f(S) = r³/6`. Then:
- a ridge `h` exists: on `(-5r/2, 5r/2)`, `h x` is a critical point of its slice in the open ball;
- for any such ridge (unique by `slice_critical_unique`) with L12's `hconv` for
  `F x = ∂_x f(x, h x)`, `h(∓r/2) = 0`, and (C1)–(C3) and (C2⁺) hold on the cap with `M = (-r/2, 0)`,
  `S = (r/2, 0)`, `b = f(M)` and `s = f(S)`.

L8 puts `h` in the open ball on `[-2r, 2r]` (`‖w‖ ≤ 3mr² < 2rδ`); the implicit function theorem
(`ridge_differentiable`) makes `h` differentiable there, with negative definite `DG ∘ inr = H`;
`ridge_inputs_of_joint` gives L11; `ridge_capHyp_H1` gives the cap. Only L12 (`hconv`) remains an
input, with L6 and L9–L11's quantitative bounds that SP uses to prove it. -/
theorem ridge_capHyp_source [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r)
    (hDf : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt f (Df p) p)
    (hGs : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-5 * r / 2) (5 * r / 2))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-5 * r / 2) (5 * r / 2))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-5 * r / 2) (5 * r / 2), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-5 * r / 2) (5 * r / 2),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        ConvexOn ℝ (Icc (-2 * r) (2 * r))
          (fun x => Df (x, h x) (1, 0) - (x + r / 2) * (x - r / 2) / 8) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hUX : Ioo (-5 * r / 2) (5 * r / 2) ⊆ Icc (-5 * r / 2) (5 * r / 2) := Ioo_subset_Icc_self
  have hcapX : Icc (-2 * r) (r / 2) ⊆ Icc (-5 * r / 2) (5 * r / 2) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-5 * r / 2) (5 * r / 2) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  -- the slice derivatives are the transverse parts of `Df`
  have hD : ∀ p ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) ((Df p).comp (ContinuousLinearMap.inr ℝ ℝ E)) p.2 := by
    rintro ⟨x, y⟩ hp
    exact slice_hasFDerivAt (hDf (x, y) hp)
  -- L1–L2: `m ≥ 2`
  have hg0 : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    have hx' : (x, (0 : E)) ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
      ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, h0Y⟩
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    exact (hDf _ hx').comp_hasDerivAt x hp
  have hG0a : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hG0c : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hm : 2 ≤ m := hermite_m_ge_two (g := fun x => f (x, 0)) hr hg0 hG0cv hG0cc hG0a hG0c hnorm
  have hm0 : 0 ≤ m := by linarith
  -- L3–L4 on the full domain
  have hlip := hessian_increment_D hr hHx hHxb hHr hHrb
  have hneg := transverse_hessian_le_D hr hm0 hlip hlam
  have hconcX := slices_strongConcave_D hr hm0 hD hH hlip hlam
  have h11 : 11 * (r * m) < lam - 5 * r * m := by nlinarith
  have hrm : 0 < r * m := by have : 0 < m := by linarith
                             positivity
  have hδ0 : 0 < lam - 5 * r * m := by linarith
  -- L5 on the full domain: `‖w x‖ ≤ 3mr² < 2rδ`
  have hwv : ∀ (t : ℝ) (v : E),
      ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v = Df (t, 0) (0, v) := fun t v => by simp
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-5 * r / 2) (5 * r / 2))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp only [hwv]
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-5 * r / 2) (5 * r / 2))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp only [hwv]
  have hwa : (Df (-r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hM]
  have hwc : (Df (r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hS]
  have hwlt : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ < (lam - 5 * r * m) * (2 * r) := by
    intro x hx
    have h1 := slice_gradient_bound (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      hm0 hwcv' hwcc' (show -r / 2 ∈ Icc (-5 * r / 2) (5 * r / 2) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-5 * r / 2) (5 * r / 2) from ⟨by linarith, by linarith⟩) (by linarith)
      hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 6 * r ^ 2 := by
      have hxx : 0 ≤ (x - -5 * r / 2) * (5 * r / 2 - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      rw [abs_le]; constructor <;> nlinarith
    have h3 : ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 3 * m * r ^ 2 :=
      calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
          ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
        _ ≤ m / 2 * (6 * r ^ 2) := by gcongr
        _ = 3 * m * r ^ 2 := by ring
    have h4 : 22 * r * (r * m) < (lam - 5 * r * m) * (2 * r) := by nlinarith
    nlinarith
  have hslice0 : ∀ x ∈ Icc (-5 * r / 2) (5 * r / 2),
      HasFDerivAt (fun y => f (x, y)) ((Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 :=
    fun x hx => hD (x, 0) ⟨hx, h0Y⟩
  refine ⟨ridge_exists (Dφ := fun x y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) hr
    (fun x hx => hconcX x (hUX hx)) (fun x hx y hy => hD (x, y) ⟨hUX hx, hy⟩)
    (fun x hx => hwlt x (hUX hx)), ?_⟩
  intro h hhU hconv
  have hball : ∀ x ∈ Ioo (-5 * r / 2) (5 * r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-5 * r / 2) (5 * r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
  -- L7 uniqueness: the pins are on the ridge
  have hMU : -r / 2 ∈ Ioo (-5 * r / 2) (5 * r / 2) := ⟨by linarith, by linarith⟩
  have hSU : r / 2 ∈ Ioo (-5 * r / 2) (5 * r / 2) := ⟨by linarith, by linarith⟩
  have hha : h (-r / 2) = 0 :=
    slice_critical_unique (hconcX _ (hUX hMU)) hδ0 (hball _ hMU) h0Y (hcrit _ hMU)
      (by simpa [hwa] using hslice0 _ (hUX hMU))
  have hhc : h (r / 2) = 0 :=
    slice_critical_unique (hconcX _ (hUX hSU)) hδ0 (hball _ hSU) h0Y (hcrit _ hSU)
      (by simpa [hwc] using hslice0 _ (hUX hSU))
  -- L8: the ridge is in the open ball on `[-2r, 2r]`
  have hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r) := by
    intro x hx
    have hxU := hIU hx
    have h8 := slice_ridge_bound (hconcX x (hUX hxU)) h0Y (hball x hxU) (hcrit x hxU)
      (hslice0 x (hUX hxU))
    have h9 := hwlt x (hUX hxU)
    rw [Metric.mem_ball, dist_zero_right]
    have : (lam - 5 * r * m) * ‖h x‖ < (lam - 5 * r * m) * (2 * r) := by linarith
    exact lt_of_mul_lt_mul_left this hδ0.le
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-5 * r / 2) (5 * r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  -- L7 regularity: the transverse part of `DG` is `H`, negative definite
  have hDGH : ∀ x ∈ Icc (-2 * r) (2 * r),
      DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E = H (x, h x) := by
    intro x hx
    have h1 : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((DG (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (h x) :=
      (hGs _ (hridgeX x hx)).hasFDerivAt.comp (h x) (hasFDerivAt_prodMk_right x (h x))
    exact h1.unique (hH _ (hridgeX x hx))
  have hnegR : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [hDGH x hx]
    exact hneg _ (hridgeX x hx) v
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ᶠ v in 𝓝 (x, h x),
      HasFDerivAt (fun y => f (v.1, y)) ((Df v).comp (ContinuousLinearMap.inr ℝ ℝ E)) v.2 := by
    intro x hx
    have hopen : IsOpen (Ioo (-5 * r / 2) (5 * r / 2) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with v hv
    exact hD v ⟨hUX hv.1, Metric.ball_subset_closedBall hv.2⟩
  have hstrict := ridge_differentiable (f := f)
    (G := fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG := fun x => DG (x, h x))
    (h := h) (U := Ioo (-5 * r / 2) (5 * r / 2)) hδ0 isOpen_Ioo hIU
    (fun x hx => hconcX x (hUX hx)) hball hcrit hint hG
    (fun x hx => hGs _ (hridgeX x hx)) hnegR
  -- L11
  have hDr : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt f (Df (x, h x)) (x, h x) :=
    fun x hx => hDf _ (hridgeX x hx)
  have hridge : ∀ x ∈ Icc (-2 * r) (2 * r),
      (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 :=
    fun x hx => (slice_hasFDerivAt (hDr x hx)).unique (hcrit x (hIU hx))
  obtain ⟨hh, hg, -⟩ := ridge_inputs_of_joint hr hDr
    (fun x hx => (hstrict x hx).hasFDerivAt.hasDerivAt) hridge
  have hFa : Df (-r / 2, h (-r / 2)) (1, 0) = 0 := by simp [hha, hM]
  have hFc : Df (r / 2, h (r / 2)) (1, 0) = 0 := by simp [hhc, hS]
  have hcap : ∀ x ∈ Icc (-2 * r) (r / 2), x ∈ Ioo (-5 * r / 2) (5 * r / 2) :=
    fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have key := ridge_capHyp_H1 (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
    hr hg0 hG0cv hG0cc hG0a hG0c
    (fun p hp => hD p ⟨hcapX hp.1, hp.2⟩) (fun p hp => hH p ⟨hcapX hp.1, hp.2⟩)
    (fun x hx => hHx x (hcapX hx)) (fun x hx => hHxb x (hcapX hx))
    (fun p hp => hHr p ⟨hcapX hp.1, hp.2⟩) (fun p hp => hHrb p ⟨hcapX hp.1, hp.2⟩) hlam hH1
    hh hg hFa hFc hconv (fun x hx => hball x (hcap x hx)) (fun x hx => hcrit x (hcap x hx))
    hha hhc (fun x hx => hslice0 x (hcapX hx))
    (fun v => (hwcv' v).subset hcapX (convex_Icc _ _))
    (fun v => (hwcc' v).subset hcapX (convex_Icc _ _)) (by rw [hha, hhc]; exact hnorm)
  rw [hha, hhc] at key
  exact ⟨hha, hhc, key⟩
