import CapFirstExit
open Set CapFirstExit

section W
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
open Filter Topology

/-- **The cap from SP's `C³` source data (L1–L20 composed; L12 proved).** As
`ridge_capHyp_source`, with L12's `hconv` derived instead of assumed. Additionally assume, on the
collar: a second derivative `D2f` of `f`, third derivatives `T3 p a b` of `p ↦ D²f p [a, b]`, every
third-derivative block other than `f_xxx` of norm at most `m` (combined form), and `f_xxx ≥ 7/4`
(SP's L4, from L2 and (H2)). Then the ridge exists, and every ridge has `h(∓r/2) = 0` and yields
(C1)–(C3) and (C2⁺) on the cap. Inside: L5 gives `‖w‖ ≤ 2mr²` on `[-2r, 2r]`, L8 `‖h‖ ≤ 2mr²/δ`, L6
`‖w'‖ ≤ 2mr`, L3 in `y` `‖∂_x∇_y f(x, h)‖ ≤ ‖w'‖ + m‖h‖`, L9 `‖h'‖ ≤ 2s + 2s²` with `s = mr/δ`,
`trilinear_lower` and `l12_constants` give `F'' ≥ 1/4` in the form of `ridge_hconv`. -/
theorem ridge_capHyp_C3 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ T3 p (1, 0) (1, 0) (1, 0))
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
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
  obtain ⟨hex, hcapS⟩ := ridge_capHyp_source hr hε hεr hDf hGs hH hHx hHxb hHr hHrb hlam hH1 hM hS
    hG0cv hG0cc hwcv hwcc hnorm
  refine ⟨hex, fun h hhU => hcapS h hhU ?_⟩
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hcapX : Icc (-2 * r) (r / 2) ⊆ Icc (-2 * r - ε) (2 * r + ε) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  -- the slice derivatives are the transverse parts of `Df`
  have hD : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) ((Df p).comp (ContinuousLinearMap.inr ℝ ℝ E)) p.2 := by
    rintro ⟨x, y⟩ hp
    exact slice_hasFDerivAt (hDf (x, y) hp)
  -- L1–L2: `m ≥ 2`
  have hg0 : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    have hx' : (x, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
      ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, h0Y⟩
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    exact (hDf _ hx').comp_hasDerivAt x hp
  have hG0a : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hG0c : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hm : 2 ≤ m := hermite_m_ge_two (g := fun x => f (x, 0)) hr hg0 hG0cv hG0cc hG0a hG0c hnorm
  have hm0 : 0 ≤ m := by linarith
  -- L3–L4 on the full domain
  have hlip := hessian_increment_D hr hε hHx hHxb hHr hHrb
  have hneg := transverse_hessian_le_D hm0 hε hεr hlip hlam
  have hconcX := slices_strongConcave_D hm0 hε hεr hD hH hlip hlam
  have h11 : 11 * (r * m) < lam - 5 * r * m := by
    nlinarith [mul_nonneg (mul_nonneg hr.le hm0) (sub_nonneg.2 hm)]
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have hδ0 : 0 < lam - 5 * r * m := by linarith
  -- L5 on the full domain: `‖w x‖ ≤ 3mr² < 2rδ`
  have hwv : ∀ (t : ℝ) (v : E),
      ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v = Df (t, 0) (0, v) := fun t v => by simp
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp only [hwv]
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp only [hwv]
  have hwa : (Df (-r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hM]
  have hwc : (Df (r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hS]
  have hwlt : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ < (lam - 5 * r * m) * (2 * r) := by
    intro x hx
    have h1 := slice_gradient_bound (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      hm0 hwcv' hwcc' (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩) (by linarith)
      hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 6 * r ^ 2 := by
      have hxx : 0 ≤ (x - (-2 * r - ε)) * (2 * r + ε - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      have hee : 0 ≤ (r / 2 - ε) * (9 * r / 2 + ε) :=
        mul_nonneg (by linarith) (by linarith)
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    have h3 : ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 3 * m * r ^ 2 :=
      calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
          ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
        _ ≤ m / 2 * (6 * r ^ 2) := by gcongr
        _ = 3 * m * r ^ 2 := by ring
    have h4 : 22 * r * (r * m) < (lam - 5 * r * m) * (2 * r) := by nlinarith
    nlinarith
  have hslice0 : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) ((Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 :=
    fun x hx => hD (x, 0) ⟨hx, h0Y⟩
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
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
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
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
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with v hv
    exact hD v ⟨hUX hv.1, Metric.ball_subset_closedBall hv.2⟩
  have hstrict := ridge_differentiable (f := f)
    (G := fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG := fun x => DG (x, h x))
    (h := h) (U := Ioo (-2 * r - ε) (2 * r + ε)) hδ0 isOpen_Ioo hIU
    (fun x hx => hconcX x (hUX hx)) hball hcrit hint hG
    (fun x hx => hGs _ (hridgeX x hx)) hnegR
  have hδ : r * m * (8 * m - 5) < lam - 5 * r * m := by nlinarith
  set h' : ℝ → E := fun x => (-(DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E).inverse ∘L
    (DG (x, h x) ∘L ContinuousLinearMap.inl ℝ ℝ E)) 1 with hh'def
  have hhd : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x :=
    fun x hx => (hstrict x hx).hasFDerivAt.hasDerivAt
  have hridgeU : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := fun x hx =>
    (slice_hasFDerivAt (hDf _ ⟨hUX hx, hball x hx⟩)).unique (hcrit x hx)
  have hD2r : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt Df (D2f (x, h x)) (x, h x) :=
    fun x hx => hD2 _ (hridgeX x hx)
  have hsym : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v w, D2f (x, h x) v w = D2f (x, h x) w v := by
    intro x hx v w
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    have hev : ∀ᶠ q in 𝓝 (x, h x), HasFDerivAt f (Df q) q := by
      filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with q hq
      exact hDf q ⟨hUX hq.1, Metric.ball_subset_closedBall hq.2⟩
    exact second_derivative_symmetric_of_eventually hev (hD2r x hx) v w
  have hnegD : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      D2f (x, h x) (0, v) (0, v) ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [← transverse_hessian_eq (hH _ (hridgeX x hx)) (hD2r x hx) v v]
    exact hneg _ (hridgeX x hx) v
  sorry

end W
