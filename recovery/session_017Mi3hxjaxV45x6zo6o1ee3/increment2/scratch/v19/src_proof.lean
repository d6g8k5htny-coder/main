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
  refine ⟨ridge_exists (Dφ := fun x y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) hr
    (fun x hx => hconcX x (hUX hx)) (fun x hx y hy => hD (x, y) ⟨hUX hx, hy⟩)
    (fun x hx => hwlt x (hUX hx)), ?_⟩
  intro h hhU hconv
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
  -- L7 uniqueness: the pins are on the ridge
  have hMU : -r / 2 ∈ Ioo (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hSU : r / 2 ∈ Ioo (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
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
  have hcap : ∀ x ∈ Icc (-2 * r) (r / 2), x ∈ Ioo (-2 * r - ε) (2 * r + ε) :=
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
