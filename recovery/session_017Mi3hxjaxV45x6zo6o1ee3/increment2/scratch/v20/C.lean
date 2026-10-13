import CapFirstExit
open Set CapFirstExit

section Z
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
open Filter Topology

/-- **L1, one-sided.** As `hermite_gap_le`, from the concave half alone (`G'' ≤ m` weakly):
`g(a) - g(c) ≤ m (c - a)³/12`. -/
theorem hermite_gap_le_concave {g G : ℝ → ℝ} {a c m : ℝ} (hac : a < c)
    (hg : ∀ x ∈ Icc a c, HasDerivAt g (G x) x)
    (hcc : ConcaveOn ℝ (Icc a c) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G a = 0) (hGc : G c = 0) :
    g a - g c ≤ m * (c - a) ^ 3 / 12 := by
  have hpoly : ∀ x, HasDerivAt (fun t => (c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
      ((x - a) * (c - x)) x := by
    intro x
    have h1 : HasDerivAt (fun t => t - a) 1 x := (hasDerivAt_id x).sub_const a
    have h2 := ((h1.pow 2).const_mul (c - a)).div_const 2
    have h3 := (h1.pow 3).div_const 3
    convert h2.sub h3 using 1
    push_cast
    ring
  set k : ℝ → ℝ := fun t => g t + m / 2 * ((c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
    with hk
  have hkd : ∀ x ∈ Icc a c, HasDerivAt k (G x + m / 2 * ((x - a) * (c - x))) x :=
    fun x hx => (hg x hx).add ((hpoly x).const_mul (m / 2))
  have hcv' : ConvexOn ℝ (Icc a c) (fun t => -G t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hχ := convexOn_add_node (a := a) (c := c) hcv'
  have hmono : MonotoneOn k (Icc a c) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a c)
      (fun x hx => (hkd x hx).continuousAt.continuousWithinAt)
      (fun x hx => (hkd x (interior_subset hx)).hasDerivWithinAt) (fun x hx => ?_)
    have hx := interior_subset hx
    have h0 := convex_zeros_between hχ (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
      (by rw [hGa]; ring) (by rw [hGc]; ring) hx
    nlinarith
  have hkac := hmono (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
  simp only [hk] at hkac
  have e0 : m / 2 * ((c - a) * (a - a) ^ 2 / 2 - (a - a) ^ 3 / 3) = 0 := by ring
  have e1 : m / 2 * ((c - a) * (c - a) ^ 2 / 2 - (c - a) ^ 3 / 3) = m * (c - a) ^ 3 / 12 := by
    ring
  linarith

/-- **L2, second statement.** Under SP's normalization `g(-r/2) - g(r/2) = r³/6` with critical pins,
if `g''' = φ` is continuous on the pin interval, then `φ(x₀) ≥ 2` for some `x₀` there (SP: the
`φ`-weighted average of `f_xxx` is `2`). -/
theorem third_deriv_exists_two {g G G' φ : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt g (G x) x)
    (hG : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt G (G' x) x)
    (hG' : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt G' (φ x) x)
    (hφ : ContinuousOn φ (Icc (-r / 2) (r / 2)))
    (hGa : G (-r / 2) = 0) (hGc : G (r / 2) = 0) (hgap : g (-r / 2) - g (r / 2) = r ^ 3 / 6) :
    ∃ x₀ ∈ Icc (-r / 2) (r / 2), 2 ≤ φ x₀ := by
  by_contra hcon
  simp only [not_exists, not_and, not_le] at hcon
  have hne : (Icc (-r / 2) (r / 2)).Nonempty := ⟨-r / 2, left_mem_Icc.mpr (by linarith)⟩
  obtain ⟨x₁, hx₁, hmax⟩ := isCompact_Icc.exists_isMaxOn hne hφ
  set M := φ x₁ with hM
  have hcc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t - M / 2 * t ^ 2) := by
    refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc _ _) (f' := fun t => G' t - M * t)
      (f'' := fun t => φ t - M)
      (fun t ht => ((hG t ht).sub ((hasDerivAt_pow 2 t).const_mul (M / 2))).continuousAt.continuousWithinAt)
      (fun t ht => ?_) (fun t ht => ?_) (fun t ht => ?_)
    · have := (hG t (interior_subset ht)).sub ((hasDerivAt_pow 2 t).const_mul (M / 2))
      convert this.hasDerivWithinAt using 1
      push_cast
      ring
    · have := (hG' t (interior_subset ht)).sub ((hasDerivAt_id' t).const_mul M)
      convert this.hasDerivWithinAt using 1
      ring
    · have h' : φ t ≤ φ x₁ := hmax (interior_subset ht)
      show φ t - M ≤ 0
      linarith
  have h1 := hermite_gap_le_concave (by linarith) hg hcc hGa hGc
  have hcr : r / 2 - -r / 2 = r := by ring
  rw [hgap, hcr] at h1
  have h2 := hcon x₁ hx₁
  have hr3 : 0 < r ^ 3 := by positivity
  nlinarith

/-- **L4 (`f_xxx ≥ 7/4`).** If `ψ = f_xxx` has derivative of norm at most `n` on the collar
`[-2r - ε, 2r + ε] × B̄(0, 2r)` (`0 < ε ≤ r/2`), `ψ(x₀, 0) ≥ 2` at a point of the pin interval and
(H2) `rn ≤ 1/20`, then `ψ ≥ 2 - 3rn ≥ 7/4` on the collar (SP: `2 - 5rn`; the product norm is the
maximum of `|Δx| ≤ 3r` and `‖Δy‖ ≤ 2r`). -/
theorem fxxx_lower {ψ : ℝ × E → ℝ} {Dψ : ℝ × E → ℝ × E →L[ℝ] ℝ} {r n ε x₀ : ℝ} (hr : 0 < r)
    (hε : 0 < ε) (hεr : ε ≤ r / 2)
    (hψ : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt ψ (Dψ p) p) (hn : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖Dψ p‖ ≤ n)
    (hx₀ : x₀ ∈ Icc (-r / 2) (r / 2)) (h2 : 2 ≤ ψ (x₀, 0)) (hH2 : r * n ≤ 1 / 20) :
    ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ ψ p := by
  intro p hp
  have hcvx : Convex ℝ (Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r)) := (convex_Icc _ _).prod (convex_closedBall _ _)
  have h0 : ((x₀ : ℝ), (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    ⟨⟨by linarith [hx₀.1], by linarith [hx₀.2]⟩, Metric.mem_closedBall_self (by positivity)⟩
  have hmv := hcvx.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun q hq => (hψ q hq).hasFDerivWithinAt) hn h0 hp
  have hdist : ‖p - (x₀, 0)‖ ≤ 3 * r := by
    obtain ⟨⟨hp1, hp2⟩, hp3⟩ := hp
    have hy : ‖p.2‖ ≤ 2 * r := by simpa [Metric.mem_closedBall, dist_zero_right] using hp3
    rw [Prod.norm_def]
    refine max_le ?_ ?_
    · simp only [Prod.fst_sub, Real.norm_eq_abs]
      rw [abs_le]; constructor <;> linarith [hx₀.1, hx₀.2]
    · simp only [Prod.snd_sub, sub_zero]
      linarith
  have hn0 : 0 ≤ n := (norm_nonneg _).trans (hn p hp)
  rw [Real.norm_eq_abs] at hmv
  have h3 := (abs_le.mp hmv).1
  have h4 := mul_le_mul_of_nonneg_left hdist hn0
  nlinarith

/-- **The cap from SP's hypotheses (H1), (H2) and the derivative bounds (L1–L20 composed).** As
`ridge_capHyp_C3`, with `f_xxx ≥ 7/4` derived instead of assumed: from a derivative `Φ4` of `f_xxx`
with `‖Φ4‖ ≤ n` on the collar (SP's `n = M_4` on that block) and (H2) `rn ≤ 1/20`. L2's second
statement gives `f_xxx(x₀, 0) ≥ 2` on the pin interval (`third_deriv_exists_two`), and
`fxxx_lower` gives `f_xxx ≥ 7/4` on the collar. -/
theorem ridge_capHyp_C4 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
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
    (hD4 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖Φ4 p‖ ≤ n) (hH2 : r * n ≤ 1 / 20)
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
  have hmem : ∀ t ∈ Icc (-r / 2) (r / 2), (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := fun t ht =>
    ⟨⟨by linarith [ht.1], by linarith [ht.2]⟩, Metric.mem_closedBall_self (by positivity)⟩
  have hp : ∀ t : ℝ, HasDerivAt (fun s : ℝ => (s, (0 : E))) ((1 : ℝ), (0 : E)) t := fun t =>
    (hasDerivAt_id t).prodMk (hasDerivAt_const t (0 : E))
  have hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    exact (hDf _ (hmem x hx)).comp_hasDerivAt x (hp x)
  have hG : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun t => Df (t, 0) (1, 0)) (D2f (x, 0) (1, 0) (1, 0)) x := by
    intro x hx
    have h1 : HasDerivAt (fun t => Df (t, 0)) (D2f (x, 0) (1, 0)) x :=
      (hD2 _ (hmem x hx)).comp_hasDerivAt x (hp x)
    simpa using h1.clm_apply (hasDerivAt_const x ((1 : ℝ), (0 : E)))
  have hG' : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun t => D2f (t, 0) (1, 0) (1, 0)) (T3 (x, 0) (1, 0) (1, 0) (1, 0)) x := by
    intro x hx
    exact (hD3 _ (hmem x hx) (1, 0) (1, 0)).comp_hasDerivAt x (hp x)
  have hc : Continuous (fun s : ℝ => (s, (0 : E))) := continuous_id.prodMk continuous_const
  have hφc : ContinuousOn (fun t => T3 (t, 0) (1, 0) (1, 0) (1, 0)) (Icc (-r / 2) (r / 2)) := by
    intro t ht
    have h1 : ContinuousAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (t, (0 : E)) :=
      (hD4 _ (hmem t ht)).continuousAt
    have h2 : ContinuousAt (fun s : ℝ => T3 (s, 0) (1, 0) (1, 0) (1, 0)) t :=
      ContinuousAt.comp (g := fun q => T3 q (1, 0) (1, 0) (1, 0)) (f := fun s : ℝ => (s, (0 : E)))
        h1 hc.continuousAt
    exact h2.continuousWithinAt
  have hGa : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hGc : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hxxx : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ T3 p (1, 0) (1, 0) (1, 0) := by
    obtain ⟨x₀, hx₀, h2⟩ := third_deriv_exists_two hr hg hG hG' hφc hGa hGc hnorm
    exact fxxx_lower (ψ := fun q => T3 q (1, 0) (1, 0) (1, 0)) hr hε hεr hD4 hn hx₀ h2 hH2
  exact ridge_capHyp_C3 hr hε hεr hDf hD2 hD3 hT hxxx hGs hH hHx hHxb hHr hHrb hlam hH1 hM hS
    hG0cv hG0cc hwcv hwcc hnorm

end Z
