import CapFirstExit

open Set

namespace CapFirstExit

theorem addCircle_locallyConnectedSpace (L : ℝ) : LocallyConnectedSpace (AddCircle L) :=
  (QuotientAddGroup.isQuotientMap_mk (AddSubgroup.zmultiples L)).isCoinducing.locallyConnectedSpace

theorem torus_locallyConnectedSpace (L : ℝ) (d : ℕ) :
    LocallyConnectedSpace (Fin d → AddCircle L) :=
  haveI := addCircle_locallyConnectedSpace L
  Pi.locallyConnectedSpace_of_finite_not_preconnectedSpace (Set.toFinite _)

section Torus

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem torus_capHyp_blocks [FiniteDimensional ℝ E] {L : ℝ} {d : ℕ}
    (φ : ℝ × E ≃ₜ (Fin d → ℝ)) {f : (Fin d → AddCircle L) → ℝ} {g : ℝ × E → ℝ}
    (hg : ∀ p, g p = f (torusCover L d (φ p))) {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 g U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 g (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 g p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 g p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 g (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ g (-r / 2, 0) = 0) (hS : fderiv ℝ g (r / 2, 0) = 0)
    (hnorm : g (-r / 2, 0) - g (r / 2, 0) = k * r ^ 3) :
    ∃ z, CapHyp f ((torusCover L d ∘ φ) '' (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)))
        (torusCover L d (φ (-r / 2, 0))) z (f (torusCover L d (φ (-r / 2, 0))))
        (f (torusCover L d (φ (r / 2, 0)))) ∧
      ∀ x ∈ frontier ((torusCover L d ∘ φ) ''
          (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r))),
        x ≠ torusCover L d (φ (r / 2, 0)) → f x < f (torusCover L d (φ (r / 2, 0))) := by
  obtain ⟨ε, -, ⟨h0, hh0⟩, hall⟩ := ridge_capHyp_blocks hr hk hU hDU hf h30 h31 h32 h33 h40 h41
    hlam hH1 hH2 hM hS hnorm
  obtain ⟨-, -, hC, hSf⟩ := hall h0 fun x hx => ⟨Metric.ball_subset_closedBall (hh0 x hx).1,
    (hh0 x hx).2⟩
  have hgf : g = f ∘ (torusCover L d ∘ φ) := funext hg
  subst hgf
  have hK : IsCompact (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) :=
    isCompact_Icc.prod (isCompact_closedBall 0 (2 * r))
  exact ⟨_, hC.toTorus φ hK, CapHyp.map_saddle ((torusCover_continuous L d).comp φ.continuous)
    ((torusCover_isOpenMap L d).comp φ.isOpenMap) hK hSf⟩

theorem torus_elder_death_level_blocks [FiniteDimensional ℝ E] {L : ℝ} [Fact (0 < L)] {d : ℕ}
    (φ : ℝ × E ≃ₜ (Fin d → ℝ)) {f : (Fin d → AddCircle L) → ℝ} (hfc : Continuous f)
    {g : ℝ × E → ℝ} (hg : ∀ p, g p = f (torusCover L d (φ p))) {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 g U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 g (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 g p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 g p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 g (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ g (-r / 2, 0) = 0) (hS : fderiv ℝ g (r / 2, 0) = 0)
    (hnorm : g (-r / 2, 0) - g (r / 2, 0) = k * r ^ 3) :
    IsGreatest (CapHyp.olderPeakLevels f (torusCover L d (φ (-r / 2, 0))))
      (f (torusCover L d (φ (r / 2, 0)))) := by
  obtain ⟨z, H, -⟩ := torus_capHyp_blocks φ hg hr hk hU hDU hf h30 h31 h32 h33 h40 h41 hlam hH1
    hH2 hM hS hnorm
  have := torus_locallyConnectedSpace L d
  exact H.elder_death_level_peak hfc rfl

end Torus


noncomputable def circleChart (L : ℝ) (θ : AddCircle L) : ℝ :=
  ‖θ + ((L / 4 : ℝ) : AddCircle L)‖ - L / 4

theorem circleChart_continuous (L : ℝ) : Continuous (circleChart L) :=
  (continuous_norm.comp (continuous_id.add continuous_const)).sub continuous_const

theorem circleChart_coe {L : ℝ} (hL : 0 < L) {x : ℝ} (hx : |x| ≤ L / 4) :
    circleChart L (x : AddCircle L) = x := by
  have h1 := (abs_le.mp hx).1
  have h2 := (abs_le.mp hx).2
  have hn : ‖((x + L / 4 : ℝ) : AddCircle L)‖ = |x + L / 4| :=
    (AddCircle.norm_coe_eq_abs_iff L hL.ne').mpr (by
      rw [abs_of_nonneg (by linarith), abs_of_pos hL]; linarith)
  unfold circleChart
  rw [← AddCircle.coe_add, hn, abs_of_nonneg (by linarith)]
  ring

theorem mixedToy_hasFDerivAt (p : ℝ × ℝ) : HasFDerivAt mixedToyF
    ((p.1 ^ 2 - 1 / 4 + p.2 ^ 2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p := hasFDerivAt_fst
  have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p := hasFDerivAt_snd
  have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
    (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3))
    (HasFDerivAt.const_mul hf (1 / 4)))
    (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20))
    (HasFDerivAt.mul hf (HasFDerivAt.pow hs 2))
  convert h1 using 1
  · funext q
    simp only [mixedToyF, Pi.sub_apply, Pi.add_apply, Pi.mul_apply]
    ring
  · ext <;> simp
    ring

theorem mixedToy_contDiff : ContDiff ℝ 4 mixedToyF := by unfold mixedToyF; fun_prop

theorem mixedToy_iteratedFDeriv_two (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 mixedToyF y m =
      2 * y.1 * (m 0).1 * (m 1).1 + 2 * y.2 * ((m 0).1 * (m 1).2 + (m 0).2 * (m 1).1) +
        (-40 + 2 * y.1) * (m 0).2 * (m 1).2 := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + q.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2 + 2 * q.1 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * y.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * y.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((2 * y.2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (-40 + 2 * y.1) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.snd ℝ ℝ ℝ)) y := by
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + q.2 ^ 2)
        ((2 * y.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * y.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) y := by
      have := ((HasFDerivAt.pow (hf y) 2).sub_const (1 / 4)).add (HasFDerivAt.pow (hs y) 2)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2 + 2 * q.1 * q.2)
        ((2 * y.2) • ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 + 2 * y.1) • ContinuousLinearMap.snd ℝ ℝ ℝ)
        y := by
      have := (HasFDerivAt.const_mul (hs y) (-40)).add
        ((HasFDerivAt.const_mul (hf y) 2).mul (hs y))
      convert this using 1
      ext <;> simp
      ring
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hfd : fderiv ℝ mixedToyF = fun p => (p.1 ^ 2 - 1 / 4 + p.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (mixedToy_hasFDerivAt p).fderiv
  rw [iteratedFDeriv_two_apply, hfd, hD2.fderiv]
  simp
  ring

theorem mixedToy_iteratedFDeriv_three (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ) :
    iteratedFDeriv ℝ 3 mixedToyF x m =
      2 * (m 0).1 * (m 1).1 * (m 2).1 + 2 * (m 0).2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1) +
        2 * (m 0).1 * (m 1).2 * (m 2).2 := by
  rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
    (mixedToy_contDiff.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
  have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 mixedToyF y (Fin.tail m))
      ((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) • ContinuousLinearMap.snd ℝ ℝ ℝ) x := by
    have := (((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) •
          ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt (x := x)).add_const (-40 * (m 1).2 * (m 2).2)
    convert this using 1
    funext y; rw [mixedToy_iteratedFDeriv_two]; simp [Fin.tail]; ring
  rw [hg.fderiv]
  simp
  ring

theorem mixedToy_iteratedFDeriv_four (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ) :
    iteratedFDeriv ℝ 4 mixedToyF x m = 0 := by
  rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
    (mixedToy_contDiff.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
  have hc : (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m)) =
      fun _ => iteratedFDeriv ℝ 3 mixedToyF x (Fin.tail m) := by
    funext y; rw [mixedToy_iteratedFDeriv_three, mixedToy_iteratedFDeriv_three]
  have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m))
      (0 : ℝ × ℝ →L[ℝ] ℝ) x := by
    rw [hc]; exact hasFDerivAt_const _ _
  rw [hg.fderiv]
  simp

noncomputable def torusToyF (θ : Fin 2 → AddCircle (16 : ℝ)) : ℝ :=
  mixedToyF (circleChart 16 (θ 0), circleChart 16 (θ 1))

theorem torusToy_elder :
    IsGreatest (CapHyp.olderPeakLevels torusToyF
        (torusCover 16 2 ((Homeomorph.finTwoArrow (X := ℝ)).symm (-1 / 2, 0))))
      (torusToyF (torusCover 16 2 ((Homeomorph.finTwoArrow (X := ℝ)).symm (1 / 2, 0)))) := by
  have : Fact ((0 : ℝ) < 16) := ⟨by norm_num⟩
  set φ : ℝ × ℝ ≃ₜ (Fin 2 → ℝ) := (Homeomorph.finTwoArrow (X := ℝ)).symm with hφ
  set U : Set (ℝ × ℝ) := Ioo (-4) 4 ×ˢ Ioo (-4) 4 with hU_def
  have hU : IsOpen U := isOpen_Ioo.prod isOpen_Ioo
  have hGU : ∀ p ∈ U, torusToyF (torusCover 16 2 (φ p)) = mixedToyF p := by
    rintro ⟨a, b⟩ ⟨ha, hb⟩
    have ha' : |a| ≤ 16 / 4 := by rw [abs_le]; constructor <;> linarith [ha.1, ha.2]
    have hb' : |b| ≤ 16 / 4 := by rw [abs_le]; constructor <;> linarith [hb.1, hb.2]
    simp only [torusToyF, torusCover, hφ, Homeomorph.finTwoArrow_symm_apply, Pi.map_apply]
    simp [circleChart_coe (by norm_num : (0 : ℝ) < 16) ha',
      circleChart_coe (by norm_num : (0 : ℝ) < 16) hb']
  have hev : ∀ p ∈ U, (fun q => torusToyF (torusCover 16 2 (φ q))) =ᶠ[nhds p] mixedToyF :=
    fun p hp => Filter.eventuallyEq_of_mem (hU.mem_nhds hp) hGU
  have hD : ∀ n p, p ∈ U → iteratedFDeriv ℝ n (fun q => torusToyF (torusCover 16 2 (φ q))) p =
      iteratedFDeriv ℝ n mixedToyF p :=
    fun n p hp => ((hev p hp).iteratedFDeriv ℝ n).eq_of_nhds
  have hDU : Icc (-2 * 1) (2 * 1) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1) ⊆ U := by
    rintro ⟨a, b⟩ ⟨ha, hb⟩
    rw [Metric.mem_closedBall, Real.dist_eq, sub_zero, abs_le] at hb
    exact ⟨⟨by linarith [ha.1], by linarith [ha.2]⟩, ⟨by linarith [hb.1], by linarith [hb.2]⟩⟩
  have hseg : ∀ x ∈ Icc (-1 / 2 : ℝ) (1 / 2), ((x, 0) : ℝ × ℝ) ∈ U := fun x hx =>
    ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, ⟨by norm_num, by norm_num⟩⟩
  have hM : ((-1 / 2, 0) : ℝ × ℝ) ∈ U := hseg _ ⟨by norm_num, by norm_num⟩
  have hS : ((1 / 2, 0) : ℝ × ℝ) ∈ U := hseg _ ⟨by norm_num, by norm_num⟩
  refine torus_elder_death_level_blocks (E := ℝ) (L := 16) (d := 2) φ
    (mixedToy_contDiff.continuous.comp ((circleChart_continuous 16).comp (continuous_apply 0) |>.prodMk
      ((circleChart_continuous 16).comp (continuous_apply 1))))
    (g := fun q => torusToyF (torusCover 16 2 (φ q))) (fun p => rfl) (U := U)
    (r := 1) (k := 1 / 6) (M3 := 2) (M4 := 0) (lam := 41) one_pos (by norm_num) hU hDU
    (mixedToy_contDiff.contDiffOn.congr fun p hp => hGU p hp)
    (fun x hx => ?_) (fun p hp y => ?_) (fun p hp y₁ y₂ => ?_) (fun p hp y₁ y₂ y₃ => ?_)
    (fun p hp => ?_) (fun p hp v => ?_) (fun v => ?_) (by norm_num) (by norm_num) ?_ ?_ ?_
  · rw [hD 3 _ (hseg _ (by simpa using hx)), mixedToy_iteratedFDeriv_three]; simp
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]; simp
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
      Matrix.head_cons, Real.norm_eq_abs]
    rw [show (2 * (0 : ℝ) * 0 * 1 + 2 * y₁ * (0 * 0 + y₂ * 1) + 2 * 0 * y₂ * 0) = 2 * (y₁ * y₂) by
      ring, abs_mul, abs_mul]
    norm_num [mul_assoc]
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
      Matrix.head_cons, Real.norm_eq_abs]
    rw [show (2 * (0 : ℝ) * 0 * 0 + 2 * y₁ * (0 * y₃ + y₂ * 0) + 2 * 0 * y₂ * y₃) = 0 by ring,
      abs_zero]
    positivity
  · rw [hD 4 _ (hDU hp), mixedToy_iteratedFDeriv_four]; simp
  · rw [hD 4 _ (hDU hp), mixedToy_iteratedFDeriv_four]; simp
  · rw [hD 2 _ (by simpa using hM), mixedToy_iteratedFDeriv_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg v]
  · rw [(hev _ (by simpa using hM)).fderiv_eq, (mixedToy_hasFDerivAt _).fderiv]
    ext <;> norm_num
  · rw [(hev _ (by simpa using hS)).fderiv_eq, (mixedToy_hasFDerivAt _).fderiv]
    ext <;> norm_num
  · rw [hGU _ (by simpa using hM), hGU _ (by simpa using hS)]
    simp only [mixedToyF]; norm_num

#print axioms torus_capHyp_blocks
#print axioms torus_elder_death_level_blocks
#print axioms torusToy_elder

end CapFirstExit
