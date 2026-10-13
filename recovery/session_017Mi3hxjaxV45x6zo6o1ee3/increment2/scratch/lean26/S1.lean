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

#print axioms torus_capHyp_blocks
#print axioms torus_elder_death_level_blocks
#print axioms torus_locallyConnectedSpace

end CapFirstExit
