import Specifications.RN

/-! RN-A probe 1 (triage only; `sorry` permitted here and nowhere else).
Purpose: (a) confirm the Mathlib names a counter-inhabitant construction would use exist at the
pinned revision; (b) check that under a Dirac tilted law the mean counts reduce to the pathwise
count, so that a null `E0` containing one critical point violates every `C r^3 |E0|` bound;
(c) check `omegaD 2 r p = 1`, so the `d = 2` weight in `DimensionLiftPuncturedPinBallP` is the
plain volume. -/

open MeasureTheory UniversalLaw.Spec.RN

#check @MeasureTheory.integral_dirac
#check @MeasureTheory.integral_dirac'
#check @MeasureTheory.withDensity_one
#check @MeasureTheory.withDensity_congr_ae
#check @MeasureTheory.ae_dirac_iff
#check @MeasureTheory.setIntegral_measure_zero
#check @Set.ncard_singleton
#check @Set.Infinite.ncard
#check @ContDiffBump
#check @ContDiffBump.contDiff
#check @ContDiffBump.one_of_mem_closedBall
#check @Real.rpow_zero
#check @MeasureTheory.Measure.dirac.isProbabilityMeasure
#check @iteratedFDeriv_two_apply

/-- (b): with `QW r b k R = dirac ω0` the expected all-height count is the pathwise count. -/
example (L : ℝ) (Ω : Type) [MeasurableSpace Ω] [MeasurableSingletonClass Ω]
    (I : PinnedLawInterface 2 L Ω) (r b k : ℝ) (R : OrthonormalBasis (Fin 2) ℝ (E 2)) (j : ℕ)
    (A : Set (E 2)) (ω0 : Ω) (h : I.QW r b k R = Measure.dirac ω0) :
    meanAllHeightCount 2 L Ω I r b k R j A = (allHeightCount 2 (I.F ω0) j A : ℝ) := by
  unfold meanAllHeightCount
  rw [h, integral_dirac]

/-- (b'): a singleton has volume zero, so the right side `C * r^3 * (volume {p}).toReal` is `0`. -/
example (C r : ℝ) (p : E 2) : C * r ^ 3 * (volume ({p} : Set (E 2))).toReal = 0 := by
  simp

/-- (b''): if the single path has an index-`j` critical point at `x0 ∈ A`, the count is `≥ 1`
provided the critical set in `A` is finite. -/
example (g : E 2 → ℝ) (j : ℕ) (A : Set (E 2)) (x0 : E 2) (hx : x0 ∈ A)
    (hc : IsCriticalPoint 2 g x0) (hi : HessianHasInertia 2 g x0 j)
    (hfin : {x : E 2 | x ∈ A ∧ IsCriticalPoint 2 g x ∧ HessianHasInertia 2 g x j}.Finite) :
    1 ≤ allHeightCount 2 g j A := by
  unfold allHeightCount
  have hmem : x0 ∈ {x : E 2 | x ∈ A ∧ IsCriticalPoint 2 g x ∧ HessianHasInertia 2 g x j} :=
    ⟨hx, hc, hi⟩
  exact (Set.ncard_pos hfin).mpr ⟨x0, hmem⟩

/-- (c): the `d = 2` weight is identically `1`. -/
example (r : ℝ) (p : E 2) : omegaD 2 r p = 1 := by
  unfold omegaD
  simp

/-- (d): the pin regime gate is nonvacuous for `0 < L`: `r = L/2` is admissible. -/
example (L : ℝ) (hL : 0 < L) : 0 < L / 2 ∧ L / 2 < L := by constructor <;> linarith
