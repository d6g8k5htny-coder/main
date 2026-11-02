import ProjectedMorseRank

noncomputable section
open I4Source I4Jets I4MorseRank MeasureTheory ProbabilityTheory
open scoped BigOperators
namespace MorseRankTests

private theorem single_ne (p : ℕ) (i : Fin (p+1)) :
    (Pi.single i 1 : Fin (p+1) → ℝ) ≠ 0 := by
  intro h
  have hi := congrFun h i
  simp at hi

-- Frame construction must work even when u's first coordinate is ZERO.
example : ∃ T : (Fin 2 → ℝ) ≃L[ℝ] (Fin 2 → ℝ),
    T (Pi.single 0 1) = Pi.single 1 1 :=
  exists_axial_frame 1 _ (single_ne 1 1)
example (p : ℕ) (u : Fin (p+1) → ℝ) (hu : u ≠ 0) :
    ∃ T : (Fin (p+1) → ℝ) ≃L[ℝ] (Fin (p+1) → ℝ),
      Function.Bijective T ∧ T (Pi.single 0 1) = u := by
  obtain ⟨T,hT⟩ := exists_axial_frame p u hu
  exact ⟨T,T.bijective,hT⟩
example (x : Fin 2 → ℝ) {L : ℝ} (hL : 0 < L) :
    0 < Var[morseObservable 1 L x (Pi.single 0 1) (Pi.single 1 1) 0;sourceLaw 2] :=
  morseObservable_variance_pos 1 (by omega) hL x _ _ _ (single_ne 1 0)
    (Or.inl (single_ne 1 1))
example (x : Fin 2 → ℝ) {L : ℝ} (hL : 0 < L) :
    0 < Var[morseObservable 1 L x (Pi.single 1 1) 0 (Pi.single 0 1);sourceLaw 2] :=
  morseObservable_variance_pos 1 (by omega) hL x _ _ _ (single_ne 1 1)
    (Or.inr (single_ne 1 0))
example (p : ℕ) (hp : 1 ≤ p) (x u : Fin (p+1) → ℝ) {L : ℝ} (hL : 0 < L)
    (hu : u ≠ 0) : (morseCovariance p L x u).PosDef :=
  morseCovariance_posDef p hp hL x u hu

private theorem zero_weights (p : ℕ) (L : ℝ) (x u : Fin (p+1) → ℝ) :
    morseObservable p L x u 0 0 = 0 := by
  funext sample
  unfold morseObservable sourceJet
  have h1 := (iteratedFDeriv ℝ 1 (sourceField (p+1) L sample) x).map_coord_zero
    (m := fun _ : Fin 1 => (0 : Fin (p+1) → ℝ)) 0 rfl
  have h2 := (iteratedFDeriv ℝ 2 (sourceField (p+1) L sample) x).map_coord_zero
    (m := fun k : Fin 2 => if k=0 then u else 0) 1 (by norm_num)
  rw [h1,h2]
  simp
example (p : ℕ) (L : ℝ) (x u : Fin (p+1) → ℝ) :
    Var[morseObservable p L x u 0 0;sourceLaw (p+1)] = 0 := by
  rw [zero_weights]
  simp
example (p : ℕ) (L : ℝ) (x u : Fin (p+1) → ℝ) :
    ¬ 0 < Var[morseObservable p L x u 0 0;sourceLaw (p+1)] := by
  rw [zero_weights]
  simp
example (p : ℕ) (L : ℝ) (x beta : Fin (p+1) → ℝ) :
    Var[morseObservable p L x 0 0 beta;sourceLaw (p+1)] = 0 := by
  have he : morseObservable p L x 0 0 beta = 0 := by
    funext sample
    unfold morseObservable sourceJet
    have h1 := (iteratedFDeriv ℝ 1 (sourceField (p+1) L sample) x).map_coord_zero
      (m := fun _ : Fin 1 => (0 : Fin (p+1) → ℝ)) 0 rfl
    have h2 := (iteratedFDeriv ℝ 2 (sourceField (p+1) L sample) x).map_coord_zero
      (m := fun k : Fin 2 => if k=0 then (0 : Fin (p+1) → ℝ) else beta) 0 (by simp)
    rw [h1,h2]
    simp
  rw [he]
  simp
-- Gaussianity includes degenerate u=0; it cannot replace the variance-rank theorem.
example (p : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin (p+1) → ℝ) :
    HasGaussianLaw (projectedMorseVector p L x 0) (sourceLaw (p+1)) :=
  projectedMorseVector_hasGaussianLaw p hL x 0
example (p : ℕ) {L : ℝ} (hL : L ≠ 0) (x u : Fin (p+1) → ℝ) :
    HasGaussianLaw (projectedMorseVector p L x u) (sourceLaw (p+1)) :=
  projectedMorseVector_hasGaussianLaw p hL x u
end MorseRankTests

