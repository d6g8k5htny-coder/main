import TwoPinRank

noncomputable section
open I4TwoPinRank I4PinMajorant I4Source I4Torus MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix
namespace TwoPinRankTests

def separatedSites (d : ℕ) : Fin 2 → (Fin (d+1) → ℝ) := ![0,fun _ => 1/2]
private theorem separated (d : ℕ) :
    torusProjection (d+1) 1 (separatedSites d 0) ≠
      torusProjection (d+1) 1 (separatedSites d 1) := by
  let : Fact (0 < (1:ℝ)) := ⟨by norm_num⟩
  intro h
  have h0 := congrFun h (0 : Fin (d+1))
  change ((0:ℝ) : AddCircle (1:ℝ)) = ((1/2:ℝ) : AddCircle (1:ℝ)) at h0
  have he := (AddCircle.coe_eq_coe_iff_of_mem_Ico
    (p := (1:ℝ)) (a := 0) (by norm_num : (0:ℝ) ∈ Set.Ico 0 (0+1))
    (by norm_num : (1/2:ℝ) ∈ Set.Ico 0 (0+1))).mp h0
  norm_num at he

example : ∃ i, Real.cos (I4Weights.sourceC 1 *
    (separatedSites 0 0 i-separatedSites 0 1 i)) ≠ 1 :=
  torus_distinct_phase 1 (by norm_num) _ _ (separated 0)

-- One-dimensional nonzero physical derivative functional.
example : 0 < Var[twoPinObservable 1 1 (separatedSites 0)
    (Pi.single (0,some 0) 1);sourceLaw 1] := by
  apply twoPinObservable_variance_pos 1 (by norm_num) _ (separated 0)
  intro h
  have h0 := congrFun h (0,some (0:Fin 1))
  norm_num at h0

-- Required physical dimension two, complete value/gradient covariance.
example : (pinCovariance 2 1 (separatedSites 1)).PosDef :=
  pinCovariance_posDef 2 (by norm_num) _ (separated 1)

example : ∃ m, combinedCoefficient 2 1 (separatedSites 1)
    (Pi.single (1,none) 1) m ≠ 0 := by
  apply combinedCoefficient_nonzero 2 (by norm_num) _ (separated 1)
  intro h
  have h0 := congrFun h (1,none)
  norm_num at h0

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) (omega : SourceMode d → ℝ) :
    twoPinObservable d L sites 0 omega = 0 := by simp [twoPinObservable]

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Var[twoPinObservable d L sites 0;sourceLaw d] = 0 := by
  have he : twoPinObservable d L sites 0 = 0 := by
    funext omega; simp [twoPinObservable]
  rw [he,variance_zero]

def valueDifference (d : ℕ) : PinIndex d → ℝ :=
  fun i => if i.2 = none then if i.1 = 0 then 1 else -1 else 0

example (d : ℕ) : valueDifference d ≠ 0 := by
  intro h
  have h0 := congrFun h (0,none)
  norm_num [valueDifference] at h0

-- Coinciding pins admit a nonzero annihilating functional on actual samples.
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (omega : SourceMode d → ℝ) :
    twoPinObservable d L (fun _ => x) (valueDifference d) omega = 0 := by
  simp [twoPinObservable,Fintype.sum_prod_type,Fin.sum_univ_two,
    valueDifference,actualPinObservation]

example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) :
    Var[twoPinObservable d L (fun _ => x) (valueDifference d);sourceLaw d] = 0 := by
  have he : twoPinObservable d L (fun _ => x) (valueDifference d) = 0 := by
    funext omega
    simp [twoPinObservable,Fintype.sum_prod_type,Fin.sum_univ_two,
      valueDifference,actualPinObservation]
  rw [he,variance_zero]

example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (m : SourceMode d) :
    combinedCoefficient d L (fun _ => x) (valueDifference d) m = 0 := by
  simp [combinedCoefficient,Fintype.sum_prod_type,Fin.sum_univ_two,
    valueDifference,pinCoefficient,pinOrder,pinDirections]

-- Distinct cover points separated by a period are one torus point.
example : torusProjection 1 1 (0 : Fin 1 → ℝ) = torusProjection 1 1 (fun _ => 1) := by
  funext i
  simp [torusProjection,AddCircle.coe_period]

example (omega : SourceMode 1 → ℝ) :
    twoPinObservable 1 1 ![0,fun _ => 1] (valueDifference 1) omega = 0 := by
  have hfield := sourceField_eq_of_torusProjection_eq 1 (by norm_num : (1:ℝ) ≠ 0)
    omega (0 : Fin 1 → ℝ) (fun _ => 1) (by
      funext i; simp [torusProjection,AddCircle.coe_period])
  simpa [twoPinObservable,Fintype.sum_prod_type,Fin.sum_univ_two,
    valueDifference,actualPinObservation,sub_eq_add_neg] using sub_eq_zero.mpr hfield

-- The empty physical coordinate space cannot satisfy torus-distinctness.
example (L : ℝ) (x y : Fin 0 → ℝ) : torusProjection 0 L x = torusProjection 0 L y := by
  funext i; exact Fin.elim0 i

end TwoPinRankTests

