import OrdinaryH0Split

noncomputable section
set_option backward.isDefEq.respectTransparency false
open CategoryTheory OrdinaryH0Elder OrdinaryH0Split
open scoped Classical

namespace OrdinaryH0Split.Tests

def highToActive : OrderDual.toDual (CapFirstExit.toyF 3) ⟶ OrderDual.toDual (-1 / 2 : ℝ) :=
  @homOfLE ℝᵒᵈ _ (CapFirstExit.toyF 3) (-1 / 2)
    ((show (-1 / 2 : ℝ) ≤ 0 by norm_num).trans (le_of_lt CapFirstExit.toy_capHyp.older))

def birthToActive : OrderDual.toDual (0 : ℝ) ⟶ OrderDual.toDual (-1 / 2 : ℝ) :=
  @homOfLE ℝᵒᵈ _ 0 (-1 / 2) (show (-1 / 2 : ℝ) ≤ 0 by norm_num)

def activeToDeath : OrderDual.toDual (-1 / 2 : ℝ) ⟶ OrderDual.toDual (-1 : ℝ) :=
  @homOfLE ℝᵒᵈ _ (-1 / 2) (-1) (show (-1 : ℝ) ≤ -1 / 2 by norm_num)

example (K : Type) [Field K] :
    delta K CapFirstExit.toy_capHyp CapFirstExit.toy_f_M 0 ≠ 0 :=
  delta_ne_zero K CapFirstExit.toy_capHyp CapFirstExit.toy_f_M ⟨by norm_num, le_rfl⟩

example :
    intervalLine ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1) = ⊥ :=
  intervalLine_inactive ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (by simp [Active])

example :
    delta ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1) = 0 :=
  delta_eq_zero_below ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M le_rfl (by norm_num)

example :
    intervalInclusion ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M ≫
      intervalRetraction ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M =
      𝟙 (intervalLineFunctor ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M) :=
  interval_split ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (by norm_num)

/-- This is the actual incoming arrow from above the cap birth, for every source vector. -/
example (q : H0 ℚ (superlevel CapFirstExit.toyF (CapFirstExit.toyF 3))) :
    (intervalRetraction ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M).app
        (OrderDual.toDual (-1 / 2 : ℝ))
        ((closedH0Persistence ℚ CapFirstExit.toyF).map highToActive q) = 0 := by
  apply Subtype.ext
  change projectionLinear ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1 / 2)
    ((h0Functor ℚ).map (superlevelInclusion CapFirstExit.toyF
      ((show (-1 / 2 : ℝ) ≤ 0 by norm_num).trans (le_of_lt CapFirstExit.toy_capHyp.older))) q) = 0
  rw [projection_natural]
  have hi : ¬ Active (-1) 0 (CapFirstExit.toyF 3) := by
    intro ha
    exact (not_le_of_gt CapFirstExit.toy_capHyp.older) ha.2
  simp [projectionLinear, hi]

/-- The original closed-diagram map sends the active difference to zero at the ridge level. -/
example :
    (closedH0Persistence ℚ CapFirstExit.toyF).map activeToDeath
      (delta ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1 / 2)) = 0 := by
  rw [show (closedH0Persistence ℚ CapFirstExit.toyF).map activeToDeath
      (delta ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1 / 2)) =
      delta ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1) from
    delta_map ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M
      (r := (-1 / 2 : ℝ)) (h := (-1 : ℝ))
      (show (-1 : ℝ) ≤ -1 / 2 by norm_num) (show (-1 / 2 : ℝ) ≤ 0 by norm_num)]
  exact delta_eq_zero_below ℚ CapFirstExit.toy_capHyp CapFirstExit.toy_f_M le_rfl (by norm_num)

local instance : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩

example : delta (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1 / 2) ≠ 0 :=
  delta_ne_zero (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M ⟨by norm_num, by norm_num⟩

example : delta (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (-1) = 0 :=
  delta_eq_zero_below (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M le_rfl (by norm_num)

/-- Actual active coordinate maps are identities, including characteristic two. -/
example (a : ZMod 2) :
    (intervalLineFunctor (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M).map birthToActive
      (activeFiberEquiv (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M
        (show Active (-1) 0 0 from ⟨by norm_num, le_rfl⟩) a) =
      activeFiberEquiv (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M
        (show Active (-1) 0 (-1 / 2) from ⟨by norm_num, by norm_num⟩) a :=
  active_transport (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M birthToActive _ _ a

example :
    pointClass (ZMod 2) (superlevel CapFirstExit.toyF (-1))
      ⟨0, by rw [CapFirstExit.toy_f_M]; norm_num⟩ ≠ 0 :=
  pointClass_ne_zero (ZMod 2) _ _

example : SplitMono
    (intervalInclusion (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M) :=
  capSplitMono (ZMod 2) CapFirstExit.toy_capHyp CapFirstExit.toy_f_M (by norm_num)

end OrdinaryH0Split.Tests

