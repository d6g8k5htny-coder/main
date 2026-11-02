import StrictParentBasis
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Tactic.NormNum

noncomputable section
open StrictParentBasis

private def nested : Fin 3 → Option (Fin 3) :=
  fun i => if i = 0 then none else if i = 1 then some 0 else some 1

private theorem nested_strict : Strict nested := by
  unfold Strict nested
  decide

local instance : Fact (Nat.Prime 2) := ⟨by decide⟩

-- A root is a point vector, not zero.
example : basis ℚ nested nested_strict 0 = Finsupp.single 0 1 := by
  simp [basis_apply, difference, nested]

-- The fixed immediate parent is 1, not the ultimate root 0.
example : basis ℚ nested nested_strict 2 =
    Finsupp.single 2 1 - Finsupp.single 1 1 := by
  simp [basis_apply, difference, nested]

example : basis ℚ nested nested_strict 2 ≠
    Finsupp.single 2 1 - Finsupp.single 0 1 := by
  rw [basis_apply]
  intro h
  have hz := congrArg (fun v : Fin 3 →₀ ℚ => v 1) h
  norm_num [difference, nested] at hz

example : LinearIndependent ℚ (difference ℚ nested) :=
  family_linearIndependent ℚ nested nested_strict

example : Submodule.span ℚ (Set.range (difference ℚ nested)) = ⊤ :=
  family_span ℚ nested nested_strict

example : equiv ℚ nested nested_strict (Finsupp.single 2 1) =
    Finsupp.single 2 1 - Finsupp.single 1 1 := by
  simp [equiv_single, difference, nested]

-- Characteristic two preserves the basis despite minus equalling plus.
example : LinearIndependent (ZMod 2) (difference (ZMod 2) nested) :=
  family_linearIndependent (ZMod 2) nested nested_strict

example : basis (ZMod 2) nested nested_strict 2 =
    Finsupp.single 2 1 + Finsupp.single 1 1 := by
  rw [basis_apply]
  ext i
  fin_cases i
  · norm_num [difference, nested]
  · norm_num [difference, nested]
    decide
  · norm_num [difference, nested]

private def emptyParent : Fin 0 → Option (Fin 0) := fun i => Fin.elim0 i
private theorem empty_strict : Strict emptyParent := by
  intro i
  exact Fin.elim0 i

example : LinearIndependent ℚ (difference ℚ emptyParent) :=
  family_linearIndependent ℚ emptyParent empty_strict

example : Submodule.span ℚ (Set.range (difference ℚ emptyParent)) = ⊤ :=
  family_span ℚ emptyParent empty_strict

-- Without strictness, a self-parent gives the zero vector and cannot be a basis.
private def selfParent : Fin 1 → Option (Fin 1) := fun i => some i
example : ¬ Strict selfParent := by
  intro h
  exact (lt_irrefl (0 : Fin 1)) (h 0 0 rfl)

example : difference ℚ selfParent 0 = 0 := by
  simp [difference, selfParent]

example : ¬ LinearIndependent ℚ (difference ℚ selfParent) := by
  intro h
  have hn := h.ne_zero 0
  simp [difference, selfParent] at hn

-- A genuine two-cycle is excluded; mere finite parent data is insufficient.
private def cycle : Fin 2 → Option (Fin 2) :=
  fun i => if i = 0 then some 1 else some 0
example : ¬ Strict cycle := by
  intro h
  have hn := h 0 1 (by decide)
  exact (by decide : ¬ ((1 : Fin 2) < 0)) hn

example : (parentMatrix ℚ cycle).det = 0 := by
  norm_num [Matrix.det_fin_two, parentMatrix, difference, cycle]

