import TorusFiltration
open TorusFiltration CategoryTheory Limits AlgebraicTopology

-- (S1) the zero coefficient object really is a zero/one-point module, and its identity is zero
example (x y : ordinaryH0Control_zeroCoeff) : x = y := Subsingleton.elim x y
example : IsZero ordinaryH0Control_zeroCoeff := ModuleCat.isZero_of_subsingleton _
-- (S2) the collapse merges: both points have the same image
example : ordinaryH0Control_collapse true = ordinaryH0Control_collapse false := rfl
-- (S3) the projection is non-injective and separates the second pair
example : ordinaryH0Control_fst ((true, false) : Bool × Bool) = ordinaryH0Control_fst ((true, true) : Bool × Bool) := rfl
example : ((true, false) : Bool × Bool) ≠ (true, true) := by decide
example : ordinaryH0Control_fst ((true, false) : Bool × Bool) = true := rfl
example : ordinaryH0Control_fst ((false, true) : Bool × Bool) = false := rfl
-- (S4) the closed superlevels are the intended sets, the points are -1 and 1
example : superlevel_set (fun x : ℝ => x ^ 2) (-1) = {x | 1 ≤ x ^ 2} := by
  ext x; simp [superlevel_set]
example : superlevel_set (fun x : ℝ => x ^ 2) (-1 / 4) = {x | 1 / 4 ≤ x ^ 2} := by
  ext x; norm_num [superlevel_set]
example : ordinaryH0Control_parabolaLeft.val = -1 := rfl
example : ordinaryH0Control_parabolaRight.val = 1 := rfl
-- (S5) the char-2 object: 1 + 1 = 0 in ZMod 2, and it is not a zero module
example : (1 : ZMod 2) + 1 = 0 := by decide
example : ¬ IsZero ordinaryH0Control_mod2Coeff := fun h =>
  ordinaryH0Control_mod2Coeff_id_ne_zero (h.eq_of_src _ _)

-- (F1) merger control is false for the intended reason, re-derived from the MODULE's augmentation (not the fixture lemma)
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0) := by
  intro h
  rw [ordinaryH0_pointClass_map] at h
  have hε := ordinaryH0_pointClass_augmentation (TopCat.of Unit) ordinaryH0Control_intCoeff
    (ordinaryH0Control_collapse true)
  rw [h, zero_comp] at hε
  exact ordinaryH0Control_intCoeff_id_ne_zero hε.symm
-- (F1') with zero coefficients the very same merger proposition is TRUE: nonzero coefficients are operative
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_zeroCoeff).map ordinaryH0Control_collapse) = 0 :=
  (ModuleCat.isZero_of_subsingleton ordinaryH0Control_zeroCoeff).eq_of_src _ _
-- (F2) separation-without-hR control is false: zero coefficient object (identity zero) forces equality
example : ¬ (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false) :=
  fun h => h ((ModuleCat.isZero_of_subsingleton ordinaryH0Control_zeroCoeff).eq_of_src _ _)
-- (F2') ... while every other premise of the module's separation theorem holds: distinct components
example : ZerothHomotopy.mk true ≠ ZerothHomotopy.mk false := ordinaryH0Control_bool_component_ne
-- (F3) nonmerger control is false, re-derived from the module's difference map and separation theorems
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0) := by
  intro h
  rw [ordinaryH0_difference_map, sub_eq_zero] at h
  exact ordinaryH0_pointClass_ne_of_component_ne (TopCat.of Bool) ordinaryH0Control_intCoeff
    ordinaryH0Control_intCoeff_id_ne_zero _ _ ordinaryH0Control_bool_component_ne h
-- (F3') the merging pair under the same map gives a TRUE vanishing statement
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0 := by
  rw [ordinaryH0_difference_map, sub_eq_zero]
  rfl
-- (Z) the natural isomorphism instantiates at the zero coefficient object
example : Nonempty ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_zeroCoeff ≅
    ordinaryH0_componentCoproductFunctor ordinaryH0Control_zeroCoeff) := ⟨ordinaryH0_naturalIso _⟩
