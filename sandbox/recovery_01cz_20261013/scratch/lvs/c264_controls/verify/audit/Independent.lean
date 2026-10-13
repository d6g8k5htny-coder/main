import OrdinaryH0Controls
/- Verifier's independent checks (not companion source). Only the author's fixture DEFINITIONS are used;
   every fact below is proved from Mathlib and the OrdinaryH0 API, never from an ordinaryH0Control_ theorem. -/
open CategoryTheory Limits AlgebraicTopology TorusFiltration

-- (F1) the zero-coefficient fixture really is a zero object with a subsingleton carrier
example : Subsingleton (ordinaryH0Control_zeroCoeff : Type) := inferInstance
example : IsZero ordinaryH0Control_zeroCoeff := ModuleCat.isZero_of_subsingleton _
-- (F2) the integer coefficient object is not a zero object; own proof of nonzero identity
theorem v_id_ne_zero : (𝟙 ordinaryH0Control_intCoeff : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff) ≠ 0 := by
  intro h
  have := congrArg (fun f : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff => f.hom (1 : ℤ)) h
  simp at this
example : ¬ IsZero ordinaryH0Control_intCoeff := fun h => v_id_ne_zero (h.eq_of_src _ _)
-- (F3) the merge fixture's map merges: both points go to the single point of Unit
example : ordinaryH0Control_collapse true = ordinaryH0Control_collapse false := rfl
-- (F4) the nonmerge fixture's map is the swap, a bijection true <-> false
example : ordinaryH0Control_swap true = false ∧ ordinaryH0Control_swap false = true := ⟨rfl, rfl⟩
-- (F5) two components of Bool, by a different route than the author's (range of a path is a preconnected subset
-- of a discrete space, hence subsingleton)
theorem v_bool_ne (a b : Bool) (hab : a ≠ b) : ZerothHomotopy.mk a ≠ ZerothHomotopy.mk b := by
  intro h
  obtain ⟨γ⟩ : Joined a b := Quotient.exact h
  exact hab ((isPreconnected_range γ.continuous).subsingleton ⟨0, γ.source⟩ ⟨1, γ.target⟩)
-- (F6) parabola levels: membership semantics
example : (superlevel_set (fun x : ℝ => x ^ 2) (-1)) = {x | 1 ≤ x ^ 2} := by
  ext x; simp [superlevel_set]
example : (superlevel_set (fun x : ℝ => x ^ 2) 0) = Set.univ := by
  ext x; simp [superlevel_set, sq_nonneg]
example : (0 : ℝ) ∉ superlevel_set (fun x : ℝ => x ^ 2) (-1) := by norm_num [superlevel_set]

-- (N1) RejectMergerKillsPointClass proposition is FALSE: the image class is the point class of Unit,
-- which augments to the nonzero identity of ℤ
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0) := by
  rw [ordinaryH0_pointClass_map]
  exact ordinaryH0_pointClass_ne_zero _ _ v_id_ne_zero _
-- (N2) RejectSeparationWithoutNonzeroCoeff proposition is FALSE: every pair of morphisms out of a zero object agrees
example : ¬ (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false) :=
  fun h => h ((ModuleCat.isZero_of_subsingleton ordinaryH0Control_zeroCoeff).eq_of_src _ _)
-- ... although the two components are distinct (the premise of the separation theorem holds)
example : ZerothHomotopy.mk true ≠ ZerothHomotopy.mk false := v_bool_ne true false (by decide)
-- (N3) RejectNonmergerKillsDifference proposition is FALSE: the image points false/true stay in distinct components
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true - ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_swap) = 0) := by
  rw [ordinaryH0_difference_map, sub_eq_zero]
  exact ordinaryH0_pointClass_ne_of_component_ne _ _ v_id_ne_zero _ _ (v_bool_ne false true (by decide))

-- (D1..D3) discrimination: deleting the intended reason makes each control's shape TRUE
-- D1: the merger kills the DIFFERENCE (not the class)
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true - ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0 :=
  (ordinaryH0_difference_map_eq_zero_iff _ v_id_ne_zero _ _ _).2 rfl
-- D2: with NONZERO coefficients separation holds
example : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false :=
  ordinaryH0_pointClass_ne_of_component_ne _ _ v_id_ne_zero _ _ (v_bool_ne true false (by decide))
-- D3: replacing the nonmerging swap by the merging collapse kills the difference (same as D1);
-- the swap with zero coefficients also kills it, so the nonmerge control needs the nonzero coefficient
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true - ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_zeroCoeff).map ordinaryH0Control_swap) = 0 :=
  (ModuleCat.isZero_of_subsingleton ordinaryH0Control_zeroCoeff).eq_of_src _ _
-- the merger keeps each individual image class alive for both points
example (b : Bool) : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff b) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) ≠ 0 := by
  rw [ordinaryH0_pointClass_map]
  exact ordinaryH0_pointClass_ne_zero _ _ v_id_ne_zero _
