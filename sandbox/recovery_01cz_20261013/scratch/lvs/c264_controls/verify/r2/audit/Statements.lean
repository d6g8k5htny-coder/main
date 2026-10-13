import TorusFiltration
open CategoryTheory Limits AlgebraicTopology TorusFiltration

/- Verifier restatements: each fixture's statement written independently with the fixtures unfolded,
   closed only by the author's declaration (or by rfl/decide for definitional identities). -/

-- the coefficient objects are what their names say
example : ordinaryH0Control_intCoeff = ModuleCat.of ℤ ℤ := rfl
example : ordinaryH0Control_zeroCoeff = ModuleCat.of ℤ PUnit.{1} := rfl
example : ordinaryH0Control_mod2Coeff = ModuleCat.of ℤ (ZMod 2) := rfl
-- the zero-coefficient object really is a zero object, and ZMod 2 really has characteristic two
example : IsZero (ModuleCat.of ℤ PUnit.{1}) := ordinaryH0Control_zeroCoeff_isZero
example : ringChar (ZMod 2) = 2 := ZMod.ringChar_zmod_n 2
example : (𝟙 (ModuleCat.of ℤ (ZMod 2)) : _ ⟶ _) ≠ 0 := ordinaryH0Control_mod2Coeff_id_ne_zero
-- zero coefficients: H0 of every space is a zero object, via the natural iso at the zero module
example : ∀ X : TopCat.{0}, IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj (ModuleCat.of ℤ PUnit.{1})).obj X) :=
  ordinaryH0Control_zeroCoeff_h0_isZero
example : (ordinaryH0_naturalIso (ModuleCat.of ℤ PUnit.{1})).app (TopCat.of Bool) =
    TopCat.singularHomology₀Iso (TopCat.of Bool) (ModuleCat.of ℤ PUnit.{1}) := rfl
-- empty
example : IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj (ModuleCat.of ℤ ℤ)).obj (TopCat.of Empty)) :=
  ordinaryH0Control_empty_h0_isZero
-- point
example : ¬ IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj (ModuleCat.of ℤ ℤ)).obj (TopCat.of Unit)) :=
  ordinaryH0Control_point_h0_not_isZero
-- the merge map merges: it is the constant map, and the two image components coincide by computation
example : ordinaryH0Control_collapse = TopCat.ofHom (ContinuousMap.const Bool ()) := rfl
example : ZerothHomotopy.mk (ordinaryH0Control_collapse true) = ZerothHomotopy.mk (ordinaryH0Control_collapse false) := rfl
-- the projection is the first projection, and is not injective
example : ∀ p : Bool × Bool, ordinaryH0Control_fst p = p.1 := fun _ => rfl
example : ¬ Function.Injective (ordinaryH0Control_fst : (Bool × Bool) → Bool) := fun h =>
  absurd (h (a₁ := (true, false)) (a₂ := (true, true)) rfl) (by decide)
-- two-component premise is a theorem, not a hypothesis
example : ZerothHomotopy.mk true ≠ ZerothHomotopy.mk false := ordinaryH0Control_bool_component_ne
example : ZerothHomotopy.mk ((true, false) : Bool × Bool) ≠ ZerothHomotopy.mk ((false, true) : Bool × Bool) :=
  ordinaryH0Control_discrete_component_ne (by decide)
-- the strict inclusion is strict and the target level is {x | 1/4 ≤ x^2}
example : (-1 : ℝ) < -1 / 4 := ordinaryH0Control_parabola_strict_growth.1
example : superlevel_set (fun x : ℝ => x ^ 2) (-1 / 4) = {x | 1 / 4 ≤ x ^ 2} := by
  ext x; show -(-1 / 4 : ℝ) ≤ x ^ 2 ↔ 1 / 4 ≤ x ^ 2; norm_num
example : (ordinaryH0Control_parabolaLeft : ℝ) = -1 ∧ (ordinaryH0Control_parabolaRight : ℝ) = 1 := ⟨rfl, rfl⟩

-- each expected-failure proposition is FALSE: its negation is a theorem (kernel-checked here)
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0) :=
  ordinaryH0Control_collapse_image_ne_zero true
example : ¬ (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false) :=
  fun h => h ordinaryH0Control_bool_zeroCoeff_eq
example : ¬ (CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0) :=
  ordinaryH0Control_fst_difference_ne_zero
-- and the true siblings hold: the merger kills the difference; the projection kills the merged pair's difference
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true - ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0 :=
  ordinaryH0Control_collapse_difference_zero
example : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false :=
  ordinaryH0Control_bool_pointClass_ne
