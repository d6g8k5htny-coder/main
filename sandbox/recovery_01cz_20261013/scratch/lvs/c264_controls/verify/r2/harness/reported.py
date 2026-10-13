# Transcribed from the updated author report in the task text (not read from the author's files).
CONTROLS = {
 "RejectMergerKillsPointClass": (
  "CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0",
  "simp only [ordinaryH0Control_collapse_image_ne_zero]",
  "import TorusFiltration\nopen TorusFiltration\nexample : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0 := by\n  simp only [ordinaryH0Control_collapse_image_ne_zero]\n",
  "3:255"),
 "RejectSeparationWithoutNonzeroCoeff": (
  "ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false",
  "simp [ordinaryH0Control_bool_zeroCoeff_eq]",
  "import TorusFiltration\nopen TorusFiltration\nexample : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false := by\n  simp [ordinaryH0Control_bool_zeroCoeff_eq]\n",
  "3:160"),
 "RejectNonmergerKillsDifference": (
  "CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0",
  "simp only [ordinaryH0Control_fst_difference_ne_zero]",
  "import TorusFiltration\nopen TorusFiltration\nexample : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0 := by\n  simp only [ordinaryH0Control_fst_difference_ne_zero]\n",
  "3:391"),
}
