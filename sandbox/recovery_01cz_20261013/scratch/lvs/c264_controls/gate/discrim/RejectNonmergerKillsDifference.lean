import TorusFiltration
open TorusFiltration
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) - ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, true) : Bool × Bool)) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_fst) = 0 := by
  simp only [ordinaryH0Control_fst_difference_ne_zero]
