import Mathlib
import OrdinaryH0
import OrdinaryH0Controls
open TorusFiltration
theorem rejected : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true - ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_swap) = 0 := by
  simp only [ordinaryH0Control_swap_difference_ne_zero]
