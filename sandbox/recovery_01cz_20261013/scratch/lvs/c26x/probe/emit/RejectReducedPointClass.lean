import TorusFiltration
open TorusFiltration
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Unit) (ModuleCat.of ℤ ℤ) ()) (TopCat.singularHomology₀ε (TopCat.of Unit) (ModuleCat.of ℤ ℤ)) ≠ CategoryTheory.CategoryStruct.id (ModuleCat.of ℤ ℤ) := by
  simp [ordinaryH0_pointClass_augmentation]
