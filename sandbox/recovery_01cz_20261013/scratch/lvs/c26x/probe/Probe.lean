import TorusFiltration
open TorusFiltration
-- A: C263 proposal, norm_num
example : superlevelBar_deathHeight (fun _ : superlevelBar_E 1 => (-1 : ℝ)) 0 ≤ (-1 : ℝ) := by
  norm_num [superlevelBar_death_const]
-- B: C263 proposal, simp
example : superlevelBar_deathHeight (fun _ : superlevelBar_E 1 => (-1 : ℝ)) 0 ≤ (-1 : ℝ) := by
  simp [superlevelBar_death_const]
-- C: C264 reduced point class, fully qualified names
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Unit) (ModuleCat.of ℤ ℤ) ()) (TopCat.singularHomology₀ε (TopCat.of Unit) (ModuleCat.of ℤ ℤ)) ≠ CategoryTheory.CategoryStruct.id (ModuleCat.of ℤ ℤ) := by
  simp [ordinaryH0_pointClass_augmentation]
-- D: C264 with open ... in term
example : open CategoryTheory in ordinaryH0_pointClass (TopCat.of Unit) (ModuleCat.of ℤ ℤ) () ≫ TopCat.singularHomology₀ε (TopCat.of Unit) (ModuleCat.of ℤ ℤ) ≠ 𝟙 (ModuleCat.of ℤ ℤ) := by
  simp [ordinaryH0_pointClass_augmentation]
-- E: C264 component functor object
example : ¬ (ordinaryH0_componentFunctor.obj (TopCat.of Unit) = ZerothHomotopy Unit) := by
  simp [ordinaryH0_componentFunctor]
-- F: C264 AddCommGrp coefficients
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Unit) (AddCommGrp.of ℤ) ()) (TopCat.singularHomology₀ε (TopCat.of Unit) (AddCommGrp.of ℤ)) ≠ CategoryTheory.CategoryStruct.id (AddCommGrp.of ℤ) := by
  simp [ordinaryH0_pointClass_augmentation]
