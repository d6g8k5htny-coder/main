import ElderInterval
open CategoryTheory TorusFiltration
#check elderInterval
#check elderSection
#check elderRetraction
#check elderSection_retraction
#check elderDirectSummand
#check elder_control_birth
#check elder_control_death
#check elder_control_ordinary_point_survives

/- These pin actual interval endpoints and map direction, not source spelling. -/
example : elderActive 1 0 (-1) := elder_control_birth (by norm_num)
example : ¬ elderActive 1 0 0 := by
  simpa only [neg_zero] using elder_control_death 1 0
example : ¬ elderActive 1 0 (-2) := by norm_num [elderActive]
example : elderActive 1 0 (-(1 / 2)) := by norm_num [elderActive]

example (x : elderFiber ℚ 1 0 (-2)) : x = 0 :=
  elderFiber_eq_zero ℚ (by norm_num [elderActive]) x

example (x : elderFiber ℚ 1 0 (-1)) :
    (elderFiberMap ℚ 1 0 (show (-1 : ℝ) ≤ -(1 / 2) by norm_num) x).val = x.val := by
  simp [elderFiberMap]

example (x : elderFiber ℚ 1 0 (-(1 / 2))) :
    elderFiberMap ℚ 1 0 (show (-(1 / 2) : ℝ) ≤ 0 by norm_num) x = 0 := by
  apply Subtype.ext
  simp [elderFiberMap]

/- The actual singular-H0 split also elaborates in characteristic two. -/
example {X : Type} [TopologicalSpace X] {g : X → ℝ} {M P : X}
    {m d : ℝ} {D : Set X} (T : ElderTrap g M P m d D) :
    elderSection (k := ZMod 2) T ≫ elderRetraction T =
      𝟙 (elderInterval (ZMod 2) m d) := elderSection_retraction T

example {X : Type} [TopologicalSpace X] {g : X → ℝ} {M P : X}
    {m d : ℝ} {D : Set X} (T : ElderTrap g M P m d D) :
    Nonempty ((superlevel_h0Filtration g (ModuleCat.of ℚ ℚ)) ≅
      elderInterval ℚ m d ⊞ Limits.kernel (elderRetraction (k := ℚ) T)) :=
  ⟨elderDirectSummand T⟩

/- Complete explicit declaration audit plus generated reassociation. -/

#print axioms TorusFiltration.elderActive
#print axioms TorusFiltration.elderFiber
#print axioms TorusFiltration.elderFiber_eq_zero
#print axioms TorusFiltration.elderFiberMap
#print axioms TorusFiltration.elderFiberMap_val
#print axioms TorusFiltration.elderInterval
#print axioms TorusFiltration.elderFiberLift
#print axioms TorusFiltration.elderFiberLift_val
#print axioms TorusFiltration.elder_control_birth
#print axioms TorusFiltration.elder_control_death
#print axioms TorusFiltration.ElderTrap
#print axioms TorusFiltration.ElderTrap.maximum_mem
#print axioms TorusFiltration.ElderTrap.older_mem
#print axioms TorusFiltration.elder_clopen_path_iff
#print axioms TorusFiltration.ElderTrap.componentInside
#print axioms TorusFiltration.ElderTrap.componentInside_mk
#print axioms TorusFiltration.elderDifference
#print axioms TorusFiltration.elderDifference_map
#print axioms TorusFiltration.elderDifference_map_dead
#print axioms TorusFiltration.elderProjection
#print axioms TorusFiltration.elderProjection_point
#print axioms TorusFiltration.elderDifference_projection
#print axioms TorusFiltration.elderSectionApp
#print axioms TorusFiltration.elderRetractionApp
#print axioms TorusFiltration.elderSectionApp_naturality
#print axioms TorusFiltration.elderSection
#print axioms TorusFiltration.elderProjection_naturality_active
#print axioms TorusFiltration.elderProjection_prebirth
#print axioms TorusFiltration.elderRetractionApp_naturality
#print axioms TorusFiltration.elderRetraction
#print axioms TorusFiltration.elderSection_retraction
#print axioms TorusFiltration.elderKernelProjection
#print axioms TorusFiltration.elderDirectSummand
#print axioms TorusFiltration.elder_control_ordinary_point_survives
#print axioms TorusFiltration.elderSection_retraction_assoc
