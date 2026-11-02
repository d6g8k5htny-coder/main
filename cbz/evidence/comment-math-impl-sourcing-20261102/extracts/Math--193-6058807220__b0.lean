import ClosedRealization

noncomputable section
open CategoryTheory OrderDual OrdinaryH0Elder GlobalComponentMaps
universe u
namespace ElementaryHistory
namespace ClosedComponentCertificate
variable {n : ℕ} {H : History.{u} n} {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f)

theorem stage_at_critical (i : Fin n) :
    C.stage.obj (OrderDual.toDual (C.criticalHeight i)) = i.succ := by
  classical
  apply Fin.ext
  rw [C.stage_count]
  change (Finset.univ.filter (fun k => C.criticalHeight i ≤ C.criticalHeight k)).card = i.val + 1
  have hs : Finset.univ.filter (fun k => C.criticalHeight i ≤ C.criticalHeight k) =
      Finset.Iic i := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic]
    exact C.ordered.le_iff_ge
  rw [hs, Fin.card_Iic]

private theorem event_birthPoint_new {A B : Type u} {g : A → B}
    (e : Event g) (hb : e.isBirth) : e.birthPoint hb ∉ Set.range g := by
  cases e with
  | birth fresh hi hn hc => exact hn
  | merge a b hab hs hf => exact False.elim hb
  | neutral hb' => exact False.elim hb

theorem birthPoint_not_range (j : H.BirthTag) :
    H.birthPoint j ∉ Set.range (H.step j.1) := by
  exact event_birthPoint_new (H.event j.1) j.2

private theorem map_eqToHom_heq {t s : Fin (n + 1)} (h : t = s)
    (x : H.Component t) : HEq (H.diagram.map (eqToHom h) x) x := by
  cases h
  simpa only [eqToHom_refl, H.diagram.map_id_apply] using (HEq.refl x)

theorem representative_height (j : H.BirthTag)
    (x : superlevel f (C.criticalHeight j.1))
    (hx : HEq (C.chart (OrderDual.toDual (C.criticalHeight j.1)) (ZerothHomotopy.mk x))
      (H.birthPoint j)) :
    f x.1 = C.criticalHeight j.1 := by
  apply le_antisymm _ x.2
  by_contra hnot
  have hgt : C.criticalHeight j.1 < f x.1 := lt_of_not_ge hnot
  let b := OrderDual.toDual (C.criticalHeight j.1)
  let d := OrderDual.toDual (f x.1)
  let a : d ⟶ b := homOfLE (show C.criticalHeight j.1 ≤ f x.1 from hgt.le)
  let xh : superlevel f (f x.1) := ⟨x.1, le_rfl⟩
  have hs : C.stage.obj b = H.bornStage j := C.stage_at_critical j.1
  have hpre : C.stage.obj d ≤ j.1.castSucc := by
    have hn : ¬ H.bornStage j ≤ C.stage.obj d := by
      rw [C.born_level_iff]
      exact not_le_of_gt hgt
    change ¬ j.1.val + 1 ≤ (C.stage.obj d).val at hn
    change (C.stage.obj d).val ≤ j.1.val
    omega
  have hchart := C.commute a (ZerothHomotopy.mk xh)
  change C.chart b (ZerothHomotopy.mk x) =
    H.diagram.map (C.stage.map a) (C.chart d (ZerothHomotopy.mk xh)) at hchart
  have hw : H.diagram.map (eqToHom hs) (C.chart b (ZerothHomotopy.mk x)) =
      H.birthPoint j :=
    eq_of_heq ((map_eqToHom_heq hs (C.chart b (ZerothHomotopy.mk x))).trans hx)
  let w := H.diagram.map (homOfLE hpre) (C.chart d (ZerothHomotopy.mk xh))
  have hm : homOfLE hpre ≫
      homOfLE (show j.1.castSucc ≤ j.1.succ by simp [Fin.le_def]) =
      C.stage.map a ≫ eqToHom hs := Subsingleton.elim _ _
  have he : H.step j.1 w = H.birthPoint j := by
    change H.diagram.map (homOfLE (show j.1.castSucc ≤ j.1.succ by simp [Fin.le_def]))
      (H.diagram.map (homOfLE hpre) (C.chart d (ZerothHomotopy.mk xh))) = _
    rw [← H.diagram.map_comp_apply, hm]
    exact (H.diagram.map_comp_apply (C.stage.map a) (eqToHom hs)
      (C.chart d (ZerothHomotopy.mk xh))).trans
      ((congrArg (H.diagram.map (eqToHom hs)) hchart).symm.trans hw)
  exact (birthPoint_not_range j) ⟨w, he⟩
end ClosedComponentCertificate
end ElementaryHistory
