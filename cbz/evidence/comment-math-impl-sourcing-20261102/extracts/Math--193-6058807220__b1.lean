import BirthPhysical
import Mathlib.Order.ConditionallyCompleteLattice.Basic

noncomputable section
open CategoryTheory OrderDual OrdinaryH0Elder GlobalComponentMaps
universe u
namespace ElementaryHistory.ClosedComponentCertificate
variable {n : ℕ} {H : History.{u} n} {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f)

private def birthChartComponent (j : H.BirthTag) :
    H.Component (C.stage.obj (OrderDual.toDual (C.criticalHeight j.1))) :=
  H.diagram.map (eqToHom (C.stage_at_critical j.1).symm) (H.birthPoint j)

/-- Chosen actual point in the genuine critical closed superlevel. No height equality is input. -/
def birthRepresentative (j : H.BirthTag) : superlevel f (C.criticalHeight j.1) :=
  Classical.choose (ZerothHomotopy.mk_surjective
    ((C.chart (OrderDual.toDual (C.criticalHeight j.1))).symm (C.birthChartComponent j)))

private theorem representative_chart_eq (j : H.BirthTag) :
    C.chart (OrderDual.toDual (C.criticalHeight j.1))
      (ZerothHomotopy.mk (C.birthRepresentative j)) = C.birthChartComponent j := by
  have hx : ZerothHomotopy.mk (C.birthRepresentative j) =
      (C.chart (OrderDual.toDual (C.criticalHeight j.1))).symm (C.birthChartComponent j) :=
    Classical.choose_spec (ZerothHomotopy.mk_surjective
      ((C.chart (OrderDual.toDual (C.criticalHeight j.1))).symm (C.birthChartComponent j)))
  exact (congrArg (C.chart (OrderDual.toDual (C.criticalHeight j.1))) hx).trans
    ((C.chart (OrderDual.toDual (C.criticalHeight j.1))).apply_symm_apply (C.birthChartComponent j))

private theorem component_cast_heq {t s : Fin (n + 1)} (hs : t = s)
    (x : H.Component t) : HEq (H.diagram.map (eqToHom hs) x) x := by
  cases hs
  simpa only [eqToHom_refl, H.diagram.map_id_apply] using (HEq.refl x)

theorem birthRepresentative_chart (j : H.BirthTag) :
    HEq (C.chart (OrderDual.toDual (C.criticalHeight j.1))
      (ZerothHomotopy.mk (C.birthRepresentative j))) (H.birthPoint j) := by
  exact (heq_of_eq (C.representative_chart_eq j)).trans
    (component_cast_heq (C.stage_at_critical j.1).symm (H.birthPoint j))


theorem birthRepresentative_height (j : H.BirthTag) :
    f (C.birthRepresentative j).1 = C.criticalHeight j.1 := by
  exact C.representative_height j (C.birthRepresentative j) (C.birthRepresentative_chart j)


theorem representative_chart_below (j : H.BirthTag) (h : ℝ)
    (hh : h ≤ C.criticalHeight j.1) :
    C.chart (OrderDual.toDual h)
      (ZerothHomotopy.mk (show superlevel f h from
        ⟨(C.birthRepresentative j).1, hh.trans (C.birthRepresentative j).2⟩)) =
      H.origin (C.stage.obj (OrderDual.toDual h))
        ⟨j, (C.born_level_iff j (OrderDual.toDual h)).2 hh⟩ := by
  let b := OrderDual.toDual (C.criticalHeight j.1)
  let a : b ⟶ OrderDual.toDual h := homOfLE hh
  have hc := C.commute a (ZerothHomotopy.mk (C.birthRepresentative j))
  change C.chart (OrderDual.toDual h)
    (ZerothHomotopy.mk (show superlevel f h from
      ⟨(C.birthRepresentative j).1, hh.trans (C.birthRepresentative j).2⟩)) =
    H.diagram.map (C.stage.map a) (C.chart b (ZerothHomotopy.mk (C.birthRepresentative j))) at hc
  rw [C.representative_chart_eq] at hc
  refine hc.trans ?_
  change H.diagram.map (C.stage.map a)
    (H.diagram.map (eqToHom (C.stage_at_critical j.1).symm) (H.birthPoint j)) =
      H.diagram.map (homOfLE ((C.born_level_iff j (OrderDual.toDual h)).2 hh)) (H.birthPoint j)
  have hm : eqToHom (C.stage_at_critical j.1).symm ≫ C.stage.map a =
      homOfLE ((C.born_level_iff j (OrderDual.toDual h)).2 hh) := Subsingleton.elim _ _
  exact (H.diagram.map_comp_apply (eqToHom (C.stage_at_critical j.1).symm)
    (C.stage.map a) (H.birthPoint j)).symm.trans
    (congrArg (fun z => H.diagram.map z (H.birthPoint j)) hm)


end ElementaryHistory.ClosedComponentCertificate
