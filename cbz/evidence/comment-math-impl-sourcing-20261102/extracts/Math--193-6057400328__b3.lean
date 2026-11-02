import EventPersistence
import GlobalFiniteComponents
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.CategoryTheory.Whiskering

noncomputable section
open CategoryTheory Module OrderDual OrdinaryH0Elder GlobalMergeSemantics GlobalComponentMaps
universe u
namespace ElementaryHistory

/-- Primitive geometric realization data only: ordered critical levels, actual component
charts, and commutation with EVERY genuine closed-superlevel inclusion. The construction
of this certificate from Morse geometry is a separate obligation. -/
structure ClosedComponentCertificate {n : ℕ} (H : History.{u} n)
    {X : Type u} [TopologicalSpace X] (f : X → ℝ) where
  stage : ℝᵒᵈ ⥤ Fin (n + 1)
  criticalHeight : Fin n → ℝ
  ordered : StrictAnti criticalHeight
  stage_count : ∀ h, (stage.obj h).val =
    (Finset.univ.filter (fun i => OrderDual.ofDual h ≤ criticalHeight i)).card
  chart : ∀ h, ZerothHomotopy (superlevel f (OrderDual.ofDual h)) ≃ H.Component (stage.obj h)
  commute : ∀ {h s} (a : h ⟶ s) (c : ZerothHomotopy (superlevel f (OrderDual.ofDual h))),
    chart s (componentMap ((superlevelFunctor f).map a) c) =
      H.diagram.map (stage.map a) (chart h c)

namespace ClosedComponentCertificate
variable {n : ℕ} {H : History.{u} n} {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f) (K : Type u) [Field K]

def intervalDiagram : ℝᵒᵈ ⥤ ModuleCat K := C.stage ⋙ H.intervalDiagram K

private theorem stage_level_iff (i : Fin n) (h : ℝᵒᵈ) :
    i.succ ≤ C.stage.obj h ↔ OrderDual.ofDual h ≤ C.criticalHeight i := by
  classical
  let S := Finset.univ.filter (fun k => OrderDual.ofDual h ≤ C.criticalHeight k)
  have hc : (C.stage.obj h).val = S.card := C.stage_count h
  constructor
  · intro hs
    by_contra hn
    have hlt : C.criticalHeight i < OrderDual.ofDual h := lt_of_not_ge hn
    have hsub : S ⊆ Finset.Iio i := by
      intro k hk
      have hkh : OrderDual.ofDual h ≤ C.criticalHeight k := (Finset.mem_filter.mp hk).2
      apply Finset.mem_Iio.mpr
      by_contra hki
      have hik : i ≤ k := le_of_not_gt hki
      have hfk := C.ordered.antitone hik
      exact (not_lt_of_ge (hkh.trans hfk)) hlt
    have hcard := Finset.card_le_card hsub
    rw [Fin.card_Iio] at hcard
    change i.val + 1 ≤ (C.stage.obj h).val at hs
    omega
  · intro hh
    have hsub : Finset.Iic i ⊆ S := by
      intro k hk
      have hki := Finset.mem_Iic.mp hk
      simp only [S, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hh.trans (C.ordered.antitone hki)
    have hcard := Finset.card_le_card hsub
    rw [Fin.card_Iic] at hcard
    change i.val + 1 ≤ (C.stage.obj h).val
    omega

theorem born_level_iff (j : H.BirthTag) (h : ℝᵒᵈ) :
    H.bornStage j ≤ C.stage.obj h ↔ OrderDual.ofDual h ≤ C.criticalHeight j.1 := by
  exact C.stage_level_iff j.1 h

/-- Essential roots have a closed birth endpoint and no finite death. -/
theorem essential_support {j : H.BirthTag} (hp : H.parent j = none) (h : ℝᵒᵈ) :
    H.Active (C.stage.obj h) j ↔ OrderDual.ofDual h ≤ C.criticalHeight j.1 := by
  constructor
  · intro ha
    exact (C.born_level_iff j h).1 ha.1
  · intro hb
    exact H.essential_active hp _ ((C.born_level_iff j h).2 hb)

/-- A finite derived parent supplies the OPEN death endpoint and CLOSED birth endpoint. -/
theorem finite_support {j p : H.BirthTag} (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧ ∀ h : ℝᵒᵈ,
      H.Active (C.stage.obj h) j ↔
        C.criticalHeight d < OrderDual.ofDual h ∧ OrderDual.ofDual h ≤ C.criticalHeight j.1 := by
  have hb := (H.birth_active j).2
  have hbpos : 0 < (H.bornStage j).val := by change 0 < j.1.val + 1; omega
  obtain ⟨t, hdt, _, _⟩ := H.parent_collision hp
  have hdn : H.death j ≤ n := by rw [hdt]; exact Fin.le_last t
  let d : Fin n := ⟨H.death j - 1, by omega⟩
  have hd : H.death j = d.succ.val := by dsimp [d]; omega
  refine ⟨d, hd, ?_⟩
  intro h
  constructor
  · intro ha
    have hn : ¬ d.succ ≤ C.stage.obj h := by
      intro hi
      have hv : H.death j ≤ (C.stage.obj h).val := by rw [hd]; exact hi
      exact (not_le_of_gt ha.2) hv
    refine ⟨?_, (C.born_level_iff j h).1 ha.1⟩
    exact lt_of_not_ge (fun hh => hn ((C.stage_level_iff d h).2 hh))
  · rintro ⟨hs, hb⟩
    refine ⟨(C.born_level_iff j h).2 hb, ?_⟩
    have hn : ¬ d.succ ≤ C.stage.obj h := fun hi =>
      (not_le_of_gt hs) ((C.stage_level_iff d h).1 hi)
    rw [hd]
    exact lt_of_not_ge hn

/-- Naturality uses the actual zeroth singular homology functor and actual inclusions. -/
def componentIso : closedH0Persistence K f ≅ C.stage ⋙ H.componentDiagram K := by
  refine NatIso.ofComponents (fun h =>
    (componentChart K (superlevel f (OrderDual.ofDual h)) (C.chart h)).toModuleIso) ?_
  intro h s a
  apply ModuleCat.hom_ext
  apply LinearMap.ext
  intro v
  change componentChart K (superlevel f (OrderDual.ofDual s)) (C.chart s)
    ((h0Functor K).map ((superlevelFunctor f).map a) v) =
    Finsupp.mapDomain (H.diagram.map (C.stage.map a))
      (componentChart K (superlevel f (OrderDual.ofDual h)) (C.chart h) v)
  exact chart_map K ((superlevelFunctor f).map a) (C.chart h) (C.chart s)
    (H.diagram.map (C.stage.map a)) (C.commute a) v

/-- Conditional global decomposition of GENUINE ordinary closed H0.
No parent, basis or interval data are fields of C or H. -/
def closedIntervalIso : C.intervalDiagram K ≅ closedH0Persistence K f := by
  exact (Functor.isoWhiskerLeft C.stage (H.finiteIntervalIso K)) ≪≫ (C.componentIso K).symm

end ClosedComponentCertificate
end ElementaryHistory

