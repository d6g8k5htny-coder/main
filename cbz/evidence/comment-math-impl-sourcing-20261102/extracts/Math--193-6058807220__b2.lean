import BirthRepresentatives
import Mathlib.Order.ConditionallyCompleteLattice.Basic

noncomputable section
open CategoryTheory OrderDual OrdinaryH0Elder GlobalComponentMaps
universe u
namespace ElementaryHistory.ClosedComponentCertificate
variable {n : ℕ} {H : History.{u} n} {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f)

private theorem joinedIn_iff_component_eq {h : ℝ} {x y : X}
    (hx : h ≤ f x) (hy : h ≤ f y) :
    JoinedIn {z : X | h ≤ f z} x y ↔
      ZerothHomotopy.mk (show superlevel f h from ⟨x, hx⟩) =
        ZerothHomotopy.mk (show superlevel f h from ⟨y, hy⟩) := by
  constructor
  · intro hp
    exact ZerothHomotopy.sound hp.joined_subtype.somePath
  · intro he
    have hj : Joined (show superlevel f h from ⟨x, hx⟩)
        (show superlevel f h from ⟨y, hy⟩) := Quotient.exact he
    exact (joinedIn_iff_joined (F := {z : X | h ≤ f z}) hx hy).2 hj

def olderPathAt (j : H.BirthTag) (h : ℝ) : Prop :=
  ∃ k : H.BirthTag, k < j ∧ JoinedIn {x : X | h ≤ f x}
    (C.birthRepresentative j).1 (C.birthRepresentative k).1

theorem olderPathAt_iff (j : H.BirthTag) (h : ℝ) :
    C.olderPathAt j h ↔ H.OlderAt (C.stage.obj (OrderDual.toDual h)) j := by
  by_cases hh : h ≤ C.criticalHeight j.1
  · have hjb := (C.born_level_iff j (OrderDual.toDual h)).2 hh
    have hjm := hh.trans (C.birthRepresentative j).2
    constructor
    · rintro ⟨k, hk, hp⟩
      have hkj : k.1 ≤ j.1 := le_of_lt (show k.1 < j.1 from hk)
      have hhk := hh.trans (C.ordered.antitone hkj)
      have hkb := (C.born_level_iff k (OrderDual.toDual h)).2 hhk
      have hkm := hhk.trans (C.birthRepresentative k).2
      have he := (joinedIn_iff_component_eq hjm hkm).1 hp
      have ho : H.origin (C.stage.obj (OrderDual.toDual h)) ⟨j, hjb⟩ =
          H.origin (C.stage.obj (OrderDual.toDual h)) ⟨k, hkb⟩ := by
        calc
          _ = C.chart (OrderDual.toDual h)
              (ZerothHomotopy.mk (show superlevel f h from
                ⟨(C.birthRepresentative j).1, hjm⟩)) := (C.representative_chart_below j h hh).symm
          _ = C.chart (OrderDual.toDual h)
              (ZerothHomotopy.mk (show superlevel f h from
                ⟨(C.birthRepresentative k).1, hkm⟩)) :=
            congrArg (C.chart (OrderDual.toDual h)) he
          _ = _ := C.representative_chart_below k h hhk
      refine ⟨hjb, k, hk, ?_⟩
      simpa only [History.location, dite_eq_left hjb, dite_eq_left hkb] using
        congrArg Option.some ho.symm
    · rintro ⟨_, k, hk, hl⟩
      have hkj : k.1 ≤ j.1 := le_of_lt (show k.1 < j.1 from hk)
      have hhk := hh.trans (C.ordered.antitone hkj)
      have hkb := (C.born_level_iff k (OrderDual.toDual h)).2 hhk
      have hkm := hhk.trans (C.birthRepresentative k).2
      have ho : H.origin (C.stage.obj (OrderDual.toDual h)) ⟨k, hkb⟩ =
          H.origin (C.stage.obj (OrderDual.toDual h)) ⟨j, hjb⟩ :=
        Option.some.inj (by
          simpa only [History.location, dite_eq_left hjb, dite_eq_left hkb] using hl)
      refine ⟨k, hk, (joinedIn_iff_component_eq hjm hkm).2 ?_⟩
      apply (C.chart (OrderDual.toDual h)).injective
      exact (C.representative_chart_below j h hh).trans
        (ho.symm.trans (C.representative_chart_below k h hhk).symm)
  · constructor
    · rintro ⟨k, hk, hp⟩
      have hm := hp.source_mem
      change h ≤ f (C.birthRepresentative j).1 at hm
      rw [C.birthRepresentative_height] at hm
      exact False.elim (hh hm)
    · rintro ⟨hb, _⟩
      exact False.elim (hh ((C.born_level_iff j (OrderDual.toDual h)).1 hb))


theorem olderPathAt_threshold {j p : H.BirthTag} (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧ ∀ h : ℝ,
      C.olderPathAt j h ↔ h ≤ C.criticalHeight d := by
  obtain ⟨d, hd, hs⟩ := C.finite_support hp
  have hb : H.Active (C.stage.obj (OrderDual.toDual (C.criticalHeight j.1))) j := by
    rw [C.stage_at_critical]
    exact H.birth_active j
  have hgap : C.criticalHeight d < C.criticalHeight j.1 :=
    ((hs (OrderDual.toDual (C.criticalHeight j.1))).1 hb).1
  refine ⟨d, hd, ?_⟩
  intro h
  rw [C.olderPathAt_iff]
  by_cases hh : h ≤ C.criticalHeight j.1
  · have hjb := (C.born_level_iff j (OrderDual.toDual h)).2 hh
    constructor
    · intro ho
      by_contra hn
      have hlt : C.criticalHeight d < h := lt_of_not_ge hn
      have ha := (hs (OrderDual.toDual h)).2 ⟨hlt, hh⟩
      exact ((H.active_iff_no_older _ j).1 ha).2 ho
    · intro hle
      by_contra hn
      have ha := (H.active_iff_no_older _ j).2 ⟨hjb, hn⟩
      exact (not_lt_of_ge hle) ((hs (OrderDual.toDual h)).1 ha).1
  · constructor
    · rintro ⟨hjb, _⟩
      exact False.elim (hh ((C.born_level_iff j (OrderDual.toDual h)).1 hjb))
    · intro hle
      exact False.elim (hh (hle.trans hgap.le))


/-- Used with a proved finite parent; no essential-bar death value is asserted.
Supremum of actual path bottleneck thresholds to every chronologically older physical birth representative. -/
def physicalOlderMaximin (j : H.BirthTag) : ℝ := sSup {h : ℝ | C.olderPathAt j h}

theorem physicalOlderMaximin_eq_death {j p : H.BirthTag} (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧ C.physicalOlderMaximin j = C.criticalHeight d := by
  obtain ⟨d, hd, ht⟩ := C.olderPathAt_threshold hp
  refine ⟨d, hd, ?_⟩
  change sSup {h : ℝ | C.olderPathAt j h} = C.criticalHeight d
  apply le_antisymm
  · apply csSup_le ⟨C.criticalHeight d, (ht (C.criticalHeight d)).2 le_rfl⟩
    intro h hh
    exact (ht h).1 hh
  · apply le_csSup
    · exact ⟨C.criticalHeight d, fun h hh => (ht h).1 hh⟩
    · exact (ht (C.criticalHeight d)).2 le_rfl

end ElementaryHistory.ClosedComponentCertificate
