import HigherPointMaximin

noncomputable section
open CategoryTheory OrderDual OrdinaryH0Elder GlobalComponentMaps
namespace PhysicalValidation
universe u
variable {n : ℕ} {H : ElementaryHistory.History.{u} n}
  {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ElementaryHistory.ClosedComponentCertificate H f)

-- The physical birth point cannot connect to a strictly higher endpoint at birth.
example (j : H.BirthTag) : ¬ C.higherPointPathAt j (C.criticalHeight j.1) := by
  intro hp
  have ha : H.Active (C.stage.obj (OrderDual.toDual (C.criticalHeight j.1))) j := by
    rw [C.stage_at_critical]
    exact H.birth_active j
  exact ((H.active_iff_no_older _ j).1 ha).2
    ((C.olderPathAt_iff _ _).1 ((C.higherPointPathAt_iff _ _).1 hp))

-- Replacing strictly higher by at least as high admits the constant self-path.
example (j : H.BirthTag) :
    ∃ z : X, f (C.birthRepresentative j).1 ≤ f z ∧
      JoinedIn {x : X | C.criticalHeight j.1 ≤ f x} (C.birthRepresentative j).1 z := by
  refine ⟨(C.birthRepresentative j).1, le_rfl, ?_⟩
  exact JoinedIn.refl (C.birthRepresentative j).2

-- A threshold strictly above its actual function value excludes all paths.
example (j : H.BirthTag) (h : ℝ) (hh : C.criticalHeight j.1 < h) :
    ¬ C.higherPointPathAt j h := by
  rintro ⟨z, hz, hp⟩
  have hm := hp.source_mem
  change h ≤ f (C.birthRepresentative j).1 at hm
  rw [C.birthRepresentative_height] at hm
  exact (not_le_of_gt hh) hm

-- The death level has an actual path to a higher endpoint: closed bottleneck.
example {j p : H.BirthTag} (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧ C.higherPointPathAt j (C.criticalHeight d) := by
  obtain ⟨d, hd, ht⟩ := C.olderPathAt_threshold hp
  exact ⟨d, hd, (C.higherPointPathAt_iff _ _).2 ((ht _).2 le_rfl)⟩

-- Above death there is no such path, even though the interval birth is closed.
example {j p : H.BirthTag} (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧
      ∀ h : ℝ, C.criticalHeight d < h → ¬ C.higherPointPathAt j h := by
  obtain ⟨d, hd, ht⟩ := C.olderPathAt_threshold hp
  refine ⟨d, hd, ?_⟩
  intro h hh he
  exact (not_le_of_gt hh) ((ht _).1 ((C.higherPointPathAt_iff _ _).1 he))

-- Essential roots have no fabricated finite older-path death at any level.
example {j : H.BirthTag} (hp : H.parent j = none) (h : ℝ) :
    ¬ C.higherPointPathAt j h := by
  intro he
  have ho := (C.olderPathAt_iff _ _).1 ((C.higherPointPathAt_iff _ _).1 he)
  have ha := H.essential_active hp _ ho.1
  exact ((H.active_iff_no_older _ j).1 ha).2 ho

-- Dead born labels still chart actual points; they are not removed from BornAt.
example (j : H.BirthTag) (h : ℝ) (hh : h ≤ C.criticalHeight j.1)
    (hd : H.death j ≤ (C.stage.obj (OrderDual.toDual h)).val) :
    C.chart (OrderDual.toDual h)
      (ZerothHomotopy.mk (show superlevel f h from
        ⟨(C.birthRepresentative j).1, hh.trans (C.birthRepresentative j).2⟩)) =
      H.origin (C.stage.obj (OrderDual.toDual h))
        ⟨j, (C.born_level_iff j (OrderDual.toDual h)).2 hh⟩ ∧
      ¬ H.Active (C.stage.obj (OrderDual.toDual h)) j := by
  refine ⟨C.representative_chart_below j h hh, ?_⟩
  intro ha
  exact (not_lt_of_ge hd) ha.2

end PhysicalValidation

#print axioms ElementaryHistory.ClosedComponentCertificate.stage_at_critical
#print axioms ElementaryHistory.ClosedComponentCertificate.birthPoint_not_range
#print axioms ElementaryHistory.ClosedComponentCertificate.representative_height
#print axioms ElementaryHistory.ClosedComponentCertificate.birthRepresentative
#print axioms ElementaryHistory.ClosedComponentCertificate.birthRepresentative_chart
#print axioms ElementaryHistory.ClosedComponentCertificate.birthRepresentative_height
#print axioms ElementaryHistory.ClosedComponentCertificate.representative_chart_below
#print axioms ElementaryHistory.ClosedComponentCertificate.olderPathAt_iff
#print axioms ElementaryHistory.ClosedComponentCertificate.olderPathAt_threshold
#print axioms ElementaryHistory.ClosedComponentCertificate.physicalOlderMaximin_eq_death
#print axioms ElementaryHistory.ClosedComponentCertificate.higherPointPathAt_iff
#print axioms ElementaryHistory.ClosedComponentCertificate.allHigherPointMaximin
#print axioms ElementaryHistory.ClosedComponentCertificate.allHigherPointMaximin_eq_death
