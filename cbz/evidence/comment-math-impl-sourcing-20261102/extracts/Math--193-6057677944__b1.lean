import ClosedRealization
import EventDeaths
import Mathlib.Algebra.Field.ZMod

noncomputable section
open CategoryTheory Module ElementaryHistory
local instance : Fact (Nat.Prime 2) := ⟨by decide⟩
namespace EventSemanticsTests
universe u
variable {n : ℕ} (H : History.{u} n) (K : Type u) [Field K]

-- A fresh tag is active at its actual birth and has a positive stage lifetime.
example (j : H.BirthTag) : (H.bornStage j).val < H.death j := (H.birth_active j).2

-- Its fixed parent survives every stage at which the child is active.
example {t : Fin (n + 1)} {j p : H.BirthTag} (hp : H.parent j = some p)
    (hj : H.Active t j) : H.bornStage p ≤ t ∧ t.val < H.death p := H.parent_active hp hj

-- Ordinary component POINT vectors survive a merge, even though a difference may vanish.
example {t s : Fin (n + 1)} (h : t ≤ s) (j : H.Alive t) :
    Finsupp.lmapDomain K K (H.diagram.map (homOfLE h))
      (Finsupp.single (H.aliveOrigin t j) (1 : K)) ≠ 0 := by
  rw [Finsupp.lmapDomain_apply, Finsupp.mapDomain_single]
  exact Finsupp.single_ne_zero.mpr one_ne_zero

-- Actual fixed generators vanish on every arrow crossing their first death.
example {t s : Fin (n + 1)} (h : t ≤ s) (j : H.Alive t) (hd : H.death j.1 ≤ s.val) :
    Finsupp.lmapDomain K K (H.diagram.map (homOfLE h)) (H.fixedGenerator K t j) = 0 := by
  have hn : ¬ H.Active s j.1 := fun ha => (not_le_of_gt ha.2) hd
  simpa only [dite_eq_right hn] using H.generator_transport K h j

-- Independent triangular coordinates, including empty and disconnected states.
example (t : Fin (n + 1)) : LinearIndependent K (H.fixedGenerator K t) := by
  have he : (H.stateBasis K t : H.Alive t → H.Component t →₀ K) = H.fixedGenerator K t := by
    funext j
    exact H.stateBasis_apply K t j
  rw [← he]
  exact (H.stateBasis K t).linearIndependent

example (t : Fin (n + 1)) : Submodule.span K (Set.range (H.fixedGenerator K t)) = ⊤ := by
  have he : (H.stateBasis K t : H.Alive t → H.Component t →₀ K) = H.fixedGenerator K t := by
    funext j
    exact H.stateBasis_apply K t j
  rw [← he]
  exact (H.stateBasis K t).span_eq

example : {j : H.BirthTag // H.parent j = none} ≃ H.Component (Fin.last n) := H.terminalRootsEquiv

example (j : H.BirthTag) (hp : H.parent j = none) : H.Active (Fin.last n) j :=
  (H.terminal_active_iff_root j).2 hp

example (H : History 0) : IsEmpty H.BirthTag := ⟨fun b => Fin.elim0 b.1⟩

-- Future birth tags have no interval coordinate before their actual birth.
example (j : H.BirthTag) (t : Fin (n + 1)) (ht : t < H.bornStage j) : ¬ H.Active t j := by
  intro ha
  exact (not_le_of_gt ht) ha.1

-- Omitting coverage/surjectivity is not an admissible neutral/merge event.
example : ¬ Function.Surjective (fun _ : Fin 1 => (0 : Fin 2)) := by decide

-- A bijective nonmerge transition has no first death.
example (i : Fin n) (hi : Function.Bijective (H.step i)) (j : H.BirthTag) :
    H.death j ≠ i.succ.val := H.injective_no_death i hi.1 j

-- Characteristic 2 uses the SAME interval maps; no division by 2 occurs.
example {n : ℕ} (H : History.{0} n) : H.intervalDiagram (ZMod 2) ≅ H.componentDiagram (ZMod 2) :=
  H.finiteIntervalIso (ZMod 2)

variable {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f)

-- CLOSED birth boundary for essential roots.
example (j : H.BirthTag) (hp : H.parent j = none) :
    H.Active (C.stage.obj (OrderDual.toDual (C.criticalHeight j.1))) j := by
  exact (C.essential_support hp _).2 le_rfl

-- An essential root is absent strictly above its birth.
example (j : H.BirthTag) (hp : H.parent j = none) (h : ℝ)
    (hh : C.criticalHeight j.1 < h) :
    ¬ H.Active (C.stage.obj (OrderDual.toDual h)) j := by
  intro ha
  exact (not_le_of_gt hh) ((C.essential_support hp _).1 ha)

-- OPEN death boundary for every finite derived interval.
example (j p : H.BirthTag) (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧
      ¬ H.Active (C.stage.obj (OrderDual.toDual (C.criticalHeight d))) j := by
  obtain ⟨d, hd, hs⟩ := C.finite_support hp
  refine ⟨d, hd, ?_⟩
  intro ha
  exact (lt_irrefl _) ((hs _).1 ha).1

-- A finite interval is already active AT its birth; the positive gap is derived.
example (j p : H.BirthTag) (hp : H.parent j = some p) :
    H.Active (C.stage.obj (OrderDual.toDual (C.criticalHeight j.1))) j := by
  obtain ⟨d, hd, hs⟩ := C.finite_support hp
  have hactive := (H.birth_active j).2
  have hi : j.1 < d := by
    change j.1.val < d.val
    change j.1.val + 1 < H.death j at hactive
    rw [hd] at hactive
    change j.1.val + 1 < d.val + 1 at hactive
    omega
  exact (hs _).2 ⟨C.ordered hi, le_rfl⟩

-- The output is a natural ISO of actual ordinary closed H0, not an assumed barcode.
example : C.intervalDiagram K ≅ OrdinaryH0Elder.closedH0Persistence K f := C.closedIntervalIso K

example {h s : ℝᵒᵈ} (a : h ⟶ s) :
    (C.intervalDiagram K).map a ≫ (C.closedIntervalIso K).hom.app s =
      (C.closedIntervalIso K).hom.app h ≫ (OrdinaryH0Elder.closedH0Persistence K f).map a :=
  (C.closedIntervalIso K).hom.naturality a

end EventSemanticsTests

-- Exact declaration audit from EventAxioms.lean follows.
#print axioms ElementaryHistory.History.origin_map
#print axioms ElementaryHistory.History.origin_surjective
#print axioms ElementaryHistory.History.collision_persistent
#print axioms ElementaryHistory.History.active_iff_no_older
#print axioms ElementaryHistory.History.parent_strict
#print axioms ElementaryHistory.History.parent_later_death
#print axioms ElementaryHistory.History.parent_active
#print axioms ElementaryHistory.History.parent_collision
#print axioms ElementaryHistory.History.birth_active
#print axioms ElementaryHistory.History.aliveOrigin_bijective
#print axioms ElementaryHistory.History.parent_points_coincide_after
#print axioms ElementaryHistory.History.essential_active
#print axioms ElementaryHistory.History.aliveParent_strict
#print axioms ElementaryHistory.History.stateBasis_apply
#print axioms ElementaryHistory.History.intervalMap_id
#print axioms ElementaryHistory.History.intervalMap_comp
#print axioms ElementaryHistory.History.generator_transport
#print axioms ElementaryHistory.History.coordinates_natural
#print axioms ElementaryHistory.History.finiteIntervalIso
#print axioms ElementaryHistory.History.terminal_active_iff_root
#print axioms ElementaryHistory.History.terminalRootsEquiv
#print axioms ElementaryHistory.ClosedComponentCertificate.born_level_iff
#print axioms ElementaryHistory.ClosedComponentCertificate.essential_support
#print axioms ElementaryHistory.ClosedComponentCertificate.finite_support
#print axioms ElementaryHistory.ClosedComponentCertificate.componentIso
#print axioms ElementaryHistory.ClosedComponentCertificate.closedIntervalIso
#print axioms ElementaryHistory.History.injective_no_death
#print axioms ElementaryHistory.History.merge_death_iff
#print axioms ElementaryHistory.History.merge_parent
#print axioms ElementaryHistory.History.merge_iff_unique_death

