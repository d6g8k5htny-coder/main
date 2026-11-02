import PhysicalMaximin

noncomputable section
open CategoryTheory OrderDual OrdinaryH0Elder GlobalComponentMaps
universe u
namespace ElementaryHistory.ClosedComponentCertificate
variable {n : ℕ} {H : History.{u} n} {X : Type u} [TopologicalSpace X] {f : X → ℝ}
variable (C : ClosedComponentCertificate H f)

/-- Actual paths to ALL endpoints higher than the chosen physical point. -/
def higherPointPathAt (j : H.BirthTag) (h : ℝ) : Prop :=
  ∃ z : X, f (C.birthRepresentative j).1 < f z ∧
    JoinedIn {x : X | h ≤ f x} (C.birthRepresentative j).1 z

theorem higherPointPathAt_iff (j : H.BirthTag) (h : ℝ) :
    C.higherPointPathAt j h ↔ C.olderPathAt j h := by
  constructor
  · rintro ⟨z, hz, hp⟩
    rw [C.birthRepresentative_height j] at hz
    let t := C.stage.obj (OrderDual.toDual (f z))
    obtain ⟨a, ha⟩ := H.origin_surjective t
      (C.chart (OrderDual.toDual (f z))
        (ZerothHomotopy.mk (show superlevel f (f z) from ⟨z, le_rfl⟩)))
    let k : H.BirthTag := a.1
    have hkb : H.bornStage k ≤ t := a.2
    have hheight : f z ≤ C.criticalHeight k.1 :=
      (C.born_level_iff k (OrderDual.toDual (f z))).1 hkb
    have hk : k < j := by
      by_contra hn
      have hle : j.1 ≤ k.1 := le_of_not_gt hn
      exact (not_lt_of_ge (hheight.trans (C.ordered.antitone hle))) hz
    have hkm : f z ≤ f (C.birthRepresentative k).1 :=
      hheight.trans (C.birthRepresentative k).2
    have hcomp :
        ZerothHomotopy.mk (show superlevel f (f z) from ⟨z, le_rfl⟩) =
        ZerothHomotopy.mk (show superlevel f (f z) from
          ⟨(C.birthRepresentative k).1, hkm⟩) := by
      apply (C.chart (OrderDual.toDual (f z))).injective
      calc
        _ = H.origin t a := ha.symm
        _ = H.origin t ⟨k, hkb⟩ := rfl
        _ = _ := (C.representative_chart_below k (f z) hheight).symm
    have hj : Joined (show superlevel f (f z) from ⟨z, le_rfl⟩)
        (show superlevel f (f z) from ⟨(C.birthRepresentative k).1, hkm⟩) :=
      Quotient.exact hcomp
    have hpath : JoinedIn {x : X | f z ≤ f x} z (C.birthRepresentative k).1 :=
      (joinedIn_iff_joined (F := {x : X | f z ≤ f x})
        (by change f z ≤ f z; exact le_rfl) hkm).2 hj
    have hlow : JoinedIn {x : X | h ≤ f x} z (C.birthRepresentative k).1 :=
      hpath.mono (by
        intro x hx
        change h ≤ f x
        change f z ≤ f x at hx
        have hhz : h ≤ f z := hp.target_mem
        exact hhz.trans hx)
    exact ⟨k, hk, hp.trans hlow⟩
  · rintro ⟨k, hk, hp⟩
    refine ⟨(C.birthRepresentative k).1, ?_, hp⟩
    rw [C.birthRepresentative_height j, C.birthRepresentative_height k]
    exact C.ordered (show k.1 < j.1 from hk)

/-- Global path bottleneck supremum against all strictly higher endpoints. -/
def allHigherPointMaximin (j : H.BirthTag) : ℝ :=
  sSup {h : ℝ | C.higherPointPathAt j h}

theorem allHigherPointMaximin_eq_death {j p : H.BirthTag}
    (hp : H.parent j = some p) :
    ∃ d : Fin n, H.death j = d.succ.val ∧
      C.allHigherPointMaximin j = C.criticalHeight d := by
  obtain ⟨d, hd, he⟩ := C.physicalOlderMaximin_eq_death hp
  refine ⟨d, hd, ?_⟩
  have hs : {h : ℝ | C.higherPointPathAt j h} = {h : ℝ | C.olderPathAt j h} := by
    ext h
    exact C.higherPointPathAt_iff j h
  change sSup {h : ℝ | C.higherPointPathAt j h} = C.criticalHeight d
  rw [hs]
  exact he

end ElementaryHistory.ClosedComponentCertificate
