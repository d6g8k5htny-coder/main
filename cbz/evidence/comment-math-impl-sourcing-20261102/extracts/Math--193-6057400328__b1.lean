import ActiveComponents
import StrictParentBasis
import Mathlib.LinearAlgebra.Finsupp.LinearCombination
import Mathlib.Algebra.Category.ModuleCat.Basic

noncomputable section
open CategoryTheory Module
universe u v
namespace ElementaryHistory.History
attribute [local instance] Classical.propDecidable
variable {n : ℕ} (H : ElementaryHistory.History.{u} n)
variable (K : Type v) [Field K]

def aliveParent (t : Fin (n + 1)) (j : H.Alive t) : Option (H.Alive t) :=
  Option.pmap (fun p hp => (⟨p, hp⟩ : H.Alive t)) (H.parent j.1)
    (fun _p hp => H.parent_active hp j.2)

theorem aliveParent_strict (t : Fin (n + 1)) : StrictParentBasis.Strict (H.aliveParent t) := by
  intro i j hij
  obtain ⟨p, ha, hp, he⟩ := Option.pmap_eq_some_iff.mp hij
  subst j
  exact H.parent_strict hp

private theorem aliveParent_none (t : Fin (n + 1)) (j : H.Alive t)
    (hp : H.parent j.1 = none) : H.aliveParent t j = none :=
  Option.pmap_none' hp

private theorem aliveParent_some (t : Fin (n + 1)) (j : H.Alive t)
    {p : H.BirthTag} (hp : H.parent j.1 = some p) :
    H.aliveParent t j = some ⟨p, H.parent_active hp j.2⟩ :=
  Option.pmap_some' hp

/-- Point differences use the FIRST-death parent, never a later current-fiber minimum. -/
def fixedGenerator (t : Fin (n + 1)) (j : H.Alive t) : H.Component t →₀ K :=
  Finsupp.mapDomain (H.aliveOrigin t) (StrictParentBasis.difference K (H.aliveParent t) j)

def stateBasis (t : Fin (n + 1)) : Basis (H.Alive t) K (H.Component t →₀ K) :=
  (StrictParentBasis.basis K (H.aliveParent t) (H.aliveParent_strict t)).map
    (Finsupp.mapDomain.linearEquiv K K (H.aliveEquiv t))



theorem stateBasis_apply (t : Fin (n + 1)) (j : H.Alive t) :
    H.stateBasis K t j = H.fixedGenerator K t j := by
  rw [stateBasis, Basis.map_apply, StrictParentBasis.basis_apply]
  rfl

def toComponents (t : Fin (n + 1)) : (H.Alive t →₀ K) ≃ₗ[K] (H.Component t →₀ K) :=
  (H.stateBasis K t).repr.symm

def intervalMap {t s : Fin (n + 1)} (_h : t ≤ s) :
    (H.Alive t →₀ K) →ₗ[K] (H.Alive s →₀ K) := by
  classical
  exact Finsupp.linearCombination K (fun j : H.Alive t =>
    if hj : H.Active s j.1 then Finsupp.single (⟨j.1, hj⟩ : H.Alive s) (1 : K) else 0)



theorem intervalMap_id (t : Fin (n + 1)) :
    H.intervalMap K (le_refl t) = LinearMap.id := by
  classical
  apply (Finsupp.basisSingleOne : Basis (H.Alive t) K (H.Alive t →₀ K)).ext
  intro j
  simp [intervalMap, j.2]

theorem intervalMap_comp {t s r : Fin (n + 1)} (h : t ≤ s) (h' : s ≤ r) :
    H.intervalMap K (h.trans h') = (H.intervalMap K h').comp (H.intervalMap K h) := by
  classical
  apply (Finsupp.basisSingleOne : Basis (H.Alive t) K (H.Alive t →₀ K)).ext
  intro j
  by_cases hs : H.Active s j.1
  · simp [intervalMap, hs]
  · have hr : ¬ H.Active r j.1 := by
      intro ha
      exact hs ⟨j.2.1.trans h, lt_of_le_of_lt h' ha.2⟩
    simp [intervalMap, hs, hr]

private theorem fixedGenerator_none (t : Fin (n + 1)) (j : H.Alive t)
    (hp : H.parent j.1 = none) :
    H.fixedGenerator K t j = Finsupp.single (H.aliveOrigin t j) (1 : K) := by
  classical
  simp [fixedGenerator, StrictParentBasis.difference, H.aliveParent_none t j hp]

private theorem fixedGenerator_some (t : Fin (n + 1)) (j : H.Alive t)
    {p : H.BirthTag} (hp : H.parent j.1 = some p) :
    H.fixedGenerator K t j = Finsupp.single (H.aliveOrigin t j) (1 : K) -
      Finsupp.single (H.origin t ⟨p, (H.parent_active hp j.2).1⟩) (1 : K) := by
  classical
  simp [fixedGenerator, StrictParentBasis.difference, H.aliveParent_some t j hp,
    Finsupp.mapDomain_sub, aliveOrigin]

theorem generator_transport {t s : Fin (n + 1)} (h : t ≤ s) (j : H.Alive t) :
    Finsupp.lmapDomain K K (H.diagram.map (homOfLE h)) (H.fixedGenerator K t j) =
      if hj : H.Active s j.1 then H.fixedGenerator K s ⟨j.1, hj⟩ else 0 := by
  classical
  have hbj : H.bornStage j.1 ≤ s := j.2.1.trans h
  cases hp : H.parent j.1 with
  | none =>
    have ha := H.essential_active hp s hbj
    rw [dite_eq_left ha, H.fixedGenerator_none K t j hp,
      H.fixedGenerator_none K s ⟨j.1, ha⟩ hp]
    simp only [Finsupp.lmapDomain_apply, Finsupp.mapDomain_single, aliveOrigin]
    congr 1
    exact H.origin_map h ⟨j.1, j.2.1⟩
  | some p =>
    have hpt := H.parent_active hp j.2
    have hps : H.bornStage p ≤ s := hpt.1.trans h
    rw [H.fixedGenerator_some K t j hp]
    simp only [map_sub, Finsupp.lmapDomain_apply, Finsupp.mapDomain_single, aliveOrigin]
    rw [H.origin_map h ⟨j.1, j.2.1⟩, H.origin_map h ⟨p, hpt.1⟩]
    by_cases ha : H.Active s j.1
    · rw [dite_eq_left ha, H.fixedGenerator_some K s ⟨j.1, ha⟩ hp]
      rfl
    · rw [dite_eq_right ha]
      have hdead : H.death j.1 ≤ s.val := by
        by_contra hn
        exact ha ⟨hbj, by omega⟩
      have he := H.parent_points_coincide_after hp s hdead
      simp only [location, dite_eq_left hps, dite_eq_left hbj, Option.some.injEq] at he
      rw [he]
      exact sub_self _

theorem coordinates_natural {t s : Fin (n + 1)} (h : t ≤ s) (v : H.Alive t →₀ K) :
    Finsupp.lmapDomain K K (H.diagram.map (homOfLE h)) (H.toComponents K t v) =
      H.toComponents K s (H.intervalMap K h v) := by
  classical
  have he : (Finsupp.lmapDomain K K (H.diagram.map (homOfLE h))).comp
      (H.toComponents K t).toLinearMap =
      (H.toComponents K s).toLinearMap.comp (H.intervalMap K h) := by
    apply (Finsupp.basisSingleOne : Basis (H.Alive t) K (H.Alive t →₀ K)).ext
    intro j
    change Finsupp.lmapDomain K K (H.diagram.map (homOfLE h))
      ((H.stateBasis K t).repr.symm (Finsupp.single j 1)) =
      (H.stateBasis K s).repr.symm (H.intervalMap K h (Finsupp.single j 1))
    rw [Basis.repr_symm_single_one, H.stateBasis_apply, H.generator_transport K h j]
    by_cases ha : H.Active s j.1
    · simp only [intervalMap, dite_eq_left ha, Finsupp.linearCombination_single, one_smul]
      rw [Basis.repr_symm_single_one, H.stateBasis_apply]
    · simp [intervalMap, ha]
  exact LinearMap.congr_fun he v

end ElementaryHistory.History

