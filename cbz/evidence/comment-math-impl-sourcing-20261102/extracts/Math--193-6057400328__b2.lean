import IntervalDecomposition
import Mathlib.CategoryTheory.NatIso

noncomputable section
open CategoryTheory Module
universe u v
namespace ElementaryHistory.History
variable {n : ℕ} (H : ElementaryHistory.History.{u} n)
variable (K : Type u) [Field K]

def intervalDiagram : Fin (n + 1) ⥤ ModuleCat K where
  obj t := ModuleCat.of K (H.Alive t →₀ K)
  map f := ModuleCat.ofHom (H.intervalMap K (leOfHom f))
  map_id t := by
    change ModuleCat.ofHom (H.intervalMap K (le_refl t)) = _
    rw [H.intervalMap_id K]
    rfl
  map_comp f g := by
    change ModuleCat.ofHom (H.intervalMap K ((leOfHom f).trans (leOfHom g))) = _
    rw [H.intervalMap_comp K (leOfHom f) (leOfHom g)]
    rfl

def componentDiagram : Fin (n + 1) ⥤ ModuleCat K where
  obj t := ModuleCat.of K (H.Component t →₀ K)
  map f := ModuleCat.ofHom (Finsupp.lmapDomain K K (H.diagram.map f))
  map_id t := by
    simp only [H.diagram.map_id]
    change ModuleCat.ofHom (Finsupp.lmapDomain K K (_root_.id : H.Component t → H.Component t)) = _
    rw [Finsupp.lmapDomain_id]
    rfl
  map_comp f g := by
    rw [H.diagram.map_comp]
    change ModuleCat.ofHom (Finsupp.lmapDomain K K
      (H.diagram.map g ∘ H.diagram.map f)) = _
    rw [Finsupp.lmapDomain_comp]
    rfl

/-- Actual all-arrow interval decomposition of the primitive history's component module. -/
def finiteIntervalIso : H.intervalDiagram K ≅ H.componentDiagram K := by
  refine NatIso.ofComponents (fun t => (H.toComponents K t).toModuleIso) ?_
  intro t s a
  apply ModuleCat.hom_ext
  apply LinearMap.ext
  intro v
  change H.toComponents K s (H.intervalMap K (leOfHom a) v) =
    Finsupp.lmapDomain K K (H.diagram.map a) (H.toComponents K t v)
  have he := (H.coordinates_natural K (leOfHom a) v).symm
  have ha : (homOfLE (leOfHom a) : t ⟶ s) = a := Subsingleton.elim _ _
  rw [ha] at he
  exact he

theorem terminal_active_iff_root (j : H.BirthTag) :
    H.Active (Fin.last n) j ↔ H.parent j = none := by
  constructor
  · intro ha
    by_contra hn
    obtain ⟨p, hp⟩ := Option.ne_none_iff_exists'.mp hn
    obtain ⟨d, hd, _, _⟩ := H.parent_collision hp
    have hdn : H.death j ≤ (Fin.last n).val := by rw [hd]; exact Fin.le_last d
    exact (not_le_of_gt ha.2) hdn
  · intro hp
    exact H.essential_active hp (Fin.last n) (Fin.le_last _)

/-- Terminal component count is derived, with no connectedness or one-root premise. -/
def terminalRootsEquiv : {j : H.BirthTag // H.parent j = none} ≃ H.Component (Fin.last n) := by
  exact (Equiv.subtypeEquivProp (by funext j; exact propext (H.terminal_active_iff_root j).symm)).trans
    (H.aliveEquiv (Fin.last n))

end ElementaryHistory.History

