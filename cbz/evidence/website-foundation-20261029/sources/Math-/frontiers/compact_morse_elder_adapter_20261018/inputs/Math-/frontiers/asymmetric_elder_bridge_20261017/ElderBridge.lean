import ElderClearance
import ElderInterval

/-!
Geometric input join for the actual ordinary-H0 interval construction.
Root01a11cbf, local author candidate, 17 October 2026.
No analytic fold, Morse uniqueness or torus theorem is assumed proved here.
-/

noncomputable section
open CategoryTheory Limits

namespace TorusFiltration
universe u

/-- Every active closed level includes continuously into the strict level
above d. Pulling back the proved strict separation gives closed-level separation. -/
theorem elderTrap_closedTrace_isClopen {X : Type u} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (d t : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d)) (ht : t < -d) :
    IsClopen {x : superlevel_set g t | x.val ∈ D} := by
  have hdt : d < -t := by linarith
  let i : superlevel_set g t → {x : X // d < g x} :=
    fun x => ⟨x.val, lt_of_lt_of_le hdt x.property⟩
  have hi : Continuous i := continuous_subtype_val.subtype_mk _
  exact htrace.preimage hi

/-- Actual geometric boundary, ceiling and closed path data inhabit the interval
trap. There is no H0 equation, interval split, clearance or death among inputs. -/
theorem elderTrap_of_boundary {X : Type u} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (m d : ℝ) (M P : X)
    (hD : IsClosed D) (hboundary : ∀ x ∈ frontier D, g x ≤ d)
    (hdm : d < m) (hMheight : g M = m) (hM : M ∈ D)
    (hheight : ∀ x ∈ D, g x ≤ m) (hP : m < g P)
    (γ : Path M P) (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    ElderTrap g M P m d D where
  death_lt_birth := hdm
  maximum_height := hMheight
  older_height := hP
  maximum_inside := hM
  older_outside := by
    intro hPD
    exact (not_lt_of_ge (hheight P hPD)) hP
  height_ceiling := hheight
  active_clopen := fun t ht => elderTrap_closedTrace_isClopen g D d t
    (elderTrap_strictTrace_isClopen g D d hD hboundary) ht.2
  closed_path := by
    refine ⟨γ, ?_⟩
    intro t
    change -(-d) ≤ g (γ t)
    simpa only [neg_neg] using hγ t

/-- The constructed singular-H0 direct summand, instantiated from geometric
inputs, rather than a premise asserting that a persistence summand exists. -/
def elderBoundaryDirectSummand (k : Type u) [Field k]
    {X : Type u} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (m d : ℝ) (M P : X)
    (hD : IsClosed D) (hboundary : ∀ x ∈ frontier D, g x ≤ d)
    (hdm : d < m) (hMheight : g M = m) (hM : M ∈ D)
    (hheight : ∀ x ∈ D, g x ≤ m) (hP : m < g P)
    (γ : Path M P) (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    superlevel_h0Filtration g (ModuleCat.of k k) ≅
      elderInterval k m d ⊞ kernel (elderRetraction (k := k)
        (elderTrap_of_boundary g D m d M P hD hboundary hdm hMheight hM hheight hP γ hγ)) :=
  elderDirectSummand (k := k)
    (elderTrap_of_boundary g D m d M P hD hboundary hdm hMheight hM hheight hP γ hγ)

/-- On Euclidean heights the same geometric inputs also give the literal C263
death. This is a proved join, not identification of an abstract death field. -/
theorem elderBoundary_death_eq {n : ℕ}
    (g : superlevelBar_E n → ℝ) (hg : Continuous g)
    (D : Set (superlevelBar_E n)) (m d : ℝ) (M P : superlevelBar_E n)
    (hD : IsClosed D) (hboundary : ∀ x ∈ frontier D, g x ≤ d)
    (hMheight : g M = m) (hM : M ∈ D)
    (hheight : ∀ x ∈ D, g x ≤ m) (hP : m < g P)
    (γ : Path M P) (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    superlevelBar_deathHeight g M = d := by
  apply elderTrap_death_eq g hg D d
    (elderTrap_strictTrace_isClopen g D d hD hboundary) M hM
    (fun x hx => (hheight x hx).trans_eq hMheight.symm) P
    (hMheight.trans_lt hP) γ hγ

end TorusFiltration
