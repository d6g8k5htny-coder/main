import SuperlevelBar

/-!
# From a geometric exit trap to the literal strict-clearance death

Author: OpenAI/Codex ROOT01a11cbf. Local candidate, 17 October 2026.
This proves a topological interface used by the reviewed asymmetric fold trap.
It does not assume the desired elder relation, clearance cut or death value.
The hypotheses are an actual closed domain with boundary below the threshold,
a bound on all heights inside, and an actual higher-endpoint path at the closed
threshold. The analytic fold construction of these hypotheses, the torus lift,
Morse saddle uniqueness and the H0 interval split are separate obligations.
Compilation, kernel checking and alignment are recorded separately.
-/

noncomputable section

namespace TorusFiltration

/-- The trace of a geometric domain in the strict superlevel above d. -/
def elderTrap_strictTrace {X : Type*} (g : X → ℝ) (D : Set X) (d : ℝ) :
    Set {x : X // d < g x} := {x | x.val ∈ D}

/-- A boundary height bound makes the trace clopen in the strict superlevel.
This derives separation from a geometric domain; it does not assume a path cut. -/
theorem elderTrap_strictTrace_isClopen {X : Type*} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (d : ℝ) (hD : IsClosed D)
    (hboundary : ∀ x ∈ frontier D, g x ≤ d) :
    IsClopen (elderTrap_strictTrace g D d) := by
  have heq : elderTrap_strictTrace g D d =
      (Subtype.val : {x : X // d < g x} → X) ⁻¹' interior D := by
    ext x
    constructor
    · intro hx
      change x.val ∈ D at hx
      change x.val ∈ interior D
      apply (mem_interior_iff_notMem_frontier hx).2
      intro hfrontier
      exact (not_lt_of_ge (hboundary x.val hfrontier)) x.property
    · intro hx
      change x.val ∈ interior D at hx
      change x.val ∈ D
      exact interior_subset hx
  refine ⟨hD.preimage continuous_subtype_val, ?_⟩
  rw [heq]
  exact isOpen_interior.preimage continuous_subtype_val

/-- A path strictly above the exit threshold cannot leave the geometric trap. -/
theorem elderTrap_path_stays {X : Type*} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (d : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d))
    {M y : X} (hM : M ∈ D) (γ : Path M y)
    (hγ : ∀ t : unitInterval, d < g (γ t)) : y ∈ D := by
  let δ : unitInterval → {x : X // d < g x} := fun t => ⟨γ t, hγ t⟩
  have hδ : Continuous δ := γ.continuous.subtype_mk _
  have hpre := htrace.preimage hδ
  have hzero : (0 : unitInterval) ∈ δ ⁻¹' elderTrap_strictTrace g D d := by
    change γ 0 ∈ D
    simpa only [γ.source] using hM
  have hall := hpre.eq_univ ⟨0, hzero⟩
  have hone : (1 : unitInterval) ∈ δ ⁻¹' elderTrap_strictTrace g D d := by
    rw [hall]
    exact Set.mem_univ _
  change γ 1 ∈ D at hone
  simpa only [γ.target] using hone

/-- Every higher-endpoint path must meet or fall below the exit threshold.
The conclusion quantifies over all ambient paths, including remote detours. -/
theorem elderTrap_higher_path_meets_threshold {X : Type*} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (d : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d))
    (M : X) (hM : M ∈ D) (hheight : ∀ x ∈ D, g x ≤ g M)
    {y : X} (hy : g M < g y) (γ : Path M y) :
    ∃ t : unitInterval, g (γ t) ≤ d := by
  by_contra hnone
  push Not at hnone
  have hyD := elderTrap_path_stays g D d htrace hM γ hnone
  exact (not_lt_of_ge (hheight y hyD)) hy

/-- The literal C263 clearance is exactly the strict interval below d.
The closed path is supplied by the geometric construction, rather than inferred
from a supremum over paths or from a compactness assertion about path space. -/
theorem elderTrap_clearance_eq_Iio {n : ℕ}
    (g : superlevelBar_E n → ℝ) (D : Set (superlevelBar_E n)) (d : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d))
    (M : superlevelBar_E n) (hM : M ∈ D)
    (hheight : ∀ x ∈ D, g x ≤ g M)
    (P : superlevelBar_E n) (hP : g M < g P) (γ : Path M P)
    (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    superlevelBar_clearanceSet g M = Set.Iio d := by
  ext h
  constructor
  · rintro ⟨y, hy, η, hη⟩
    obtain ⟨t, ht⟩ := elderTrap_higher_path_meets_threshold g D d htrace M hM hheight hy η
    exact lt_of_lt_of_le (hη t) ht
  · intro hh
    exact ⟨P, hP, γ, fun t => lt_of_lt_of_le hh (hγ t)⟩

/-- With continuity, the new geometric cut identifies the existing C263 death.
Nonessentiality is derived from the actual higher point P. -/
theorem elderTrap_death_eq {n : ℕ}
    (g : superlevelBar_E n → ℝ) (hg : Continuous g)
    (D : Set (superlevelBar_E n)) (d : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d))
    (M : superlevelBar_E n) (hM : M ∈ D)
    (hheight : ∀ x ∈ D, g x ≤ g M)
    (P : superlevelBar_E n) (hP : g M < g P) (γ : Path M P)
    (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    superlevelBar_deathHeight g M = d := by
  have hne : ¬ superlevelBar_isEssential g M := fun hess =>
    (not_lt_of_ge (hess P)) hP
  apply (superlevelBar_cut_unique g hg M hne d ?_).symm
  intro h
  rw [elderTrap_clearance_eq_Iio g D d htrace M hM hheight P hP γ hγ]
  rfl

/-- The displayed closed path actually attains d. This does not replace strict
clearance by a closed inequality or infer an optimal path from a supremum. -/
theorem elderTrap_closed_path_attains {X : Type*} [TopologicalSpace X]
    (g : X → ℝ) (D : Set X) (d : ℝ)
    (htrace : IsClopen (elderTrap_strictTrace g D d))
    (M : X) (hM : M ∈ D) (hheight : ∀ x ∈ D, g x ≤ g M)
    {P : X} (hP : g M < g P) (γ : Path M P)
    (hγ : ∀ t : unitInterval, d ≤ g (γ t)) :
    ∃ t : unitInterval, g (γ t) = d := by
  obtain ⟨t, ht⟩ := elderTrap_higher_path_meets_threshold g D d htrace M hM hheight hP γ
  exact ⟨t, le_antisymm ht (hγ t)⟩

end TorusFiltration
