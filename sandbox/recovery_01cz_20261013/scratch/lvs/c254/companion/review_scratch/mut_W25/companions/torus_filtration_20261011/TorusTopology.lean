import Mathlib

/-!
# The actual period-lattice quotient and specified-start path compatibility

C254 proposed source. Compilation and kernel audit: NOT_RUN.
The ambient space and lattice-vector formula below are exactly those in the
source-bound Field.lean; this module does not import or alter that owner's file.
All quotient, compactness, covering and descent properties are constructed.
The final lift endpoint is existential. No critical-level attainment is claimed.
Compile fixes at the pin (Lean v4.34.1, Mathlib d13f23b7), no statement altered: `Basis` → `Module.Basis`, `Basis.unitsSMul_apply` → `Module.Basis.unitsSMul_apply`, lattice discreteness passed via `SetLike.isDiscrete_iff_discreteTopology.mpr` instead of `DiscreteTopology.isDiscrete`, quotient path-connectedness via `inferInstanceAs (PathConnectedSpace (Quotient _))` instead of `infer_instance`.
Three unused simp arguments (`PiLp.single_apply` twice, `hij`) removed from `torus_basis_sum_eq_latticeVec` so the
file elaborates with warnings as errors; namespace `TorusFiltration` in this companion (handoff: `ResearchFormalCoreR1`).
-/

noncomputable section

open Topology

namespace TorusFiltration
abbrev torus_E (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- The exact `Field.latticeVec` formula, retained without any coordinate rescaling. -/
def torus_latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) : torus_E d :=
  ∑ i, EuclideanSpace.single i (L * (n i : ℝ))

theorem torus_field_latticeVec_formula (d : ℕ) (L : ℝ) (n : Fin d → ℤ) :
    torus_latticeVec d L n = ∑ i, EuclideanSpace.single i (L * (n i : ℝ)) := rfl

def torus_periodBasis (d : ℕ) (L : ℝ) (hL : 0 < L) :
    Module.Basis (Fin d) ℝ (torus_E d) :=
  (EuclideanSpace.basisFun (Fin d) ℝ).toBasis.unitsSMul
    (fun _ => Units.mk0 L hL.ne')

def torus_periodLattice (d : ℕ) (L : ℝ) (hL : 0 < L) : Submodule ℤ (torus_E d) :=
  Submodule.span ℤ (Set.range (torus_periodBasis d L hL))

abbrev torus_quotient (d : ℕ) (L : ℝ) (hL : 0 < L) :=
  torus_E d ⧸ (torus_periodLattice d L hL).toAddSubgroup

def torus_proj (d : ℕ) (L : ℝ) (hL : 0 < L) : torus_E d → torus_quotient d L hL :=
  QuotientAddGroup.mk

/-- Concrete adapter from integer basis coefficients to Field's exact vector formula. -/
theorem torus_basis_sum_eq_latticeVec (d : ℕ) (L : ℝ) (hL : 0 < L)
    (n : Fin d → ℤ) :
    ∑ i, n i • torus_periodBasis d L hL i = torus_latticeVec d L n := by
  classical
  unfold torus_latticeVec
  apply Finset.sum_congr rfl
  intro i hi
  ext j
  by_cases hij : i = j
  · subst j
    simp [torus_periodBasis, Module.Basis.unitsSMul_apply, OrthonormalBasis.coe_toBasis,
      EuclideanSpace.basisFun_apply, Units.smul_def, PiLp.smul_apply,
      zsmul_eq_mul, mul_comm]
  · simp [torus_periodBasis, Module.Basis.unitsSMul_apply, OrthonormalBasis.coe_toBasis,
      EuclideanSpace.basisFun_apply, Units.smul_def, PiLp.smul_apply,
      Ne.symm hij]

theorem torus_lattice_mem_iff (d : ℕ) (L : ℝ) (hL : 0 < L) (x : torus_E d) :
    x ∈ torus_periodLattice d L hL ↔ ∃ n : Fin d → ℤ, torus_latticeVec d L n = x := by
  classical
  change x ∈ Submodule.span ℤ (Set.range (torus_periodBasis d L hL)) ↔ _
  rw [Submodule.mem_span_range_iff_exists_fun]
  simp only [torus_basis_sum_eq_latticeVec]

instance torus_lattice_discrete (d : ℕ) (L : ℝ) (hL : 0 < L) :
    DiscreteTopology (torus_periodLattice d L hL) := by
  unfold torus_periodLattice
  infer_instance

instance torus_lattice_addSubgroup_discrete (d : ℕ) (L : ℝ) (hL : 0 < L) :
    DiscreteTopology (torus_periodLattice d L hL).toAddSubgroup := by
  unfold torus_periodLattice
  infer_instance

instance torus_lattice_isZLattice (d : ℕ) (L : ℝ) (hL : 0 < L) :
    IsZLattice ℝ (torus_periodLattice d L hL) := by
  unfold torus_periodLattice
  infer_instance

theorem torus_proj_surjective (d : ℕ) (L : ℝ) (hL : 0 < L) :
    Function.Surjective (torus_proj d L hL) := QuotientAddGroup.mk_surjective

theorem torus_proj_continuous (d : ℕ) (L : ℝ) (hL : 0 < L) :
    Continuous (torus_proj d L hL) := continuous_quotient_mk'

theorem torus_proj_eq_iff (d : ℕ) (L : ℝ) (hL : 0 < L) (x y : torus_E d) :
    torus_proj d L hL x = torus_proj d L hL y ↔
      ∃ n : Fin d → ℤ, y = x + torus_latticeVec d L n := by
  change (QuotientAddGroup.mk x : torus_quotient d L hL) = QuotientAddGroup.mk y ↔ _
  rw [QuotientAddGroup.eq]
  change -x + y ∈ torus_periodLattice d L hL ↔ _
  rw [torus_lattice_mem_iff]
  constructor
  · rintro ⟨n, hn⟩
    exact ⟨n, by rw [hn]; abel⟩
  · rintro ⟨n, rfl⟩
    exact ⟨n, by abel⟩

/-- Duplicate representatives have exactly the same quotient image. -/
theorem torus_control_lattice_translate (d : ℕ) (L : ℝ) (hL : 0 < L)
    (x : torus_E d) (n : Fin d → ℤ) :
    torus_proj d L hL (x + torus_latticeVec d L n) = torus_proj d L hL x :=
  ((torus_proj_eq_iff d L hL x _).2 ⟨n, rfl⟩).symm

theorem torus_proj_isCoveringMap (d : ℕ) (L : ℝ) (hL : 0 < L) :
    IsCoveringMap (torus_proj d L hL) :=
  ((torus_periodLattice d L hL).toAddSubgroup.isAddQuotientCoveringMap_of_comm
    (SetLike.isDiscrete_iff_discreteTopology.mpr
      (torus_lattice_addSubgroup_discrete d L hL))).isCoveringMap

theorem torus_compact (d : ℕ) (L : ℝ) (hL : 0 < L) :
    IsCompact (Set.univ : Set (torus_quotient d L hL)) := by
  have hc := IsZLattice.isCompact_range_of_periodic (torus_periodLattice d L hL)
    (torus_proj d L hL) (torus_proj_continuous d L hL)
    (fun z w hw => QuotientAddGroup.mk_add_of_mem z hw)
  rwa [Set.range_eq_univ.mpr (torus_proj_surjective d L hL)] at hc

theorem torus_pathConnectedSpace (d : ℕ) (L : ℝ) (hL : 0 < L) :
    PathConnectedSpace (torus_quotient d L hL) := by
  exact inferInstanceAs <| PathConnectedSpace (Quotient _)

/-- Exact Field periodicity predicate with the literal lattice-vector adapter above. -/
def torus_isPeriodic (d : ℕ) (L : ℝ) (g : torus_E d → ℝ) : Prop :=
  ∀ (x : torus_E d) (n : Fin d → ℤ), g (x + torus_latticeVec d L n) = g x

theorem torus_field_periodic_formula (d : ℕ) (L : ℝ) (g : torus_E d → ℝ) :
    torus_isPeriodic d L g ↔
      ∀ (x : EuclideanSpace ℝ (Fin d)) (n : Fin d → ℤ),
        g (x + ∑ i, EuclideanSpace.single i (L * (n i : ℝ))) = g x := Iff.rfl

theorem torus_periodic_respects (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (x y : torus_E d)
    (hxy : QuotientAddGroup.leftRel (torus_periodLattice d L hL).toAddSubgroup x y) :
    g x = g y := by
  have hq : torus_proj d L hL x = torus_proj d L hL y := Quotient.sound hxy
  obtain ⟨n, rfl⟩ := (torus_proj_eq_iff d L hL x y).1 hq
  exact (hg x n).symm

def torus_descend (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) : torus_quotient d L hL → ℝ :=
  Quotient.lift g (torus_periodic_respects d L hL g hg)

theorem torus_descend_proj (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (x : torus_E d) :
    torus_descend d L hL g hg (torus_proj d L hL x) = g x := rfl

theorem torus_descend_continuous (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (hgc : Continuous g) :
    Continuous (torus_descend d L hL g hg) :=
  hgc.quotient_lift (torus_periodic_respects d L hL g hg)

/-- Lift from a prescribed representative; the endpoint is determined existentially. -/
theorem torus_path_lift_from (d : ℕ) (L : ℝ) (hL : 0 < L)
    {M : torus_E d} {y : torus_quotient d L hL}
    (γ : Path (torus_proj d L hL M) y) :
    ∃ z : torus_E d, ∃ Γ : Path M z,
      torus_proj d L hL z = y ∧
        ∀ t : unitInterval, torus_proj d L hL (Γ t) = γ t := by
  let cov := torus_proj_isCoveringMap d L hL
  let Γ := cov.liftPath γ M γ.source
  refine ⟨Γ 1, ⟨Γ, cov.liftPath_zero γ M γ.source, rfl⟩, ?_, ?_⟩
  · exact (congrFun (cov.liftPath_lifts γ M γ.source) 1).trans γ.target
  · intro t
    exact congrFun (cov.liftPath_lifts γ M γ.source) t

/-- Control: a constant projected path cannot be lifted from M to a different lift.
This refutes replacing the existential endpoint by an arbitrary chosen representative. -/
theorem torus_control_constant_path_endpoint (d : ℕ) (L : ℝ) (hL : 0 < L)
    {M z : torus_E d} (Γ : Path M z)
    (hΓ : ∀ t : unitInterval, torus_proj d L hL (Γ t) = torus_proj d L hL M) : z = M := by
  have heq : (Γ : unitInterval → torus_E d) = fun _ => M :=
    (torus_proj_isCoveringMap d L hL).eq_of_comp_eq Γ.continuous continuous_const
      (funext hΓ) 0 Γ.source
  exact Γ.target.symm.trans (congrFun heq 1)

/-- Combining the duplicate-image and uniqueness controls gives a direct rejection
of any different preselected endpoint for a constant quotient path. -/
theorem torus_control_no_arbitrary_endpoint (d : ℕ) (L : ℝ) (hL : 0 < L)
    (M : torus_E d) (n : Fin d → ℤ) (hn : torus_latticeVec d L n ≠ 0) :
    torus_proj d L hL (M + torus_latticeVec d L n) = torus_proj d L hL M ∧
      ¬ ∃ Γ : Path M (M + torus_latticeVec d L n),
        ∀ t : unitInterval, torus_proj d L hL (Γ t) = torus_proj d L hL M := by
  refine ⟨torus_control_lattice_translate d L hL M n, ?_⟩
  rintro ⟨Γ, hΓ⟩
  have heq := torus_control_constant_path_endpoint d L hL Γ hΓ
  apply hn
  exact add_left_cancel (show M + torus_latticeVec d L n = M + 0 by simpa using heq)

/-- Higher-endpoint compatibility for any scalar path predicate, including strict
and closed thresholds. No conclusion here asserts critical-level attainment. -/
theorem torus_higher_path_iff (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (M : torus_E d) (P : ℝ → Prop) :
    (∃ y : torus_E d, g M < g y ∧ ∃ γ : Path M y, ∀ t : unitInterval, P (g (γ t))) ↔
      (∃ y : torus_quotient d L hL,
        torus_descend d L hL g hg (torus_proj d L hL M) < torus_descend d L hL g hg y ∧
        ∃ γ : Path (torus_proj d L hL M) y,
          ∀ t : unitInterval, P (torus_descend d L hL g hg (γ t))) := by
  constructor
  · rintro ⟨y, hy, γ, hγ⟩
    exact ⟨torus_proj d L hL y, hy, γ.map (torus_proj_continuous d L hL), hγ⟩
  · rintro ⟨y, hy, γ, hγ⟩
    obtain ⟨z, Γ, hz, hΓ⟩ := torus_path_lift_from d L hL γ
    refine ⟨z, ?_, Γ, ?_⟩
    · have hz' := congrArg (torus_descend d L hL g hg) hz
      change g z = torus_descend d L hL g hg y at hz'
      change g M < torus_descend d L hL g hg y at hy
      rwa [← hz'] at hy
    · intro t
      have ht := hγ t
      rw [← hΓ t] at ht
      exact ht

theorem torus_strict_higher_path_iff (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (M : torus_E d) (h : ℝ) :
    (∃ y : torus_E d, g M < g y ∧ ∃ γ : Path M y, ∀ t : unitInterval, h < g (γ t)) ↔
      (∃ y : torus_quotient d L hL,
        torus_descend d L hL g hg (torus_proj d L hL M) < torus_descend d L hL g hg y ∧
        ∃ γ : Path (torus_proj d L hL M) y,
          ∀ t : unitInterval, h < torus_descend d L hL g hg (γ t)) :=
  torus_higher_path_iff d L hL g hg M (fun a => h < a)

theorem torus_closed_higher_path_iff (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (M : torus_E d) (h : ℝ) :
    (∃ y : torus_E d, g M < g y ∧ ∃ γ : Path M y, ∀ t : unitInterval, h ≤ g (γ t)) ↔
      (∃ y : torus_quotient d L hL,
        torus_descend d L hL g hg (torus_proj d L hL M) < torus_descend d L hL g hg y ∧
        ∃ γ : Path (torus_proj d L hL M) y,
          ∀ t : unitInterval, h ≤ torus_descend d L hL g hg (γ t)) :=
  torus_higher_path_iff d L hL g hg M (fun a => h ≤ a)

theorem torus_global_max_iff (d : ℕ) (L : ℝ) (hL : 0 < L)
    (g : torus_E d → ℝ) (hg : torus_isPeriodic d L g) (M : torus_E d) :
    (∀ y : torus_E d, g y ≤ g M) ↔
      ∀ y : torus_quotient d L hL,
        torus_descend d L hL g hg y ≤ torus_descend d L hL g hg (torus_proj d L hL M) := by
  constructor
  · intro h y
    obtain ⟨z, rfl⟩ := torus_proj_surjective d L hL y
    exact h z
  · intro h y
    exact h (torus_proj d L hL y)

end TorusFiltration