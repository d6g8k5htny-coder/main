import Mathlib

/-!
# Closed superlevels and their actual component and singular H0 functors

C254 proposed source. Compilation and kernel audit: NOT_RUN.
The parameter is `t = -h`; endpoint membership uses `≤`.
This file supplies no Morse theorem, labels, bars, or H0/component naturality.
Compile fix at the pin (Lean v4.34.1, Mathlib d13f23b7), altering no statement: in
`superlevel_componentMap` the `ZerothHomotopy.lift` witness binds its two strict-implicit
points before the `Path` argument (`fun _ _ p => ...`), as `ZerothHomotopy.lift` requires here.
-/

noncomputable section

open CategoryTheory

universe w v u

namespace TorusFiltration
/-- The increasing closed-superlevel family, with parameter `t = -h`. -/
def superlevel_set {X : Type*} (g : X → ℝ) (t : ℝ) : Set X :=
  {x | -t ≤ g x}

/-- Control: the point at the threshold belongs to the closed superlevel.
Changing `≤` to `<` invalidates this statement. -/
theorem superlevel_control_closed_endpoint (b : ℝ) :
    () ∈ superlevel_set (fun _ : Unit => b) (-b) := by
  simp [superlevel_set]

/-- Control: the strict condition fails at that same threshold. -/
theorem superlevel_control_strict_endpoint (b : ℝ) :
    ¬ (-(-b) < (fun _ : Unit => b) ()) := by
  simp

/-- Control: at parameter `1` the constant height `0` is present.
Changing `-t` to `t` invalidates this statement. -/
theorem superlevel_control_parameter_sign :
    () ∈ superlevel_set (fun _ : Unit => (0 : ℝ)) 1 := by
  norm_num [superlevel_set]

/-- Control: inclusion cannot run from parameter `1` back to `-1`. -/
theorem superlevel_control_inclusion_direction :
    ¬ superlevel_set (fun _ : Unit => (0 : ℝ)) 1 ⊆
      superlevel_set (fun _ : Unit => (0 : ℝ)) (-1) := by
  intro h
  have hc := h superlevel_control_parameter_sign
  norm_num [superlevel_set] at hc

/-- A global maximum has no strictly higher endpoint, regardless of paths. -/
theorem superlevel_control_essential {X : Type*} (g : X → ℝ) (M : X)
    (hM : ∀ y, g y ≤ g M) : ¬ ∃ y, g M < g y := by
  rintro ⟨y, hy⟩
  exact (not_lt_of_ge (hM y)) hy

/-- Increasing parameters give increasing closed superlevels. -/
theorem superlevel_mono {X : Type*} (g : X → ℝ) {s t : ℝ} (hst : s ≤ t) :
    superlevel_set g s ⊆ superlevel_set g t := by
  intro x hx
  exact le_trans (neg_le_neg hst) hx

/-- Each level is closed when the function is continuous. -/
theorem superlevel_isClosed {X : Type*} [TopologicalSpace X]
    (g : X → ℝ) (hg : Continuous g) (t : ℝ) : IsClosed (superlevel_set g t) := by
  exact isClosed_le continuous_const hg

/-- Continuity of the inclusion is independent of continuity of the height function. -/
def superlevel_inclusion {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) {s t : ℝ} (hst : s ≤ t) :
    C(superlevel_set g s, superlevel_set g t) :=
  ContinuousMap.inclusion (superlevel_mono g hst)

theorem superlevel_inclusion_id {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (t : ℝ) :
    superlevel_inclusion g (le_refl t) = ContinuousMap.id (superlevel_set g t) := by
  ext x
  rfl

theorem superlevel_inclusion_comp {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) {r s t : ℝ} (hrs : r ≤ s) (hst : s ≤ t) :
    (superlevel_inclusion g hst).comp (superlevel_inclusion g hrs) =
      superlevel_inclusion g (hrs.trans hst) := by
  ext x
  rfl

/-- Point membership at its own height; this is not a component-birth theorem. -/
theorem superlevel_mem_at_own_height {X : Type*} (g : X → ℝ) (x : X) :
    x ∈ superlevel_set g (-g x) := by
  simp [superlevel_set]

theorem superlevel_neg_height {X : Type*} (g : X → ℝ) (h : ℝ) :
    superlevel_set g (-h) = {x | h ≤ g x} := by
  ext x
  simp [superlevel_set]

/-- An explicit threshold countercontrol; it is not a compact Morse counterexample. -/
theorem superlevel_control_strict_thresholds :
    (∀ h : ℝ, h < 0 → ∃ k : ℝ, h < k ∧ k < 0) ∧
      ¬ ∃ k : ℝ, 0 ≤ k ∧ k < 0 := by
  constructor
  · intro h hh
    exact ⟨h / 2, by linarith, by linarith⟩
  · rintro ⟨k, hk, hk'⟩
    exact (not_lt_of_ge hk) hk'

/-- The induced map on actual path-component quotients. -/
def superlevel_componentMap {X Y : Type w} [TopologicalSpace X] [TopologicalSpace Y]
    (f : C(X, Y)) : ZerothHomotopy X → ZerothHomotopy Y :=
  ZerothHomotopy.lift (fun x => ZerothHomotopy.mk (f x))
    (fun _ _ p => ZerothHomotopy.sound (p.map f.continuous))

theorem superlevel_componentMap_mk {X Y : Type w}
    [TopologicalSpace X] [TopologicalSpace Y] (f : C(X, Y)) (x : X) :
    superlevel_componentMap f (ZerothHomotopy.mk x) = ZerothHomotopy.mk (f x) := rfl

theorem superlevel_componentMap_id {X : Type w} [TopologicalSpace X] :
    superlevel_componentMap (ContinuousMap.id X) = id := by
  funext q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  rfl

theorem superlevel_componentMap_comp {X Y Z : Type w}
    [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z]
    (f : C(X, Y)) (g : C(Y, Z)) :
    superlevel_componentMap (g.comp f) =
      superlevel_componentMap g ∘ superlevel_componentMap f := by
  funext q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  rfl

def superlevel_componentInclusion {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) {s t : ℝ} (hst : s ≤ t) :
    ZerothHomotopy (superlevel_set g s) → ZerothHomotopy (superlevel_set g t) :=
  superlevel_componentMap (superlevel_inclusion g hst)

theorem superlevel_componentInclusion_id {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (t : ℝ) : superlevel_componentInclusion g (le_refl t) = id := by
  funext q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  rfl

theorem superlevel_componentInclusion_comp {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) {r s t : ℝ} (hrs : r ≤ s) (hst : s ≤ t) :
    superlevel_componentInclusion g (hrs.trans hst) =
      superlevel_componentInclusion g hst ∘ superlevel_componentInclusion g hrs := by
  funext q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  rfl

/-- Actual component equality is precisely a path in the ambient closed superlevel. -/
theorem superlevel_component_eq_iff_joinedIn {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (t : ℝ) (x y : superlevel_set g t) :
    ZerothHomotopy.mk x = ZerothHomotopy.mk y ↔
      JoinedIn (superlevel_set g t) x.val y.val := by
  constructor
  · intro h
    have hxy : Joined x y := Quotient.exact h
    exact (joinedIn_iff_joined x.property y.property).2 hxy
  · intro h
    obtain ⟨p⟩ := (joinedIn_iff_joined x.property y.property).1 h
    exact ZerothHomotopy.sound p

/-- The topological filtration in the preorder category of real numbers. -/
def superlevel_filtration {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) : ℝ ⥤ TopCat.{w} where
  obj t := TopCat.of (superlevel_set g t)
  map f := TopCat.ofHom (superlevel_inclusion g (leOfHom f))
  map_id t := by
    ext x
    rfl
  map_comp f k := by
    ext x
    rfl

variable {C : Type u} [Category.{v} C]
    [CategoryTheory.Limits.HasCoproducts.{w} C]
    [Preadditive C] [CategoryWithHomology C]

/-- Actual singular H0 with coefficient object R, including the inclusion maps. -/
def superlevel_h0Filtration {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) : ℝ ⥤ C :=
  superlevel_filtration g ⋙ (AlgebraicTopology.singularHomologyFunctor C 0).obj R

/-- This target includes the actual H0 definition in the transitive dependency audit. -/
theorem superlevel_h0Filtration_map_id {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) (t : ℝ) :
    (superlevel_h0Filtration g R).map (𝟙 t) =
      𝟙 ((superlevel_h0Filtration g R).obj t) :=
  (superlevel_h0Filtration g R).map_id t

theorem superlevel_h0Filtration_map_comp {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) {r s t : ℝ} (f : r ⟶ s) (k : s ⟶ t) :
    (superlevel_h0Filtration g R).map (f ≫ k) =
      (superlevel_h0Filtration g R).map f ≫ (superlevel_h0Filtration g R).map k :=
  (superlevel_h0Filtration g R).map_comp f k

end TorusFiltration