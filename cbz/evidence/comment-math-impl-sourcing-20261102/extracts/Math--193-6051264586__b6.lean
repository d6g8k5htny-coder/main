import Mathlib.AlgebraicTopology.SingularHomology.HomologyZero
import Mathlib.Algebra.Category.ModuleCat.Colimits
import Mathlib.Algebra.Category.ModuleCat.Abelian
import Mathlib.LinearAlgebra.Finsupp.VectorSpace
import CapFirstExit

/-!
Actual ordinary singular H₀ point generators and elder-span criterion.
No interval decomposition or Morse/barcode interface is assumed.
-/

noncomputable section

open CategoryTheory Limits AlgebraicTopology Set
open scoped Simplicial

universe u

namespace OrdinaryH0Elder

variable (K : Type u) [Field K]

/-- Ordinary singular H₀ with coefficients in the one-dimensional K-module. -/
def h0Functor : TopCat.{u} ⥤ ModuleCat.{u} K :=
  (singularHomologyFunctor (ModuleCat.{u} K) 0).obj (ModuleCat.of K K)

abbrev H0 (X : TopCat.{u}) := (h0Functor K).obj X

/-- The existing singular H₀ isomorphism followed by the concrete coproduct model. -/
def componentIso (X : TopCat.{u}) :
    H0 K X ≅ ModuleCat.of K (ZerothHomotopy X →₀ K) :=
  TopCat.singularHomology₀Iso X (ModuleCat.of K K) ≪≫
    (coproductIsCoproduct _).coconePointUniqueUpToIso
      (ModuleCat.finsuppCoconeIsColimit K K (ZerothHomotopy X))

def componentEquiv (X : TopCat.{u}) :
    H0 K X ≃ₗ[K] (ZerothHomotopy X →₀ K) :=
  (componentIso K X).toLinearEquiv

/-- The actual ordinary H₀ class of a point, corresponding to its component basis vector. -/
def pointClass (X : TopCat.{u}) (x : X) : H0 K X :=
  (componentEquiv K X).symm (Finsupp.single (ZerothHomotopy.mk x) 1)

def olderSpan (X : TopCat.{u}) (older : X → Prop) : Submodule K (H0 K X) :=
  Submodule.span K (pointClass K X '' {x | older x})

theorem pointClass_eq_iff_joined (X : TopCat.{u}) (x y : X) :
    pointClass K X x = pointClass K X y ↔ Joined x y := by
  rw [pointClass, pointClass, (componentEquiv K X).symm.injective.eq_iff,
    Finsupp.single_left_inj (one_ne_zero : (1 : K) ≠ 0)]
  exact Quotient.eq

theorem pointClass_ne_zero (X : TopCat.{u}) (x : X) :
    pointClass K X x ≠ 0 := by
  intro h
  have hs : Finsupp.single (ZerothHomotopy.mk x) (1 : K) = 0 := by
    simpa [pointClass] using congrArg (componentEquiv K X) h
  exact one_ne_zero (Finsupp.single_eq_zero.mp hs)

/-- The actual zero-simplex cycle, projected into singular H₀. -/
def pointCycleMorph (X : TopCat.{u}) (x : X) : ModuleCat.of K K ⟶ H0 K X :=
  ((TopCat.toSSet.obj X).chainComplex (ModuleCat.of K K)).liftCycles
      ((TopCat.toSSet.obj X).ιChainComplex (TopCat.toSSetObj₀Equiv.symm x))
      0 (by simp) (by simp) ≫
    ((TopCat.toSSet.obj X).chainComplex (ModuleCat.of K K)).homologyπ 0

set_option backward.isDefEq.respectTransparency false in
lemma pointCycleMorph_componentIso (X : TopCat.{u}) (x : X) :
    pointCycleMorph K X x ≫ (componentIso K X).hom =
      ModuleCat.ofHom (Finsupp.lsingle (ZerothHomotopy.mk x)) := by
  simp [pointCycleMorph, componentIso, TopCat.singularHomology₀Iso, sigmaConst,
    ModuleCat.finsuppCocone]
  exact IsColimit.comp_coconePointUniqueUpToIso_hom
    (coproductIsCoproduct _) (ModuleCat.finsuppCoconeIsColimit K K (ZerothHomotopy X))
    ⟨ZerothHomotopy.mk x⟩

lemma pointClass_eq_pointCycleMorph (X : TopCat.{u}) (x : X) :
    pointClass K X x = pointCycleMorph K X x (1 : K) := by
  apply (componentEquiv K X).injective
  simp only [pointClass, LinearEquiv.apply_symm_apply]
  change Finsupp.single (ZerothHomotopy.mk x) (1 : K) =
    (componentIso K X).hom (pointCycleMorph K X x (1 : K))
  have h := congrArg (fun g : ModuleCat.of K K ⟶
      ModuleCat.of K (ZerothHomotopy X →₀ K) => g (1 : K))
    (pointCycleMorph_componentIso K X x)
  exact h.symm

set_option backward.isDefEq.respectTransparency false in
lemma pointCycleMorph_map {X Y : TopCat.{u}} (f : X ⟶ Y) (x : X) :
    pointCycleMorph K X x ≫ (h0Functor K).map f = pointCycleMorph K Y (f x) := by
  change pointCycleMorph K X x ≫
    SSet.homologyMap (TopCat.toSSet.map f) (ModuleCat.of K K) 0 = _
  simp only [pointCycleMorph, SSet.homologyMap, Category.assoc,
    HomologicalComplex.homologyπ_naturality]
  rw [← Category.assoc, HomologicalComplex.liftCycles_comp_cyclesMap,
    SSet.ι_chainComplexMap_f]
  rfl

/-- Naturality under genuine singular homology maps. -/
theorem pointClass_map {X Y : TopCat.{u}} (f : X ⟶ Y) (x : X) :
    (h0Functor K).map f (pointClass K X x) = pointClass K Y (f x) := by
  rw [pointClass_eq_pointCycleMorph, pointClass_eq_pointCycleMorph]
  change (pointCycleMorph K X x ≫ (h0Functor K).map f) (1 : K) = _
  rw [pointCycleMorph_map]

theorem mem_olderSpan_iff_joinedOlder (X : TopCat.{u}) (older : X → Prop) (x : X) :
    pointClass K X x ∈ olderSpan K X older ↔ ∃ y, older y ∧ Joined x y := by
  classical
  constructor
  · intro hm
    by_contra h
    have hne : ∀ y, older y → ZerothHomotopy.mk x ≠ ZerothHomotopy.mk y := by
      intro y hy he
      exact h ⟨y, hy, Quotient.exact he⟩
    let ev : H0 K X →ₗ[K] K :=
      (Finsupp.lapply (ZerothHomotopy.mk x)).comp (componentEquiv K X).toLinearMap
    have hs : olderSpan K X older ≤ LinearMap.ker ev := by
      apply Submodule.span_le.mpr
      rintro _ ⟨y, hy, rfl⟩
      change ((componentEquiv K X) (pointClass K X y)) (ZerothHomotopy.mk x) = 0
      simp only [pointClass, LinearEquiv.apply_symm_apply]
      exact Finsupp.single_eq_of_ne (hne y hy)
    have hc := hs hm
    change ((componentEquiv K X) (pointClass K X x)) (ZerothHomotopy.mk x) = 0 at hc
    simp [pointClass] at hc
  · rintro ⟨y, hy, hxy⟩
    rw [(pointClass_eq_iff_joined K X x y).mpr hxy]
    exact Submodule.subset_span ⟨y, hy, rfl⟩

section Superlevel

variable {X : Type u} [TopologicalSpace X]

def superlevel (f : X → ℝ) (h : ℝ) : TopCat.{u} :=
  TopCat.of {x : X // h ≤ f x}

/-- The continuous closed-superlevel inclusion from a higher level to a lower level. -/
def superlevelInclusion (f : X → ℝ) {high low : ℝ} (hle : low ≤ high) :
    superlevel f high ⟶ superlevel f low :=
  TopCat.ofHom ⟨fun x => ⟨x.1, hle.trans x.2⟩, continuous_subtype_val.subtype_mk _⟩

/-- The genuine closed-superlevel diagram, indexed by decreasing levels. -/
def superlevelFunctor (f : X → ℝ) : ℝᵒᵈ ⥤ TopCat.{u} where
  obj h := superlevel f h
  map {a b} g := superlevelInclusion f (@leOfHom ℝᵒᵈ _ a b g)
  map_id _ := rfl
  map_comp _ _ := rfl

/-- The actual ordinary singular H₀ persistence diagram of closed superlevels. -/
def closedH0Persistence (f : X → ℝ) : ℝᵒᵈ ⥤ ModuleCat.{u} K :=
  superlevelFunctor f ⋙ h0Functor K

theorem pointClass_superlevelInclusion (f : X → ℝ) {high low : ℝ}
    (hle : low ≤ high) (x : X) (hx : high ≤ f x) :
    (h0Functor K).map (superlevelInclusion f hle)
      (pointClass K (superlevel f high) ⟨x, hx⟩) =
        pointClass K (superlevel f low) ⟨x, hle.trans hx⟩ :=
  pointClass_map K (superlevelInclusion f hle) ⟨x, hx⟩

set_option backward.isDefEq.respectTransparency false in
/-- A cap's point class becomes dependent on strictly older point classes exactly at s. -/
theorem cap_mem_olderSpan_iff {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}
    (H : CapFirstExit.CapHyp f C M z b s) (hMb : f M = b) {h : ℝ} (hh : h ≤ b) :
    pointClass K (superlevel f h) ⟨M, by rwa [hMb]⟩ ∈
      olderSpan K (superlevel f h) (fun y => b < f y.1) ↔ h ≤ s := by
  rw [mem_olderSpan_iff_joinedOlder]
  constructor
  · rintro ⟨y, hy, hjoin⟩
    have hj : JoinedIn {x | h ≤ f x} M y.1 :=
      (joinedIn_iff_joined (F := {x | h ≤ f x}) (x := M) (y := y.1)
        (by simpa [hMb] using hh) y.2).mpr hjoin
    exact (H.elder_merge_iff_closed h).mp
      ⟨y.1, CapFirstExit.component_joined hj.somePath hj.somePath_mem, hy⟩
  · intro hhs
    obtain ⟨γ, hγ⟩ := H.ridge
    have hz : h ≤ f z := by simpa using hhs.trans (hγ 1)
    have hj : JoinedIn {x | h ≤ f x} M z := ⟨γ, fun t => hhs.trans (hγ t)⟩
    exact ⟨⟨z, hz⟩, H.older,
      (joinedIn_iff_joined (F := {x | h ≤ f x}) (x := M) (y := z)
        (by simpa [hMb] using hh) hz).mp hj⟩

end Superlevel
end OrdinaryH0Elder

