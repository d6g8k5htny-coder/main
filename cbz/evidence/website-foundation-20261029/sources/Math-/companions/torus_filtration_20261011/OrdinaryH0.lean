import Mathlib
import ClosedSuperlevel

/-!
# Actual ordinary singular H0 and path components, naturally

C264 isolated source candidate. Compilation and kernel audit: NOT_RUN.
The homology objects and maps below are Mathlib's singular homology functor.
The comparison is its existing objectwise `TopCat.singularHomology₀Iso`;
its naturality is proved here from actual zero-cycle generators.

The coefficient object belongs to an arbitrary preadditive category with
the required coproducts and homology. Nonzero/distinction controls explicitly
require `𝟙 R ≠ 0`; no nonzero claim is made for a zero coefficient object.

This supplies the component/H0 comparison for the existing closed-superlevel
filtration. It supplies no Morse chart, critical-level attainment theorem,
interval decomposition, elder correspondence, Field adapter, count identity,
stochastic law, or scientific-status promotion.

Owner repairs at the pin (Anthropic/Claude lane, companion owner, recorded as the
actual repair author; handoff bytes SHA-256
261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e did not elaborate
with `-DwarningAsError=true`, Math- #428 comments 6113300903 and 6113372506): the
homology instances `[Preadditive C] [CategoryWithHomology C]` enter the `variable`
scope only after `ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentFiltration`
and `ordinaryH0_componentFiltration_map_iota`, which do not use them (the two filtration
declarations therefore precede the homology ones; no statement changed);
`ordinaryH0_iso_naturality` is proved by explicit `rw`/`erw` reassociation of both sides
(the source author's five-step `calc` elaborated with an unresolved `Trans` instance
at the pin) instead of a `simp only` that left the right side unreassociated; `isoWhiskerLeft` is referred to as `Functor.isoWhiskerLeft`; the
three declaration names ending in `ι` are spelled `…_iota` / `…Iota…` for the
companion's ASCII identifier rule. No statement changed.
-/

noncomputable section

open CategoryTheory Limits AlgebraicTopology
open scoped Simplicial

universe w v u

namespace TorusFiltration

/-- Proves: actual path-component quotients form a functor on topological spaces.
Method: use the existing continuous-map quotient map and its identity/composition laws.
Does not establish: equality of connected and path-connected components. -/
def ordinaryH0_componentFunctor : TopCat.{w} ⥤ Type w where
  obj X := ZerothHomotopy X
  map f := TypeCat.ofHom (superlevel_componentMap f.hom)
  map_id X := by
    apply ConcreteCategory.hom_ext
    intro q
    exact congrFun superlevel_componentMap_id q
  map_comp f g := by
    apply ConcreteCategory.hom_ext
    intro q
    exact congrFun (superlevel_componentMap_comp f.hom g.hom) q

variable {C : Type u} [Category.{v} C] [HasCoproducts.{w} C]

/-- Proves: the component-indexed coproduct construction is a functor.
Method: compose the actual component functor with the categorical coproduct functor.
Does not establish: an identification with homology until naturality is proved below. -/
def ordinaryH0_componentCoproductFunctor (R : C) : TopCat.{w} ⥤ C :=
  ordinaryH0_componentFunctor ⋙ sigmaConst.obj R

/-- Proves: a coproduct generator maps to the component of its continuous image.
Method: the universal coproduct map sends each inclusion to its indexed image.
Does not establish: that distinct source components stay distinct. -/
theorem ordinaryH0_componentCoproductMap_iota (R : C) {X Y : TopCat.{w}}
    (f : X ⟶ Y) (q : ZerothHomotopy X) :
    Sigma.ι (fun _ : ZerothHomotopy X => R) q ≫
        (ordinaryH0_componentCoproductFunctor R).map f =
      Sigma.ι (fun _ : ZerothHomotopy Y => R) (superlevel_componentMap f.hom q) := by
  simp [ordinaryH0_componentCoproductFunctor, ordinaryH0_componentFunctor, sigmaConst]

/-- Proves: the existing closed-superlevel family has a component-indexed coproduct filtration.
Method: compose its actual continuous inclusions with the proved component coproduct functor.
Does not establish: regularity, finite dimensionality, or Morse hypotheses for the height function. -/
def ordinaryH0_componentFiltration {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) : ℝ ⥤ C :=
  superlevel_filtration g ⋙ ordinaryH0_componentCoproductFunctor R

/-- Proves: a filtration component generator maps by the existing actual component inclusion.
Method: specialize the coproduct generator map formula to a closed-superlevel inclusion.
Does not establish: a reverse inclusion or endpoint deletion. -/
theorem ordinaryH0_componentFiltration_map_iota {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) {s t : ℝ} (hst : s ≤ t)
    (q : ZerothHomotopy (superlevel_set g s)) :
    Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g s) => R) q ≫
        (ordinaryH0_componentFiltration g R).map (homOfLE hst) =
      Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g t) => R)
        (superlevel_componentInclusion g hst q) := by
  exact ordinaryH0_componentCoproductMap_iota R
    (TopCat.ofHom (superlevel_inclusion g hst)) q

variable [Preadditive C] [CategoryWithHomology C]

/-- Proves: a zero-simplex defines a morphism into actual simplicial homology.
Method: include its chain generator, lift through the zero differential, then take its class.
Does not establish: a replacement definition of homology or any naturality premise. -/
def ordinaryH0_simplicialPointClass (S : SSet.{w}) (R : C) (x : S _⦋0⦌) :
    R ⟶ S.homology R 0 :=
  (S.chainComplex R).liftCycles (S.ιChainComplex x) 0 (by simp) (by simp) ≫
    (S.chainComplex R).homologyπ 0

/-- Proves: the existing simplicial H0 isomorphism sends a point class to its component.
Method: apply Mathlib's zero-cycle generator formula.
Does not establish: naturality merely from the existence of the objectwise isomorphism. -/
theorem ordinaryH0_simplicialPointClass_iso (S : SSet.{w}) (R : C)
    (x : S _⦋0⦌) :
    ordinaryH0_simplicialPointClass S R x ≫ (S.homology₀Iso R).hom =
      Sigma.ι (fun _ : S.π₀ => R) (SSet.π₀.mk x) := by
  simpa only [ordinaryH0_simplicialPointClass, Category.assoc] using
    SSet.liftCycles_ιChainComplex_homologyπ_homology₀Iso_hom S R x

/-- Proves: the actual simplicial homology map carries a point class to its image class.
Method: homology projection naturality, cycle-lift naturality, and the chain generator formula.
Does not establish: a transported or postulated substitute for the actual homology map. -/
theorem ordinaryH0_simplicialPointClass_map (R : C) {S T : SSet.{w}}
    (f : S ⟶ T) (x : S _⦋0⦌) :
    ordinaryH0_simplicialPointClass S R x ≫ SSet.homologyMap f R 0 =
      ordinaryH0_simplicialPointClass T R (f.app _ x) := by
  simp only [ordinaryH0_simplicialPointClass, SSet.homologyMap, Category.assoc,
    HomologicalComplex.homologyπ_naturality,
    HomologicalComplex.liftCycles_comp_cyclesMap_assoc, SSet.ι_chainComplexMap_f]

/-- Proves: the singular zero-simplex of a point maps to that of its continuous image.
Method: unfold the point/simplex correspondence and the singular-set map.
Does not establish: a chain or homology identification by itself. -/
theorem ordinaryH0_toSSet_pointMap {X Y : TopCat.{w}} (f : X ⟶ Y) (x : X) :
    (TopCat.toSSet.map f).app _ (TopCat.toSSetObj₀Equiv.symm x) =
      TopCat.toSSetObj₀Equiv.symm (f x) := by
  rfl

/-- Proves: each point has its actual ordinary singular zero-homology class morphism.
Method: use the actual singular zero-simplex and the cycle/projection construction above.
Does not establish: a free-module definition standing in for singular H0. -/
def ordinaryH0_pointClass (X : TopCat.{w}) (R : C) (x : X) :
    R ⟶ ((singularHomologyFunctor C 0).obj R).obj X :=
  ordinaryH0_simplicialPointClass (TopCat.toSSet.obj X) R
    (TopCat.toSSetObj₀Equiv.symm x)

/-- Proves: actual continuous-map-induced singular H0 maps transport point classes.
Method: specialize the proved simplicial generator naturality to the singular-set functor.
Does not establish: injectivity of the induced map. -/
theorem ordinaryH0_pointClass_map (R : C) {X Y : TopCat.{w}}
    (f : X ⟶ Y) (x : X) :
    ordinaryH0_pointClass X R x ≫ ((singularHomologyFunctor C 0).obj R).map f =
      ordinaryH0_pointClass Y R (f x) := by
  change ordinaryH0_simplicialPointClass (TopCat.toSSet.obj X) R
      (TopCat.toSSetObj₀Equiv.symm x) ≫
      SSet.homologyMap (TopCat.toSSet.map f) R 0 =
    ordinaryH0_simplicialPointClass (TopCat.toSSet.obj Y) R
      (TopCat.toSSetObj₀Equiv.symm (f x))
  rw [ordinaryH0_simplicialPointClass_map, ordinaryH0_toSSet_pointMap]

/-- Proves: Mathlib's existing topological comparison sends actual point classes to components.
Method: combine the simplicial generator formula with the actual path-component equivalence.
Does not establish: any connected-component substitution. -/
theorem ordinaryH0_pointClass_iso (X : TopCat.{w}) (R : C) (x : X) :
    ordinaryH0_pointClass X R x ≫ (TopCat.singularHomology₀Iso X R).hom =
      Sigma.ι (fun _ : ZerothHomotopy X => R) (ZerothHomotopy.mk x) := by
  change ordinaryH0_simplicialPointClass (TopCat.toSSet.obj X) R
      (TopCat.toSSetObj₀Equiv.symm x) ≫
      ((SSet.homology₀Iso (TopCat.toSSet.obj X) R).hom ≫
        (sigmaConst.obj R).map TopCat.zerothHomotopyEquiv.toIso.inv) = _
  rw [← Category.assoc, ordinaryH0_simplicialPointClass_iso]
  simp [sigmaConst]

/-- Proves: inverse comparison takes each component generator to any representing point class.
Method: cancel the comparison isomorphism and use the point-class formula.
Does not establish: a choice of distinguished geometric representative. -/
theorem ordinaryH0_componentIota_isoInv (X : TopCat.{w}) (R : C) (x : X) :
    Sigma.ι (fun _ : ZerothHomotopy X => R) (ZerothHomotopy.mk x) ≫
        (TopCat.singularHomology₀Iso X R).inv = ordinaryH0_pointClass X R x := by
  apply (cancel_mono (TopCat.singularHomology₀Iso X R).hom).1
  simp only [Category.assoc, Iso.inv_hom_id, Category.comp_id, ordinaryH0_pointClass_iso]

/-- Proves: morphisms out of actual singular H0 agree if they agree on all point classes.
Method: cancel the inverse comparison, use coproduct extensionality and quotient surjectivity.
Does not establish: a basis of geometric maxima or finiteness of components. -/
theorem ordinaryH0_hom_ext (X : TopCat.{w}) (R : C) {A : C}
    {f g : ((singularHomologyFunctor C 0).obj R).obj X ⟶ A}
    (h : ∀ x : X, ordinaryH0_pointClass X R x ≫ f = ordinaryH0_pointClass X R x ≫ g) :
    f = g := by
  apply (cancel_epi (TopCat.singularHomology₀Iso X R).inv).1
  apply Sigma.hom_ext
  intro q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  simpa only [← Category.assoc, ordinaryH0_componentIota_isoInv] using h x

/-- Proves: the existing actual singular H0 comparison commutes with every continuous map.
Method: test both composites on actual point classes, transport them, and compare component images.
Does not establish: naturality by adding an assumption or a compatibility record. -/
theorem ordinaryH0_iso_naturality (R : C) {X Y : TopCat.{w}} (f : X ⟶ Y) :
    ((singularHomologyFunctor C 0).obj R).map f ≫
        (TopCat.singularHomology₀Iso Y R).hom =
      (TopCat.singularHomology₀Iso X R).hom ≫
        (ordinaryH0_componentCoproductFunctor R).map f := by
  apply ordinaryH0_hom_ext X R
  intro x
  rw [← Category.assoc, ordinaryH0_pointClass_map, ordinaryH0_pointClass_iso]
  -- the right side composes through `(ordinaryH0_componentCoproductFunctor R).obj X`, which is
  -- `∐ fun _ => R` only after unfolding, so its reassociation is rewritten at default transparency
  erw [← Category.assoc, ordinaryH0_pointClass_iso, ordinaryH0_componentCoproductMap_iota]
  rfl

/-- Proves: actual ordinary singular H0 is naturally isomorphic to component-indexed coproducts.
Method: assemble the existing objectwise isomorphisms using the commuting square proved above.
Does not establish: an interval decomposition or persistence-bar classification. -/
def ordinaryH0_naturalIso (R : C) :
    (singularHomologyFunctor C 0).obj R ≅ ordinaryH0_componentCoproductFunctor R :=
  NatIso.ofComponents (fun X => TopCat.singularHomology₀Iso X R)
    (fun f => ordinaryH0_iso_naturality R f)

/-- Proves: ordinary augmentation sends every actual point-class morphism to the coefficient identity.
Method: use the comparison's augmentation formula and the coproduct universal property.
Does not establish: reduced-H0 vanishing; this retains the ordinary point class. -/
theorem ordinaryH0_pointClass_augmentation (X : TopCat.{w}) (R : C) (x : X) :
    ordinaryH0_pointClass X R x ≫ TopCat.singularHomology₀ε X R = 𝟙 R := by
  rw [← TopCat.singularHomology₀Iso_sigma_desc_id X R, ← Category.assoc,
    ordinaryH0_pointClass_iso]
  simp

/-- Proves: an actual point class is nonzero for a coefficient object with nonzero identity.
Method: compose a proposed zero class with ordinary augmentation.
Does not establish: nonzero classes for a zero coefficient object. -/
theorem ordinaryH0_pointClass_ne_zero (X : TopCat.{w}) (R : C)
    (hR : (𝟙 R : R ⟶ R) ≠ 0) (x : X) : ordinaryH0_pointClass X R x ≠ 0 := by
  intro h
  have hε := ordinaryH0_pointClass_augmentation X R x
  rw [h, zero_comp] at hε
  exact hR hε.symm

/-- Proves: points in the same actual path component give equal ordinary H0 class morphisms.
Method: cancel the objectwise comparison and compare their component indices.
Does not establish: a converse for zero coefficients. -/
theorem ordinaryH0_pointClass_eq_of_component_eq (X : TopCat.{w}) (R : C)
    (x y : X) (h : ZerothHomotopy.mk x = ZerothHomotopy.mk y) :
    ordinaryH0_pointClass X R x = ordinaryH0_pointClass X R y := by
  apply (cancel_mono (TopCat.singularHomology₀Iso X R).hom).1
  rw [ordinaryH0_pointClass_iso, ordinaryH0_pointClass_iso, h]

/-- Proves: the difference of same-component point classes is zero for arbitrary coefficients.
Method: subtract the equality proved from actual path-component equality.
Does not establish: that either individual point class dies. -/
theorem ordinaryH0_sameComponent_difference_zero (X : TopCat.{w}) (R : C)
    (x y : X) (h : ZerothHomotopy.mk x = ZerothHomotopy.mk y) :
    ordinaryH0_pointClass X R x - ordinaryH0_pointClass X R y = 0 := by
  rw [ordinaryH0_pointClass_eq_of_component_eq X R x y h, sub_self]

/-- Proves: distinct actual path components have distinct ordinary H0 class morphisms if coefficients are nonzero.
Method: separate the two coproduct injections by an identity/zero component projection.
Does not establish: distinction with zero coefficients or distinction of points within one component. -/
theorem ordinaryH0_pointClass_ne_of_component_ne (X : TopCat.{w}) (R : C)
    (hR : (𝟙 R : R ⟶ R) ≠ 0) (x y : X)
    (hxy : ZerothHomotopy.mk x ≠ ZerothHomotopy.mk y) :
    ordinaryH0_pointClass X R x ≠ ordinaryH0_pointClass X R y := by
  classical
  intro h
  have hc : Sigma.ι (fun _ : ZerothHomotopy X => R) (ZerothHomotopy.mk x) =
      Sigma.ι (fun _ : ZerothHomotopy X => R) (ZerothHomotopy.mk y) := by
    simpa only [ordinaryH0_pointClass_iso] using
      congrArg (fun k => k ≫ (TopCat.singularHomology₀Iso X R).hom) h
  let p : (∐ fun _ : ZerothHomotopy X => R) ⟶ R :=
    Sigma.desc (fun q => if q = ZerothHomotopy.mk x then 𝟙 R else 0)
  apply hR
  simpa [p, Ne.symm hxy] using congrArg (fun k => k ≫ p) hc

/-- Proves: with nonzero coefficients, actual point-class equality is exactly path-component equality.
Method: combine the same-component formula with the separating projection control.
Does not establish: equality of geometric points or connected components. -/
theorem ordinaryH0_pointClass_eq_iff (X : TopCat.{w}) (R : C)
    (hR : (𝟙 R : R ⟶ R) ≠ 0) (x y : X) :
    ordinaryH0_pointClass X R x = ordinaryH0_pointClass X R y ↔
      ZerothHomotopy.mk x = ZerothHomotopy.mk y := by
  constructor
  · intro h
    by_contra hn
    exact ordinaryH0_pointClass_ne_of_component_ne X R hR x y hn h
  · exact ordinaryH0_pointClass_eq_of_component_eq X R x y

/-- Proves: the actual H0 map sends a difference of point classes to the difference of image classes.
Method: distribute composition over subtraction and use proved actual point-class transport.
Does not establish: death of the individual ordinary point classes. -/
theorem ordinaryH0_difference_map (R : C) {X Y : TopCat.{w}} (f : X ⟶ Y) (x y : X) :
    (ordinaryH0_pointClass X R x - ordinaryH0_pointClass X R y) ≫
        ((singularHomologyFunctor C 0).obj R).map f =
      ordinaryH0_pointClass Y R (f x) - ordinaryH0_pointClass Y R (f y) := by
  rw [Preadditive.sub_comp, ordinaryH0_pointClass_map, ordinaryH0_pointClass_map]

/-- Proves: with nonzero coefficients, a mapped difference vanishes exactly when the image points merge in path components.
Method: actual homology-map transport, subtraction cancellation, and the component equality criterion.
Does not establish: a Morse merger event, elder selection, death height, or barcode. -/
theorem ordinaryH0_difference_map_eq_zero_iff (R : C) (hR : (𝟙 R : R ⟶ R) ≠ 0)
    {X Y : TopCat.{w}} (f : X ⟶ Y) (x y : X) :
    (ordinaryH0_pointClass X R x - ordinaryH0_pointClass X R y) ≫
        ((singularHomologyFunctor C 0).obj R).map f = 0 ↔
      ZerothHomotopy.mk (f x) = ZerothHomotopy.mk (f y) := by
  rw [ordinaryH0_difference_map, sub_eq_zero, ordinaryH0_pointClass_eq_iff Y R hR]

/-- Proves: the existing actual singular H0 filtration is naturally isomorphic to its component coproduct filtration.
Method: restrict the proved natural comparison along the existing topological filtration.
Does not establish: the later interval, elder, maximum/saddle, count, Field, or stochastic bridges. -/
def ordinaryH0_filtrationIso {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) :
    superlevel_h0Filtration g R ≅ ordinaryH0_componentFiltration g R :=
  Functor.isoWhiskerLeft (superlevel_filtration g) (ordinaryH0_naturalIso R)

/-- Proves: all ordered closed-superlevel inclusion maps satisfy the actual H0/component square.
Method: read the naturality of the proved filtration isomorphism at the preorder morphism.
Does not establish: strict-level conventions or any unproved geometric persistence hypothesis. -/
theorem ordinaryH0_filtration_naturality {X : Type w} [TopologicalSpace X]
    (g : X → ℝ) (R : C) {s t : ℝ} (hst : s ≤ t) :
    (superlevel_h0Filtration g R).map (homOfLE hst) ≫
        (ordinaryH0_filtrationIso g R).hom.app t =
      (ordinaryH0_filtrationIso g R).hom.app s ≫
        (ordinaryH0_componentFiltration g R).map (homOfLE hst) :=
  (ordinaryH0_filtrationIso g R).hom.naturality (homOfLE hst)

/-- Proves: under a closed-superlevel inclusion, a difference vanishes exactly when the two included points are joined in the target closed superlevel.
Method: apply the actual H0 difference control and the existing component/JoinedIn equivalence.
Does not establish: attainment of a critical death threshold or an elder/barcode theorem. -/
theorem ordinaryH0_filtration_difference_zero_iff_joinedIn
    {X : Type w} [TopologicalSpace X] (g : X → ℝ) (R : C)
    (hR : (𝟙 R : R ⟶ R) ≠ 0) {s t : ℝ} (hst : s ≤ t)
    (x y : superlevel_set g s) :
    (ordinaryH0_pointClass (TopCat.of (superlevel_set g s)) R x -
        ordinaryH0_pointClass (TopCat.of (superlevel_set g s)) R y) ≫
        (superlevel_h0Filtration g R).map (homOfLE hst) = 0 ↔
      JoinedIn (superlevel_set g t) x.val y.val := by
  exact (ordinaryH0_difference_map_eq_zero_iff R hR
    (TopCat.ofHom (superlevel_inclusion g hst)) x y).trans
      (superlevel_component_eq_iff_joinedIn g t
        (superlevel_inclusion g hst x) (superlevel_inclusion g hst y))

end TorusFiltration
