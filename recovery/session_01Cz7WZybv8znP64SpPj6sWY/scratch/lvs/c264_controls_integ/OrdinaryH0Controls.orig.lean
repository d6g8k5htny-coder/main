import Mathlib
import OrdinaryH0

/-!
# Instantiated ordinary-H0 controls for the C264 module

C264 owner task item 4 (Math- #428 comment 6113277485): concrete fixtures for the
CONTROL_OBLIGATIONS matrix quoted in the C264 nonauthor source review (Math- #428 comment
6113277173), exercised against the registered `OrdinaryH0` successor that this module
imports: SHA-256 3d6b0d6b53413c484b1d010bccf0d53ae7f75becb2d747868f22dbd673fe0a88,
19803 bytes, the owner repair of the handoff bytes SHA-256
261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e (18469 bytes). That
successor is FORMAL HOLD until a nonauthor exact-source review of the compiled successor;
registration is not review. The C264 nonauthor review is source-only: it binds the handoff
bytes, not the successor, and leaves formal elaboration pending. These fixtures are not a
review of `OrdinaryH0` and do not lift FORMAL HOLD.
Universe 0 throughout: spaces in `TopCat.{0}`, coefficients in `ModuleCat.{0} ℤ`.

Every premise is proved inside the fixture: the nonzero identities of `ℤ` and of `ZMod 2`,
the zero identity of the zero module, the path joining `0` and `1` in `ℝ`, the separation
of distinct points of a discrete space, the target-component equalities and separations
under the collapse `Bool → Unit` and the first projection `Bool × Bool → Bool`, and the
absence of a path through `0` in the closed superlevel `{x | 1/4 ≤ x ^ 2}`. No hypothesis,
compatibility field or axiom states a naturality, merger or separation fact.

The fixtures keep the death of a difference apart from the survival of each individual
point class: a merger kills the difference while each image class stays nonzero. One
non-injective map (the projection) kills one difference and keeps another, and a strict
closed-superlevel inclusion keeps a difference alive. With zero coefficients ordinary H0
of every space in `TopCat.{0}` is a zero object through the module's natural comparison,
so the two point classes of `Bool` coincide although their path components differ: the
nonzero-coefficient hypothesis of the separation statements cannot be dropped. With
`ZMod 2` coefficients, a characteristic-two object, the separation holds.

Author: Anthropic/Claude lane (companion owner), delegated AI work for Dylan Roy.
Compilation and kernel audit of these bytes are recorded only where a run is reported.
Organizational-independence credit 0. Scientific effect NONE.

These are instantiated controls of the categorical H0/path-component comparison only.
They supply no Morse chart, critical-level attainment theorem, interval decomposition,
elder correspondence, Field adapter, count identity, stochastic law, or scientific-status
promotion.
-/

noncomputable section

open CategoryTheory Limits AlgebraicTopology

namespace TorusFiltration

/-- Proves: nothing; it names the integer coefficient object `ℤ` as a `ℤ`-module at universe 0.
Method: Mathlib's `ModuleCat.of`.
Does not establish: any homology computation. -/
abbrev ordinaryH0Control_intCoeff : ModuleCat.{0} ℤ := ModuleCat.of ℤ ℤ

/-- Proves: nothing; it names the zero coefficient object, the one-point `ℤ`-module at universe 0.
Method: Mathlib's `ModuleCat.of` on `PUnit`.
Does not establish: any homology computation. -/
abbrev ordinaryH0Control_zeroCoeff : ModuleCat.{0} ℤ := ModuleCat.of ℤ PUnit.{1}

/-- Proves: nothing; it names the characteristic-two coefficient object `ZMod 2` as a `ℤ`-module at universe 0.
Method: Mathlib's `ModuleCat.of`.
Does not establish: any homology computation. -/
abbrev ordinaryH0Control_mod2Coeff : ModuleCat.{0} ℤ := ModuleCat.of ℤ (ZMod 2)

/-- Proves: the identity of the integer coefficient object is not the zero morphism.
Method: evaluate both morphisms at `1 : ℤ`, where they give `1` and `0`.
Does not establish: nonvanishing of any homology class by itself. -/
theorem ordinaryH0Control_intCoeff_id_ne_zero :
    (𝟙 ordinaryH0Control_intCoeff : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff) ≠ 0 := by
  intro h
  have h1 : (1 : ℤ) = 0 :=
    congrArg (fun f : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff => f.hom (1 : ℤ)) h
  exact one_ne_zero h1

/-- Proves: the identity of the `ZMod 2` coefficient object is not the zero morphism.
Method: evaluate both morphisms at `1 : ZMod 2`, where they give `1` and `0`, which differ by `decide`.
Does not establish: nonvanishing of any homology class by itself. -/
theorem ordinaryH0Control_mod2Coeff_id_ne_zero :
    (𝟙 ordinaryH0Control_mod2Coeff : ordinaryH0Control_mod2Coeff ⟶ ordinaryH0Control_mod2Coeff) ≠ 0 := by
  intro h
  have h1 : (1 : ZMod 2) = 0 :=
    congrArg (fun f : ordinaryH0Control_mod2Coeff ⟶ ordinaryH0Control_mod2Coeff => f.hom (1 : ZMod 2)) h
  exact absurd h1 (by decide)

/-- Proves: the `ZMod 2` coefficient object has characteristic two: its identity added to itself is zero.
Method: pointwise, `x + x = 0` in `ZMod 2` by Mathlib's characteristic-two identity.
Does not establish: any homology computation. -/
theorem ordinaryH0Control_mod2Coeff_id_add_self :
    (𝟙 ordinaryH0Control_mod2Coeff : ordinaryH0Control_mod2Coeff ⟶ ordinaryH0Control_mod2Coeff) +
      𝟙 ordinaryH0Control_mod2Coeff = 0 := by
  ext x
  change x + x = 0
  exact CharTwo.add_self_eq_zero x

/-- Proves: the one-point module is a zero object of `ModuleCat ℤ`.
Method: Mathlib's zero-object criterion for a subsingleton module.
Does not establish: vanishing of homology with nonzero coefficients. -/
theorem ordinaryH0Control_zeroCoeff_isZero : IsZero ordinaryH0Control_zeroCoeff :=
  ModuleCat.isZero_of_subsingleton _

/-- Proves: the identity of the zero coefficient object is the zero morphism.
Method: any two morphisms out of a zero object agree.
Does not establish: that the zero coefficient object satisfies the nonzero-coefficient hypothesis. -/
theorem ordinaryH0Control_zeroCoeff_id_eq_zero :
    (𝟙 ordinaryH0Control_zeroCoeff : ordinaryH0Control_zeroCoeff ⟶ ordinaryH0Control_zeroCoeff) = 0 :=
  ordinaryH0Control_zeroCoeff_isZero.eq_of_src _ _

/-- Proves: the empty space has no path component.
Method: every component is the class of a point, and the empty space has no point.
Does not establish: anything about a nonempty space. -/
theorem ordinaryH0Control_empty_components : IsEmpty (ZerothHomotopy (TopCat.of Empty)) := by
  constructor
  intro q
  obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
  exact Empty.elim x

/-- Proves: the component-indexed coproduct of the empty space is a zero object for every coefficient module.
Method: its identity and zero morphism agree on every (absent) coproduct inclusion.
Does not establish: vanishing for a nonempty space with nonzero coefficients. -/
theorem ordinaryH0Control_empty_componentCoproduct_isZero (R : ModuleCat.{0} ℤ) :
    IsZero ((ordinaryH0_componentCoproductFunctor R).obj (TopCat.of Empty)) := by
  rw [IsZero.iff_id_eq_zero]
  change 𝟙 (∐ fun _ : ZerothHomotopy (TopCat.of Empty) => R) = 0
  exact Sigma.hom_ext _ _ (fun q => (ordinaryH0Control_empty_components.false q).elim)

/-- Proves: actual ordinary singular H0 of the empty space with integer coefficients is a zero object.
Method: transport the empty-coproduct zero object along the proved natural comparison `ordinaryH0_naturalIso`.
Does not establish: H0 of the empty space for coefficients other than `ℤ` (the zero-module case is `ordinaryH0Control_zeroCoeff_h0_isZero`), or anything about a nonempty space; no nonempty or `PathConnectedSpace` hypothesis is used. -/
theorem ordinaryH0Control_empty_h0_isZero :
    IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).obj
      (TopCat.of Empty)) :=
  (ordinaryH0Control_empty_componentCoproduct_isZero ordinaryH0Control_intCoeff).of_iso
    ((ordinaryH0_naturalIso ordinaryH0Control_intCoeff).app (TopCat.of Empty))

/-- Proves: with zero coefficients the component-indexed coproduct of every space is a zero object.
Method: its identity and zero morphism agree on every coproduct inclusion, each out of the zero module.
Does not establish: vanishing with nonzero coefficients. -/
theorem ordinaryH0Control_zeroCoeff_componentCoproduct_isZero (X : TopCat.{0}) :
    IsZero ((ordinaryH0_componentCoproductFunctor ordinaryH0Control_zeroCoeff).obj X) := by
  rw [IsZero.iff_id_eq_zero]
  change 𝟙 (∐ fun _ : ZerothHomotopy X => ordinaryH0Control_zeroCoeff) = 0
  exact Sigma.hom_ext _ _ (fun _ => ordinaryH0Control_zeroCoeff_isZero.eq_of_src _ _)

/-- Proves: with zero coefficients actual ordinary singular H0 of every space is a zero object.
Method: transport along the module's natural comparison `ordinaryH0_naturalIso` instantiated at the zero module.
Does not establish: any statement for nonzero coefficients, or that component distinctions are visible with zero coefficients. -/
theorem ordinaryH0Control_zeroCoeff_h0_isZero (X : TopCat.{0}) :
    IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_zeroCoeff).obj X) :=
  (ordinaryH0Control_zeroCoeff_componentCoproduct_isZero X).of_iso
    ((ordinaryH0_naturalIso ordinaryH0Control_zeroCoeff).app X)

/-- Proves: the point class of the one-point space with integer coefficients augments to the identity of `ℤ`.
Method: specialize the module's ordinary augmentation formula.
Does not establish: reduced-H0 vanishing; this is the ordinary point class. -/
theorem ordinaryH0Control_point_augmentation :
    ordinaryH0_pointClass (TopCat.of Unit) ordinaryH0Control_intCoeff () ≫
        TopCat.singularHomology₀ε (TopCat.of Unit) ordinaryH0Control_intCoeff =
      𝟙 ordinaryH0Control_intCoeff :=
  ordinaryH0_pointClass_augmentation _ _ _

/-- Proves: the point class of the one-point space with integer coefficients is nonzero.
Method: the module's nonvanishing theorem with the proved nonzero identity of `ℤ`.
Does not establish: nonvanishing with zero coefficients. -/
theorem ordinaryH0Control_point_ne_zero :
    ordinaryH0_pointClass (TopCat.of Unit) ordinaryH0Control_intCoeff () ≠ 0 :=
  ordinaryH0_pointClass_ne_zero (TopCat.of Unit) ordinaryH0Control_intCoeff
    ordinaryH0Control_intCoeff_id_ne_zero ()

/-- Proves: actual ordinary singular H0 of the one-point space with integer coefficients is not a zero object.
Method: a zero target would force the nonzero point class to vanish.
Does not establish: a computation of that H0 beyond its nonvanishing. -/
theorem ordinaryH0Control_point_h0_not_isZero :
    ¬ IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).obj
      (TopCat.of Unit)) :=
  fun h => ordinaryH0Control_point_ne_zero (h.eq_of_tgt _ _)

/-- Proves: `0` and `1` lie in the same actual path component of the real line.
Method: the explicit path `t ↦ t` along the unit interval.
Does not establish: path-connectedness of any closed superlevel. -/
theorem ordinaryH0Control_line_component_eq :
    ZerothHomotopy.mk (0 : ℝ) = ZerothHomotopy.mk (1 : ℝ) :=
  ZerothHomotopy.sound
    { toFun := fun t => (t : ℝ)
      continuous_toFun := continuous_subtype_val
      source' := rfl
      target' := rfl }

/-- Proves: with integer coefficients the point classes of `0` and `1` in the real line have zero difference.
Method: the module's same-component theorem with the proved path premise.
Does not establish: that either individual point class dies. -/
theorem ordinaryH0Control_line_difference_zero_int :
    ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_intCoeff (0 : ℝ) -
        ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_intCoeff (1 : ℝ) = 0 :=
  ordinaryH0_sameComponent_difference_zero _ _ _ _ ordinaryH0Control_line_component_eq

/-- Proves: with zero coefficients the point classes of `0` and `1` in the real line have zero difference.
Method: the module's same-component theorem, which carries no coefficient hypothesis.
Does not establish: any separation statement for zero coefficients. -/
theorem ordinaryH0Control_line_difference_zero_zeroCoeff :
    ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_zeroCoeff (0 : ℝ) -
        ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_zeroCoeff (1 : ℝ) = 0 :=
  ordinaryH0_sameComponent_difference_zero _ _ _ _ ordinaryH0Control_line_component_eq

/-- Proves: with integer coefficients each of the two same-component point classes in the real line is nonzero.
Method: the module's nonvanishing theorem with the proved nonzero identity of `ℤ`.
Does not establish: distinctness of the two classes, which are equal. -/
theorem ordinaryH0Control_line_classes_ne_zero :
    ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_intCoeff (0 : ℝ) ≠ 0 ∧
      ordinaryH0_pointClass (TopCat.of ℝ) ordinaryH0Control_intCoeff (1 : ℝ) ≠ 0 :=
  ⟨ordinaryH0_pointClass_ne_zero _ _ ordinaryH0Control_intCoeff_id_ne_zero _,
    ordinaryH0_pointClass_ne_zero _ _ ordinaryH0Control_intCoeff_id_ne_zero _⟩

/-- Proves: distinct points of a discrete space lie in distinct actual path components.
Method: a path is a continuous map from the connected unit interval into a discrete space, hence constant.
Does not establish: anything about path components of a non-discrete space. -/
theorem ordinaryH0Control_discrete_component_ne {X : Type} [TopologicalSpace X] [DiscreteTopology X]
    {x y : X} (hxy : x ≠ y) : ZerothHomotopy.mk x ≠ ZerothHomotopy.mk y := by
  intro h
  obtain ⟨p⟩ : Joined x y := Quotient.exact h
  have hc : p 0 = p 1 := PreconnectedSpace.constant inferInstance p.continuous
  rw [p.source, p.target] at hc
  exact hxy hc

/-- Proves: `true` and `false` lie in distinct actual path components of the discrete space `Bool`.
Method: the discrete-space separation above with `true ≠ false` by `decide`.
Does not establish: anything about connected components of other spaces. -/
theorem ordinaryH0Control_bool_component_ne :
    ZerothHomotopy.mk true ≠ ZerothHomotopy.mk false :=
  ordinaryH0Control_discrete_component_ne (by decide)

/-- Proves: with integer coefficients the point classes of `true` and `false` in `Bool` differ.
Method: the module's separation theorem with the two proved premises.
Does not establish: separation with zero coefficients. -/
theorem ordinaryH0Control_bool_pointClass_ne :
    ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true ≠
      ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false :=
  ordinaryH0_pointClass_ne_of_component_ne _ _ ordinaryH0Control_intCoeff_id_ne_zero _ _
    ordinaryH0Control_bool_component_ne

/-- Proves: with `ZMod 2` coefficients, a characteristic-two object, the point classes of `true` and `false` in `Bool` differ.
Method: the module's separation theorem with the proved nonzero identity of `ZMod 2` and the proved component separation.
Does not establish: separation for every characteristic-two coefficient object, or with zero coefficients. -/
theorem ordinaryH0Control_bool_pointClass_ne_mod2 :
    ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_mod2Coeff true ≠
      ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_mod2Coeff false :=
  ordinaryH0_pointClass_ne_of_component_ne _ _ ordinaryH0Control_mod2Coeff_id_ne_zero _ _
    ordinaryH0Control_bool_component_ne

/-- Proves: nothing; it names the continuous collapse of `Bool` onto the one-point space.
Method: Mathlib's constant continuous map.
Does not establish: any homology statement by itself. -/
def ordinaryH0Control_collapse : TopCat.of Bool ⟶ TopCat.of Unit :=
  TopCat.ofHom (ContinuousMap.const Bool ())

/-- Proves: with integer coefficients the collapse kills the difference of the two point classes of `Bool`.
Method: the module's merger criterion with the proved target-component equality in the one-point space.
Does not establish: that an individual point class dies. -/
theorem ordinaryH0Control_collapse_difference_zero :
    (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true -
        ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_collapse = 0 :=
  (ordinaryH0_difference_map_eq_zero_iff _ ordinaryH0Control_intCoeff_id_ne_zero _ _ _).2 rfl

/-- Proves: with integer coefficients each image point class under the collapse is nonzero.
Method: actual point-class transport followed by the module's nonvanishing theorem.
Does not establish: distinctness of the two image classes, which are equal. -/
theorem ordinaryH0Control_collapse_image_ne_zero (b : Bool) :
    ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff b ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_collapse ≠ 0 := by
  rw [ordinaryH0_pointClass_map]
  exact ordinaryH0_pointClass_ne_zero _ _ ordinaryH0Control_intCoeff_id_ne_zero _

/-- Proves: a merger kills the difference of the two point classes of `Bool` while each image class survives.
Method: combine the two collapse controls above.
Does not establish: a Morse merger event, elder selection or death height. -/
theorem ordinaryH0Control_collapse_difference_dies_classes_survive :
    (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true -
        ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_collapse = 0 ∧
      ∀ b : Bool, ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff b ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_collapse ≠ 0 :=
  ⟨ordinaryH0Control_collapse_difference_zero, ordinaryH0Control_collapse_image_ne_zero⟩

/-- Proves: nothing; it names the continuous first projection of the discrete square `Bool × Bool` onto `Bool`, a non-injective map.
Method: Mathlib's continuous first projection.
Does not establish: any homology statement by itself. -/
def ordinaryH0Control_fst : TopCat.of (Bool × Bool) ⟶ TopCat.of Bool :=
  TopCat.ofHom ⟨Prod.fst, continuous_fst⟩

/-- Proves: with integer coefficients, before any map, the differences of the classes of `(true, false)` and `(true, true)`, and of `(true, false)` and `(false, true)`, in the discrete square are both nonzero.
Method: the module's separation theorem with the proved discrete-space component separation, then `sub_ne_zero`.
Does not establish: any statement about their images. -/
theorem ordinaryH0Control_square_differences_ne_zero :
    ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) -
        ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, true) : Bool × Bool) ≠ 0 ∧
      ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) -
        ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool) ≠ 0 :=
  ⟨sub_ne_zero.mpr (ordinaryH0_pointClass_ne_of_component_ne _ _ ordinaryH0Control_intCoeff_id_ne_zero _ _
      (ordinaryH0Control_discrete_component_ne (X := Bool × Bool) (by decide))),
    sub_ne_zero.mpr (ordinaryH0_pointClass_ne_of_component_ne _ _ ordinaryH0Control_intCoeff_id_ne_zero _ _
      (ordinaryH0Control_discrete_component_ne (X := Bool × Bool) (by decide)))⟩

/-- Proves: with integer coefficients the projection kills the difference of the classes of `(true, false)` and `(true, true)`, whose images coincide.
Method: the module's merger criterion with the proved target-component equality.
Does not establish: that an individual point class dies. -/
theorem ordinaryH0Control_fst_difference_zero :
    (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) -
        ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, true) : Bool × Bool)) ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_fst = 0 :=
  (ordinaryH0_difference_map_eq_zero_iff _ ordinaryH0Control_intCoeff_id_ne_zero _ _ _).2 rfl

/-- Proves: with integer coefficients the same non-injective projection keeps the difference of the classes of `(true, false)` and `(false, true)` nonzero, since their images `true` and `false` stay in distinct components.
Method: the module's merger criterion, used contrapositively with the proved separation of the image components.
Does not establish: injectivity of the induced H0 map, which fails on the first pair. -/
theorem ordinaryH0Control_fst_difference_ne_zero :
    (ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((true, false) : Bool × Bool) -
        ordinaryH0_pointClass (TopCat.of (Bool × Bool)) ordinaryH0Control_intCoeff ((false, true) : Bool × Bool)) ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_fst ≠ 0 := by
  rw [Ne, ordinaryH0_difference_map_eq_zero_iff _ ordinaryH0Control_intCoeff_id_ne_zero]
  exact ordinaryH0Control_bool_component_ne

/-- Proves: with zero coefficients the point classes of `true` and `false` in `Bool` are equal.
Method: any two morphisms out of the zero coefficient object agree.
Does not establish: equality of their path components, which differ. -/
theorem ordinaryH0Control_bool_zeroCoeff_eq :
    ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true =
      ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false :=
  ordinaryH0Control_zeroCoeff_isZero.eq_of_src _ _

/-- Proves: with zero coefficients the point class of the one-point space is zero.
Method: any two morphisms out of the zero coefficient object agree.
Does not establish: vanishing of ordinary H0 with nonzero coefficients. -/
theorem ordinaryH0Control_point_zeroCoeff_eq_zero :
    ordinaryH0_pointClass (TopCat.of Unit) ordinaryH0Control_zeroCoeff () = 0 :=
  ordinaryH0Control_zeroCoeff_isZero.eq_of_src _ _

/-- Proves: the separation statement is false once its nonzero-coefficient hypothesis is deleted.
Method: the zero-coefficient fixture on `Bool` has distinct components and equal point classes.
Does not establish: any failure of the module's statements, which retain that hypothesis. -/
theorem ordinaryH0Control_separation_needs_nonzero :
    ¬ ∀ (X : TopCat.{0}) (R : ModuleCat.{0} ℤ) (x y : X),
      ZerothHomotopy.mk x ≠ ZerothHomotopy.mk y →
        ordinaryH0_pointClass X R x ≠ ordinaryH0_pointClass X R y :=
  fun h => h (TopCat.of Bool) ordinaryH0Control_zeroCoeff true false
    ordinaryH0Control_bool_component_ne ordinaryH0Control_bool_zeroCoeff_eq

/-- Proves: the nonvanishing statement is false once its nonzero-coefficient hypothesis is deleted.
Method: the zero-coefficient fixture on the one-point space has a zero point class.
Does not establish: any failure of the module's statements, which retain that hypothesis. -/
theorem ordinaryH0Control_nonvanishing_needs_nonzero :
    ¬ ∀ (X : TopCat.{0}) (R : ModuleCat.{0} ℤ) (x : X), ordinaryH0_pointClass X R x ≠ 0 :=
  fun h => h (TopCat.of Unit) ordinaryH0Control_zeroCoeff () ordinaryH0Control_point_zeroCoeff_eq_zero

/-- Proves: nothing; it names the point `-1` of the closed superlevel of `x ↦ x ^ 2` at parameter `-1`.
Method: membership is `-(-1) ≤ (-1) ^ 2`, checked by `norm_num`.
Does not establish: any component statement by itself. -/
def ordinaryH0Control_parabolaLeft : superlevel_set (fun x : ℝ => x ^ 2) (-1) :=
  ⟨-1, by norm_num [superlevel_set]⟩

/-- Proves: nothing; it names the point `1` of the closed superlevel of `x ↦ x ^ 2` at parameter `-1`.
Method: membership is `-(-1) ≤ 1 ^ 2`, checked by `norm_num`.
Does not establish: any component statement by itself. -/
def ordinaryH0Control_parabolaRight : superlevel_set (fun x : ℝ => x ^ 2) (-1) :=
  ⟨1, by norm_num [superlevel_set]⟩

/-- Proves: the closed-superlevel parameter `-1` lies below `-1/4`, so the inclusion of the level at `-1` into the level at `-1/4` exists.
Method: `norm_num`.
Does not establish: any component or homology statement by itself. -/
theorem ordinaryH0Control_parabola_le : (-1 : ℝ) ≤ -1 / 4 := by
  norm_num

/-- Proves: the inclusion from parameter `-1` to `-1/4` is strict: `-1 < -1/4`, and `1/2` lies in the level `{x | 1/4 ≤ x ^ 2}` but not in `{x | 1 ≤ x ^ 2}`.
Method: `norm_num` on the three numeric facts.
Does not establish: any component or homology statement by itself. -/
theorem ordinaryH0Control_parabola_strict_growth :
    (-1 : ℝ) < -1 / 4 ∧ (1 / 2 : ℝ) ∈ superlevel_set (fun x : ℝ => x ^ 2) (-1 / 4) ∧
      (1 / 2 : ℝ) ∉ superlevel_set (fun x : ℝ => x ^ 2) (-1) := by
  refine ⟨by norm_num, ?_, ?_⟩ <;> norm_num [superlevel_set]

/-- Proves: `-1` and `1` are not joined inside the closed superlevel `{x | 1/4 ≤ x ^ 2}` of `x ↦ x ^ 2` at parameter `-1/4`.
Method: intermediate value theorem; a joining path would pass through `0`, which lies outside that level.
Does not establish: a first merger level or threshold attainment. -/
theorem ordinaryH0Control_parabola_not_joined_quarter :
    ¬ JoinedIn (superlevel_set (fun x : ℝ => x ^ 2) (-1 / 4))
      ordinaryH0Control_parabolaLeft.val ordinaryH0Control_parabolaRight.val := by
  rintro ⟨γ, hγ⟩
  obtain ⟨t, ht⟩ : (0 : ℝ) ∈ Set.range γ :=
    intermediate_value_univ 0 1 γ.continuous
      ⟨by norm_num [ordinaryH0Control_parabolaLeft], by norm_num [ordinaryH0Control_parabolaRight]⟩
  have hmem := hγ t
  rw [ht] at hmem
  norm_num [superlevel_set] at hmem

/-- Proves: `-1` and `1` are joined inside the closed superlevel of `x ↦ x ^ 2` at parameter `0`, which is the whole line.
Method: the straight segment lies in that level because squares are nonnegative.
Does not establish: path-connectedness of any other level. -/
theorem ordinaryH0Control_parabola_joined :
    JoinedIn (superlevel_set (fun x : ℝ => x ^ 2) 0)
      ordinaryH0Control_parabolaLeft.val ordinaryH0Control_parabolaRight.val :=
  JoinedIn.of_segment_subset (by
    intro z _
    show -(0 : ℝ) ≤ z ^ 2
    rw [neg_zero]
    exact sq_nonneg z)

/-- Proves: with integer coefficients the closed-superlevel inclusion from parameter `-1` to `0` kills the difference of the classes of `-1` and `1`.
Method: the module's filtration criterion with the proved target-level path.
Does not establish: that either included point class dies, or that `0` is the first merger level. -/
theorem ordinaryH0Control_parabola_difference_zero :
    (ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaLeft -
        ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaRight) ≫
        (superlevel_h0Filtration (fun x : ℝ => x ^ 2) ordinaryH0Control_intCoeff).map
          (homOfLE (neg_nonpos.mpr zero_le_one : (-1 : ℝ) ≤ 0)) = 0 :=
  (ordinaryH0_filtration_difference_zero_iff_joinedIn _ _ ordinaryH0Control_intCoeff_id_ne_zero
    _ _ _).2 ordinaryH0Control_parabola_joined

/-- Proves: with integer coefficients the strict closed-superlevel inclusion from parameter `-1` to `-1/4` keeps the difference of the classes of `-1` and `1` nonzero.
Method: the module's filtration criterion, used contrapositively with the proved absence of a path in the target level.
Does not establish: the behaviour at any parameter strictly between `-1/4` and `0`, or the first merger level. -/
theorem ordinaryH0Control_parabola_strict_difference_ne_zero :
    (ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaLeft -
        ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaRight) ≫
        (superlevel_h0Filtration (fun x : ℝ => x ^ 2) ordinaryH0Control_intCoeff).map
          (homOfLE ordinaryH0Control_parabola_le) ≠ 0 := fun h =>
  ordinaryH0Control_parabola_not_joined_quarter
    ((ordinaryH0_filtration_difference_zero_iff_joinedIn _ _ ordinaryH0Control_intCoeff_id_ne_zero
      _ _ _).1 h)

end TorusFiltration
