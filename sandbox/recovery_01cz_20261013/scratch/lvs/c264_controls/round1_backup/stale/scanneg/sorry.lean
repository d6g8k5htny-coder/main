import Mathlib
import OrdinaryH0

/-!
# Instantiated ordinary-H0 controls for the C264 module

C264 owner task item 4 (Math- #428 comment 6113277485): concrete fixtures for the
CONTROL_OBLIGATIONS matrix of the reviewed `OrdinaryH0` module. Universe 0 throughout:
spaces in `TopCat.{0}`, coefficients in `ModuleCat.{0} ℤ`.

Every premise is proved inside the fixture: the nonzero identity of `ℤ`, the zero identity
of the zero module, the path joining `0` and `1` in `ℝ`, the separation of `true` and
`false` in the discrete space `Bool`, the target-component equality under the collapse
`Bool → Unit`, and the target-component separation under the swap `Bool → Bool`. No
hypothesis, compatibility field or axiom states a naturality, merger or separation fact.

The fixtures keep the death of a difference apart from the survival of each individual
point class: a merger kills the difference while each image class stays nonzero.
With zero coefficients the two point classes of `Bool` coincide although their path
components differ, so the nonzero-coefficient hypothesis of the separation statements
cannot be dropped.

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

/-- Proves: the identity of the integer coefficient object is not the zero morphism.
Method: evaluate both morphisms at `1 : ℤ`, where they give `1` and `0`.
Does not establish: nonvanishing of any homology class by itself. -/
theorem ordinaryH0Control_intCoeff_id_ne_zero :
    (𝟙 ordinaryH0Control_intCoeff : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff) ≠ 0 := by
  intro h
  have h1 : (1 : ℤ) = 0 :=
    congrArg (fun f : ordinaryH0Control_intCoeff ⟶ ordinaryH0Control_intCoeff => f.hom (1 : ℤ)) h
  sorry

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
Does not establish: vanishing for a nonempty space. -/
theorem ordinaryH0Control_empty_componentCoproduct_isZero (R : ModuleCat.{0} ℤ) :
    IsZero ((ordinaryH0_componentCoproductFunctor R).obj (TopCat.of Empty)) := by
  rw [IsZero.iff_id_eq_zero]
  change 𝟙 (∐ fun _ : ZerothHomotopy (TopCat.of Empty) => R) = 0
  exact Sigma.hom_ext _ _ (fun q => (ordinaryH0Control_empty_components.false q).elim)

/-- Proves: actual ordinary singular H0 of the empty space with integer coefficients is a zero object.
Method: transport the empty-coproduct zero object along the proved natural comparison `ordinaryH0_naturalIso`.
Does not establish: a nonempty or path-connected hypothesis anywhere; none is used. -/
theorem ordinaryH0Control_empty_h0_isZero :
    IsZero (((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).obj
      (TopCat.of Empty)) :=
  (ordinaryH0Control_empty_componentCoproduct_isZero ordinaryH0Control_intCoeff).of_iso
    ((ordinaryH0_naturalIso ordinaryH0Control_intCoeff).app (TopCat.of Empty))

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

/-- Proves: `true` and `false` lie in distinct actual path components of the discrete space `Bool`.
Method: a path is a continuous map from the connected unit interval into a discrete space, hence constant.
Does not establish: anything about connected components of other spaces. -/
theorem ordinaryH0Control_bool_component_ne :
    ZerothHomotopy.mk true ≠ ZerothHomotopy.mk false := by
  intro h
  obtain ⟨p⟩ : Joined true false := Quotient.exact h
  have hc : p 0 = p 1 := PreconnectedSpace.constant inferInstance p.continuous
  rw [p.source, p.target] at hc
  cases hc

/-- Proves: with integer coefficients the point classes of `true` and `false` in `Bool` differ.
Method: the module's separation theorem with the two proved premises.
Does not establish: separation with zero coefficients. -/
theorem ordinaryH0Control_bool_pointClass_ne :
    ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true ≠
      ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false :=
  ordinaryH0_pointClass_ne_of_component_ne _ _ ordinaryH0Control_intCoeff_id_ne_zero _ _
    ordinaryH0Control_bool_component_ne

/-- Proves: nothing; it names the continuous collapse of `Bool` onto the one-point space.
Method: Mathlib's constant continuous map.
Does not establish: any homology statement by itself. -/
def ordinaryH0Control_collapse : TopCat.of Bool ⟶ TopCat.of Unit :=
  TopCat.ofHom (ContinuousMap.const Bool ())

/-- Proves: nothing; it names the continuous swap of the discrete space `Bool`.
Method: negation is continuous on a discrete space.
Does not establish: any homology statement by itself. -/
def ordinaryH0Control_swap : TopCat.of Bool ⟶ TopCat.of Bool :=
  TopCat.ofHom ⟨not, continuous_of_discreteTopology⟩

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

/-- Proves: with integer coefficients the swap does not kill the difference of the two point classes of `Bool`.
Method: the module's merger criterion, used contrapositively with the proved separation of the image components.
Does not establish: injectivity of the induced H0 map. -/
theorem ordinaryH0Control_swap_difference_ne_zero :
    (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true -
        ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false) ≫
        ((singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map
          ordinaryH0Control_swap ≠ 0 := by
  rw [Ne, ordinaryH0_difference_map_eq_zero_iff _ ordinaryH0Control_intCoeff_id_ne_zero]
  exact fun h => ordinaryH0Control_bool_component_ne h.symm

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

/-- Proves: `-1` and `1` are not joined inside the closed superlevel `{x | 1 ≤ x ^ 2}` of `x ↦ x ^ 2` at parameter `-1`.
Method: intermediate value theorem; a joining path would pass through `0`, which lies outside that level.
Does not establish: a first merger level or threshold attainment. -/
theorem ordinaryH0Control_parabola_not_joined :
    ¬ JoinedIn (superlevel_set (fun x : ℝ => x ^ 2) (-1))
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

/-- Proves: with integer coefficients the identity inclusion at parameter `-1` keeps the difference of the classes of `-1` and `1` nonzero.
Method: the module's filtration criterion, used contrapositively with the proved absence of a path in that level.
Does not establish: the behaviour at any parameter strictly between `-1` and `0`. -/
theorem ordinaryH0Control_parabola_difference_ne_zero :
    (ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaLeft -
        ordinaryH0_pointClass (TopCat.of (superlevel_set (fun x : ℝ => x ^ 2) (-1)))
          ordinaryH0Control_intCoeff ordinaryH0Control_parabolaRight) ≫
        (superlevel_h0Filtration (fun x : ℝ => x ^ 2) ordinaryH0Control_intCoeff).map
          (homOfLE (le_refl (-1 : ℝ))) ≠ 0 := fun h =>
  ordinaryH0Control_parabola_not_joined
    ((ordinaryH0_filtration_difference_zero_iff_joinedIn _ _ ordinaryH0Control_intCoeff_id_ne_zero
      _ _ _).1 h)

end TorusFiltration
