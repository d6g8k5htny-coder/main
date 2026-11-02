import ConeFootprint
import Mathlib.Topology.Instances.AddCircle.Defs
import Mathlib.Topology.OpenPartialHomeomorph.Composition
import Mathlib.Algebra.Field.ZMod

noncomputable section
open Set MorseCone CategoryTheory
namespace MorseConeTests
abbrev Z := EuclideanSpace ℝ (Fin 0)

private def identityForm (E F : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] : NormalForm (E × F) E F where
  f := quadratic
  p := 0
  c := 0
  ρ := 1
  hρ := by norm_num
  chart := OpenPartialHomeomorph.refl (E × F)
  center_mem := by trivial
  center_eq := rfl
  ball_target := by intro x hx; trivial
  equation := by intro x hx; simp

example : IsPathConnected (cone (E := EuclideanSpace ℝ (Fin 3)) (F := EuclideanSpace ℝ (Fin 2)) 1) :=
  cone_pathConnected 1 (by norm_num)
example : cone (E := Z) (F := ℝ) 1 = {0} := cone_zero_positive 1 (by norm_num)
example : puncture (E := Z) (F := ℝ) 1 = ∅ := by
  simp [puncture,cone_zero_positive 1 (by norm_num : (0:ℝ)<1)]
example : IsPathConnected (positiveLobe (F := EuclideanSpace ℝ (Fin 4)) 1) :=
  one_positive_lobe 1 (by norm_num)
example : IsPathConnected (negativeLobe (F := EuclideanSpace ℝ (Fin 4)) 1) :=
  one_negative_lobe 1 (by norm_num)
example : ¬ JoinedIn (puncture (F := ℝ) 1) ((1/2:ℝ),0) ((-1/2:ℝ),0) := by
  apply one_lobes_separated
  · norm_num [positiveLobe,cone,radiusBall,Real.norm_eq_abs]
  · norm_num [negativeLobe,cone,radiusBall,Real.norm_eq_abs]
example : IsPathConnected (puncture (E := EuclideanSpace ℝ (Fin 2)) (F := ℝ) 1) := by
  apply higher_puncture_pathConnected 1 (by norm_num)
  rw [← Module.finrank_eq_rank',finrank_euclideanSpace_fin]
  norm_num

-- The sum-of-squares radius is not the product sup-norm ball.
example : ‖((4/5:ℝ),(4/5:ℝ))‖ < 1 ∧
    ((4/5:ℝ),(4/5:ℝ)) ∉ radiusBall 1 := by
  norm_num [Prod.norm_def,radiusBall,Real.norm_eq_abs]
-- Equality Q=0 belongs to the actual CLOSED cone.
example : ((1/3:ℝ),(1/3:ℝ)) ∈ cone 1 := by
  norm_num [cone,radiusBall,Real.norm_eq_abs]
example : cone (E := ℝ) (F := ℝ) 0 = ∅ := by
  ext x
  constructor
  · intro hx
    have hh : ‖x.1‖^2+‖x.2‖^2<(0:ℝ)^2 := hx.2
    nlinarith [sq_nonneg ‖x.1‖,sq_nonneg ‖x.2‖]
  · intro hx; exact hx.elim

-- Literal local normal-form chart; no component conclusion is supplied.
example : oldSet (identityForm Z ℝ) ∪ patchSet (identityForm Z ℝ) = univ :=
  critical_cover _
example : ∀ a, ¬ MorseOpenGluing.Foot (oldSet (identityForm Z ℝ))
    (patchSet (identityForm Z ℝ)) a := zero_foot _
example : ElementaryHistory.Event (MorseOpenGluing.inclusionMap (oldSet (identityForm Z ℝ))) :=
  zero_event _
example : ∃ a b, ∀ q, MorseOpenGluing.Foot (oldSet (identityForm ℝ ℝ))
    (patchSet (identityForm ℝ ℝ)) q ↔ q=a ∨ q=b := one_foot _
example : ElementaryHistory.Event (MorseOpenGluing.inclusionMap (oldSet (identityForm ℝ ℝ))) :=
  one_event _
example : ElementaryHistory.Event
    (MorseOpenGluing.inclusionMap (oldSet (identityForm (EuclideanSpace ℝ (Fin 2)) ℝ))) := by
  apply higher_event
  rw [← Module.finrank_eq_rank',finrank_euclideanSpace_fin]
  norm_num

-- Opposite literal old components of the identity normal form genuinely MERGE.
private def linePlus : oldSet (identityForm ℝ ℝ) :=
  ⟨⟨((1/2:ℝ),0),by norm_num [closedLevel,identityForm,quadratic,Real.norm_eq_abs]⟩,
    by norm_num [oldSet,identityForm,Prod.ext_iff]⟩
private def lineMinus : oldSet (identityForm ℝ ℝ) :=
  ⟨⟨((-1/2:ℝ),0),by norm_num [closedLevel,identityForm,quadratic,Real.norm_eq_abs]⟩,
    by norm_num [oldSet,identityForm,Prod.ext_iff]⟩
private theorem line_distinct : ZerothHomotopy.mk linePlus ≠ ZerothHomotopy.mk lineMinus := by
  intro heq
  have hj : Joined linePlus lineMinus := Quotient.exact heq
  let γ := hj.somePath
  have hc : Continuous (fun t : ℝ => (γ.extend t).val.val.1) :=
    continuous_fst.comp (continuous_subtype_val.comp (continuous_subtype_val.comp γ.continuous_extend))
  have hi := isPreconnected_Icc.intermediate_value
    (show (1:ℝ)∈Icc 0 1 by simp) (show (0:ℝ)∈Icc 0 1 by simp) hc.continuousOn
  have hm : (0:ℝ)∈Icc (γ.extend 1).val.val.1 (γ.extend 0).val.val.1 := by
    norm_num [γ,linePlus,lineMinus]
  obtain ⟨t,ht,he⟩ := hi hm
  have hh : (0:ℝ) ≤ ‖(γ.extend t).val.val.1‖^2 - ‖(γ.extend t).val.val.2‖^2 :=
    (γ.extend t).val.property
  have hz : (γ.extend t).val.val.2=0 := by
    have hn : ‖(γ.extend t).val.val.2‖^2=0 := by
      simp only [he,norm_zero,zero_pow (by decide : 2≠0),zero_sub] at hh
      nlinarith [sq_nonneg ‖(γ.extend t).val.val.2‖]
    exact norm_eq_zero.mp (sq_eq_zero_iff.mp hn)
  exact (γ.extend t).property (Prod.ext he hz)
private theorem line_plus_patch : linePlus.val ∈ patchSet (identityForm ℝ ℝ) := by
  norm_num [patchSet,identityForm,linePlus,radiusBall,Real.norm_eq_abs]
private theorem line_minus_patch : lineMinus.val ∈ patchSet (identityForm ℝ ℝ) := by
  norm_num [patchSet,identityForm,lineMinus,radiusBall,Real.norm_eq_abs]
example : ∃ a b, a≠b ∧ MorseOpenGluing.Foot (oldSet (identityForm ℝ ℝ))
    (patchSet (identityForm ℝ ℝ)) a ∧ MorseOpenGluing.Foot (oldSet (identityForm ℝ ℝ))
    (patchSet (identityForm ℝ ℝ)) b :=
  ⟨ZerothHomotopy.mk linePlus,ZerothHomotopy.mk lineMinus,line_distinct,
    ⟨linePlus,line_plus_patch,rfl⟩,⟨lineMinus,line_minus_patch,rfl⟩⟩
example : ¬ Function.Injective (MorseOpenGluing.inclusionMap (oldSet (identityForm ℝ ℝ))) := by
  intro hi
  apply line_distinct
  apply hi
  exact ZerothHomotopy.sound
    ((patch_pathConnected (identityForm ℝ ℝ)).joinedIn _ line_plus_patch _ line_minus_patch).somePath

-- Local opposite signs can become the SAME global punctured-circle component.
local instance : Fact ((0:ℝ)<1) := ⟨by norm_num⟩
private theorem circle_puncture_pc : IsPathConnected ({(0:AddCircle (1:ℝ))}ᶜ) := by
  let e := AddCircle.openPartialHomeomorphCoe (1:ℝ) 0
  have hs0 : IsPathConnected (Ioo (0:ℝ) 1) :=
    (convex_Ioo (𝕜:=ℝ) (0:ℝ) 1).isPathConnected ⟨1/2,by norm_num⟩
  have hs : IsPathConnected e.source := by simpa [e] using hs0
  have hi := hs.image' e.continuousOn
  change IsPathConnected e.target
  simpa only [e.image_source_eq_target] using hi
private def circlePlus : ↥({(0:AddCircle (1:ℝ))}ᶜ : Set (AddCircle (1:ℝ))) :=
  ⟨((1/8:ℝ):AddCircle (1:ℝ)),(AddCircle.openPartialHomeomorphCoe (1:ℝ) 0).map_source (by norm_num : (1/8:ℝ) ∈ Ioo 0 (0+1))⟩
private def circleMinus : ↥({(0:AddCircle (1:ℝ))}ᶜ : Set (AddCircle (1:ℝ))) :=
  ⟨((7/8:ℝ):AddCircle (1:ℝ)),(AddCircle.openPartialHomeomorphCoe (1:ℝ) 0).map_source (by norm_num : (7/8:ℝ) ∈ Ioo 0 (0+1))⟩
example : ((-1/8:ℝ):AddCircle (1:ℝ)) = circleMinus.val := by
  have h := AddCircle.coe_add_period (1:ℝ) (-1/8)
  norm_num at h
  change ((-1/8:ℝ):AddCircle (1:ℝ)) = ((7/8:ℝ):AddCircle (1:ℝ))
  convert h.symm using 1 <;> norm_num
example : ZerothHomotopy.mk circlePlus = ZerothHomotopy.mk circleMinus := by
  letI : PathConnectedSpace ↥({(0:AddCircle (1:ℝ))}ᶜ : Set (AddCircle (1:ℝ))) :=
    isPathConnected_iff_pathConnectedSpace.mp circle_puncture_pc
  exact Subsingleton.elim _ _

-- Genuine ordinary H0 on the literal critical-level inclusion in characteristic2.
local instance : Fact (Nat.Prime 2) := ⟨by decide⟩
example (x : oldSet (identityForm ℝ ℝ)) :
    (OrdinaryH0Elder.h0Functor (ZMod 2)).map (MorseOpenGluing.inclusion (oldSet (identityForm ℝ ℝ)))
      (OrdinaryH0Elder.pointClass (ZMod 2) (TopCat.of (oldSet (identityForm ℝ ℝ))) x) ≠ 0 := by
  rw [OrdinaryH0Elder.pointClass_map]
  exact OrdinaryH0Elder.pointClass_ne_zero _ _ _

end MorseConeTests
