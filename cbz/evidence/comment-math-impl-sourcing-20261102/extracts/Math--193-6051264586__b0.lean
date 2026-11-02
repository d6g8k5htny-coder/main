import OrdinaryH0Image
import Mathlib.LinearAlgebra.Finsupp.LinearCombination

noncomputable section
set_option backward.isDefEq.respectTransparency false
open CategoryTheory Set OrdinaryH0Elder OrdinaryH0Image
open scoped Classical
universe u

namespace OrdinaryH0Split

variable (K : Type u) [Field K]
variable {X : Type u} [TopologicalSpace X]
variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}
variable (H : CapFirstExit.CapHyp f C M z b s) (hMb : f M = b)

def Active (s b h : ℝ) : Prop := s < h ∧ h ≤ b

theorem path_mem_cap_iff (H : CapFirstExit.CapHyp f C M z b s) {h : ℝ} (hsh : s < h)
    {x y : superlevel f h} (p : Path x y) : x.1 ∈ C ↔ y.1 ∈ C := by
  constructor
  · intro hx
    by_contra hy
    obtain ⟨t, ht⟩ := CapFirstExit.path_meets_frontier (p.map continuous_subtype_val) hx hy
    exact (not_le_of_gt hsh) ((p t).2.trans (H.frontier_le _ ht))
  · intro hy
    by_contra hx
    obtain ⟨t, ht⟩ := CapFirstExit.path_meets_frontier (p.symm.map continuous_subtype_val) hy hx
    exact (not_le_of_gt hsh) ((p.symm t).2.trans (H.frontier_le _ ht))

def capIndicator (H : CapFirstExit.CapHyp f C M z b s) {h : ℝ} (hsh : s < h) :
    ZerothHomotopy (superlevel f h) → K :=
  ZerothHomotopy.lift (fun x => if x.1 ∈ C then (1 : K) else 0)
    (fun {_ _} p => by
      rw [propext (path_mem_cap_iff H hsh p)]
    )

def capFunctional {h : ℝ} (hsh : s < h) : H0 K (superlevel f h) →ₗ[K] K :=
  (Finsupp.linearCombination K (capIndicator K H hsh)).comp
    (componentEquiv K (superlevel f h)).toLinearMap

theorem capFunctional_point {h : ℝ} (hsh : s < h) (x : superlevel f h) :
    capFunctional K H hsh (pointClass K (superlevel f h) x) =
      if x.1 ∈ C then 1 else 0 := by
  simp [capFunctional, pointClass, capIndicator, Finsupp.linearCombination_single]

def delta (h : ℝ) : H0 K (superlevel f h) :=
  if hh : h ≤ b then
    pointClass K (superlevel f h) ⟨M, by rwa [hMb]⟩ -
      pointClass K (superlevel f h) ⟨z, hh.trans (le_of_lt H.older)⟩
  else 0

theorem delta_map {r h : ℝ} (hle : h ≤ r) (hrb : r ≤ b) :
    (h0Functor K).map (superlevelInclusion f hle) (delta K H hMb r) =
      delta K H hMb h := by
  have hhb : h ≤ b := hle.trans hrb
  simp only [delta, dite_eq_left hrb, dite_eq_left hhb, map_sub, pointClass_map]
  rfl

theorem delta_eq_zero_below {h : ℝ} (hhs : h ≤ s) (hhb : h ≤ b) :
    delta K H hMb h = 0 := by
  obtain ⟨γ, hγ⟩ := H.ridge
  have hz : h ≤ f z := hhb.trans (le_of_lt H.older)
  have hj : JoinedIn {x | h ≤ f x} M z := ⟨γ, fun t => hhs.trans (hγ t)⟩
  have hp := (joinedIn_iff_joined (F := {x | h ≤ f x}) (x := M) (y := z)
    (by simpa [hMb] using hhb) hz).mp hj
  rw [delta, dite_eq_left hhb]
  exact sub_eq_zero.mpr ((pointClass_eq_iff_joined K (superlevel f h) _ _).mpr hp)

theorem capFunctional_delta {h : ℝ} (ha : Active s b h) :
    capFunctional K H ha.1 (delta K H hMb h) = 1 := by
  have hz : z ∉ C := fun hz => (not_le_of_gt H.older) (H.ceiling z hz)
  simp [delta, ha.2, map_sub, capFunctional_point, H.mem, hz]

theorem delta_ne_zero {h : ℝ} (ha : Active s b h) : delta K H hMb h ≠ 0 := by
  intro hz
  have hc := capFunctional_delta K H hMb ha
  rw [hz, map_zero] at hc
  exact zero_ne_one hc

theorem delta_eq_zero_inactive {h : ℝ} (ha : ¬ Active s b h) :
    delta K H hMb h = 0 := by
  by_cases hh : h ≤ b
  · exact delta_eq_zero_below K H hMb (le_of_not_gt (fun hs => ha ⟨hs, hh⟩)) hh
  · simp [delta, hh]

theorem delta_support (h : ℝ) : delta K H hMb h ≠ 0 ↔ Active s b h := by
  constructor
  · intro hn
    by_contra ha
    exact hn (delta_eq_zero_inactive K H hMb ha)
  · exact delta_ne_zero K H hMb

def intervalLine (h : ℝ) : Submodule K (H0 K (superlevel f h)) :=
  Submodule.span K ({delta K H hMb h} : Set _)

theorem intervalLine_inactive {h : ℝ} (ha : ¬ Active s b h) :
    intervalLine K H hMb h = ⊥ := by
  simp [intervalLine, delta_eq_zero_inactive K H hMb ha]

theorem line_map_mem {r h : ℝ} (hle : h ≤ r) {q : H0 K (superlevel f r)}
    (hq : q ∈ intervalLine K H hMb r) :
    (h0Functor K).map (superlevelInclusion f hle) q ∈ intervalLine K H hMb h := by
  obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hq
  rw [map_smul]
  by_cases hrb : r ≤ b
  · rw [delta_map K H hMb hle hrb]
    exact Submodule.smul_mem _ a (Submodule.mem_span_singleton_self _)
  · simp [delta, hrb]

def intervalLineFunctor : ℝᵒᵈ ⥤ ModuleCat.{u} K where
  obj h := ModuleCat.of K (intervalLine K H hMb h)
  map {r h} g := ModuleCat.ofHom
    (((closedH0Persistence K f).map g).hom.restrict
      (fun _ hq => line_map_mem K H hMb (@leOfHom ℝᵒᵈ _ r h g) hq))
  map_id h := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro q
    apply Subtype.ext
    change ((closedH0Persistence K f).map (𝟙 h)) q.1 = q.1
    rw [CategoryTheory.Functor.map_id]
    rfl
  map_comp g j := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro q
    apply Subtype.ext
    change ((closedH0Persistence K f).map (g ≫ j)) q.1 =
      ((closedH0Persistence K f).map j) (((closedH0Persistence K f).map g) q.1)
    rw [CategoryTheory.Functor.map_comp]
    rfl

def intervalInclusion : intervalLineFunctor K H hMb ⟶ closedH0Persistence K f where
  app h := ModuleCat.ofHom (intervalLine K H hMb h).subtype
  naturality {r h} g := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro q
    rfl

def activeFiberEquiv {h : ℝ} (ha : Active s b h) :
    K ≃ₗ[K] intervalLine K H hMb h :=
  LinearEquiv.toSpanNonzeroSingleton K _ (delta K H hMb h) (delta_ne_zero K H hMb ha)

theorem active_transport {r h : ℝᵒᵈ} (g : r ⟶ h)
    (ar : Active s b r) (ah : Active s b h) (a : K) :
    (intervalLineFunctor K H hMb).map g (activeFiberEquiv K H hMb ar a) =
      activeFiberEquiv K H hMb ah a := by
  apply Subtype.ext
  change (h0Functor K).map (superlevelInclusion f (@leOfHom ℝᵒᵈ _ r h g))
    (a • delta K H hMb r) = a • delta K H hMb h
  rw [map_smul, delta_map K H hMb _ ar.2]

theorem capFunctional_map {r h : ℝ} (hle : h ≤ r) (hsh : s < h)
    (hsr : s < r) (q : H0 K (superlevel f r)) :
    capFunctional K H hsh ((h0Functor K).map (superlevelInclusion f hle) q) =
      capFunctional K H hsr q := by
  have he : (capFunctional K H hsh).comp
      ((h0Functor K).map (superlevelInclusion f hle)).hom = capFunctional K H hsr := by
    apply LinearMap.ext_on_range (pointClasses_span_top K (superlevel f r))
    intro x
    simp only [LinearMap.comp_apply, pointClass_map, capFunctional_point]
    rfl
  exact LinearMap.congr_fun he q

theorem capFunctional_map_above {r h : ℝ} (hle : h ≤ r) (hsh : s < h)
    (hbr : b < r) (q : H0 K (superlevel f r)) :
    capFunctional K H hsh ((h0Functor K).map (superlevelInclusion f hle) q) = 0 := by
  have he : (capFunctional K H hsh).comp
      ((h0Functor K).map (superlevelInclusion f hle)).hom = 0 := by
    apply LinearMap.ext_on_range (pointClasses_span_top K (superlevel f r))
    intro x
    have hx : x.1 ∉ C := fun hx => (not_le_of_gt (hbr.trans_le x.2)) (H.ceiling _ hx)
    change capFunctional K H hsh
      ((h0Functor K).map (superlevelInclusion f hle) (pointClass K (superlevel f r) x)) = 0
    rw [pointClass_map, capFunctional_point]
    change (if x.1 ∈ C then (1 : K) else 0) = 0
    simp [hx]
  exact LinearMap.congr_fun he q

def projectionLinear (h : ℝ) : H0 K (superlevel f h) →ₗ[K] H0 K (superlevel f h) :=
  if ha : Active s b h then (capFunctional K H ha.1).smulRight (delta K H hMb h)
  else 0

theorem projection_mem (h : ℝ) (q : H0 K (superlevel f h)) :
    projectionLinear K H hMb h q ∈ intervalLine K H hMb h := by
  by_cases ha : Active s b h
  · simp only [projectionLinear, dite_eq_left ha, LinearMap.smulRight_apply]
    exact Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _)
  · simp [projectionLinear, ha]

def projection (h : ℝ) : H0 K (superlevel f h) →ₗ[K] intervalLine K H hMb h :=
  (projectionLinear K H hMb h).codRestrict (intervalLine K H hMb h)
    (projection_mem K H hMb h)

theorem projection_natural {r h : ℝ} (hle : h ≤ r) (q : H0 K (superlevel f r)) :
    projectionLinear K H hMb h ((h0Functor K).map (superlevelInclusion f hle) q) =
      (h0Functor K).map (superlevelInclusion f hle) (projectionLinear K H hMb r q) := by
  by_cases ar : Active s b r
  · by_cases ah : Active s b h
    · simp only [projectionLinear, dite_eq_left ar, dite_eq_left ah, LinearMap.smulRight_apply, map_smul]
      rw [capFunctional_map K H hle ah.1 ar.1, delta_map K H hMb hle ar.2]
    · simp only [projectionLinear, dite_eq_left ar, dite_eq_right ah, LinearMap.zero_apply,
        LinearMap.smulRight_apply, map_smul]
      rw [delta_map K H hMb hle ar.2, delta_eq_zero_inactive K H hMb ah, smul_zero]
  · by_cases ah : Active s b h
    · have hbr : b < r := lt_of_not_ge (fun hrb => ar ⟨ah.1.trans_le hle, hrb⟩)
      simp only [projectionLinear, dite_eq_right ar, dite_eq_left ah, LinearMap.zero_apply,
        LinearMap.smulRight_apply, map_zero]
      rw [capFunctional_map_above K H hle ah.1 hbr, zero_smul]
    · simp [projectionLinear, ar, ah]

def intervalRetraction : closedH0Persistence K f ⟶ intervalLineFunctor K H hMb where
  app h := ModuleCat.ofHom (projection K H hMb h)
  naturality {r h} g := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro q
    apply Subtype.ext
    exact projection_natural K H hMb (@leOfHom ℝᵒᵈ _ r h g) q

theorem interval_split (hsb : s < b) :
    intervalInclusion K H hMb ≫ intervalRetraction K H hMb =
      𝟙 (intervalLineFunctor K H hMb) := by
  apply NatTrans.ext
  funext h
  apply ModuleCat.hom_ext
  apply LinearMap.ext
  intro q
  apply Subtype.ext
  change projectionLinear K H hMb h q.1 = q.1
  by_cases ha : Active s b h
  · obtain ⟨a, he⟩ := Submodule.mem_span_singleton.mp q.2
    rw [← he]
    simp [projectionLinear, ha, LinearMap.smulRight_apply, map_smul,
      capFunctional_delta K H hMb ha]
  · have hq : q.1 = 0 := by
      have hm : q.1 ∈ (⊥ : Submodule K (H0 K (superlevel f h))) :=
        (le_of_eq (intervalLine_inactive K H hMb ha)) q.2
      exact (Submodule.mem_bot K).mp hm
    simp [projectionLinear, ha, hq]

def capSplitMono (hsb : s < b) : SplitMono (intervalInclusion K H hMb) :=
  { retraction := intervalRetraction K H hMb, id := interval_split K H hMb hsb }

theorem birth_active (hsb : s < b) : Active s b b := ⟨hsb, le_rfl⟩

end OrdinaryH0Split

