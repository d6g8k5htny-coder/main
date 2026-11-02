import OrdinaryH0Image

noncomputable section
open CategoryTheory OrdinaryH0Elder OrdinaryH0Image
universe u
namespace GlobalComponentMaps

variable (K : Type u) [Field K]

def componentMap {X Y : TopCat.{u}} (g : X ⟶ Y) :
    ZerothHomotopy X → ZerothHomotopy Y :=
  ZerothHomotopy.lift (fun x => ZerothHomotopy.mk (g x))
    (fun {_ _} p => ZerothHomotopy.sound (p.map g.hom.continuous))

theorem componentEquiv_map {X Y : TopCat.{u}} (g : X ⟶ Y) (v : H0 K X) :
    componentEquiv K Y ((h0Functor K).map g v) =
      Finsupp.mapDomain (componentMap g) (componentEquiv K X v) := by
  have he : (componentEquiv K Y).toLinearMap.comp ((h0Functor K).map g).hom =
      (Finsupp.lmapDomain K K (componentMap g)).comp (componentEquiv K X).toLinearMap := by
    apply LinearMap.ext_on_range (pointClasses_span_top K X)
    intro x
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, pointClass_map]
    simp [pointClass, componentMap]
  exact LinearMap.congr_fun he v

def binaryMerge {ι : Type u} : Option (Option ι) → Option ι
  | none => none
  | some none => none
  | some (some i) => some i

def mergeDifference (ι : Type u) : Option (Option ι) →₀ K :=
  Finsupp.single none 1 - Finsupp.single (some none) 1

private theorem merge_none {ι : Type u} (v : Option (Option ι) →₀ K) :
    Finsupp.mapDomain binaryMerge v none = v none + v (some none) := by
  classical
  induction v using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => simp only [Finsupp.mapDomain_add, Finsupp.add_apply, ha, hb]; ring
  | single j a => cases j with
    | none => simp [binaryMerge]
    | some j => cases j <;> simp [binaryMerge]

private theorem merge_some {ι : Type u} (v : Option (Option ι) →₀ K) (i : ι) :
    Finsupp.mapDomain binaryMerge v (some i) = v (some (some i)) := by
  classical
  induction v using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => simp only [Finsupp.mapDomain_add, Finsupp.add_apply, ha, hb]
  | single j a => cases j with
    | none => simp [binaryMerge]
    | some j => cases j <;> simp [binaryMerge, Finsupp.single_apply]

theorem binaryMerge_kernel (ι : Type u) :
    LinearMap.ker (Finsupp.lmapDomain K K (binaryMerge (ι := ι))) =
      Submodule.span K {mergeDifference K ι} := by
  classical
  ext v
  rw [LinearMap.mem_ker, Submodule.mem_span_singleton]
  constructor
  · intro hv
    refine ⟨v none, ?_⟩
    have hn : v none + v (some none) = 0 := by
      simpa only [Finsupp.lmapDomain_apply, merge_none, Finsupp.zero_apply] using
        congrArg (fun w : Option ι →₀ K => w none) hv
    ext j
    cases j with
    | none => simp [mergeDifference]
    | some j => cases j with
      | none => simp [mergeDifference, smul_eq_mul]; linear_combination -hn
      | some i =>
        have hi : v (some (some i)) = 0 := by
          simpa only [Finsupp.lmapDomain_apply, merge_some, Finsupp.zero_apply] using
            congrArg (fun w : Option ι →₀ K => w (some i)) hv
        simp [mergeDifference, hi]
  · rintro ⟨a, rfl⟩
    simp [mergeDifference, binaryMerge, map_sub]

theorem binaryMerge_point_survives (ι : Type u) (i : Option (Option ι)) :
    Finsupp.lmapDomain K K binaryMerge (Finsupp.single i 1) ≠ 0 := by
  simp

end GlobalComponentMaps
