import OrdinaryH0

/-!+# One actual ordinary-H0 interval summand from an asymmetric trap

Author implementation. The geometric premises remain visible; this file does not
prove the analytic ridge estimates, Morse labeling, or an exhaustive barcode.
The coefficient object is the field itself, and H0 is the existing singular
homology functor, compared with components by the proved C264 isomorphism.
-/

noncomputable section

open CategoryTheory Limits AlgebraicTopology
open scoped Classical

namespace TorusFiltration

universe u

/-- Increasing parameter `t=-height`: closed birth and open death. -/
def elderActive (m d t : ℝ) : Prop := -m ≤ t ∧ t < -d

variable (k : Type u) [Field k]

/-- The fiber is k on the interval and the zero submodule elsewhere. -/
def elderFiber (m d t : ℝ) : Submodule k k where
  carrier := {a | ¬ elderActive m d t → a = 0}
  zero_mem' := by simp
  add_mem' := by intro a b ha hb h; simp [ha h, hb h]
  smul_mem' := by intro a b hb h; simp [hb h]

theorem elderFiber_eq_zero {m d t : ℝ} (ht : ¬ elderActive m d t)
    (x : elderFiber k m d t) : x = 0 := by
  apply Subtype.ext
  exact x.property ht

def elderFiberMap (m d : ℝ) {s t : ℝ} (hst : s ≤ t) :
    elderFiber k m d s →ₗ[k] elderFiber k m d t where
  toFun x := ⟨if t < -d then x.val else 0, by
    intro ht
    by_cases htd : t < -d
    · have hsa : ¬ elderActive m d s := by
        intro hs
        exact ht ⟨hs.1.trans hst, htd⟩
      simp [htd, x.property hsa]
    · simp [htd]⟩
  map_add' := by intro x y; apply Subtype.ext; dsimp; split_ifs <;> simp
  map_smul' := by intro a x; apply Subtype.ext; dsimp; split_ifs <;> simp

@[simp] theorem elderFiberMap_val (m d : ℝ) {s t : ℝ} (hst : s ≤ t)
    (x : elderFiber k m d s) :
    (elderFiberMap k m d hst x).val = if t < -d then x.val else 0 := rfl

/-- The actual interval persistence functor; no maps are assumed. -/
def elderInterval (m d : ℝ) : ℝ ⥤ ModuleCat k where
  obj t := ModuleCat.of k (elderFiber k m d t)
  map f := ModuleCat.ofHom (elderFiberMap k m d (leOfHom f))
  map_id t := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro x
    apply Subtype.ext
    change (if t < -d then x.val else 0) = x.val
    by_cases ht : t < -d
    · simp [ht]
    · have hn : ¬ elderActive m d t := fun h => ht h.2
      simp [ht, x.property hn]
  map_comp {r s t} f g := by
    apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro x
    apply Subtype.ext
    change (if t < -d then x.val else 0) =
      if t < -d then (if s < -d then x.val else 0) else 0
    by_cases ht : t < -d
    · have hs : s < -d := (leOfHom g).trans_lt ht
      simp [ht, hs]
    · simp [ht]

/-- A canonical lift of a coefficient to any active interval fiber. -/
def elderFiberLift {m d t : ℝ} (ht : elderActive m d t) :
    k →ₗ[k] elderFiber k m d t where
  toFun a := ⟨a, fun hn => (hn ht).elim⟩
  map_add' := by intros; rfl
  map_smul' := by intros; rfl

@[simp] theorem elderFiberLift_val {m d t : ℝ} (ht : elderActive m d t) (a : k) :
    (elderFiberLift k ht a).val = a := rfl

/-- Endpoint control: birth is included. -/
theorem elder_control_birth {m d : ℝ} (hdm : d < m) : elderActive m d (-m) := by
  constructor <;> linarith

/-- Endpoint control: death is excluded, despite closed superlevels. -/
theorem elder_control_death (m d : ℝ) : ¬ elderActive m d (-d) := by
  intro h
  exact (lt_irrefl _) h.2

variable {X : Type u} [TopologicalSpace X] (g : X → ℝ) (M P : X) (m d : ℝ)

/-- The topological consequences of the asymmetric ridge trap. No H0 maps or
splitting conclusions occur among the premises. The path at d is actual closed
superlevel connectivity, while active cuts are clopen. -/
structure ElderTrap (D : Set X) : Prop where
  death_lt_birth : d < m
  maximum_height : g M = m
  older_height : m < g P
  maximum_inside : M ∈ D
  older_outside : P ∉ D
  height_ceiling : ∀ x ∈ D, g x ≤ m
  active_clopen : ∀ t, elderActive m d t →
    IsClopen {x : superlevel_set g t | x.val ∈ D}
  closed_path : JoinedIn (superlevel_set g (-d)) M P

variable {g M P m d} {D : Set X}

theorem ElderTrap.maximum_mem (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) : M ∈ superlevel_set g t := by
  change -t ≤ g M
  erw [T.maximum_height]
  linarith [ht.1]

theorem ElderTrap.older_mem (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) : P ∈ superlevel_set g t := by
  change -t ≤ g P
  linarith [ht.1, T.older_height]

/-- Clopen cuts are constant on actual path components, with no local path
connectivity assumption on the ambient space. -/
theorem elder_clopen_path_iff {Y : Type u} [TopologicalSpace Y] {U : Set Y}
    (hU : IsClopen U) {x y : Y} (p : Path x y) : x ∈ U ↔ y ∈ U := by
  have hxy : y ∈ connectedComponent x :=
    pathComponent_subset_component x (show y ∈ pathComponent x from ⟨p⟩)
  have hyx : x ∈ connectedComponent y :=
    pathComponent_subset_component y (show x ∈ pathComponent y from ⟨p.symm⟩)
  exact ⟨fun hx => hU.connectedComponent_subset hx hxy,
    fun hy => hU.connectedComponent_subset hy hyx⟩

/-- The D indicator is genuinely descended to the path-component quotient. -/
def ElderTrap.componentInside (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) : ZerothHomotopy (superlevel_set g t) → Prop :=
  ZerothHomotopy.lift (fun x => x.val ∈ D) (fun _ _ p =>
    propext (elder_clopen_path_iff (T.active_clopen t ht) p))

@[simp] theorem ElderTrap.componentInside_mk (T : ElderTrap g M P m d D)
    {t : ℝ} (ht : elderActive m d t) (x : superlevel_set g t) :
    T.componentInside ht (ZerothHomotopy.mk x) ↔ x.val ∈ D := Iff.rfl

variable {k}

/-- The actual singular-H0 point-class difference on an active level. -/
def elderDifference (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) :
    ModuleCat.of k k ⟶ (superlevel_h0Filtration g (ModuleCat.of k k)).obj t :=
  ordinaryH0_pointClass (TopCat.of (superlevel_set g t)) (ModuleCat.of k k)
      ⟨M, T.maximum_mem ht⟩ -
    ordinaryH0_pointClass (TopCat.of (superlevel_set g t)) (ModuleCat.of k k)
      ⟨P, T.older_mem ht⟩

theorem elderDifference_map (T : ElderTrap g M P m d D) {s t : ℝ}
    (hs : elderActive m d s) (ht : elderActive m d t) (hst : s ≤ t) :
    elderDifference (k := k) T hs ≫
        (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) =
      elderDifference T ht := by
  exact ordinaryH0_difference_map (ModuleCat.of k k)
    (TopCat.ofHom (superlevel_inclusion g hst))
    ⟨M, T.maximum_mem hs⟩ ⟨P, T.older_mem hs⟩

theorem elderDifference_map_dead (T : ElderTrap g M P m d D) {s t : ℝ}
    (hs : elderActive m d s) (ht : -d ≤ t) (hst : s ≤ t) :
    elderDifference (k := k) T hs ≫
        (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) = 0 := by
  unfold elderDifference
  erw [ordinaryH0_difference_map]
  apply ordinaryH0_sameComponent_difference_zero
  apply (superlevel_component_eq_iff_joinedIn g t _ _).2
  exact T.closed_path.mono (superlevel_mono g ht)

/-- The component indicator transported by the existing actual H0 comparison. -/
def elderProjection (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) :
    (superlevel_h0Filtration g (ModuleCat.of k k)).obj t ⟶ ModuleCat.of k k := by
  classical
  exact (ordinaryH0_filtrationIso g (ModuleCat.of k k)).hom.app t ≫
    Sigma.desc (fun q => if T.componentInside ht q then 𝟙 _ else 0)

theorem elderProjection_point (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) (x : superlevel_set g t) :
    ordinaryH0_pointClass (TopCat.of (superlevel_set g t)) (ModuleCat.of k k) x ≫
        elderProjection (k := k) T ht =
      if x.val ∈ D then 𝟙 _ else 0 := by
  classical
  unfold elderProjection
  erw [← Category.assoc]
  erw [ordinaryH0_pointClass_iso]
  erw [Sigma.ι_comp_desc]
  rfl

theorem elderDifference_projection (T : ElderTrap g M P m d D) {t : ℝ}
    (ht : elderActive m d t) :
    elderDifference (k := k) T ht ≫ elderProjection T ht = 𝟙 _ := by
  classical
  unfold elderDifference
  erw [Preadditive.sub_comp,
    elderProjection_point, elderProjection_point]
  simp [T.maximum_inside, T.older_outside]

def elderSectionApp (T : ElderTrap g M P m d D) (t : ℝ) :
    (elderInterval k m d).obj t ⟶
      (superlevel_h0Filtration g (ModuleCat.of k k)).obj t := by
  classical
  exact if ht : elderActive m d t then
    ModuleCat.ofHom (elderFiber k m d t).subtype ≫ elderDifference T ht else 0

def elderRetractionApp (T : ElderTrap g M P m d D) (t : ℝ) :
    (superlevel_h0Filtration g (ModuleCat.of k k)).obj t ⟶
      (elderInterval k m d).obj t := by
  classical
  exact if ht : elderActive m d t then
    elderProjection T ht ≫ ModuleCat.ofHom (elderFiberLift k ht) else 0

theorem elderSectionApp_naturality (T : ElderTrap g M P m d D) {s t : ℝ}
    (hst : s ≤ t) :
    (elderInterval k m d).map (homOfLE hst) ≫ elderSectionApp T t =
      elderSectionApp T s ≫
        (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) := by
  classical
  by_cases hs : elderActive m d s
  · by_cases ht : elderActive m d t
    · simp only [elderSectionApp, dite_eq_left hs, dite_eq_left ht]
      erw [Category.assoc, elderDifference_map T hs ht hst, ← Category.assoc]
      congr 1
      ext x
      change (if t < -d then x.val else 0) = x.val
      simp [ht.2]
    · have htd : -d ≤ t := by
        by_contra h
        exact ht ⟨hs.1.trans hst, lt_of_not_ge h⟩
      simp only [elderSectionApp, dite_eq_left hs, dite_eq_right ht, comp_zero]
      erw [Category.assoc, elderDifference_map_dead T hs htd hst, comp_zero]
  · apply ModuleCat.hom_ext
    apply LinearMap.ext
    intro x
    have hx : x = 0 := elderFiber_eq_zero k hs x
    simp [hx]

/-- Section from the interval to actual singular H0. -/
def elderSection (T : ElderTrap g M P m d D) :
    elderInterval k m d ⟶ superlevel_h0Filtration g (ModuleCat.of k k) where
  app := elderSectionApp T
  naturality := by
    intro s t f
    exact elderSectionApp_naturality T (leOfHom f)

theorem elderProjection_naturality_active (T : ElderTrap g M P m d D) {s t : ℝ}
    (hs : elderActive m d s) (ht : elderActive m d t) (hst : s ≤ t) :
    (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) ≫
        elderProjection (k := k) T ht = elderProjection T hs := by
  classical
  apply ordinaryH0_hom_ext (TopCat.of (superlevel_set g s)) (ModuleCat.of k k)
  intro x
  erw [← Category.assoc]
  erw [ordinaryH0_pointClass_map]
  erw [elderProjection_point, elderProjection_point]
  rfl

/-- The birth-crossing square follows from the geometric height ceiling:
there is no source point in D above m. It is not an assumed naturality field. -/
theorem elderProjection_prebirth (T : ElderTrap g M P m d D) {s t : ℝ}
    (hs : s < -m) (ht : elderActive m d t) (hst : s ≤ t) :
    (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) ≫
        elderProjection (k := k) T ht = 0 := by
  classical
  apply ordinaryH0_hom_ext (TopCat.of (superlevel_set g s)) (ModuleCat.of k k)
  intro x
  erw [← Category.assoc]
  erw [ordinaryH0_pointClass_map]
  erw [elderProjection_point]
  change (if x.val ∈ D then 𝟙 _ else 0) = _
  have hx : x.val ∉ D := by
    intro hx
    have hceil := T.height_ceiling x.val hx
    have hlevel : -s ≤ g x.val := x.property
    linarith
  simp only [ite_eq_right hx]
  erw [comp_zero]

theorem elderRetractionApp_naturality (T : ElderTrap g M P m d D) {s t : ℝ}
    (hst : s ≤ t) :
    (superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) ≫
        elderRetractionApp T t =
      elderRetractionApp T s ≫ (elderInterval k m d).map (homOfLE hst) := by
  classical
  by_cases ht : elderActive m d t
  · by_cases hs : elderActive m d s
    · simp only [elderRetractionApp, dite_eq_left hs, dite_eq_left ht]
      apply ModuleCat.hom_ext
      apply LinearMap.ext
      intro x
      apply Subtype.ext
      change elderProjection T ht
          ((superlevel_h0Filtration g (ModuleCat.of k k)).map (homOfLE hst) x) =
        if t < -d then elderProjection T hs x else 0
      rw [ite_eq_left ht.2]
      have he := congrArg
        (fun f : (superlevel_h0Filtration g (ModuleCat.of k k)).obj s ⟶
            ModuleCat.of k k => f x)
        (elderProjection_naturality_active (k := k) T hs ht hst)
      exact he
    · have hsm : s < -m := by
        by_contra h
        exact hs ⟨le_of_not_gt h, hst.trans_lt ht.2⟩
      simp only [elderRetractionApp, dite_eq_right hs, dite_eq_left ht, zero_comp]
      erw [← Category.assoc, elderProjection_prebirth T hsm ht hst, zero_comp]
  · simp only [elderRetractionApp, dite_eq_right ht, comp_zero]
    by_cases hs : elderActive m d s
    · simp only [dite_eq_left hs]
      have htd : ¬ t < -d := by
        intro h
        exact ht ⟨hs.1.trans hst, h⟩
      apply ModuleCat.hom_ext
      apply LinearMap.ext
      intro x
      apply Subtype.ext
      change 0 = if t < -d then _ else 0
      simp [htd]
    · simp [hs]

/-- Retraction from actual singular H0 to the interval, using component descent. -/
def elderRetraction (T : ElderTrap g M P m d D) :
    superlevel_h0Filtration g (ModuleCat.of k k) ⟶ elderInterval k m d where
  app := elderRetractionApp T
  naturality := by
    intro s t f
    exact elderRetractionApp_naturality T (leOfHom f)

/-- The constructed section and retraction split the actual interval functor. -/
@[reassoc (attr := simp)]
theorem elderSection_retraction (T : ElderTrap g M P m d D) :
    elderSection (k := k) T ≫ elderRetraction T = 𝟙 (elderInterval k m d) := by
  classical
  apply NatTrans.ext
  funext t
  change elderSectionApp (k := k) T t ≫ elderRetractionApp (k := k) T t = 𝟙 _
  by_cases ht : elderActive m d t
  · simp only [elderSectionApp, elderRetractionApp, dite_eq_left ht]
    erw [Category.assoc, ← Category.assoc (elderDifference T ht),
      elderDifference_projection, Category.id_comp]
    ext x
    rfl
  · ext x
    have hx : x = 0 := elderFiber_eq_zero k ht x
    simp [hx]

def elderKernelProjection (T : ElderTrap g M P m d D) :
    superlevel_h0Filtration g (ModuleCat.of k k) ⟶ kernel (elderRetraction (k := k) T) :=
  kernel.lift (elderRetraction T)
    (𝟙 _ - elderRetraction T ≫ elderSection T) (by
      erw [Preadditive.sub_comp, Category.id_comp, Category.assoc,
        elderSection_retraction, Category.comp_id, sub_self])

/-- An actual natural direct-sum isomorphism, not a postulated decomposition.
Its maps are v ↦ (qv,v-jqv) and (c,w) ↦ jc+w. -/
def elderDirectSummand (T : ElderTrap g M P m d D) :
    superlevel_h0Filtration g (ModuleCat.of k k) ≅
      elderInterval k m d ⊞ kernel (elderRetraction (k := k) T) where
  hom := biprod.lift (elderRetraction T) (elderKernelProjection T)
  inv := biprod.desc (elderSection T) (kernel.ι (elderRetraction T))
  hom_inv_id := by
    erw [biprod.lift_desc]
    simp only [elderKernelProjection, kernel.lift_ι]
    abel
  inv_hom_id := by
    apply biprod.hom_ext' <;> apply biprod.hom_ext
    · simp [Category.assoc, elderSection_retraction]
    · apply (cancel_mono (kernel.ι (elderRetraction T))).1
      simp [Category.assoc, elderKernelProjection, Preadditive.comp_sub]
    · simp
    · apply (cancel_mono (kernel.ι (elderRetraction T))).1
      simp [elderKernelProjection, Category.assoc, Preadditive.comp_sub]

/-- Ordinary individual point classes remain nonzero; the dying section uses
their difference. This is valid independently of characteristic. -/
theorem elder_control_ordinary_point_survives (t : ℝ) (x : superlevel_set g t) :
    ordinaryH0_pointClass (TopCat.of (superlevel_set g t)) (ModuleCat.of k k) x ≠ 0 := by
  apply ordinaryH0_pointClass_ne_zero
  intro h
  have hl := congrArg ModuleCat.Hom.hom h
  change (LinearMap.id : k →ₗ[k] k) = 0 at hl
  exact one_ne_zero (LinearMap.congr_fun hl 1)

end TorusFiltration
