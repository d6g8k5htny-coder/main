import Lifetime

/-!
Relative topology for closed scalar superlevels.

The chart-input theorem below is a reusable topology adapter. It does not construct
a chart from a nonzero derivative: that literal scalar-chart obligation is separately
owned. No critical-point Morse chart or global closed-level LPC is asserted here.
-/

open Set Filter Topology
open UniversalLaw.Spec.Lifetime

namespace TorusRegularSuperlevel

/-- An open embedding carries a path-connected neighborhood basis to its image point. -/
theorem path_connected_basis_of_openEmbedding
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    [LocallyPathConnectedSpace X] {e : X → Y}
    (he : IsOpenEmbedding e) (x : X) :
    (𝓝 (e x)).HasBasis
      (fun V : Set Y => V ∈ 𝓝 (e x) ∧ IsPathConnected V) id := by
  rw [hasBasis_self]
  intro U hU
  have hpre : e ⁻¹' U ∈ 𝓝 x :=
    he.continuous.continuousAt.preimage_mem_nhds hU
  obtain ⟨V, ⟨hV, hpath⟩, hVU⟩ := (path_connected_basis x).mem_iff.mp hpre
  refine ⟨e '' V, ⟨he.image_mem_nhds.mpr hV, hpath.image he.continuous⟩, ?_⟩
  exact image_subset_iff.mpr hVU

/-- An LPC open neighborhood supplies the required basis in the ambient space. -/
theorem path_connected_basis_of_open
    {X : Type*} [TopologicalSpace X] {U : Set X}
    (hU : IsOpen U) [LocallyPathConnectedSpace U] {x : X} (hx : x ∈ U) :
    (𝓝 x).HasBasis (fun V : Set X => V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  exact path_connected_basis_of_openEmbedding hU.isOpenEmbedding_subtypeVal ⟨x, hx⟩

/-- Restrict a chart to two corresponding subsets, using their relative open
source and target. Neither subset is required to be open in its original ambient space. -/
def relative_chart_homeomorph
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    (p : OpenPartialHomeomorph X Y) (S : Set X) (T : Set Y)
    (hmem : ∀ y ∈ p.source, y ∈ S ↔ p y ∈ T) :
    {y : S // y.val ∈ p.source} ≃ₜ {z : T // z.val ∈ p.target} where
  toFun y := ⟨⟨p y.val.val, (hmem y.val.val y.property).mp y.val.property⟩,
    p.map_source y.property⟩
  invFun z := ⟨⟨p.symm z.val.val,
    (hmem (p.symm z.val.val) (p.map_target z.property)).mpr
      (by rw [p.right_inv z.property]; exact z.val.property)⟩,
    p.map_target z.property⟩
  left_inv y := by
    apply Subtype.ext
    apply Subtype.ext
    exact p.left_inv y.property
  right_inv z := by
    apply Subtype.ext
    apply Subtype.ext
    exact p.right_inv z.property
  continuous_toFun := by
    apply Continuous.subtype_mk
    apply Continuous.subtype_mk
    exact p.continuousOn.comp_continuous
      (continuous_subtype_val.comp continuous_subtype_val) (fun y => y.property)
  continuous_invFun := by
    apply Continuous.subtype_mk
    apply Continuous.subtype_mk
    exact p.continuousOn_symm.comp_continuous
      (continuous_subtype_val.comp continuous_subtype_val) (fun z => z.property)

/-- Relative chart transport derives a neighborhood basis in the source subset
from LPC of the model subset. This helper is consumed with a proved convex halfspace. -/
theorem path_connected_basis_of_relative_chart
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    (p : OpenPartialHomeomorph X Y) (S : Set X) (T : Set Y)
    [LocallyPathConnectedSpace T]
    (hmem : ∀ y ∈ p.source, y ∈ S ↔ p y ∈ T)
    (x : S) (hx : x.val ∈ p.source) :
    (𝓝 x).HasBasis (fun V : Set S => V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  let U : Set S := {y | y.val ∈ p.source}
  let V : Set T := {z | z.val ∈ p.target}
  have hU : IsOpen U := p.open_source.preimage continuous_subtype_val
  have hV : IsOpen V := p.open_target.preimage continuous_subtype_val
  letI : LocallyPathConnectedSpace V := hV.locallyPathConnectedSpace
  let h : U ≃ₜ V := relative_chart_homeomorph p S T hmem
  letI : LocallyPathConnectedSpace U := h.symm.locallyPathConnectedSpace
  exact path_connected_basis_of_open hU hx

/-- A scalar chart supplies the actual closed-superlevel neighborhood basis.

The coordinate identity is needed only on the chart source. This theorem proves
the relative-topology step; the separately owned literal regular-point constructor
must supply `p`, `hx`, and `hcoord` before a noncritical-point corollary is claimed. -/
theorem closed_superlevel_basis_of_chart
    {d : ℕ} {g : E d → ℝ} {F : Type*}
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (a : ℝ) (x : {y : E d // a ≤ g y})
    (p : OpenPartialHomeomorph (E d) (ℝ × F))
    (hx : x.val ∈ p.source)
    (hcoord : ∀ y ∈ p.source, (p y).fst = g y) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} => V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  let H : Set (ℝ × F) := {z | a ≤ z.fst}
  have hconvex : Convex ℝ H := by
    intro u hu v hv s t hs ht hst
    change a ≤ s • u.fst + t • v.fst
    calc
      a = s • a + t • a := (Convex.combo_self hst a).symm
      _ ≤ s • u.fst + t • v.fst :=
        add_le_add (smul_le_smul_of_nonneg_left hu hs)
          (smul_le_smul_of_nonneg_left hv ht)
  letI : LocallyPathConnectedSpace H := hconvex.locallyPathConnectedSpace
  apply path_connected_basis_of_relative_chart p {y | a ≤ g y} H _ x hx
  intro y hy
  change a ≤ g y ↔ a ≤ (p y).fst
  rw [hcoord y hy]

/-- A point strictly above the level has a path-connected neighborhood basis in
the closed superlevel, with continuity alone and no regularity assumption. -/
theorem closed_superlevel_interior_basis
    {d : ℕ} {g : E d → ℝ} (hg : Continuous g)
    (a : ℝ) (x : {y : E d // a ≤ g y}) (hx : a < g x.val) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} => V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  let S : Set (E d) := {y | a ≤ g y}
  let U : Set (E d) := {y | a < g y}
  have hU : IsOpen U := isOpen_lt continuous_const hg
  letI : LocallyPathConnectedSpace U := hU.locallyPathConnectedSpace
  let e : U → S := fun y => ⟨y.val, le_of_lt y.property⟩
  have he : IsOpenEmbedding e := by
    exact IsOpenEmbedding.id.restrict (fun y hy => le_of_lt hy) hU
  exact path_connected_basis_of_openEmbedding he ⟨x.val, hx⟩

end TorusRegularSuperlevel
