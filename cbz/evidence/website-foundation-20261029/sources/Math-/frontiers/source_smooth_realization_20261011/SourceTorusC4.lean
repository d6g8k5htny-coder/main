import SourceC4
import SourceTorus
import Mathlib.Topology.OpenPartialHomeomorph.Constructions
import Mathlib.Geometry.Manifold.ContMDiff.Basic
import Mathlib.Geometry.Manifold.ContMDiff.NormedSpace
import Mathlib.Topology.LocallyConstant.Basic

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Torus MeasureTheory
open scoped Manifold ContDiff Topology
namespace I4TorusC4

def quotientChart (d : ℕ) (L : ℝ) [Fact (0 < L)] (a : Fin d → ℝ) :
    OpenPartialHomeomorph (SourceTorus d L) (Fin d → ℝ) :=
  (OpenPartialHomeomorph.pi (fun i : Fin d => AddCircle.openPartialHomeomorphCoe L (a i))).symm

def chartCenter (d : ℕ) (L : ℝ) [Fact (0 < L)] (q : SourceTorus d L) : Fin d → ℝ :=
  fun i => torusRepresentative d L q i - L / 2

theorem quotientChart_symm (d : ℕ) (L : ℝ) [Fact (0 < L)] (a : Fin d → ℝ) :
    (quotientChart d L a).symm = torusProjection d L := by
  rfl

theorem centered_chart_covers (d : ℕ) (L : ℝ) [Fact (0 < L)] (q : SourceTorus d L) :
    q ∈ (quotientChart d L (chartCenter d L q)).source := by
  let e := OpenPartialHomeomorph.pi
    (fun i : Fin d => AddCircle.openPartialHomeomorphCoe L (chartCenter d L q i))
  have hs : torusRepresentative d L q ∈ e.source := by
    change torusRepresentative d L q ∈ Set.pi Set.univ
      (fun i => Set.Ioo (chartCenter d L q i) (chartCenter d L q i + L))
    intro i _
    change chartCenter d L q i < torusRepresentative d L q i ∧
      torusRepresentative d L q i < chartCenter d L q i + L
    have hL : 0 < L := Fact.out
    dsimp [chartCenter]
    constructor <;> linarith
  have he : e (torusRepresentative d L q) = q :=
    torusProjection_representative d L q
  have ht := e.map_source hs
  rw [he] at ht
  exact ht

@[instance_reducible] def actualChartedSpace (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    ChartedSpace (Fin d → ℝ) (SourceTorus d L) where
  atlas := Set.range (quotientChart d L)
  chartAt := fun q => quotientChart d L (chartCenter d L q)
  mem_chart_source := centered_chart_covers d L
  chart_mem_atlas := fun q => ⟨chartCenter d L q, rfl⟩

attribute [local instance] actualChartedSpace

theorem chart_transition_contDiff (d : ℕ) (L : ℝ) [Fact (0 < L)] (a b : Fin d → ℝ) :
    ContDiffOn ℝ ∞ ((quotientChart d L a).symm.trans (quotientChart d L b))
      ((quotientChart d L a).symm.trans (quotientChart d L b)).source := by
  let ea := quotientChart d L a
  let eb := quotientChart d L b
  let T := ea.symm.trans eb
  let S := T.source
  have hproj (x : S) : torusProjection d L (T x) = torusProjection d L x := by
    have hb : ea.symm (x : Fin d → ℝ) ∈ eb.source := x.property.2
    have hi := eb.left_inv hb
    simpa only [T, ea, eb, OpenPartialHomeomorph.trans_apply, quotientChart_symm] using hi
  have hlat (x : S) (i : Fin d) : T x i - (x : Fin d → ℝ) i ∈
      AddSubgroup.zmultiples L := by
    apply AddSubgroup.mem_zmultiples_iff.mpr
    apply (AddCircle.coe_eq_zero_iff (p := L)).mp
    rw [AddCircle.coe_sub]
    exact sub_eq_zero.mpr (congrFun (hproj x) i)
  let delta : S → Fin d → AddSubgroup.zmultiples L :=
    fun x i => ⟨T x i - (x : Fin d → ℝ) i, hlat x i⟩
  have hd : Continuous delta := by
    apply continuous_pi
    intro i
    apply Continuous.subtype_mk
    exact ((continuous_apply i).comp T.continuousOn.domRestrict).sub
      ((continuous_apply i).comp continuous_subtype_val)
  have hlc : IsLocallyConstant delta :=
    (IsLocallyConstant.iff_continuous delta).mpr hd
  apply T.open_source.contDiffOn_iff.mpr
  intro x hx
  let x0 : S := ⟨x, hx⟩
  let c : Fin d → ℝ := fun i => (delta x0 i : ℝ)
  have hsub : ∀ᶠ y : S in 𝓝 x0, T y = (y : Fin d → ℝ) + c := by
    filter_upwards [hlc.eventually_eq x0] with y hy
    funext i
    have hi := congrArg (fun z : Fin d → AddSubgroup.zmultiples L => (z i : ℝ)) hy
    change T y i - (y : Fin d → ℝ) i = c i at hi
    change T y i = (y : Fin d → ℝ) i + c i
    linarith
  have hamb : (fun y => T y) =ᶠ[𝓝 x] (fun y => y + c) := by
    have hm : ∀ᶠ y in Filter.map ((↑) : S → (Fin d → ℝ)) (𝓝 x0), T y = y + c :=
      Filter.eventually_map.mpr hsub
    have hmap : Filter.map ((↑) : S → (Fin d → ℝ)) (𝓝 x0) = 𝓝 x :=
      map_nhds_subtype_coe_eq_nhds hx (T.open_source.mem_nhds hx)
    rw [hmap] at hm
    exact hm
  exact (contDiff_id.add contDiff_const).contDiffAt.congr_of_eventuallyEq hamb

theorem torus_isManifold (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    IsManifold 𝓘(ℝ, Fin d → ℝ) ∞ (SourceTorus d L) := by
  apply isManifold_of_contDiffOn
  intro e e' he he'
  change e ∈ Set.range (quotientChart d L) at he
  change e' ∈ Set.range (quotientChart d L) at he'
  rcases he with ⟨a, rfl⟩
  rcases he' with ⟨b, rfl⟩
  simpa only [modelWithCornersSelf_coe, modelWithCornersSelf_coe_symm,
    Function.comp_id, Function.id_comp, Set.preimage_id, Set.range_id, Set.inter_univ] using
      chart_transition_contDiff d L a b

theorem torusSourceField_contMDiff_ae (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    ∀ᵐ sample ∂sourceLaw d,
      ContMDiff 𝓘(ℝ, Fin d → ℝ) 𝓘(ℝ) 4 (torusSourceField d L sample) := by
  let _ := torus_isManifold d L
  filter_upwards [sourceField_contDiff_ae d (Fact.out : 0 < L).ne'] with sample hs
  intro q
  rw [contMDiffAt_iff_source]
  have heq : torusSourceField d L sample ∘ (extChartAt 𝓘(ℝ, Fin d → ℝ) q).symm =
      sourceField d L sample := by
    funext x
    change torusSourceField d L sample ((quotientChart d L (chartCenter d L q)).symm x) = _
    rw [quotientChart_symm]
    exact torusSourceField_pullback d L sample x
  rw [heq, contMDiffWithinAt_iff_contDiffWithinAt]
  exact hs.contDiffAt.contDiffWithinAt

theorem torusSourceField_weakSuperlevel_closed_ae (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    ∀ᵐ sample ∂sourceLaw d, ∀ h : ℝ,
      IsClosed {q : SourceTorus d L | h ≤ torusSourceField d L sample q} := by
  filter_upwards [torusSourceField_contMDiff_ae d L] with sample hs
  intro h
  exact isClosed_Ici.preimage hs.continuous

end I4TorusC4
