import ResearchFormalCoreR1.GaussianNormalizer
import Mathlib.Probability.HasLaw
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.Topology.Instances.ENNReal.Lemmas
import Mathlib.Topology.Order.LeftRightNhds

/-!
# A full small-radius lower normalizer from an explicit common coupling

Nonnegative Fatou and the proved positive Gaussian reference give one lower
constant for every sufficiently small positive radius. Neither uniform
integrability nor convergence of normalizers is assumed. The literal
conditioned-field construction of the coupling remains an application join.
See ../SOURCE_GAUSSIAN_NORMALIZER.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory Filter
open scoped Topology ENNReal NNReal

set_option autoImplicit false

private theorem nfloor_eventually_lower_of_fatou {Ω ι : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (u : Filter ι) [IsCountablyGenerated u] [NeBot u]
    (f : ι → Ω → ℝ) (g : Ω → ℝ)
    (hmeas : ∀ r, Measurable (f r))
    (hnonneg : ∀ r ω, 0 ≤ f r ω)
    (hg : Integrable g P) (hgpos : 0 < ∫ ω, g ω ∂P)
    (hconv : ∀ᵐ ω ∂P, Tendsto (fun r => f r ω) u (𝓝 (g ω)))
    (hfint : ∀ᶠ r in u, Integrable (f r) P) :
    ∀ᶠ r in u, (∫ ω, g ω ∂P)/2 ≤ ∫ ω, f r ω ∂P := by
  have hg0 : 0 ≤ᵐ[P] g := hconv.mono (fun ω hω =>
    ge_of_tendsto hω (Eventually.of_forall (fun r => hnonneg r ω)))
  have hlim : (fun ω => liminf (fun r => ENNReal.ofReal (f r ω)) u) =ᵐ[P]
      (fun ω => ENNReal.ofReal (g ω)) := by
    filter_upwards [hconv] with ω hω
    exact ((ENNReal.continuous_ofReal.tendsto (g ω)).comp hω).liminf_eq
  have hfatou : ENNReal.ofReal (∫ ω, g ω ∂P) ≤
      liminf (fun r => ∫⁻ ω, ENNReal.ofReal (f r ω) ∂P) u := by
    rw [ofReal_integral_eq_lintegral_ofReal hg hg0]
    rw [← lintegral_congr_ae hlim]
    exact lintegral_liminf_le (fun r => (hmeas r).ennreal_ofReal)
  have hhalf : ENNReal.ofReal ((∫ ω, g ω ∂P)/2) <
      ENNReal.ofReal (∫ ω, g ω ∂P) := by
    exact (ENNReal.ofReal_lt_ofReal_iff hgpos).mpr (by linarith)
  have hevent := eventually_lt_of_lt_liminf (hhalf.trans_le hfatou)
  filter_upwards [hevent, hfint] with r hr hri
  rw [← ofReal_integral_eq_lintegral_ofReal hri
    (Eventually.of_forall (fun ω => hnonneg r ω))] at hr
  exact (ENNReal.ofReal_lt_ofReal_iff'.mp hr).1.le

private theorem nfloor_weight_continuous :
    Continuous (fun p : ℝ × (Fin 6 → ℝ) => gn_normalizedWeight p.1 p.2) := by
  unfold gn_normalizedWeight
  fun_prop
private theorem nfloor_contact_value (a q : ℝ) :
    gn_normalizedWeight 0 ![-1,1,-a/2,a/2,q,q] = (max (-q) 0)^2 := by
  simp [gn_normalizedWeight, pow_two]
private theorem nfloor_weight_limit (X : ℝ → Fin 6 → ℝ) (a q : ℝ)
    (hX : Tendsto X (𝓝[>] (0:ℝ)) (𝓝 ![-1,1,-a/2,a/2,q,q])) :
    Tendsto (fun r => gn_normalizedWeight r (X r)) (𝓝[>] (0:ℝ))
      (𝓝 ((max (-q) 0)^2)) := by
  have hr : Tendsto (fun r : ℝ => r) (𝓝[>] (0:ℝ)) (𝓝 0) :=
    continuousWithinAt_id
  simpa only [nfloor_contact_value, Function.comp_def] using
    (nfloor_weight_continuous.tendsto (0, ![-1,1,-a/2,a/2,q,q])).comp (hr.prodMk_nhds hX)
private theorem nfloor_reference {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (Q : Ω → ℝ) (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0)
    (hlaw : HasLaw Q (gaussianReal m v) P) :
    Integrable (fun ω => (max (-Q ω) 0)^2) P ∧
      0 < ∫ ω, (max (-Q ω) 0)^2 ∂P := by
  obtain ⟨hInt,hPos⟩ := gn_gaussian_reference_integrable_pos m v hv
  refine ⟨hlaw.integrable_comp hInt, ?_⟩
  have heq := hlaw.integral_comp
    (show AEStronglyMeasurable (fun q : ℝ => (max (-q) 0)^2) (gaussianReal m v) by fun_prop)
  exact heq.trans_gt hPos

/-- The actual integral of the physical r²-scaled continuous endpoint weight
has a positive lower constant at every sufficiently small positive radius.
The common coupling and current integrability are explicit premises; its
positive reference integral and the lower bound are proved, not premises. -/
theorem gn_normalizer_floor_of_common_coupling
    {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (X : ℝ → Ω → Fin 6 → ℝ) (hX : ∀ r, Measurable (X r))
    (A Q : Ω → ℝ) (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0)
    (hQ : HasLaw Q (gaussianReal m v) P)
    (hconv : ∀ᵐ ω ∂P, Tendsto (fun r => X r ω) (𝓝[>] (0 : ℝ))
      (𝓝 ![-1,1,-A ω/2,A ω/2,Q ω,Q ω]))
    (hint : ∀ᶠ r in 𝓝[>] (0 : ℝ),
      Integrable (fun ω => gn_normalizedWeight r (X r ω)) P) :
    ∃ c : ℝ, 0 < c ∧ ∃ δ : ℝ, 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ →
        c * r^2 ≤ ∫ ω, r^2 * gn_normalizedWeight r (X r ω) ∂P := by
  let g : Ω → ℝ := fun ω => (max (-Q ω) 0)^2
  obtain ⟨hg, hgpos⟩ := nfloor_reference P Q m v hv hQ
  have hmeas (r : ℝ) : Measurable (fun ω => gn_normalizedWeight r (X r ω)) := by
    have hc : Continuous (gn_normalizedWeight r) := by
      unfold gn_normalizedWeight
      fun_prop
    exact hc.measurable.comp (hX r)
  have hnonneg (r : ℝ) (ω : Ω) : 0 ≤ gn_normalizedWeight r (X r ω) :=
    mul_nonneg (le_max_right _ _) (le_max_right _ _)
  have hweightconv : ∀ᵐ ω ∂P,
      Tendsto (fun r => gn_normalizedWeight r (X r ω)) (𝓝[>] (0 : ℝ)) (𝓝 (g ω)) := by
    filter_upwards [hconv] with ω hω
    exact nfloor_weight_limit (fun r => X r ω) (A ω) (Q ω) hω
  have hfloor : ∀ᶠ r in 𝓝[>] (0 : ℝ),
      (∫ ω, g ω ∂P)/2 ≤ ∫ ω, gn_normalizedWeight r (X r ω) ∂P :=
    nfloor_eventually_lower_of_fatou P (𝓝[>] (0 : ℝ))
      (fun r ω => gn_normalizedWeight r (X r ω)) g hmeas hnonneg hg hgpos hweightconv hint
  obtain ⟨δ, hδ, hinterval⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp hfloor
  refine ⟨(∫ ω, g ω ∂P)/2, by positivity, δ, hδ, ?_⟩
  intro r hr hrδ
  have hbound := hinterval ⟨hr, hrδ⟩
  rw [integral_const_mul]
  simpa only [mul_comm] using mul_le_mul_of_nonneg_left hbound (sq_nonneg r)

end ResearchFormalCoreR1
