import ResearchFormalCoreR1.GaussianCoupling
import ResearchFormalCoreR1.NormalizerFloor
import ResearchFormalCoreR1.EndpointNormalizerBridge
import Mathlib.Probability.Moments.Variance
import Mathlib.MeasureTheory.Function.LpSeminorm.CompareExp

/-!
# A full normalizer floor from finite Gaussian parameter limits

Contact identities follow from the displayed zero variances. The common
coupling, noncentered nondegenerate scalar reference and weight integrability
are derived. No positive normalizer, coupling limit or moment bound is assumed.
Actual torus conditional means/covariances remain a separate source join.
See ../SOURCE_GAUSSIAN_NORMALIZER_LIMIT.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Filter Matrix WithLp
open scoped Topology ENNReal NNReal MatrixOrder Matrix.Norms.L2Operator
set_option autoImplicit false

private theorem gnl_contact_relations (m : EuclideanSpace ℝ (Fin 6)) (S : Matrix (Fin 6) (Fin 6) ℝ)
    (hS : S.PosSemidef)
    (hm0 : m 0 = -1) (hm1 : m 1 = 1)
    (hm23 : m 2 + m 3 = 0) (hm45 : m 4 - m 5 = 0)
    (hS0 : S 0 0 = 0) (hS1 : S 1 1 = 0)
    (hS23 : S 2 2 + 2*S 2 3 + S 3 3 = 0)
    (hS45 : S 4 4 - 2*S 4 5 + S 5 5 = 0) :
    ∀ᵐ x ∂multivariateGaussian m S,
      x 0 = -1 ∧ x 1 = 1 ∧ x 2 + x 3 = 0 ∧ x 4 - x 5 = 0 := by
  let P := multivariateGaussian m S
  have hLp (i : Fin 6) : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => x i) 2 P := by
    exact (IsGaussian.memLp_two_id (μ := P)).continuousLinearMap_comp (EuclideanSpace.proj (𝕜 := ℝ) i)
  have hi (i : Fin 6) : ∫ x : EuclideanSpace ℝ (Fin 6), x i ∂P = m i := by
    change ∫ x, (EuclideanSpace.proj (𝕜 := ℝ) i) x ∂P = m i
    rw [ContinuousLinearMap.integral_comp_id_comm]
    · simp [P]
    · exact IsGaussian.integrable_id
  have hvar0 : Var[fun x : EuclideanSpace ℝ (Fin 6) => x 0; P] = 0 := by
    exact (variance_eval_multivariateGaussian hS 0).trans hS0
  have hvar1 : Var[fun x : EuclideanSpace ℝ (Fin 6) => x 1; P] = 0 := by
    exact (variance_eval_multivariateGaussian hS 1).trans hS1
  have hvar23 : Var[fun x : EuclideanSpace ℝ (Fin 6) => x 2 + x 3; P] = 0 := by
    rw [variance_fun_add (hLp 2) (hLp 3)]
    simpa [P, variance_eval_multivariateGaussian hS,
      covariance_eval_multivariateGaussian hS] using hS23
  have hvar45 : Var[fun x : EuclideanSpace ℝ (Fin 6) => x 4 - x 5; P] = 0 := by
    rw [variance_fun_sub (hLp 4) (hLp 5)]
    simpa [P, variance_eval_multivariateGaussian hS,
      covariance_eval_multivariateGaussian hS] using hS45
  have hi23 : ∫ x : EuclideanSpace ℝ (Fin 6), (x 2 + x 3) ∂P = 0 := by
    rw [integral_add ((hLp 2).integrable (by norm_num)) ((hLp 3).integrable (by norm_num)), hi, hi]
    exact hm23
  have hi45 : ∫ x : EuclideanSpace ℝ (Fin 6), (x 4 - x 5) ∂P = 0 := by
    rw [integral_sub ((hLp 4).integrable (by norm_num)) ((hLp 5).integrable (by norm_num)), hi, hi]
    exact hm45
  filter_upwards [ae_eq_integral_of_variance_eq_zero (hLp 0) hvar0,
    ae_eq_integral_of_variance_eq_zero (hLp 1) hvar1,
    ae_eq_integral_of_variance_eq_zero ((hLp 2).add (hLp 3)) hvar23,
    ae_eq_integral_of_variance_eq_zero ((hLp 4).sub (hLp 5)) hvar45]
    with x hx0 hx1 hx23 hx45
  exact ⟨hx0.trans ((hi 0).trans hm0), hx1.trans ((hi 1).trans hm1),
    hx23.trans hi23, hx45.trans hi45⟩

private lemma gnl_positivePart_memLp {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {f : Ω → ℝ} {p : ℝ≥0∞} (hf : MemLp f p P) :
    MemLp (fun x => max (f x) 0) p P := by
  apply hf.norm.mono' ((hf.aestronglyMeasurable.aemeasurable.sup_const 0).aestronglyMeasurable)
  filter_upwards [] with x
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
  exact max_le (le_abs_self _) (abs_nonneg _)

private lemma gnl_gaussian_weight_integrable (r : ℝ) (m : EuclideanSpace ℝ (Fin 6))
    (S : Matrix (Fin 6) (Fin 6) ℝ) (hS : S.PosSemidef) :
    Integrable (fun x : EuclideanSpace ℝ (Fin 6) => gn_normalizedWeight r (ofLp x))
      (multivariateGaussian m S) := by
  have : ENNReal.HolderTriple 4 4 2 := ⟨by
    rw [← two_mul, show (4 : ℝ≥0∞) = 2*2 by norm_num, ENNReal.mul_inv (by simp) (by simp),
      ← mul_assoc, ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_mul]⟩
  let P := multivariateGaussian m S
  have hc (i : Fin 6) : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => x i) 4 P := by
    have hp := measurePreserving_eval_multivariateGaussian (μ := m) hS (i := i)
    have hlaw : HasLaw (fun x : EuclideanSpace ℝ (Fin 6) => x i)
        (gaussianReal (m i) (S i i).toNNReal) P := ⟨hp.measurable.aemeasurable, hp.map_eq⟩
    exact hlaw.memLp (memLp_id_gaussianReal 4)
  have hpos (i : Fin 6) : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => max (-x i) 0) 4 P :=
    gnl_positivePart_memLp (hc i).neg
  have hM : MemLp (fun x : EuclideanSpace ℝ (Fin 6) =>
      max (-x 0) 0 * max (-x 4) 0 - r*(x 2)^2) 2 P := by
    have h1 : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => max (-x 0) 0 * max (-x 4) 0) 2 P :=
      (hpos 0).fun_mul (hpos 4)
    have h2 : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => (x 2)^2) 2 P := by
      simpa only [pow_two] using (hc 2).fun_mul (hc 2)
    exact h1.sub (h2.const_mul r)
  have hSadd : MemLp (fun x : EuclideanSpace ℝ (Fin 6) =>
      r*(x 3)^2 - x 1*x 5) 2 P := by
    have h1 : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => (x 3)^2) 2 P := by
      simpa only [pow_two] using (hc 3).fun_mul (hc 3)
    have h2 : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => x 1*x 5) 2 P :=
      (hc 1).fun_mul (hc 5)
    exact (h1.const_mul r).sub h2
  have hT : MemLp (fun x : EuclideanSpace ℝ (Fin 6) => gn_normalizedWeight r (ofLp x)) 1 P :=
    (gnl_positivePart_memLp hM).fun_mul (gnl_positivePart_memLp hSadd)
  exact memLp_one_iff_integrable.mp hT


/-- Gaussian endpoint parameter convergence and concrete contact moments supply
one positive physical normalizer coefficient at every sufficiently small radius.
The covariance may be singular; the transverse reference mean is unrestricted. -/
theorem gnl_normalizer_floor_of_parameter_limits
    (m : ℝ → EuclideanSpace ℝ (Fin 6)) (S : ℝ → Matrix (Fin 6) (Fin 6) ℝ)
    (m0 : EuclideanSpace ℝ (Fin 6)) (S0 : Matrix (Fin 6) (Fin 6) ℝ)
    (hm : Tendsto m (𝓝[>] (0 : ℝ)) (𝓝 m0))
    (hS : Tendsto S (𝓝[>] (0 : ℝ)) (𝓝 S0))
    (hPSD : ∀ᶠ r in 𝓝[>] (0 : ℝ), (S r).PosSemidef) (hPSD0 : S0.PosSemidef)
    (hm0 : m0 0 = -1) (hm1 : m0 1 = 1)
    (hm23 : m0 2 + m0 3 = 0) (hm45 : m0 4 - m0 5 = 0)
    (hS0 : S0 0 0 = 0) (hS1 : S0 1 1 = 0)
    (hS23 : S0 2 2 + 2*S0 2 3 + S0 3 3 = 0)
    (hS45 : S0 4 4 - 2*S0 4 5 + S0 5 5 = 0) (hQpos : 0 < S0 4 4) :
    ∃ c : ℝ, 0 < c ∧ ∃ δ : ℝ, 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ → c*r^2 ≤
        ∫ x : EuclideanSpace ℝ (Fin 6), r^2 * gn_normalizedWeight r (ofLp x)
          ∂multivariateGaussian (m r) (S r) := by
  let P := stdGaussian (EuclideanSpace ℝ (Fin 6))
  let X : ℝ → EuclideanSpace ℝ (Fin 6) → Fin 6 → ℝ :=
    fun r z => ofLp (gc_coupledVector (m r) (S r) z)
  let A : EuclideanSpace ℝ (Fin 6) → ℝ := fun z => 2 * gc_coupledVector m0 S0 z 3
  let Q : EuclideanSpace ℝ (Fin 6) → ℝ := fun z => gc_coupledVector m0 S0 z 4
  have hX (r : ℝ) : Measurable (X r) := by
    dsimp [X, gc_coupledVector]
    fun_prop
  have hLaw0 : HasLaw (gc_coupledVector m0 S0) (multivariateGaussian m0 S0) P :=
    gc_coupledVector_hasLaw m0 S0
  have hcontacts : ∀ᵐ z ∂P,
      gc_coupledVector m0 S0 z 0 = -1 ∧ gc_coupledVector m0 S0 z 1 = 1 ∧
      gc_coupledVector m0 S0 z 2 + gc_coupledVector m0 S0 z 3 = 0 ∧
      gc_coupledVector m0 S0 z 4 - gc_coupledVector m0 S0 z 5 = 0 := by
    have hb : ∀ᵐ x ∂P.map (gc_coupledVector m0 S0),
        x 0 = -1 ∧ x 1 = 1 ∧ x 2 + x 3 = 0 ∧ x 4 - x 5 = 0 := by
      rw [hLaw0.map_eq]
      exact gnl_contact_relations m0 S0 hPSD0 hm0 hm1 hm23 hm45 hS0 hS1 hS23 hS45
    exact ae_of_ae_map hLaw0.aemeasurable hb
  have hconv : ∀ᵐ z ∂P, Tendsto (fun r => X r z) (𝓝[>] (0 : ℝ))
      (𝓝 ![-1,1,-A z/2,A z/2,Q z,Q z]) := by
    filter_upwards [hcontacts] with z hz
    have hshape : ofLp (gc_coupledVector m0 S0 z) = ![-1,1,-A z/2,A z/2,Q z,Q z] := by
      funext i
      fin_cases i <;> simp [A, Q] <;> linarith [hz.1, hz.2.1, hz.2.2.1, hz.2.2.2]
    have ht := ((PiLp.continuous_ofLp 2 (fun _ : Fin 6 => ℝ)).tendsto
      (gc_coupledVector m0 S0 z)).comp
        (gc_coupledVector_tendsto (𝓝[>] (0 : ℝ)) m S m0 S0 hm hS hPSD hPSD0 z)
    simpa only [Function.comp_def, X, hshape] using ht
  have hv : (S0 4 4).toNNReal ≠ 0 := ne_of_gt (Real.toNNReal_pos.mpr hQpos)
  have hQ : HasLaw Q (gaussianReal (m0 4) (S0 4 4).toNNReal) P := by
    exact (measurePreserving_eval_multivariateGaussian (μ := m0) hPSD0 (i := 4)).hasLaw.fun_comp hLaw0
  have hint : ∀ᶠ r in 𝓝[>] (0 : ℝ),
      Integrable (fun z => gn_normalizedWeight r (X r z)) P := by
    filter_upwards [hPSD] with r hr
    exact (gc_coupledVector_hasLaw (m r) (S r)).integrable_comp
      (gnl_gaussian_weight_integrable r (m r) (S r) hr)
  obtain ⟨c, hc, δ, hδ, hfloor⟩ :=
    gn_normalizer_floor_of_common_coupling P X hX A Q (m0 4) (S0 4 4).toNNReal hv hQ hconv hint
  refine ⟨c, hc, δ, hδ, ?_⟩
  intro r hr hrδ
  calc
    c*r^2 ≤ ∫ z, r^2 * gn_normalizedWeight r (X r z) ∂P := hfloor r hr hrδ
    _ = ∫ x : EuclideanSpace ℝ (Fin 6), r^2 * gn_normalizedWeight r (ofLp x)
        ∂multivariateGaussian (m r) (S r) := by
      simpa only [Function.comp_def, X] using
        (gc_coupledVector_hasLaw (m r) (S r)).integral_comp
          (f := fun x : EuclideanSpace ℝ (Fin 6) => r^2 * gn_normalizedWeight r (ofLp x))
          (by unfold gn_normalizedWeight; fun_prop)


/-- The same Gaussian parameter limit gives the full literal typed endpoint
normalizer, after the explicit permutation back to physical endpoint order. -/
theorem gnl_physical_normalizer_floor_of_parameter_limits
    (m : ℝ → EuclideanSpace ℝ (Fin 6)) (S : ℝ → Matrix (Fin 6) (Fin 6) ℝ)
    (m0 : EuclideanSpace ℝ (Fin 6)) (S0 : Matrix (Fin 6) (Fin 6) ℝ)
    (hm : Tendsto m (𝓝[>] (0 : ℝ)) (𝓝 m0))
    (hS : Tendsto S (𝓝[>] (0 : ℝ)) (𝓝 S0))
    (hPSD : ∀ᶠ r in 𝓝[>] (0 : ℝ), (S r).PosSemidef) (hPSD0 : S0.PosSemidef)
    (hm0 : m0 0 = -1) (hm1 : m0 1 = 1)
    (hm23 : m0 2 + m0 3 = 0) (hm45 : m0 4 - m0 5 = 0)
    (hS0 : S0 0 0 = 0) (hS1 : S0 1 1 = 0)
    (hS23 : S0 2 2 + 2*S0 2 3 + S0 3 3 = 0)
    (hS45 : S0 4 4 - 2*S0 4 5 + S0 5 5 = 0) (hQpos : 0 < S0 4 4) :
    ∃ c : ℝ, 0 < c ∧ ∃ δ : ℝ, 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ → c*r^2 ≤
        ∫ x : EuclideanSpace ℝ (Fin 6), egm_weight r ![x 0,x 2,x 4,x 1,x 3,x 5]
          ∂multivariateGaussian (m r) (S r) := by
  obtain ⟨c, hc, δ, hδ, hfloor⟩ := gnl_normalizer_floor_of_parameter_limits
    m S m0 S0 hm hS hPSD hPSD0 hm0 hm1 hm23 hm45 hS0 hS1 hS23 hS45 hQpos
  refine ⟨c, hc, δ, hδ, ?_⟩
  intro r hr hrδ
  calc
    c*r^2 ≤ ∫ x : EuclideanSpace ℝ (Fin 6), r^2 * gn_normalizedWeight r (ofLp x)
        ∂multivariateGaussian (m r) (S r) := hfloor r hr hrδ
    _ = ∫ x : EuclideanSpace ℝ (Fin 6), egm_weight r ![x 0,x 2,x 4,x 1,x 3,x 5]
        ∂multivariateGaussian (m r) (S r) := by
      apply integral_congr_ae
      filter_upwards [] with x
      have hp : enb_permutation ![x 0,x 2,x 4,x 1,x 3,x 5] = ofLp x := by
        funext i
        fin_cases i <;> rfl
      simpa only [hp] using
        (enb_weight_identity r ![x 0,x 2,x 4,x 1,x 3,x 5] hr).symm

end ResearchFormalCoreR1
