import SourceSmoothConditionalKernel
import GaussianWeightedMajorants
import SourceSmoothPalm

/-!
# Actual conditional Fourier derivative majorants

The posterior retains its literal coefficient shift and noncentered drift.
All series below are justified on the existing smooth sample carrier.
See SOURCE_CONDITIONAL_MAJORANT.md for the frozen source contract.
-/

noncomputable section

open I4Foundation I4Source I4Weights I4Smooth I4SmoothCarrier I4SourceDerivativeGaussian
open I4SourceDerivativeContinuity I4SourceConfluentLimits I4SourceEndpointRegression
open I4SourceEndpointEstimates I4SourceContact I4SourceSmoothConditional I4ContactGram
open I4GaussianMajorant
open I4SourceSmoothPalm
open MeasureTheory ProbabilityTheory Matrix Filter
open scoped BigOperators Topology

set_option autoImplicit false

namespace I4SourceConditionalMajorant

def sampleMajorant {L : ℝ} (w : SourceMode 2 → ℝ) (ω : SmoothSample 2 L) : ℝ :=
  ∑' m, w m * |ω.val m|

def sourceDerivativeMajorant (L : ℝ) (k : ℕ) (ω : SmoothSample 2 L) : ℝ :=
  sampleMajorant (orderModeBound k 2 L) ω

theorem sampleMajorant_nonneg {L : ℝ} (w : SourceMode 2 → ℝ)
    (hw : ∀ m, 0 ≤ w m) (ω : SmoothSample 2 L) : 0 ≤ sampleMajorant w ω :=
  tsum_nonneg (fun m => mul_nonneg (hw m) (abs_nonneg _))

theorem derivativeMajorant_weighted_summable {L : ℝ} (ω : SmoothSample 2 L)
    (k : ℕ) (u : Fin k → (Fin 2 → ℝ)) :
    Summable (fun m => derivativeMajorant 2 L k u m * |ω.val m|) := by
  have h := (ω.property k).mul_right (∏ j, ‖u j‖)
  simpa only [derivativeMajorant, mul_assoc, mul_left_comm, mul_comm] using h

theorem confluentMajorant_weighted_summable {L : ℝ} (ω : SmoothSample 2 L) :
    Summable (fun m => confluentModeMajorant L m * |ω.val m|) := by
  have hc (s : Fin 8) : Summable (fun m => modeComponentMajorant L m s * |ω.val m|) := by
    fin_cases s <;> dsimp [modeComponentMajorant]
    all_goals exact derivativeMajorant_weighted_summable ω _ _
  have hs := (summable_sum (fun s (_ : s ∈ (Finset.univ : Finset (Fin 8))) => hc s)).mul_left 6
  convert hs using 1
  funext m
  simp only [confluentModeMajorant, Finset.sum_mul, mul_assoc]

theorem pinCoefficient_abs_le {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0)
    (i : Fin 6) (m : SourceMode 2) : |pinCoefficient L r i m| ≤ confluentModeMajorant L m := by
  rw [pinCoefficient_eq_regularized L r hr]
  simpa only [Real.norm_eq_abs] using regularizedPinCoefficient_bound hL r i m

theorem sourceConfluentPins_abs_le {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0)
    (ω : SmoothSample 2 L) (i : Fin 6) :
    |sourceConfluentPinsAt L r ω i| ≤ sampleMajorant (confluentModeMajorant L) ω := by
  rw [congrFun (sourceConfluentPinsAt_series hL r ω) i]
  change |∑' m, pinCoefficient L r i m * ω.val m| ≤ _
  have h := tsum_of_norm_bounded (confluentMajorant_weighted_summable ω).hasSum
    (fun m => show ‖pinCoefficient L r i m * ω.val m‖ ≤
        confluentModeMajorant L m * |ω.val m| by
      rw [Real.norm_eq_abs, abs_mul]
      exact mul_le_mul_of_nonneg_right (pinCoefficient_abs_le hL r hr i m) (abs_nonneg _))
  simpa only [Real.norm_eq_abs, sampleMajorant] using h

private theorem six_mulVec_abs_le (M : Matrix (Fin 6) (Fin 6) ℝ)
    (K T : ℝ) (hK : 0 ≤ K) (_hT : 0 ≤ T)
    (hM : ∀ i j, |M i j| ≤ K) (y : Fin 6 → ℝ) (hy : ∀ j, |y j| ≤ T) (i : Fin 6) :
    |(M *ᵥ y) i| ≤ 6*K*T := by
  calc
    _ ≤ ∑ j, |M i j * y j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j : Fin 6, K*T := Finset.sum_le_sum (fun j _ => by
      rw [abs_mul]
      exact mul_le_mul (hM i j) (hy j) (abs_nonneg _) hK)
    _ = _ := by simp; ring

theorem pinLift_abs_le {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0)
    (K D : ℝ) (hK : 0 ≤ K) (hD : 0 ≤ D)
    (hInv : ∀ i j, |(sourcePinCovarianceAt L r)⁻¹ i j| ≤ K)
    (x : Fin 6 → ℝ) (hx : ∀ i, |x i| ≤ D) (ω : SmoothSample 2 L) (m : SourceMode 2) :
    |pinLift L r (x - sourceConfluentPinsAt L r ω) m| ≤
      36*K*confluentModeMajorant L m *
        (D + sampleMajorant (confluentModeMajorant L) ω) := by
  let T := D + sampleMajorant (confluentModeMajorant L) ω
  have hT : 0 ≤ T := add_nonneg hD
    (sampleMajorant_nonneg _ (confluentModeMajorant_nonneg L) ω)
  have hy (i : Fin 6) : |(x - sourceConfluentPinsAt L r ω) i| ≤ T := by
    exact (abs_sub _ _).trans (add_le_add (hx i) (sourceConfluentPins_abs_le hL r hr ω i))
  have hv := six_mulVec_abs_le ((sourcePinCovarianceAt L r)⁻¹) K T hK hT hInv
    (x - sourceConfluentPinsAt L r ω) hy
  unfold pinLift
  calc
    _ ≤ ∑ i, |pinCoefficient L r i m *
        (((sourcePinCovarianceAt L r)⁻¹ *ᵥ (x - sourceConfluentPinsAt L r ω)) i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin 6, confluentModeMajorant L m * (6*K*T) :=
      Finset.sum_le_sum (fun i _ => by
        rw [abs_mul]
        exact mul_le_mul (pinCoefficient_abs_le hL r hr i m) (hv i) (abs_nonneg _)
          (confluentModeMajorant_nonneg L m))
    _ = _ := by simp [T]; ring

/-- The deterministic posterior bound includes every required summability
premise for a general weight. Source derivative weights supply it pointwise. -/
theorem posterior_sampleMajorant_le {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0)
    (w : SourceMode 2 → ℝ) (hw0 : ∀ m, 0 ≤ w m)
    (hwA : Summable (fun m => w m * confluentModeMajorant L m))
    (K D : ℝ) (hK : 0 ≤ K) (hD : 0 ≤ D)
    (hInv : ∀ i j, |(sourcePinCovarianceAt L r)⁻¹ i j| ≤ K)
    (x : Fin 6 → ℝ) (hx : ∀ i, |x i| ≤ D) (ω : SmoothSample 2 L)
    (hwω : Summable (fun m => w m * |ω.val m|)) :
    Summable (fun m => w m * |(posteriorSample hL r x ω).val m|) ∧
      sampleMajorant w (posteriorSample hL r x ω) ≤ sampleMajorant w ω +
        36*K*(∑' m, w m * confluentModeMajorant L m)*
          (D + sampleMajorant (confluentModeMajorant L) ω) := by
  let T := D + sampleMajorant (confluentModeMajorant L) ω
  let C := 36*K*T
  have hC : 0 ≤ C := mul_nonneg (mul_nonneg (by norm_num) hK)
    (add_nonneg hD (sampleMajorant_nonneg _ (confluentModeMajorant_nonneg L) ω))
  have hmajor := hwω.add (hwA.mul_left C)
  have hpoint (m : SourceMode 2) : w m * |(posteriorSample hL r x ω).val m| ≤
      w m * |ω.val m| + C*(w m*confluentModeMajorant L m) := by
    rw [posteriorSample_val]
    calc
      _ ≤ w m * (|ω.val m| + |pinLift L r (x-sourceConfluentPinsAt L r ω) m|) :=
        mul_le_mul_of_nonneg_left (abs_add_le _ _) (hw0 m)
      _ ≤ w m * (|ω.val m| + 36*K*confluentModeMajorant L m*T) :=
        mul_le_mul_of_nonneg_left
          (add_le_add le_rfl (pinLift_abs_le hL r hr K D hK hD hInv x hx ω m)) (hw0 m)
      _ = _ := by dsimp [C]; ring
  have hpost : Summable (fun m => w m * |(posteriorSample hL r x ω).val m|) :=
    Summable.of_nonneg_of_le (fun m => mul_nonneg (hw0 m) (abs_nonneg _)) hpoint hmajor
  refine ⟨hpost, ?_⟩
  calc
    _ ≤ ∑' m, (w m * |ω.val m| + C*(w m*confluentModeMajorant L m)) :=
      hpost.tsum_le_tsum hpoint hmajor
    _ = sampleMajorant w ω + C*(∑' m, w m*confluentModeMajorant L m) := by
      rw [hwω.tsum_add (hwA.mul_left C), tsum_mul_left]
      rfl
    _ = _ := by dsimp [C, T]; ring

/-- Actual inverse continuity and the actual prescribed pin limit derive the
uniform constants used by the posterior bound; neither is a model premise. -/
theorem sourceInverse_prescribed_uniform_bounds {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ K D δ : ℝ, 0 ≤ K ∧ 0 ≤ D ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ →
        (∀ i j, |(sourcePinCovarianceAt L r)⁻¹ i j| ≤ K) ∧
        (∀ i, |sourcePrescribedPins r b i| ≤ D) := by
  have hdet : (sourcePinCovariance L).det ≠ 0 :=
    ((Matrix.isUnit_iff_isUnit_det _).mp (sourcePinCovariance_isUnit hL)).ne_zero
  have hc : ContinuousAt (Ring.inverse : ℝ → ℝ) (sourcePinCovariance L).det := by
    simpa only [Ring.inverse_eq_inv'] using continuousAt_inv₀ hdet
  have hi := (continuousAt_matrix_inv (sourcePinCovariance L) hc).tendsto.comp
    (sourcePinCovarianceAt_tendsto hL)
  have hi' : Tendsto (fun r => fun i j => (sourcePinCovarianceAt L r)⁻¹ i j)
      (𝓝[>] (0 : ℝ)) (𝓝 (fun i j => (sourcePinCovariance L)⁻¹ i j)) := hi
  have hib : ∀ᶠ r in 𝓝[>] (0 : ℝ), ‖fun i j => (sourcePinCovarianceAt L r)⁻¹ i j‖ <
      ‖fun i j => (sourcePinCovariance L)⁻¹ i j‖ + 1 :=
    hi'.norm.eventually (gt_mem_nhds (by linarith))
  have hdb : ∀ᶠ r in 𝓝[>] (0 : ℝ), ‖sourcePrescribedPins r b‖ <
      ‖(![b,0,0,0,0,2] : Fin 6 → ℝ)‖ + 1 :=
    (sourcePrescribedPins_tendsto b).norm.eventually (gt_mem_nhds (by linarith))
  have hunit := (sourceEndpoint_parameter_limits hL b).1
  obtain ⟨δ, hδ, hb⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp ((hib.and hdb).and hunit)
  refine ⟨‖fun i j => (sourcePinCovariance L)⁻¹ i j‖+1,
    ‖(![b,0,0,0,0,2] : Fin 6 → ℝ)‖+1,min δ 1, by positivity, by positivity,
    lt_min hδ (by norm_num),min_le_right _ _,?_⟩
  intro r hr hrδ
  have hh := (hb ⟨hr,lt_of_lt_of_le hrδ (min_le_left _ _)⟩).1
  constructor
  · intro i j
    exact (show |(sourcePinCovarianceAt L r)⁻¹ i j| ≤
        ‖fun i j => (sourcePinCovarianceAt L r)⁻¹ i j‖ from
      (show |(sourcePinCovarianceAt L r)⁻¹ i j| ≤ ‖(sourcePinCovarianceAt L r)⁻¹ i‖ by
        simpa only [Real.norm_eq_abs] using norm_le_pi_norm ((sourcePinCovarianceAt L r)⁻¹ i) j).trans
          (norm_le_pi_norm (fun i j => (sourcePinCovarianceAt L r)⁻¹ i j) i)).trans hh.1.le
  · intro i
    exact (show |sourcePrescribedPins r b i| ≤ ‖sourcePrescribedPins r b‖ by
      simpa only [Real.norm_eq_abs] using norm_le_pi_norm (sourcePrescribedPins r b) i).trans hh.2.le

/-- Every actual derivative is bounded globally in its evaluation point by
the literal all-order Fourier absolute majorant on the smooth carrier. -/
theorem smoothSourceField_global_derivative_bound {L : ℝ} (hL : L ≠ 0)
    (k : ℕ) (ω : SmoothSample 2 L) (x : Fin 2 → ℝ) :
    ‖iteratedFDeriv ℝ k (smoothSourceField 2 L ω) x‖ ≤ sourceDerivativeMajorant L k ω := by
  apply ContinuousMultilinearMap.opNorm_le_bound
    (sampleMajorant_nonneg _ (orderModeBound_nonneg k 2 L) ω)
  intro u
  rw [smoothSourceField_derivative_series 2 hL]
  have hs := (ω.property k).mul_right (∏ j, ‖u j‖)
  have h := tsum_of_norm_bounded hs.hasSum (fun m =>
    show ‖derivativeCoefficient 2 L k x u m * ω.val m‖ ≤
      (orderModeBound k 2 L m * |ω.val m|) * (∏ j, ‖u j‖) by
      simp only [norm_mul, Real.norm_eq_abs]
      have hb := mul_le_mul_of_nonneg_right (derivativeCoefficient_bound 2 hL k u m x)
        (abs_nonneg (ω.val m))
      simpa only [Real.norm_eq_abs, derivativeMajorant, mul_assoc, mul_left_comm, mul_comm] using hb)
  change ‖∑' m, derivativeCoefficient 2 L k x u m * ω.val m‖ ≤ _
  simpa only [sourceDerivativeMajorant, sampleMajorant, tsum_mul_right] using h

theorem sampleMajorant_measurable (L : ℝ) (w : SourceMode 2 → ℝ) :
    Measurable (sampleMajorant (L := L) w) := by
  exact weightedAbsSeries_measurable w (fun m (ω : SmoothSample 2 L) => ω.val m)
    (fun m => (measurable_pi_apply m).comp measurable_subtype_coe)

theorem sourceDerivativeMajorant_measurable (L : ℝ) (k : ℕ) :
    Measurable (sourceDerivativeMajorant L k) := sampleMajorant_measurable L _

theorem sampleMajorant_add {L : ℝ} (w v : SourceMode 2 → ℝ) (ω : SmoothSample 2 L)
    (hw : Summable (fun m => w m * |ω.val m|))
    (hv : Summable (fun m => v m * |ω.val m|)) :
    sampleMajorant (fun m => w m+v m) ω = sampleMajorant w ω + sampleMajorant v ω := by
  unfold sampleMajorant
  simp_rw [add_mul]
  exact hw.tsum_add hv

theorem sampleMajorant_const_mul {L : ℝ} (c : ℝ) (w : SourceMode 2 → ℝ)
    (ω : SmoothSample 2 L) :
    sampleMajorant (fun m => c*w m) ω = c*sampleMajorant w ω := by
  unfold sampleMajorant
  simp_rw [mul_assoc]
  exact tsum_mul_left

/-- The posterior is dominated by one fixed positive summable weight plus
its retained drift, uniformly at every sufficiently small positive radius. -/
theorem sourceDerivativeMajorant_uniform_domination {L : ℝ} (hL : L ≠ 0)
    (b : ℝ) (k : ℕ) :
    ∃ a : ℝ, ∃ w : SourceMode 2 → ℝ, ∃ δ : ℝ,
      0 ≤ a ∧ (∀ m, 0 ≤ w m) ∧ Summable w ∧ 0 < ∑' m, w m ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ → ∀ ω : SmoothSample 2 L,
        sourceDerivativeMajorant L k (posteriorSample hL r (sourcePrescribedPins r b) ω) ≤
          a + sampleMajorant w ω := by
  obtain ⟨K,D,δ,hK,hD,hδ,hδ1,hbounds⟩ := sourceInverse_prescribed_uniform_bounds hL b
  let w0 := orderModeBound k 2 L
  let A := confluentModeMajorant L
  have hw0 : ∀ m, 0 ≤ w0 m := orderModeBound_nonneg k 2 L
  have hA : ∀ m, 0 ≤ A m := confluentModeMajorant_nonneg L
  have hwA : Summable (fun m => w0 m*A m) := coefficient_product_summable
    (orderModeBound_summable k 2 hL) (confluentModeMajorant_summable hL)
  let B := ∑' m, w0 m*A m
  have hB : 0 ≤ B := tsum_nonneg (fun m => mul_nonneg (hw0 m) (hA m))
  let C := 36*K*B
  have hC : 0 ≤ C := mul_nonneg (mul_nonneg (by norm_num) hK) hB
  let w := fun m => w0 m+C*A m
  have hw : Summable w := (orderModeBound_summable k 2 hL).add
    ((confluentModeMajorant_summable hL).mul_left C)
  have hwpos : ∀ m, 0 ≤ w m := fun m => add_nonneg (hw0 m) (mul_nonneg hC (hA m))
  have hDC : 0 < w0 none := by
    change 0 < Real.sqrt (1/sourceZ 2 L)
    exact Real.sqrt_pos.mpr (one_div_pos.mpr (sourceZ_pos 2 hL))
  have hW : 0 < ∑' m, w m :=
    (hDC.trans_le (le_add_of_nonneg_right (mul_nonneg hC (hA none)))).trans_le
      (hw.le_tsum none (fun m _ => hwpos m))
  refine ⟨C*D,w,δ,mul_nonneg hC hD,hwpos,hw,hW,hδ,hδ1,?_⟩
  intro r hr hrδ ω
  obtain ⟨hInv,hx⟩ := hbounds r hr hrδ
  have hbound := (posterior_sampleMajorant_le hL r hr.ne' w0 hw0 hwA K D hK hD hInv
    (sourcePrescribedPins r b) hx ω (ω.property k)).2
  have hrec : sampleMajorant w ω = sourceDerivativeMajorant L k ω +
      C*sampleMajorant A ω := by
    rw [show w = (fun m => w0 m+C*A m) from rfl,
      sampleMajorant_add w0 (fun m => C*A m) ω (ω.property k)]
    · rw [sampleMajorant_const_mul]
      rfl
    · convert (confluentMajorant_weighted_summable ω).mul_left C using 1
      funext m
      ring
  calc
    _ ≤ sampleMajorant w0 ω + 36*K*B*(D+sampleMajorant A ω) := hbound
    _ = C*D + sampleMajorant w ω := by rw [hrec]; dsimp [C,sourceDerivativeMajorant,w0]; ring

theorem source_coordinate_hasLaw {L : ℝ} (hL : L ≠ 0) (m : SourceMode 2) :
    HasLaw (fun ω : SmoothSample 2 L => ω.val m) (gaussianReal 0 1) (smoothLaw 2 L) where
  aemeasurable := ((measurable_pi_apply m).comp measurable_subtype_coe).aemeasurable
  map_eq := I4SmoothCarrier.coordinate_marginal 2 hL m

theorem sampleMajorant_prior_moment {L : ℝ} (hL : L ≠ 0)
    (w : SourceMode 2 → ℝ) (hw0 : ∀ m, 0 ≤ w m) (hw : Summable w) (n : ℕ) :
    Integrable (fun ω : SmoothSample 2 L => sampleMajorant w ω ^ n) (smoothLaw 2 L) ∧
      (∫ ω, sampleMajorant w ω ^ n ∂smoothLaw 2 L) ≤
        (∑' m, w m)^n * standardAbsMoment n := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  exact weightedAbsSeries_moment (smoothLaw 2 L) w
    (fun m (ω : SmoothSample 2 L) => ω.val m)
    (fun m => (measurable_pi_apply m).comp measurable_subtype_coe) hw0 hw
    (source_coordinate_hasLaw hL) n

private theorem drift_majorant_integrable_pow {L : ℝ} (hL : L ≠ 0)
    (a : ℝ) (ha : 0 ≤ a) (w : SourceMode 2 → ℝ) (hw0 : ∀ m, 0 ≤ w m)
    (hw : Summable w) (n : ℕ) :
    Integrable (fun ω : SmoothSample 2 L => (a+sampleMajorant w ω)^n) (smoothLaw 2 L) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hΓ := (sampleMajorant_prior_moment hL w hw0 hw n).1
  have hmajor : Integrable (fun ω : SmoothSample 2 L =>
      (2:ℝ)^(n-1)*(a^n+sampleMajorant w ω^n)) (smoothLaw 2 L) :=
    ((integrable_const _).add hΓ).const_mul _
  apply hmajor.mono' (((sampleMajorant_measurable L w).const_add a).pow_const n).aestronglyMeasurable
  filter_upwards [] with ω
  rw [Real.norm_eq_abs,abs_of_nonneg (pow_nonneg (add_nonneg ha (sampleMajorant_nonneg w hw0 ω)) n)]
  exact add_pow_le ha (sampleMajorant_nonneg w hw0 ω) n

/-- Every natural conditional derivative-majorant moment is derived from the
actual posterior sampling map; the common small-radius interval is uniform in n. -/
theorem sourceConditionalDerivative_moments {L : ℝ} (hL : L ≠ 0) (b : ℝ) (k : ℕ) :
    ∃ δ : ℝ, 0 < δ ∧ δ ≤ 1 ∧ ∀ n : ℕ, ∃ M : ℝ, 0 ≤ M ∧
      ∀ r : ℝ, 0 < r → r < δ →
        Integrable (fun ω => sourceDerivativeMajorant L k ω ^ n)
          (sourceSmoothKernel hL r (sourcePrescribedPins r b)) ∧
        (∫ ω, sourceDerivativeMajorant L k ω ^ n
          ∂sourceSmoothKernel hL r (sourcePrescribedPins r b)) ≤ M := by
  obtain ⟨a,w,δ,ha,hw0,hw,_hW,hδ,hδ1,hdom⟩ := sourceDerivativeMajorant_uniform_domination hL b k
  refine ⟨δ,hδ,hδ1,?_⟩
  intro n
  let F : SmoothSample 2 L → ℝ := fun ω => (a+sampleMajorant w ω)^n
  have hF : Integrable F (smoothLaw 2 L) := drift_majorant_integrable_pow hL a ha w hw0 hw n
  refine ⟨∫ ω, F ω ∂smoothLaw 2 L,integral_nonneg (fun ω =>
    pow_nonneg (add_nonneg ha (sampleMajorant_nonneg w hw0 ω)) n),?_⟩
  intro r hr hrδ
  have hp : Measurable (posteriorSample hL r (sourcePrescribedPins r b)) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  have hg := (sourceDerivativeMajorant_measurable L k).pow_const n
  have hpoint (ω : SmoothSample 2 L) :
      sourceDerivativeMajorant L k (posteriorSample hL r (sourcePrescribedPins r b) ω)^n ≤ F ω :=
    pow_le_pow_left₀ (sampleMajorant_nonneg _ (orderModeBound_nonneg k 2 L) _) (hdom r hr hrδ ω) n
  have hpost : Integrable (fun ω => sourceDerivativeMajorant L k
      (posteriorSample hL r (sourcePrescribedPins r b) ω)^n) (smoothLaw 2 L) := by
    apply hF.mono' (hg.comp hp).aestronglyMeasurable
    filter_upwards [] with ω
    rw [Real.norm_eq_abs]
    change |sourceDerivativeMajorant L k (posteriorSample hL r (sourcePrescribedPins r b) ω)^n| ≤ F ω
    have hnon : 0 ≤ sourceDerivativeMajorant L k
        (posteriorSample hL r (sourcePrescribedPins r b) ω)^n :=
      pow_nonneg (sampleMajorant_nonneg _ (orderModeBound_nonneg k 2 L) _) n
    rw [abs_of_nonneg hnon]
    exact hpoint ω
  rw [sourceSmoothKernel_apply hL r (sourcePrescribedPins r b)]
  refine ⟨(integrable_map_measure hg.aestronglyMeasurable hp.aemeasurable).mpr hpost,?_⟩
  rw [integral_map hp.aemeasurable hg.aestronglyMeasurable]
  exact integral_mono hpost hF hpoint

/-- Markov gives every polynomial rate for the literal measurable majorant
cutoff. This event certifies the global derivative bound without a supremum. -/
theorem sourceConditionalDerivative_cutoff_tail {L : ℝ} (hL : L ≠ 0)
    (b : ℝ) (k n : ℕ) :
    ∃ M δ : ℝ, 0 ≤ M ∧ 0 < δ ∧ δ ≤ 1 ∧ ∀ r : ℝ, 0 < r → r < δ →
      (sourceSmoothKernel hL r (sourcePrescribedPins r b)).real
        {ω | r⁻¹ < sourceDerivativeMajorant L k ω} ≤ M*r^n := by
  obtain ⟨δ,hδ,hδ1,hmom⟩ := sourceConditionalDerivative_moments hL b k
  obtain ⟨M,hM,hm⟩ := hmom n
  refine ⟨M,δ,hM,hδ,hδ1,?_⟩
  intro r hr hrδ
  obtain ⟨hint,hbound⟩ := hm r hr hrδ
  let P := sourceSmoothKernel hL r (sourcePrescribedPins r b)
  have : IsProbabilityMeasure P := inferInstance
  have hpow : 0 ≤ᵐ[P] (fun ω => sourceDerivativeMajorant L k ω ^ n) :=
    Eventually.of_forall (fun ω => pow_nonneg
      (sampleMajorant_nonneg _ (orderModeBound_nonneg k 2 L) ω) n)
  have hMarkov := mul_meas_ge_le_integral_of_nonneg hpow hint ((r⁻¹)^n)
  have hsub : {ω : SmoothSample 2 L | r⁻¹ < sourceDerivativeMajorant L k ω} ⊆
      {ω | (r⁻¹)^n ≤ sourceDerivativeMajorant L k ω^n} :=
    fun ω hω => pow_le_pow_left₀ (inv_nonneg.mpr hr.le) hω.le n
  have hreal := measureReal_mono (μ := P) hsub
  have hprod : (r⁻¹)^n * P.real {ω | r⁻¹ < sourceDerivativeMajorant L k ω} ≤ M :=
    (mul_le_mul_of_nonneg_left hreal (pow_nonneg (inv_nonneg.mpr hr.le) n)).trans
      (hMarkov.trans hbound)
  have hcancel : ((r⁻¹)^n * P.real {ω | r⁻¹ < sourceDerivativeMajorant L k ω}) * r^n =
      P.real {ω | r⁻¹ < sourceDerivativeMajorant L k ω} := by
    rw [mul_right_comm,←mul_pow,inv_mul_cancel₀ hr.ne',one_pow,one_mul]
  rw [←hcancel]
  exact mul_le_mul_of_nonneg_right hprod (pow_nonneg hr.le n)

/-- A genuine Gaussian upper tail follows from the independently proved
folded-Gaussian exponential estimate, with the posterior drift retained. -/
theorem sourceConditionalDerivative_gaussian_upper_tail {L : ℝ} (hL : L ≠ 0)
    (b : ℝ) (k : ℕ) :
    ∃ a s δ : ℝ, 0 ≤ a ∧ 0 < s ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ → ∀ R : ℝ, 0 ≤ R →
        (sourceSmoothKernel hL r (sourcePrescribedPins r b)).real
          {ω | a+R ≤ sourceDerivativeMajorant L k ω} ≤
            2*Real.exp (-(R^2)/(2*s^2)) := by
  obtain ⟨a,w,δ,ha,hw0,hw,hW,hδ,hδ1,hdom⟩ := sourceDerivativeMajorant_uniform_domination hL b k
  refine ⟨a,∑' m,w m,δ,ha,hW,hδ,hδ1,?_⟩
  intro r hr hrδ R hR
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hp : Measurable (posteriorSample hL r (sourcePrescribedPins r b)) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  have hE : MeasurableSet {ω : SmoothSample 2 L | a+R ≤ sourceDerivativeMajorant L k ω} :=
    measurableSet_le measurable_const (sourceDerivativeMajorant_measurable L k)
  have htail := weightedAbsSeries_upper_tail (smoothLaw 2 L) w
    (fun m (ω : SmoothSample 2 L) => ω.val m)
    (fun m => (measurable_pi_apply m).comp measurable_subtype_coe) hw0 hw
    (source_coordinate_hasLaw hL) hW R hR
  rw [sourceSmoothKernel_apply hL r (sourcePrescribedPins r b),measureReal_def,
    Measure.map_apply hp hE]
  change (smoothLaw 2 L).real
    {ω | a+R ≤ sourceDerivativeMajorant L k (posteriorSample hL r (sourcePrescribedPins r b) ω)} ≤ _
  calc
    _ ≤ (smoothLaw 2 L).real {ω | R ≤ sampleMajorant w ω} := by
      apply measureReal_mono
      · intro ω hω
        have hd := hdom r hr hrδ ω
        change a+R ≤ sourceDerivativeMajorant L k (posteriorSample hL r (sourcePrescribedPins r b) ω) at hω
        change R ≤ sampleMajorant w ω
        linarith
      · exact measure_ne_top _ _
    _ ≤ _ := htail

/-- The same measurable cutoff has every polynomial rate under the actual
whole-carrier endpoint-weighted law. This is an explicit weighted probability,
with no identification with a geometric Kac–Rice/Palm law asserted. -/
theorem sourceWeightedDerivative_cutoff_tail {L : ℝ} (hL : L ≠ 0)
    (b : ℝ) (k n : ℕ) :
    ∃ M δ : ℝ, 0 ≤ M ∧ 0 < δ ∧ δ ≤ 1 ∧ ∀ r : ℝ, 0 < r → r < δ →
      IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b) ∧
      (sourceSmoothWeightedLaw hL r b).real
        {ω | r⁻¹ < sourceDerivativeMajorant L k ω} ≤ M*r^n := by
  obtain ⟨C,c,δP,hC,hc,hδP,hδP1,hPalm⟩ := sourceSmoothPalm_estimates hL b
  obtain ⟨B,δB,hB,hδB,hδB1,hbase⟩ := sourceConditionalDerivative_cutoff_tail hL b k (2*n)
  let A := Real.sqrt C/c
  have hA : 0 ≤ A := div_nonneg (Real.sqrt_nonneg _) hc.le
  refine ⟨A*Real.sqrt B,min δP δB,mul_nonneg hA (Real.sqrt_nonneg _),
    lt_min hδP hδB,(min_le_left _ _).trans hδP1,?_⟩
  intro r hr hrδ
  have hrP : r < δP := lt_of_lt_of_le hrδ (min_le_left _ _)
  have hrB : r < δB := lt_of_lt_of_le hrδ (min_le_right _ _)
  have hp := hPalm r hr hrP
  have hE : MeasurableSet {ω : SmoothSample 2 L | r⁻¹ < sourceDerivativeMajorant L k ω} :=
    measurableSet_lt measurable_const (sourceDerivativeMajorant_measurable L k)
  refine ⟨hp.2.2.2.1,?_⟩
  have he := hp.2.2.2.2.1 _ hE
  have hb := hbase r hr hrB
  have hsqrt : Real.sqrt ((sourceSmoothLaw hL r b).real
      {ω | r⁻¹ < sourceDerivativeMajorant L k ω}) ≤ Real.sqrt B*r^n := by
    calc
      _ ≤ Real.sqrt (B*r^(2*n)) := Real.sqrt_le_sqrt hb
      _ = Real.sqrt B*r^n := by
        rw [Real.sqrt_mul hB,show r^(2*n) = (r^n)^2 by rw [←pow_mul,Nat.mul_comm],
          Real.sqrt_sq (pow_nonneg hr.le n)]
  exact he.trans (by
    change A*Real.sqrt ((sourceSmoothLaw hL r b).real
      {ω | r⁻¹ < sourceDerivativeMajorant L k ω}) ≤ A*Real.sqrt B*r^n
    simpa only [mul_assoc] using mul_le_mul_of_nonneg_left hsqrt hA)

end I4SourceConditionalMajorant
