import ResearchFormalCoreR1.EndpointGaussianMoments
import ResearchFormalCoreR1.NormalizerFloor

/-!
# Literal endpoint weight and continuous normalized weight

The actual endpoint vector has order `(aM,bM,cM,aS,bS,cS)`, while the
continuous normalizer uses `(aM,aS,bM,bS,cM,cS)`. The permutation below
and the identity hold pointwise, including all determinant/type boundaries.
See ../SOURCE_ENDPOINT_NORMALIZER_BRIDGE.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1

open MeasureTheory ProbabilityTheory Filter
open scoped Topology NNReal

set_option autoImplicit false

/-- Convert the actual six endpoint entries to the continuous weight's order. -/
def enb_permutation (j : Fin 6 → ℝ) : Fin 6 → ℝ :=
  ![j 0, j 3, j 1, j 4, j 2, j 5]

theorem enb_permutation_continuous : Continuous enb_permutation := by
  unfold enb_permutation
  fun_prop

private theorem enb_maximum_factor (r a b c : ℝ) (hr : 0 < r) :
    max (max (-a) 0 * max (-c) 0 - r * b ^ 2) 0 =
      if a < 0 ∧ 0 < a * c - r * b ^ 2 then a * c - r * b ^ 2 else 0 := by
  by_cases h : a < 0 ∧ 0 < a * c - r * b ^ 2
  · have hc : c < 0 := by
      by_contra hn
      have hac : a * c ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h.1.le (le_of_not_gt hn)
      have hrb : 0 ≤ r * b ^ 2 := mul_nonneg hr.le (sq_nonneg b)
      linarith [h.2]
    rw [ite_eq_left h, max_eq_left (neg_nonneg.mpr h.1.le),
      max_eq_left (neg_nonneg.mpr hc.le)]
    have heq : -a * -c - r * b ^ 2 = a * c - r * b ^ 2 := by ring
    rw [heq, max_eq_left h.2.le]
  · rw [ite_eq_right h]
    by_cases ha : a < 0
    · have hd : a * c - r * b ^ 2 ≤ 0 :=
        le_of_not_gt (fun hp => h ⟨ha, hp⟩)
      by_cases hc : c ≤ 0
      · rw [max_eq_left (neg_nonneg.mpr ha.le), max_eq_left (neg_nonneg.mpr hc)]
        have heq : -a * -c - r * b ^ 2 = a * c - r * b ^ 2 := by ring
        rw [heq, max_eq_right hd]
      · rw [max_eq_right (by linarith : -c ≤ 0)]
        simpa using (max_eq_right (neg_nonpos.mpr (mul_nonneg hr.le (sq_nonneg b))))
    · rw [max_eq_right (by linarith : -a ≤ 0)]
      simpa using (max_eq_right (neg_nonpos.mpr (mul_nonneg hr.le (sq_nonneg b))))

/-- Strict negative definiteness in actual endpoint coordinates, with no
assumed sign of the other diagonal entry or omission of singular boundaries. -/
theorem enb_maximum_iff (r : ℝ) (j : Fin 6 → ℝ) (hr : 0 < r) :
    dm2Negative (egm_HM r j) ↔ j 0 < 0 ∧ 0 < j 0 * j 2 - r * j 1 ^ 2 := by
  rw [egm_HM, dm2_negative_iff]
  have heq : r * j 0 * j 2 - (r * j 1) ^ 2 =
      r * (j 0 * j 2 - r * j 1 ^ 2) := by ring
  have hmul (x : ℝ) : r * x < 0 ↔ x < 0 := by
    constructor <;> intro hx <;> nlinarith
  rw [heq, hmul, mul_pos_iff_of_pos_left hr]

/-- The literal strict saddle condition is the sign of its factored determinant. -/
theorem enb_saddle_iff (r : ℝ) (j : Fin 6 → ℝ) (hr : 0 < r) :
    (egm_HS r j).det < 0 ↔ j 3 * j 5 - r * j 4 ^ 2 < 0 := by
  have heq : (egm_HS r j).det = r * (j 3 * j 5 - r * j 4 ^ 2) := by
    simp [egm_HS, Matrix.det_fin_two]
    ring
  rw [heq]
  constructor <;> intro h <;> nlinarith

/-- Exact bridge from the literal typed physical endpoint weight to the
continuous expression. All excluded types and all zero boundaries give zero. -/
theorem enb_weight_identity (r : ℝ) (j : Fin 6 → ℝ) (hr : 0 < r) :
    egm_weight r j = r ^ 2 * gn_normalizedWeight r (enb_permutation j) := by
  classical
  have hM := enb_maximum_iff r j hr
  have hS := enb_saddle_iff r j hr
  have hdetM : (egm_HM r j).det = r * (j 0 * j 2 - r * j 1 ^ 2) := by
    simp [egm_HM, Matrix.det_fin_two]
    ring
  have hdetS : (egm_HS r j).det = r * (j 3 * j 5 - r * j 4 ^ 2) := by
    simp [egm_HS, Matrix.det_fin_two]
    ring
  unfold egm_weight gn_normalizedWeight
  change (if dm2Negative (egm_HM r j) ∧ (egm_HS r j).det < 0 then
    |(egm_HM r j).det * (egm_HS r j).det| else 0) =
      r ^ 2 * (max (max (-j 0) 0 * max (-j 2) 0 - r * j 1 ^ 2) 0 *
        max (r * j 4 ^ 2 - j 3 * j 5) 0)
  rw [enb_maximum_factor r (j 0) (j 1) (j 2) hr]
  by_cases hmax : j 0 < 0 ∧ 0 < j 0 * j 2 - r * j 1 ^ 2
  · rw [ite_eq_left hmax]
    by_cases hsad : j 3 * j 5 - r * j 4 ^ 2 < 0
    · rw [ite_eq_left ⟨hM.mpr hmax, hS.mpr hsad⟩, hdetM, hdetS]
      rw [max_eq_left (by linarith : 0 ≤ r * j 4 ^ 2 - j 3 * j 5)]
      have hp : r * (j 0 * j 2 - r * j 1 ^ 2) *
          (r * (j 3 * j 5 - r * j 4 ^ 2)) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hr.le hmax.2.le)
          (mul_nonpos_of_nonneg_of_nonpos hr.le hsad.le)
      rw [abs_of_nonpos hp]
      ring
    · rw [ite_eq_right (fun h => hsad (hS.mp h.2)),
        max_eq_right (by linarith : r * j 4 ^ 2 - j 3 * j 5 ≤ 0)]
      ring
  · rw [ite_eq_right hmax, ite_eq_right (fun h => hmax (hM.mp h.1))]
    ring

/-- The physical and continuous normalizers are the same integral up to the
exact radius factor, for any common random six-vector and any measure. -/
theorem enb_integral_identity {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (J : Ω → Fin 6 → ℝ) (r : ℝ) (hr : 0 < r) :
    (∫ ω, egm_weight r (J ω) ∂P) =
      r ^ 2 * ∫ ω, gn_normalizedWeight r (enb_permutation (J ω)) ∂P := by
  calc
    _ = ∫ ω, r ^ 2 * gn_normalizedWeight r (enb_permutation (J ω)) ∂P :=
      integral_congr_ae (Eventually.of_forall (fun ω => enb_weight_identity r (J ω) hr))
    _ = _ := integral_const_mul _ _

/-- Positive radius makes the exact scaling nonzero, so either weight is
integrable precisely when the other is. -/
theorem enb_integrable_iff {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (J : Ω → Fin 6 → ℝ) (r : ℝ) (hr : 0 < r) :
    Integrable (fun ω => egm_weight r (J ω)) P ↔
      Integrable (fun ω => gn_normalizedWeight r (enb_permutation (J ω))) P := by
  have heq : (fun ω => egm_weight r (J ω)) =
      (fun ω => r ^ 2 * gn_normalizedWeight r (enb_permutation (J ω))) :=
    funext (fun ω => enb_weight_identity r (J ω) hr)
  rw [heq, integrable_const_mul_iff (isUnit_iff_ne_zero.mpr (pow_ne_zero 2 hr.ne'))]

/-- On a finite measure, the actual endpoint second-moment theorem supplies
the first-moment hypotheses needed by the normalizer join. -/
theorem enb_integrable_of_integrable_sq {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsFiniteMeasure P]
    (J : Ω → Fin 6 → ℝ) (hJ : Measurable J) (r : ℝ) (hr : 0 < r)
    (hsq : Integrable (fun ω => egm_weight r (J ω) ^ 2) P) :
    Integrable (fun ω => egm_weight r (J ω)) P ∧
      Integrable (fun ω => gn_normalizedWeight r (enb_permutation (J ω))) P := by
  have hmajor : Integrable (fun ω => 1 + egm_weight r (J ω) ^ 2) P :=
    (integrable_const 1).add hsq
  have hphysical : Integrable (fun ω => egm_weight r (J ω)) P := by
    apply hmajor.mono' (((egm_weight_measurable r).comp hJ).aestronglyMeasurable)
    filter_upwards [] with ω
    rw [Real.norm_eq_abs]
    change |egm_weight r (J ω)| ≤ 1 + egm_weight r (J ω) ^ 2
    nlinarith [sq_nonneg (|egm_weight r (J ω)| - 1/2), sq_abs (egm_weight r (J ω))]
  exact ⟨hphysical, (enb_integrable_iff P J r hr).mp hphysical⟩

/-- The proved common-coupling normalizer floor transfers to the literal
physical endpoint weight, with its own actual coordinate order. Current
physical integrability and the conditioned-field convergence remain inputs. -/
theorem enb_normalizer_floor_of_common_coupling
    {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (J : ℝ → Ω → Fin 6 → ℝ) (hJ : ∀ r, Measurable (J r))
    (A Q : Ω → ℝ) (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0)
    (hQ : HasLaw Q (gaussianReal m v) P)
    (hconv : ∀ᵐ ω ∂P, Tendsto (fun r => J r ω) (𝓝[>] (0 : ℝ))
      (𝓝 ![-1,-A ω/2,Q ω,1,A ω/2,Q ω]))
    (hint : ∀ᶠ r in 𝓝[>] (0 : ℝ),
      Integrable (fun ω => egm_weight r (J r ω)) P) :
    ∃ c : ℝ, 0 < c ∧ ∃ δ : ℝ, 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ →
        c * r ^ 2 ≤ ∫ ω, egm_weight r (J r ω) ∂P := by
  let X : ℝ → Ω → Fin 6 → ℝ := fun r ω => enb_permutation (J r ω)
  have hX (r : ℝ) : Measurable (X r) :=
    enb_permutation_continuous.measurable.comp (hJ r)
  have hXconv : ∀ᵐ ω ∂P, Tendsto (fun r => X r ω) (𝓝[>] (0 : ℝ))
      (𝓝 ![-1,1,-A ω/2,A ω/2,Q ω,Q ω]) := by
    filter_upwards [hconv] with ω hω
    have hp := (enb_permutation_continuous.tendsto
      ![-1,-A ω/2,Q ω,1,A ω/2,Q ω]).comp hω
    have hcontact : enb_permutation ![-1,-A ω/2,Q ω,1,A ω/2,Q ω] =
        ![-1,1,-A ω/2,A ω/2,Q ω,Q ω] := rfl
    simpa only [Function.comp_def, hcontact, X] using hp
  have hXint : ∀ᶠ r in 𝓝[>] (0 : ℝ),
      Integrable (fun ω => gn_normalizedWeight r (X r ω)) P := by
    filter_upwards [hint, self_mem_nhdsWithin] with r hri hr
    exact (enb_integrable_iff P (J r) r hr).mp hri
  obtain ⟨c, hc, δ, hδ, hfloor⟩ :=
    gn_normalizer_floor_of_common_coupling P X hX A Q m v hv hQ hXconv hXint
  refine ⟨c, hc, δ, hδ, ?_⟩
  intro r hr hrδ
  have hbound := hfloor r hr hrδ
  calc
    c * r ^ 2 ≤ ∫ ω, r ^ 2 * gn_normalizedWeight r (X r ω) ∂P := hbound
    _ = ∫ ω, egm_weight r (J r ω) ∂P :=
      integral_congr_ae (Eventually.of_forall (fun ω => (enb_weight_identity r (J r ω) hr).symm))

end ResearchFormalCoreR1
