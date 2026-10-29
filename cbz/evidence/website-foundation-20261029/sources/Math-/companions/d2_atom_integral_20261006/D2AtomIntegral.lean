import D2MomentBridge
import D2SquareSchur
import Mathlib.MeasureTheory.Integral.Bochner.Set

/-!
Quantitative squared-radius event masses give actual Bochner-integral floors.
This supplies the six previously published contracts; no finite-model surrogate
or changed definition of moment, residual or Schur expression is introduced.
-/
noncomputable section
namespace D2AtomIntegral

open MeasureTheory D2MomentBridge ResearchFormalCoreR1
set_option autoImplicit false

variable {Ω : Type*} [MeasurableSpace Ω]

theorem two_event_integral_lower (μ : Measure Ω) [IsProbabilityMeasure μ] (f : Ω → ℝ)
    (S T : Set Ω) (hS : MeasurableSet S) (hT : MeasurableSet T)
    (hST : Disjoint S T) (hf : Integrable f μ) (hf0 : ∀ w, 0 ≤ f w)
    (a b : ℝ) (ha : ∀ w ∈ S, a ≤ f w) (hb : ∀ w ∈ T, b ≤ f w) :
    μ.real S * a + μ.real T * b ≤ ∫ w, f w ∂μ := by
  classical
  have hsi : Integrable (S.indicator (fun _ : Ω => a)) μ :=
    (integrable_const a).indicator hS
  have hti : Integrable (T.indicator (fun _ : Ω => b)) μ :=
    (integrable_const b).indicator hT
  have hle : ∀ w, S.indicator (fun _ : Ω => a) w +
      T.indicator (fun _ : Ω => b) w ≤ f w := by
    intro w
    by_cases hs : w ∈ S
    · have ht : w ∉ T := fun ht => Set.disjoint_left.mp hST hs ht
      simpa only [Set.indicator_of_mem hs, Set.indicator_of_notMem ht, add_zero]
        using ha w hs
    · by_cases ht : w ∈ T
      · simpa only [Set.indicator_of_notMem hs, Set.indicator_of_mem ht, zero_add]
          using hb w ht
      · simpa only [Set.indicator_of_notMem hs, Set.indicator_of_notMem ht, zero_add]
          using hf0 w
  calc
    μ.real S * a + μ.real T * b =
        ∫ w, S.indicator (fun _ : Ω => a) w + T.indicator (fun _ : Ω => b) w ∂μ := by
      rw [integral_add hsi hti, integral_indicator hS, integral_indicator hT]
      simp [Measure.real, smul_eq_mul]
    _ ≤ ∫ w, f w ∂μ := integral_mono (hsi.add hti) hf hle

theorem two_radius_integral_lower (μ : Measure Ω) [IsProbabilityMeasure μ] (X f : Ω → ℝ)
    (s t p q a b : ℝ)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t}) (hst : s ≠ t)
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hf : Integrable f μ) (hf0 : ∀ w, 0 ≤ f w)
    (ha : ∀ w, X w ^ 2 = s → f w = a) (hb : ∀ w, X w ^ 2 = t → f w = b) :
    p * a + q * b ≤ ∫ w, f w ∂μ := by
  have hd : Disjoint {w | X w ^ 2 = s} {w | X w ^ 2 = t} := by
    apply Set.disjoint_left.mpr
    intro w hwS hwT
    exact hst (hwS.symm.trans hwT)
  have hmass := add_le_add (mul_le_mul_of_nonneg_right hp ha0)
    (mul_le_mul_of_nonneg_right hq hb0)
  exact le_trans hmass (two_event_integral_lower μ f _ _ hS hT hd hf hf0 a b
    (fun w hw => (ha w hw).symm.le) (fun w hw => (hb w hw).symm.le))

theorem second_moment_floor (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hs : 0 < s) (ht : 0 < t) (hst : s ≠ t)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ) :
    p*s + q*t ≤ moment μ X 2 := by
  exact two_radius_integral_lower μ X (fun w => X w ^ 2) s t p q s t
    hS hT hst hp hq hs.le ht.le h2 (fun w => sq_nonneg (X w))
    (fun _ h => h) (fun _ h => h)

theorem fourth_gap_floor (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hp0 : 0 < p) (hq0 : 0 < q) (hst : s ≠ t)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) :
    p*q/(p+q)*(s-t)^2 ≤ moment μ X 4 - (moment μ X 2)^2 := by
  have h := two_radius_integral_lower μ X (fun w => (X w ^ 2 - moment μ X 2)^2)
    s t p q ((s-moment μ X 2)^2) ((t-moment μ X 2)^2)
    hS hT hst hp hq (sq_nonneg _) (sq_nonneg _)
    (square_gap_integrable μ X h2 h4) (fun _ => sq_nonneg _)
    (by intro w hw; rw [hw]) (by intro w hw; rw [hw])
  rw [square_gap_integral μ X h2 h4] at h
  exact le_trans (D2SquareSchur.weighted_square_lower p q s t (moment μ X 2) hp0 hq0) h

theorem residual_moment_floor (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hs : 0 < s) (ht : 0 < t) (hp0 : 0 < p) (hq0 : 0 < q)
    (hst : s ≠ t) (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ) :
    (p*s)*(q*t)/(p*s+q*t)*(s-t)^2 ≤
      d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) := by
  let c : ℝ := moment μ X 4 / moment μ X 2
  have hsecond := second_moment_floor μ X s t p q hs ht hst hS hT hp hq h2
  have hm2 : 0 < moment μ X 2 :=
    lt_of_lt_of_le (add_pos (mul_pos hp0 hs) (mul_pos hq0 ht)) hsecond
  have h := two_radius_integral_lower μ X (fun w => cubicResidual c (X w)^2)
    s t p q (s*(s-c)^2) (t*(t-c)^2) hS hT hst hp hq
    (mul_nonneg hs.le (sq_nonneg _)) (mul_nonneg ht.le (sq_nonneg _))
    (residual_sq_integrable μ X c h2 h4 h6) (fun _ => sq_nonneg _)
    (by
      intro w hw
      calc
        cubicResidual c (X w)^2 = X w ^ 2 * (X w ^ 2-c)^2 := by
          unfold cubicResidual
          ring
        _ = s*(s-c)^2 := by rw [hw])
    (by
      intro w hw
      calc
        cubicResidual c (X w)^2 = X w ^ 2 * (X w ^ 2-c)^2 := by
          unfold cubicResidual
          ring
        _ = t*(t-c)^2 := by rw [hw])
  have hr : (∫ w, cubicResidual c (X w)^2 ∂μ) =
      d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) :=
    residual_integral_eq_delta μ X h2 h4 h6 (ne_of_gt hm2)
  rw [hr] at h
  have hl := D2SquareSchur.weighted_square_lower (p*s) (q*t) s t c
    (mul_pos hp0 hs) (mul_pos hq0 ht)
  exact le_trans hl (by simpa only [mul_assoc] using h)

theorem schur_floor_of_radius_masses (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q u : ℝ) (hs : 0 < s) (ht : 0 < t) (hp0 : 0 < p) (hq0 : 0 < q)
    (hst : s ≠ t) (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) (hu0 : 0 ≤ u) (hu1 : u ≤ 1/4) :
    min (p*q/(p+q)*(s-t)^2) (2*(p*s+q*t)^2) ≤
      d2Schur (moment μ X 2) (moment μ X 4) u := by
  have hm := second_moment_floor μ X s t p q hs ht hst hS hT hp hq h2
  have hg := fourth_gap_floor μ X s t p q hp0 hq0 hst hS hT hp hq h2 h4
  have hm0 : 0 < p*s+q*t := add_pos (mul_pos hp0 hs) (mul_pos hq0 ht)
  have hg0 : 0 < p*q/(p+q)*(s-t)^2 :=
    mul_pos (div_pos (mul_pos hp0 hq0) (add_pos hp0 hq0))
      (sq_pos_of_ne_zero (sub_ne_zero.mpr hst))
  exact D2SquareSchur.schur_from_lower_inputs (moment μ X 2) (moment μ X 4) u
    (p*s+q*t) (p*q/(p+q)*(s-t)^2) hm0 hg0 hm hg hu0 hu1

end D2AtomIntegral
