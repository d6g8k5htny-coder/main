import D2AtomIntegral

open MeasureTheory D2MomentBridge ResearchFormalCoreR1

variable {Ω : Type*} [MeasurableSpace Ω]

example (μ : Measure Ω) [IsProbabilityMeasure μ] (f : Ω → ℝ)
    (S T : Set Ω) (hS : MeasurableSet S) (hT : MeasurableSet T)
    (hST : Disjoint S T) (hf : Integrable f μ) (hf0 : ∀ w, 0 ≤ f w)
    (a b : ℝ) (ha : ∀ w ∈ S, a ≤ f w) (hb : ∀ w ∈ T, b ≤ f w) :
    μ.real S * a + μ.real T * b ≤ ∫ w, f w ∂μ :=
  D2AtomIntegral.two_event_integral_lower μ f S T hS hT hST hf hf0 a b ha hb

example (μ : Measure Ω) [IsProbabilityMeasure μ] (X f : Ω → ℝ)
    (s t p q a b : ℝ)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t}) (hst : s ≠ t)
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hf : Integrable f μ) (hf0 : ∀ w, 0 ≤ f w)
    (ha : ∀ w, X w ^ 2 = s → f w = a) (hb : ∀ w, X w ^ 2 = t → f w = b) :
    p * a + q * b ≤ ∫ w, f w ∂μ :=
  D2AtomIntegral.two_radius_integral_lower μ X f s t p q a b hS hT hst hp hq ha0 hb0 hf hf0 ha hb

example (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hs : 0 < s) (ht : 0 < t) (hst : s ≠ t)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ) :
    p*s + q*t ≤ moment μ X 2 :=
  D2AtomIntegral.second_moment_floor μ X s t p q hs ht hst hS hT hp hq h2

example (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hp0 : 0 < p) (hq0 : 0 < q) (hst : s ≠ t)
    (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) :
    p*q/(p+q)*(s-t)^2 ≤ moment μ X 4 - (moment μ X 2)^2 :=
  D2AtomIntegral.fourth_gap_floor μ X s t p q hp0 hq0 hst hS hT hp hq h2 h4

example (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q : ℝ) (hs : 0 < s) (ht : 0 < t) (hp0 : 0 < p) (hq0 : 0 < q)
    (hst : s ≠ t) (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ)
    (h6 : Integrable (fun w => X w ^ 6) μ) :
    (p*s)*(q*t)/(p*s+q*t)*(s-t)^2 ≤
      d2Delta (moment μ X 2) (moment μ X 4) (moment μ X 6) :=
  D2AtomIntegral.residual_moment_floor μ X s t p q hs ht hp0 hq0 hst hS hT hp hq h2 h4 h6

example (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (s t p q u : ℝ) (hs : 0 < s) (ht : 0 < t) (hp0 : 0 < p) (hq0 : 0 < q)
    (hst : s ≠ t) (hS : MeasurableSet {w | X w ^ 2 = s})
    (hT : MeasurableSet {w | X w ^ 2 = t})
    (hp : p ≤ μ.real {w | X w ^ 2 = s}) (hq : q ≤ μ.real {w | X w ^ 2 = t})
    (h2 : Integrable (fun w => X w ^ 2) μ)
    (h4 : Integrable (fun w => X w ^ 4) μ) (hu0 : 0 ≤ u) (hu1 : u ≤ 1/4) :
    min (p*q/(p+q)*(s-t)^2) (2*(p*s+q*t)^2) ≤
      d2Schur (moment μ X 2) (moment μ X 4) u :=
  D2AtomIntegral.schur_floor_of_radius_masses μ X s t p q u hs ht hp0 hq0 hst hS hT hp hq h2 h4 hu0 hu1
