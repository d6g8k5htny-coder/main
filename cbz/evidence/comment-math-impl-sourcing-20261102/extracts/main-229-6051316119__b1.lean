theorem summable_weight_mul_exp {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    Summable fun k : ℤ => weight L k * Real.exp (t * atom L k)
theorem integrable_exp_mul_dualMeasure {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    Integrable (fun x : ℝ => Real.exp (t * x)) (dualMeasure L)
theorem moment_law_eq_dual {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hint : ∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) μ)
    (h : ∀ n : ℕ, ∫ x, x ^ n ∂μ = dualMoment L n) : μ = dualMeasure L
theorem eq_dual_iff_moments {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ] :
    μ = dualMeasure L ↔ (∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) μ) ∧
      ∀ n : ℕ, ∫ x, x ^ n ∂μ = dualMoment L n
theorem eq_dual_iff_charFun {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ] :
    μ = dualMeasure L ↔ ∀ t, charFun μ t = (qL L t : ℂ)          -- W3i (R5), restated
theorem eq_dual_iff_iba_moments {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ] :
    μ = dualMeasure L ↔ (∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) μ) ∧
      (∀ m : ℕ, ∫ x, x ^ (2 * m) ∂μ = ibaMoment L (2 * m)) ∧
      (∀ m : ℕ, ∫ x, x ^ (2 * m + 1) ∂μ = 0)
theorem charFun_iff_moments {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ] :
    (∀ t, charFun μ t = (qL L t : ℂ)) ↔ (∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) μ) ∧
      ∀ n : ℕ, ∫ x, x ^ n ∂μ = dualMoment L n
