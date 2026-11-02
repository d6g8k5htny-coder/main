def dualMeasure (L : ℝ) : Measure ℝ :=
  Measure.sum fun k : ℤ => ENNReal.ofReal (weight L k / dualZ L) • Measure.dirac (atom L k)

theorem dualMeasure_univ {L : ℝ} (hL : L ≠ 0) : dualMeasure L Set.univ = 1
theorem isProbabilityMeasure_dualMeasure' {L : ℝ} (hL : L ≠ 0) : IsProbabilityMeasure (dualMeasure L)

-- existence (R5)
theorem charFun_dualMeasure {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    charFun (dualMeasure L) t = (qL L t : ℂ)
-- uniqueness (R5)
theorem spectral_law_eq_dual {L : ℝ} (hL : L ≠ 0) (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (h : ∀ t, charFun μ t = (qL L t : ℂ)) : μ = dualMeasure L
theorem spectral_law_existsUnique {L : ℝ} (hL : L ≠ 0) :
    ∃! μ : Measure ℝ, IsProbabilityMeasure μ ∧ ∀ t, charFun μ t = (qL L t : ℂ)

-- (b) moments derived from the measure equality
theorem integral_pow_dualMeasure {L : ℝ} (hL : L ≠ 0) (n : ℕ) :
    ∫ x, x ^ n ∂dualMeasure L = dualMoment L n
theorem spectral_integrable_pow … (h : ∀ t, charFun μ t = (qL L t : ℂ)) (n : ℕ) :
    Integrable (fun x : ℝ => x ^ n) μ
theorem spectral_moment_eq … (h : …) (m : ℕ) :
    ∫ x, x ^ (2 * m) ∂μ = ibaMoment L (2 * m) ∧ ∫ x, x ^ (2 * m) ∂μ = dualMoment L (2 * m)
theorem spectral_odd_moment_zero … (h : …) (m : ℕ) : ∫ x, x ^ (2 * m + 1) ∂μ = 0
theorem spectral_moment_hyps … (h : …) :
    0 < ∫ x, x ^ 2 ∂μ ∧ (∫ x, x ^ 2 ∂μ) ^ 2 < ∫ x, x ^ 4 ∂μ ∧
      0 < d2Delta (∫ x, x ^ 2 ∂μ) (∫ x, x ^ 4 ∂μ) (∫ x, x ^ 6 ∂μ)
