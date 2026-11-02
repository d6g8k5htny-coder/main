theorem ext_of_moments_of_integrable_exp {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hμ : ∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) μ)
    (hν : ∀ n : ℕ, Integrable (fun x : ℝ => x ^ n) ν)
    (hmom : ∀ n : ℕ, ∫ x, x ^ n ∂μ = ∫ x, x ^ n ∂ν)
    (hexp : ∀ t : ℝ, Integrable (fun x : ℝ => Real.exp (t * x)) ν) : μ = ν
