import WholeFieldRegression
open MeasureTheory ProbabilityTheory WholeFieldRegression
-- NEGATIVE CONTROL: wrong sign on the correction must not be provable by the same proof.
example {n : ℕ} {T : Type*} (ν : Measure (T → ℝ)) [SFinite ν]
    {c : (Fin n → ℝ) → T → ℝ} (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ)
    (x : Fin n → ℝ) :
    fieldRegressionKernel ν c mX mY x = ν.map (fun r t => r t + mY t - c (x - mX) t) :=
  reconstructionKernel_apply ν (fieldReconstruction_measurable hc mX mY) x
