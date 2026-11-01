import WholeFieldRegression
open MeasureTheory ProbabilityTheory WholeFieldRegression
-- NEGATIVE CONTROL: dropping the independence hypothesis must break the proof.
example {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P] {n : ℕ}
    {T : Type*} {X : Ω → Fin n → ℝ} {R : Ω → T → ℝ}
    (hX : AEMeasurable X P) (hR : AEMeasurable R P)
    {c : (Fin n → ℝ) → T → ℝ} (hc : Measurable c) (mX : Fin n → ℝ) (mY : T → ℝ) :
    HasCondDistrib (fun ω t => R ω t + mY t + c (X ω - mX) t) X
      (fieldRegressionKernel (P.map R) c mX mY) P :=
  fieldRegression_hasCondDistrib hX hR (by assumption) hc mX mY
