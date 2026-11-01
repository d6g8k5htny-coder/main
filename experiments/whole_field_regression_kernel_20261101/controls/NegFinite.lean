import WholeFieldRegression
open MeasureTheory ProbabilityTheory WholeFieldRegression
-- NEGATIVE CONTROL: independence from ONE coordinate does not give the whole field.
example {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α] {P : Measure Ω}
    [IsProbabilityMeasure P] {T : Type*} {X : Ω → α} {R : Ω → T → ℝ}
    (hX : Measurable X) (hR : Measurable R) (t0 : T)
    (h1 : IndepFun X (fun ω => R ω t0) P) : IndepFun X R P :=
  indepFun_pi_of_finite hX hR (fun S => h1)
