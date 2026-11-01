import WholeFieldRegression
open WholeFieldRegression
-- AUDIT CONTROL: a sorry-backed theorem must show sorryAx in its axiom report.
theorem sorry_control : (1 : ℕ) = 2 := by sorry
#print axioms sorry_control
