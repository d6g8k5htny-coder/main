import PinMajorant

noncomputable section
open I4Foundation I4Source I4PinMajorant I4DeterministicC4 MeasureTheory
open scoped BigOperators Manifold ContDiff
namespace PinMajorantTests
attribute [local instance] I4TorusC4.actualChartedSpace

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) (omega : SourceMode d → ℝ) :
    regressionCoefficients d L sites 0 v omega = omega := by
  classical
  funext m
  simp [regressionCoefficients]

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (omega : SourceMode d → ℝ) :
    regressionCoefficients d L sites R (pinObservation d L sites omega) omega = omega := by
  funext m
  unfold regressionCoefficients
  ring

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) :
    regressionCoefficients d L sites R 0 0 = 0 := by
  classical
  funext m
  simp [regressionCoefficients,pinObservation,scalarSeries]

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (c : ℝ) (v : PinIndex d → ℝ) :
    regressionCoefficients d L sites (c • R) v 0 =
      c • regressionCoefficients d L sites R v 0 := by
  classical
  funext m
  simp only [regressionCoefficients,pinObservation,scalarSeries,Pi.zero_apply,
    mul_zero,tsum_zero,Finset.sum_const_zero,sub_zero,zero_add,Pi.smul_apply,smul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  simp [Matrix.smul_apply]
  ring

-- A zero matrix does not reproduce a nonzero prescribed observation.
example : pinObservation 1 1 (fun _ => 0)
    (regressionCoefficients 1 1 (fun _ => 0) 0 (fun _ => 1) 0) (0,none) ≠ 1 := by
  classical
  simp [regressionCoefficients,pinObservation,scalarSeries]

-- No distinctness hypothesis: the two physical pin sites may coincide.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    ∀ᵐ omega ∂sourceLaw d,
      pinObservation d L (fun _ => x) omega = actualPinObservation d L (fun _ => x) omega :=
  (pinObservation_eq_actual_ae d hL).mono (fun _ h => h _)

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (v : PinIndex d → ℝ) :
    Measurable (regressionCoefficients d L sites R v) :=
  regressionCoefficients_measurable d L sites R v

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (v : PinIndex d → ℝ) :
    IsProbabilityMeasure (transformedLaw d L sites R v) := by
  unfold transformedLaw
  infer_instance

-- The all-ones finite matrix is admitted; invertibility is unnecessary for regularity.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    Summable (fun m => modeMajorant d L m *
      |regressionCoefficients d L (fun _ => x) (fun _ _ => 1) v 0 m|) := by
  apply regression_majorant d hL
  simp

-- Deterministic regularity of transformed ZERO coefficients, outside any AE inference.
example (d : ℕ) (L : ℝ) [Fact (0 < L)] (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    ContMDiff 𝓘(ℝ, Fin d → ℝ) 𝓘(ℝ) 4
      (I4Torus.torusSourceField d L (regressionCoefficients d L (fun _ => x) (fun _ _ => 1) v 0)) := by
  apply torusSourceField_contMDiff_of_majorant
  apply regression_majorant d (Fact.out : 0 < L).ne'
  simp

example (d : ℕ) (L : ℝ) [Fact (0 < L)] (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    ∀ᵐ eta ∂transformedLaw d L (fun _ => x) (fun _ _ => 1) v,
      ContDiff ℝ 4 (sourceField d L eta) :=
  (transformedLaw_regular d L (fun _ => x) (fun _ _ => 1) v).mono (fun _ h => h.2.1)

example (L : ℝ) [Fact (0 < L)] (v : PinIndex 0 → ℝ) :
    ∀ᵐ eta ∂transformedLaw 0 L (fun _ => 0) 0 v,
      ContMDiff 𝓘(ℝ, Fin 0 → ℝ) 𝓘(ℝ) 4 (I4Torus.torusSourceField 0 L eta) :=
  (transformedLaw_regular 0 L (fun _ => 0) 0 v).mono (fun _ h => h.2.2)
end PinMajorantTests

