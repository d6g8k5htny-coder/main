import PinKernel

noncomputable section
open I4Foundation I4Source I4PinMajorant I4PinKernel I4DeterministicC4
  MeasureTheory ProbabilityTheory
open scoped BigOperators Manifold ContDiff
namespace PinKernelTests
attribute [local instance] I4TorusC4.actualChartedSpace

private theorem zeroR_law (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) : regressionKernel d L (sites,0,v) = sourceLaw d := by
  classical
  rw [regressionKernel_apply,transformedLaw]
  have he : regressionCoefficients d L sites 0 v = id := by
    funext omega m
    simp [regressionCoefficients]
  rw [he,Measure.map_id]

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) : regressionKernel d L (sites,0,v) = sourceLaw d :=
  zeroR_law d L sites v

example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) (omega : SourceMode d → ℝ) :
    regressionJoint d L ((sites,0,v),omega) = omega := by
  classical
  funext m
  simp [regressionJoint,regressionCoefficients]

-- The complete parameter product is admitted, with no inverse/rank premise.
example (d : ℕ) (L : ℝ) (p : PinParameters d) :
    regressionKernel d L p = transformedLaw d L p.1 p.2.1 p.2.2 :=
  regressionKernel_apply d L p

-- Singular all-ones R and colliding physical sites still give the literal law.
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    regressionKernel d L (fun _ => x,fun _ _ => 1,v) =
      transformedLaw d L (fun _ => x) (fun _ _ => 1) v :=
  regressionKernel_apply d L _

example (d : ℕ) (L : ℝ) : IsMarkovKernel (regressionKernel d L) :=
  regressionKernel_markov d L

example (d : ℕ) (L : ℝ) (p : PinParameters d) :
    IsProbabilityMeasure (regressionKernel d L p) := by
  let : IsMarkovKernel (regressionKernel d L) := regressionKernel_markov d L
  infer_instance

example (d : ℕ) (L : ℝ) (p : PinParameters d) :
    regressionKernel d L p Set.univ = 1 := by
  let : IsMarkovKernel (regressionKernel d L) := regressionKernel_markov d L
  exact measure_univ

-- This actual mapped kernel cannot be the nonmeasurable-map zero fallback.
example (d : ℕ) (L : ℝ) (p : PinParameters d) : regressionKernel d L p ≠ 0 := by
  let : IsMarkovKernel (regressionKernel d L) := regressionKernel_markov d L
  intro h
  have hu : regressionKernel d L p Set.univ = 1 := measure_univ
  simp [h] at hu

example (d : ℕ) (L : ℝ) (s : Set (SourceMode d → ℝ)) (hs : MeasurableSet s) :
    Measurable (fun p => regressionKernel d L p s) :=
  (regressionKernel d L).measurable_coe hs

example (d : ℕ) (L : ℝ) (m : SourceMode d) :
    Measurable (fun sites : Fin 2 → (Fin d → ℝ) => pinCoefficient d L sites m) :=
  pinCoefficient_joint_measurable d L m

example (d : ℕ) (L : ℝ) :
    Measurable (fun p : (Fin 2 → (Fin d → ℝ)) × (SourceMode d → ℝ) =>
      pinObservation d L p.1 p.2) := pinObservation_joint_measurable d L

example (d : ℕ) (L : ℝ) : Measurable (regressionJoint d L) :=
  regressionJoint_measurable d L

-- d0 and L0 are valid measurability/probability boundaries, not regularity claims.
example (v : PinIndex 0 → ℝ) : regressionKernel 0 0 (fun _ => 0,0,v) = sourceLaw 0 :=
  zeroR_law 0 0 _ v

-- Correct-law C4 consumer uses equality with the proved pushforward law.
example (d : ℕ) (L : ℝ) [Fact (0 < L)] (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    ∀ᵐ eta ∂regressionKernel d L (fun _ => x,fun _ _ => 1,v),
      ContDiff ℝ 4 (sourceField d L eta) ∧
      ContMDiff 𝓘(ℝ,Fin d → ℝ) 𝓘(ℝ) 4 (I4Torus.torusSourceField d L eta) := by
  rw [regressionKernel_apply]
  exact (transformedLaw_regular d L (fun _ => x) (fun _ _ => 1) v).mono
    (fun _ h => h.2)

-- Zero R need not reproduce a nonzero prescribed observation.
example : pinObservation 1 1 (fun _ => 0)
    (regressionJoint 1 1 ((fun _ => 0,0,fun _ => 1),0)) (0,none) ≠ 1 := by
  classical
  simp [regressionJoint,regressionCoefficients,pinObservation,scalarSeries]

end PinKernelTests

