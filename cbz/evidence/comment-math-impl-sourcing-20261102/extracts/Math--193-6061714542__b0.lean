import PinMajorant
import Mathlib.Analysis.Matrix.MeasurableSpace
import Mathlib.Probability.Kernel.Composition.Prod
import Mathlib.Probability.Kernel.Composition.MapComap

noncomputable section
open I4Foundation I4Weights I4Source I4Jets I4PinMajorant MeasureTheory ProbabilityTheory
open scoped BigOperators
namespace I4PinKernel

abbrev PinParameters (d : ℕ) :=
  (Fin 2 → (Fin d → ℝ)) × (Matrix (PinIndex d) (PinIndex d) ℝ) × (PinIndex d → ℝ)
def regressionJoint (d : ℕ) (L : ℝ) (p : PinParameters d × (SourceMode d → ℝ)) :
    SourceMode d → ℝ :=
  regressionCoefficients d L p.1.1 p.1.2.1 p.1.2.2 p.2
def regressionKernel (d : ℕ) (L : ℝ) : Kernel (PinParameters d) (SourceMode d → ℝ) :=
  (Kernel.id ×ₖ Kernel.const (PinParameters d) (sourceLaw d)).map (regressionJoint d L)

private theorem actualMode_contDiff (d : ℕ) (L : ℝ) (m : SourceMode d) :
    ContDiff ℝ 4 (modeCoefficient d L m) := by
  cases m with
  | none =>
    change ContDiff ℝ 4 (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L))
    fun_prop
  | some p =>
    rcases p with ⟨k,b⟩
    change ContDiff ℝ 4 (fun x : Fin d → ℝ =>
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        (if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)))
    cases b <;> simp [phase] <;> fun_prop

theorem pinCoefficient_joint_measurable (d : ℕ) (L : ℝ) (m : SourceMode d) :
    Measurable (fun sites : Fin 2 → (Fin d → ℝ) => pinCoefficient d L sites m) := by
  classical
  apply Measurable.of_eval
  intro i
  rcases i with ⟨a,j⟩
  cases j with
  | none =>
    change Measurable (fun sites : Fin 2 → (Fin d → ℝ) => modeCoefficient d L m (sites a))
    exact (actualMode_contDiff d L m).continuous.measurable.comp (measurable_pi_apply a)
  | some j =>
    change Measurable (fun sites : Fin 2 → (Fin d → ℝ) =>
      iteratedFDeriv ℝ 1 (modeCoefficient d L m) (sites a) (fun _ => Pi.single j 1))
    simp_rw [iteratedFDeriv_one_apply]
    have hh : Measurable (fun sites : Fin 2 → (Fin d → ℝ) =>
        (sites a,(Pi.single j 1 : Fin d → ℝ))) :=
      (measurable_pi_apply a).prodMk measurable_const
    exact ((actualMode_contDiff d L m).continuous_fderiv_apply (by norm_num)).measurable.comp
      hh

theorem pinObservation_joint_measurable (d : ℕ) (L : ℝ) :
    Measurable (fun p : (Fin 2 → (Fin d → ℝ)) × (SourceMode d → ℝ) =>
      pinObservation d L p.1 p.2) := by
  apply Measurable.of_eval
  intro i
  unfold pinObservation scalarSeries
  apply Measurable.tsum
  intro m
  exact (((measurable_pi_apply i).comp (pinCoefficient_joint_measurable d L m)).comp measurable_fst).mul
    ((measurable_pi_apply m).comp measurable_snd)

theorem regressionJoint_measurable (d : ℕ) (L : ℝ) :
    Measurable (regressionJoint d L) := by
  classical
  apply Measurable.of_eval
  intro m
  have hc (i : PinIndex d) : Measurable (fun p : PinParameters d × (SourceMode d → ℝ) =>
      pinCoefficient d L p.1.1 m i) :=
    ((measurable_pi_apply i).comp (pinCoefficient_joint_measurable d L m)).comp (measurable_fst.fst)
  have hu (j : PinIndex d) : Measurable (fun p : PinParameters d × (SourceMode d → ℝ) =>
      pinObservation d L p.1.1 p.2 j) :=
    ((measurable_pi_apply j).comp (pinObservation_joint_measurable d L)).comp
      (measurable_fst.fst.prodMk measurable_snd)
  unfold regressionJoint regressionCoefficients
  apply Measurable.add
  · apply Measurable.sub
    · exact (measurable_pi_apply m).comp measurable_snd
    · apply Finset.measurable_sum
      intro i _
      apply Finset.measurable_sum
      intro j _
      exact ((hc i).mul (measurable_fst.snd.fst.eval_matrix)).mul (hu j)
  · apply Finset.measurable_sum
    intro i _
    apply Finset.measurable_sum
    intro j _
    exact ((hc i).mul (measurable_fst.snd.fst.eval_matrix)).mul
      ((measurable_pi_apply j).comp measurable_fst.snd.snd)

theorem regressionKernel_markov (d : ℕ) (L : ℝ) :
    IsMarkovKernel (regressionKernel d L) := by
  unfold regressionKernel
  exact Kernel.IsMarkovKernel.map _ (regressionJoint_measurable d L)

theorem regressionKernel_apply (d : ℕ) (L : ℝ) (p : PinParameters d) :
    regressionKernel d L p = transformedLaw d L p.1 p.2.1 p.2.2 := by
  ext s hs : 1
  rw [regressionKernel,Kernel.map_apply' _ (regressionJoint_measurable d L) _ hs,
    Kernel.id_prod_apply' _ _ (regressionJoint_measurable d L hs),Kernel.const_apply,
    transformedLaw,Measure.map_apply (regressionCoefficients_measurable d L p.1 p.2.1 p.2.2) hs]
  rfl

end I4PinKernel

