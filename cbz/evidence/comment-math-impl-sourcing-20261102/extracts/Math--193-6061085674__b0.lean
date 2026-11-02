import DeterministicSourceC4
import SourceJetLaw
import Mathlib.MeasureTheory.Constructions.Polish.Basic
import Mathlib.Order.Filter.AtTopBot.CountablyGenerated

noncomputable section
open I4Foundation I4Weights I4Source I4Jets I4DeterministicC4 MeasureTheory
open scoped BigOperators Manifold ContDiff Topology
namespace I4PinMajorant
attribute [local instance] I4TorusC4.actualChartedSpace

abbrev PinIndex (d : ℕ) := Fin 2 × Option (Fin d)
def pinOrder (d : ℕ) (i : PinIndex d) : ℕ := i.2.elim 0 (fun _ => 1)
def pinDirections (d : ℕ) (i : PinIndex d) : Fin (pinOrder d i) → (Fin d → ℝ) := by
  rcases i with ⟨s,j⟩
  cases j with
  | none => exact fun k => Fin.elim0 k
  | some j => exact fun _ => Pi.single j 1
def pinCoefficient (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (m : SourceMode d) (i : PinIndex d) : ℝ :=
  modeJetCoefficient d L m (pinOrder d i) (sites i.1) (pinDirections d i)
def pinObservation (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (omega : SourceMode d → ℝ) (i : PinIndex d) : ℝ :=
  scalarSeries (fun m => pinCoefficient d L sites m i) omega
def actualPinObservation (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (omega : SourceMode d → ℝ) (i : PinIndex d) : ℝ :=
  i.2.elim (sourceField d L omega (sites i.1))
    (fun j => fderiv ℝ (sourceField d L omega) (sites i.1) (Pi.single j 1))
def regressionCoefficients (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (v : PinIndex d → ℝ)
    (omega : SourceMode d → ℝ) (m : SourceMode d) : ℝ :=
  omega m - (∑ i, ∑ j, pinCoefficient d L sites m i * R i j * pinObservation d L sites omega j)
    + (∑ i, ∑ j, pinCoefficient d L sites m i * R i j * v j)
def transformedLaw (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (R : Matrix (PinIndex d) (PinIndex d) ℝ) (v : PinIndex d → ℝ) :=
  (sourceLaw d).map (regressionCoefficients d L sites R v)

private theorem pinCoefficient_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (m : SourceMode d) (i : PinIndex d) :
    |pinCoefficient d L sites m i| ≤ modeMajorant d L m := by
  rcases i with ⟨s,j⟩
  cases j with
  | none =>
    change |(iteratedFDeriv ℝ 0 (modeCoefficient d L m) (sites s))
      (fun k : Fin 0 => Fin.elim0 k)| ≤ _
    rw [iteratedFDeriv_zero_apply]
    exact modeCoefficient_bound d hL m (sites s)
  | some j =>
    have h := modeJetCoefficient_bound d hL m 1 (by omega) (sites s)
      (fun _ => (Pi.single j 1 : Fin d → ℝ))
    simpa [pinCoefficient,pinOrder,pinDirections,Pi.norm_single] using h

private theorem majorant_event_measurable (d : ℕ) (L : ℝ) :
    MeasurableSet {eta : SourceMode d → ℝ |
      Summable (fun m => modeMajorant d L m * |eta m|)} := by
  change MeasurableSet {eta : SourceMode d → ℝ | ∃ a : ℝ,
    Filter.Tendsto (fun s : Finset (SourceMode d) => ∑ m ∈ s,
      modeMajorant d L m * |eta m|) Filter.atTop (nhds a)}
  apply measurableSet_exists_tendsto
  intro s
  fun_prop

theorem pinObservation_eq_actual_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ omega ∂sourceLaw d, ∀ sites : Fin 2 → (Fin d → ℝ),
      pinObservation d L sites omega = actualPinObservation d L sites omega := by
  filter_upwards [sourceJet_eq_series_ae d hL] with omega hjet
  intro sites
  funext i
  rcases i with ⟨s,j⟩
  cases j with
  | none =>
    have h := hjet 0 (by omega) (sites s) (fun k => Fin.elim0 k)
    simpa [pinObservation,pinCoefficient,pinOrder,pinDirections,actualPinObservation,
      sourceJet_zero] using h.symm
  | some j =>
    have h := hjet 1 (by omega) (sites s) (fun _ => (Pi.single j 1 : Fin d → ℝ))
    simpa [pinObservation,pinCoefficient,pinOrder,pinDirections,actualPinObservation,
      sourceJet,iteratedFDeriv_one_apply] using h.symm

theorem regressionCoefficients_measurable (d : ℕ) (L : ℝ)
    (sites : Fin 2 → (Fin d → ℝ)) (R : Matrix (PinIndex d) (PinIndex d) ℝ)
    (v : PinIndex d → ℝ) : Measurable (regressionCoefficients d L sites R v) := by
  classical
  apply Measurable.of_eval
  intro m
  have hobs (i : PinIndex d) : Measurable (fun omega => pinObservation d L sites omega i) :=
    scalarSeries_measurable _
  unfold regressionCoefficients
  fun_prop

theorem regression_majorant (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (R : Matrix (PinIndex d) (PinIndex d) ℝ)
    (v : PinIndex d → ℝ) (omega : SourceMode d → ℝ)
    (homega : Summable (fun m => modeMajorant d L m * |omega m|)) :
    Summable (fun m => modeMajorant d L m * |regressionCoefficients d L sites R v omega m|) := by
  classical
  have hb := modeMajorant_summable d hL
  have hbb : Summable (fun m => modeMajorant d L m * modeMajorant d L m) :=
    (hb.mul_of_nonneg hb (modeMajorant_nonneg d L) (modeMajorant_nonneg d L)).comp_injective
      (show Function.Injective (fun m : SourceMode d => (m,m)) from
        fun _ _ h => congrArg Prod.fst h)
  have ha (i : PinIndex d) : Summable (fun m => modeMajorant d L m * pinCoefficient d L sites m i) := by
    apply hbb.of_norm_bounded
    intro m
    rw [Real.norm_eq_abs,abs_mul,abs_of_nonneg (modeMajorant_nonneg d L m)]
    exact mul_le_mul_of_nonneg_left (pinCoefficient_bound d hL sites m i)
      (modeMajorant_nonneg d L m)
  have ho : Summable (fun m => modeMajorant d L m * omega m) := by
    apply Summable.of_norm
    simpa only [Real.norm_eq_abs,abs_mul,abs_of_nonneg (modeMajorant_nonneg d L _)] using homega
  have hs : Summable (fun m => ∑ i, ∑ j,
      modeMajorant d L m * pinCoefficient d L sites m i * R i j * pinObservation d L sites omega j) :=
    summable_sum (fun i _ => summable_sum (fun j _ => ((ha i).mul_right (R i j)).mul_right _))
  have ht : Summable (fun m => ∑ i, ∑ j,
      modeMajorant d L m * pinCoefficient d L sites m i * R i j * v j) :=
    summable_sum (fun i _ => summable_sum (fun j _ => ((ha i).mul_right (R i j)).mul_right _))
  have he : Summable (fun m => modeMajorant d L m * regressionCoefficients d L sites R v omega m) := by
    simpa only [regressionCoefficients,mul_add,mul_sub,Finset.mul_sum,mul_assoc] using (ho.sub hs).add ht
  simpa only [Real.norm_eq_abs,abs_mul,abs_of_nonneg (modeMajorant_nonneg d L _)] using he.norm

theorem transformedLaw_regular (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (sites : Fin 2 → (Fin d → ℝ)) (R : Matrix (PinIndex d) (PinIndex d) ℝ)
    (v : PinIndex d → ℝ) :
    ∀ᵐ eta ∂transformedLaw d L sites R v,
      Summable (fun m => modeMajorant d L m * |eta m|) ∧
      ContDiff ℝ 4 (sourceField d L eta) ∧
      ContMDiff 𝓘(ℝ, Fin d → ℝ) 𝓘(ℝ) 4 (I4Torus.torusSourceField d L eta) := by
  have hL : L ≠ 0 := (Fact.out : 0 < L).ne'
  have hmaj : ∀ᵐ eta ∂transformedLaw d L sites R v,
      Summable (fun m => modeMajorant d L m * |eta m|) := by
    change ∀ᵐ eta ∂(sourceLaw d).map (regressionCoefficients d L sites R v), _
    apply (ae_map_iff (regressionCoefficients_measurable d L sites R v).aemeasurable
      (majorant_event_measurable d L)).2
    filter_upwards [sourceMajorant_ae d hL] with omega homega
    exact regression_majorant d hL sites R v omega homega
  filter_upwards [hmaj] with eta heta
  exact ⟨heta,sourceField_contDiff_of_majorant d hL eta heta,
    torusSourceField_contMDiff_of_majorant d L eta heta⟩

end I4PinMajorant

