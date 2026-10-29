import GaussianSeriesFoundation
import SpectralWeights
import Mathlib.Logic.Equiv.Option
import Mathlib.Topology.Algebra.InfiniteSum.TsumUniformlyOn

/-! Actual canonical real Fourier series and a shared-event uniform convergence bridge.
The half-lattice partition, covariance, sample C4 and finite-jet law require separate proofs. -/

noncomputable section
open MeasureTheory ProbabilityTheory I4Foundation I4Weights
open scoped BigOperators ENNReal

namespace I4Source

/-- The representative whose first nonzero coordinate is positive. -/
def halfLattice (d : ℕ) : Set (Fin d → ℤ) :=
  {k | ∃ i, 0 < k i ∧ ∀ j, j < i → k j = 0}

abbrev SourceMode (d : ℕ) := Option (↥(halfLattice d) × Bool)

/-- The actual common law, with independent variance-one real coefficients. -/
abbrev sourceLaw (d : ℕ) : Measure (SourceMode d → ℝ) :=
  coefficientLaw (SourceMode d)

def phase (d : ℕ) (L : ℝ) (k : Fin d → ℤ) (x : Fin d → ℝ) : ℝ :=
  ∑ i, sourceC L * (k i : ℝ) * x i

/-- False is the cosine coefficient and true the independent sine coefficient. -/
def modeCoefficient (d : ℕ) (L : ℝ) (m : SourceMode d) (x : Fin d → ℝ) : ℝ :=
  match m with
  | none => Real.sqrt (1 / sourceZ d L)
  | some (k, b) => Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
      if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)

def modeMajorant (d : ℕ) (L : ℝ) (m : SourceMode d) : ℝ :=
  match m with
  | none => Real.sqrt (1 / sourceZ d L)
  | some (k, _) => pairedBound d L k

/-- Literal normalized real Fourier sum on the common coefficient samples. -/
def sourceField (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ) : ℝ :=
  scalarSeries (fun m => modeCoefficient d L m x) ω

theorem modeMajorant_nonneg (d : ℕ) (L : ℝ) (m : SourceMode d) :
    0 ≤ modeMajorant d L m := by
  cases m with
  | none => exact Real.sqrt_nonneg _
  | some p =>
    change 0 ≤ pairedBound d L p.1
    unfold pairedBound productBound coordinateBound decay
    positivity

theorem modeMajorant_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Summable (modeMajorant d L) := by
  classical
  have hs := pairedBound_subtype_summable d hL (halfLattice d)
  have hb : Summable (fun _ : Bool => (1 : ℝ)) := (hasSum_fintype _).summable
  have hp : Summable (fun p : ↥(halfLattice d) × Bool => pairedBound d L p.1) := by
    simpa only [mul_one] using hs.mul_of_nonneg hb
      (fun k => modeMajorant_nonneg d L (some (k, false))) (fun _ => zero_le_one)
  let e := Equiv.optionEquivSumPUnit.{0, 0} (↥(halfLattice d) × Bool)
  apply e.symm.summable_iff.mp
  apply Summable.sum (fun q => modeMajorant d L (e.symm q))
  · simpa [e, Function.comp_def, modeMajorant] using hp
  · exact (hasSum_fintype _).summable

theorem modeCoefficient_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (x : Fin d → ℝ) :
    |modeCoefficient d L m x| ≤ modeMajorant d L m := by
  cases m with
  | none =>
    exact (abs_of_nonneg (Real.sqrt_nonneg _)).le
  | some p =>
    rcases p with ⟨k, b⟩
    have hprod : 1 ≤ ∏ i, (1 + |sourceC L * (k.val i : ℝ)|) ^ 4 := by
      apply Finset.one_le_prod₀
      intro i _
      apply one_le_pow₀
      linarith [abs_nonneg (sourceC L * (k.val i : ℝ))]
    have hamp : Real.sqrt (2 * sourceWeight d L k / sourceZ d L) ≤
        pairedBound d L k := by
      rw [pairedBound_eq_amplitude d hL k]
      simpa only [mul_one] using mul_le_mul_of_nonneg_left hprod
        (Real.sqrt_nonneg (2 * sourceWeight d L k / sourceZ d L))
    change |Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
      (if b then Real.sin (phase d L k x) else Real.cos (phase d L k x))| ≤ _
    rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
    have ht : |if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)| ≤ 1 := by
      cases b with
      | false => exact Real.abs_cos_le_one _
      | true => exact Real.abs_sin_le_one _
    have hbase : Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        |if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)| ≤
        Real.sqrt (2 * sourceWeight d L k / sourceZ d L) := by
      simpa only [mul_one] using mul_le_mul_of_nonneg_left ht
        (Real.sqrt_nonneg (2 * sourceWeight d L k / sourceZ d L))
    exact hbase.trans hamp

theorem sourceField_eval_measurable (d : ℕ) (L : ℝ) (x : Fin d → ℝ) :
    Measurable (fun ω => sourceField d L ω x) := by
  exact scalarSeries_measurable _

theorem sourceMajorant_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, Summable (fun m => modeMajorant d L m * |ω m|) := by
  exact weighted_abs_summable_ae _ (modeMajorant_nonneg d L) (modeMajorant_summable d hL)

theorem sourceAbs_uniform_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d,
      HasSumUniformly (fun m x => |modeCoefficient d L m x * ω m|)
        (fun x => ∑' m, |modeCoefficient d L m x * ω m|) := by
  filter_upwards [sourceMajorant_ae d hL] with ω hω
  apply hasSumUniformly_iff_tendstoUniformly.mpr
  apply tendstoUniformly_tsum hω
  intro m x
  simp only [Real.norm_eq_abs, abs_abs, abs_mul]
  exact mul_le_mul_of_nonneg_right (modeCoefficient_bound d hL m x) (abs_nonneg _)

theorem sourceField_uniform_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d,
      HasSumUniformly (fun m x => modeCoefficient d L m x * ω m)
        (sourceField d L ω) := by
  filter_upwards [sourceMajorant_ae d hL] with ω hω
  apply hasSumUniformly_iff_tendstoUniformly.mpr
  apply tendstoUniformly_tsum hω
  intro m x
  simp only [Real.norm_eq_abs, abs_mul]
  exact mul_le_mul_of_nonneg_right (modeCoefficient_bound d hL m x) (abs_nonneg _)

theorem sourceField_continuous_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, Continuous (sourceField d L ω) := by
  filter_upwards [sourceMajorant_ae d hL] with ω hω
  change Continuous (fun x => ∑' m, modeCoefficient d L m x * ω m)
  have hcont (m : SourceMode d) :
      Continuous (fun x : Fin d → ℝ => modeCoefficient d L m x * ω m) := by
    cases m with
    | none =>
      change Continuous (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L) * ω none)
      exact continuous_const
    | some p =>
      rcases p with ⟨k, b⟩
      cases b <;> simp [modeCoefficient, phase] <;> fun_prop
  apply continuous_tsum hcont hω
  intro m x
  simp only [Real.norm_eq_abs, abs_mul]
  exact mul_le_mul_of_nonneg_right (modeCoefficient_bound d hL m x) (abs_nonneg _)

end I4Source
