import SourceCovariance
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic
import Mathlib.Probability.Distributions.Gaussian.IsGaussianProcess.Basic

/-! Literal descent of the existing real-cover field to a product of additive circles.
The sample law remains sourceLaw. Smooth torus and Banach-valued laws are separate. -/
noncomputable section
open I4Foundation I4Weights I4Source I4Lattice I4Kernel I4SourceGaussian I4SourceCovariance
open MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal

namespace I4Torus

abbrev SourceTorus (d : ℕ) (L : ℝ) := Fin d → AddCircle L

def torusProjection (d : ℕ) (L : ℝ) (x : Fin d → ℝ) : SourceTorus d L :=
  fun i => (x i : AddCircle L)

def torusRepresentative (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (q : SourceTorus d L) : Fin d → ℝ :=
  fun i => (AddCircle.equivIco L 0 (q i)).1

def torusSourceField (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (ω : SourceMode d → ℝ) (q : SourceTorus d L) : ℝ :=
  sourceField d L ω (torusRepresentative d L q)

theorem torusProjection_measurable (d : ℕ) (L : ℝ) : Measurable (torusProjection d L) := by
  exact Measurable.of_eval (fun i => AddCircle.measurable_mk'.comp (measurable_pi_apply i))

theorem torusRepresentative_measurable (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    Measurable (torusRepresentative d L) := by
  exact Measurable.of_eval (fun i => measurable_subtype_coe.comp
    ((AddCircle.measurableEquivIco L 0).measurable.comp (measurable_pi_apply i)))

theorem torusProjection_representative (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (q : SourceTorus d L) : torusProjection d L (torusRepresentative d L q) = q := by
  funext i
  exact AddCircle.coe_equivIco (p := L) (a := 0) (y := q i)

theorem sourceField_eq_of_torusProjection_eq (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SourceMode d → ℝ) (x y : Fin d → ℝ)
    (h : torusProjection d L x = torusProjection d L y) :
    sourceField d L ω x = sourceField d L ω y := by
  classical
  have hz (i : Fin d) : ∃ n : ℤ, n • L = y i - x i := by
    apply (AddCircle.coe_eq_zero_iff (p := L)).mp
    rw [AddCircle.coe_sub]
    exact sub_eq_zero.mpr (congrFun h i).symm
  choose n hn using hz
  have he : y = fun i => x i + L * (n i : ℝ) := by
    funext i
    have hi := hn i
    rw [zsmul_eq_mul] at hi
    linarith
  rw [he]
  exact (sourceField_lattice_shift d hL ω n x).symm

theorem torusSourceField_pullback (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (ω : SourceMode d → ℝ) (x : Fin d → ℝ) :
    torusSourceField d L ω (torusProjection d L x) = sourceField d L ω x := by
  exact sourceField_eq_of_torusProjection_eq d (Fact.out : 0 < L).ne' ω
    (torusRepresentative d L (torusProjection d L x)) x
    (torusProjection_representative d L (torusProjection d L x))

theorem torusSourceField_eval_measurable (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (q : SourceTorus d L) : Measurable (fun ω => torusSourceField d L ω q) := by
  exact sourceField_eval_measurable d L (torusRepresentative d L q)

theorem torusSourceField_joint_measurable (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    Measurable (fun p : (SourceMode d → ℝ) × SourceTorus d L =>
      torusSourceField d L p.1 p.2) := by
  unfold torusSourceField sourceField scalarSeries
  apply Measurable.tsum
  intro m
  have hm : Continuous (fun x : Fin d → ℝ => modeCoefficient d L m x) := by
    cases m with
    | none => simp only [modeCoefficient]; fun_prop
    | some p =>
      rcases p with ⟨k,b⟩
      cases b <;> simp only [modeCoefficient, Bool.false_eq_true, ite_false, ite_true] <;>
        unfold phase <;> fun_prop
  exact (hm.measurable.comp ((torusRepresentative_measurable d L).comp measurable_snd)).mul
    ((measurable_pi_apply m).comp measurable_fst)

theorem torusSourceField_finiteVector_hasGaussianLaw (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (n : ℕ) (q : Fin n → SourceTorus d L) :
    HasGaussianLaw (fun ω j => torusSourceField d L ω (q j)) (sourceLaw d) := by
  exact source_finiteVector_hasGaussianLaw d (Fact.out : 0 < L).ne' n
    (fun j => torusRepresentative d L (q j))

theorem torusSourceField_isGaussianProcess (d : ℕ) (L : ℝ) [Fact (0 < L)] :
    IsGaussianProcess (fun q ω => torusSourceField d L ω q) (sourceLaw d) := by
  exact (sourceField_isGaussianProcess d (Fact.out : 0 < L).ne').comp_right
    (torusRepresentative d L)

theorem torusSourceField_covariance (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (q p : SourceTorus d L) :
    cov[fun ω => torusSourceField d L ω q, fun ω => torusSourceField d L ω p; sourceLaw d] =
      sourceSpectralKernel d L (torusRepresentative d L q - torusRepresentative d L p) := by
  exact sourceField_covariance d (Fact.out : 0 < L).ne'
    (torusRepresentative d L q) (torusRepresentative d L p)

theorem torusSourceField_covariance_pullback (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (x y : Fin d → ℝ) :
    cov[fun ω => torusSourceField d L ω (torusProjection d L x),
      fun ω => torusSourceField d L ω (torusProjection d L y); sourceLaw d] =
      sourceSpectralKernel d L (x - y) := by
  simp_rw [torusSourceField_pullback]
  exact sourceField_covariance d (Fact.out : 0 < L).ne' x y

end I4Torus
