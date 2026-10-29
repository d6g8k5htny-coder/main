import SourceDerivativeGaussian
import ContactJetCovariance
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! The actual origin contact covariance of the literal Fourier source.
Rows are (f,fx,fy,fxx,fxy,fxxx,fyy). Selected canonical Fourier modes are
proved to give the seven-mode witness with their actual positive amplitudes. -/

noncomputable section
open I4Source I4Weights I4C4 I4SmoothCarrier I4SourceDerivativeGaussian I4ContactGram
open MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators Matrix NNReal

namespace I4SourceContact

def contactOrder : Fin 7 → ℕ := ![0,1,1,2,2,3,2]

def contactE0 : Fin 2 → ℝ := ![1,0]
def contactE1 : Fin 2 → ℝ := ![0,1]

def contactDirections : (i : Fin 7) → Fin (contactOrder i) → (Fin 2 → ℝ) :=
  Fin.cases ![]
    (Fin.cases ![contactE0]
      (Fin.cases ![contactE1]
        (Fin.cases ![contactE0,contactE0]
          (Fin.cases ![contactE0,contactE1]
            (Fin.cases ![contactE0,contactE0,contactE0]
              (Fin.cases ![contactE1,contactE1] (fun i => Fin.elim0 i)))))))

def sourceContactVector (L : ℝ) (ω : SmoothSample 2 L) (i : Fin 7) : ℝ :=
  iteratedFDeriv ℝ (contactOrder i) (smoothSourceField 2 L ω) 0 (contactDirections i)

def sourceContactCoefficient (L : ℝ) (i : Fin 7) (m : SourceMode 2) : ℝ :=
  derivativeCoefficient 2 L (contactOrder i) 0 (contactDirections i) m

def sourceContactCovariance (L : ℝ) : Matrix (Fin 7) (Fin 7) ℝ :=
  fun i j => cov[fun ω => sourceContactVector L ω i,
    fun ω => sourceContactVector L ω j; smoothLaw 2 L]

theorem sourceContactVector_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) :
    HasGaussianLaw (sourceContactVector L) (smoothLaw 2 L) :=
  smoothSourceField_derivativeVector_hasGaussianLaw 2 hL 7 contactOrder (fun _ => 0)
    contactDirections

theorem sourceContactCoefficient_summable {L : ℝ} (hL : L ≠ 0) (i : Fin 7) :
    Summable (sourceContactCoefficient L i) :=
  derivativeCoefficient_summable 2 hL (contactOrder i) 0 (contactDirections i)

theorem sourceContactCovariance_eq_coefficientGram {L : ℝ} (hL : L ≠ 0) :
    sourceContactCovariance L = coefficientGram (sourceContactCoefficient L) := by
  ext i j
  exact smoothSourceField_derivative_covariance 2 hL (contactOrder i) (contactOrder j)
    0 0 (contactDirections i) (contactDirections j)

private def frequency10 : ↥(halfLattice 2) := ⟨![1,0], by
  refine ⟨0, by norm_num, ?_⟩
  intro j hj
  fin_cases j <;> norm_num at hj⟩

private def frequency01 : ↥(halfLattice 2) := ⟨![0,1], by
  refine ⟨1, by norm_num, ?_⟩
  intro j hj
  fin_cases j <;> norm_num at *⟩

private def frequency11 : ↥(halfLattice 2) := ⟨![1,1], by
  refine ⟨0, by norm_num, ?_⟩
  intro j hj
  fin_cases j <;> norm_num at hj⟩

private def frequency20 : ↥(halfLattice 2) := ⟨![2,0], by
  refine ⟨0, by norm_num, ?_⟩
  intro j hj
  fin_cases j <;> norm_num at hj⟩

/-- DC, cos(1,0), cos(0,1), cos(1,1), sin(1,0), sin(2,0), sin(0,1).
The frequency subtypes retain the canonical half-lattice membership proofs. -/
def sourceContactMode : Fin 7 → SourceMode 2 :=
  ![none, some (frequency10,false), some (frequency01,false), some (frequency11,false),
    some (frequency10,true), some (frequency20,true), some (frequency01,true)]

theorem sourceContactMode_nonzero_frequency (s : Fin 7) :
    match sourceContactMode s with
    | none => True
    | some (k, _) => (k : Fin 2 → ℤ) ≠ 0 := by
  cases sourceContactMode s with
  | none => trivial
  | some p => exact I4Lattice.halfLattice_ne_zero 2 p.1.property

def sourceModeAmplitude (L : ℝ) (m : SourceMode 2) : ℝ := match m with
  | none => Real.sqrt (1 / sourceZ 2 L)
  | some (k, _) => Real.sqrt (2 * sourceWeight 2 L k / sourceZ 2 L)

def sourceContactAmplitude (L : ℝ) (s : Fin 7) : ℝ :=
  sourceModeAmplitude L (sourceContactMode s)

theorem sourceContactAmplitude_pos {L : ℝ} (hL : L ≠ 0) (s : Fin 7) :
    0 < sourceContactAmplitude L s := by
  unfold sourceContactAmplitude sourceModeAmplitude
  cases sourceContactMode s with
  | none => exact Real.sqrt_pos.mpr (one_div_pos.mpr (sourceZ_pos 2 hL))
  | some p =>
    apply Real.sqrt_pos.mpr
    apply div_pos _ (sourceZ_pos 2 hL)
    apply mul_pos (by norm_num)
    rw [sourceWeight_eq_physical]
    exact Real.exp_pos _

private theorem derivativeCoefficient_none (L : ℝ) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) :
    derivativeCoefficient 2 L k 0 u none = if k = 0 then Real.sqrt (1 / sourceZ 2 L) else 0 := by
  cases k with
  | zero => simp [derivativeCoefficient, modeCoefficient, iteratedFDeriv_zero_apply]
  | succ k =>
    simp only [Nat.succ_ne_zero, ite_false]
    change (iteratedFDeriv ℝ (k + 1)
      (fun _ : Fin 2 → ℝ => Real.sqrt (1 / sourceZ 2 L)) 0) u = 0
    rw [iteratedFDeriv_succ_const]
    rfl

private theorem derivativeCoefficient_some (L : ℝ) (k : ↥(halfLattice 2)) (b : Bool)
    (n : ℕ) (u : Fin n → (Fin 2 → ℝ)) :
    derivativeCoefficient 2 L n 0 u (some (k,b)) =
      Real.sqrt (2 * sourceWeight 2 L k / sourceZ 2 L) *
        ((∏ j, phaseCLM 2 L k (u j)) *
          iteratedDeriv n (if b then Real.sin else Real.cos) 0) := by
  let A := Real.sqrt (2 * sourceWeight 2 L k / sourceZ 2 L)
  let g : ℝ → ℝ := if b then Real.sin else Real.cos
  let ell := phaseCLM 2 L k
  have hg : ContDiff ℝ n g := by
    cases b <;> simp only [g, Bool.false_eq_true, ite_false, ite_true] <;> fun_prop
  have hcomp : ContDiff ℝ n (g ∘ ell) := hg.comp_continuousLinearMap
  have heq : modeCoefficient 2 L (some (k,b)) = A • (g ∘ ell) := by
    funext x
    cases b <;> simp [modeCoefficient, A, g, ell, phaseCLM_apply, Function.comp_apply]
  unfold derivativeCoefficient
  rw [heq, iteratedFDeriv_const_smul_apply hcomp.contDiffAt,
    ell.iteratedFDeriv_comp_right hg 0 le_rfl]
  simp only [_root_.smul_apply, ContinuousMultilinearMap.compContinuousLinearMap_apply,
    map_zero, iteratedFDeriv_apply_eq_iteratedDeriv_mul_prod, smul_eq_mul]
  rfl

set_option maxHeartbeats 2000000 in
/-- These are literal differentiated source columns, with the actual Fourier
amplitudes, rather than assumed witness columns or a finite-mode replacement. -/
theorem sourceContact_selected_columns (L : ℝ) (i s : Fin 7) :
    sourceContactCoefficient L i (sourceContactMode s) =
      sourceContactAmplitude L s * contactWitness (sourceC L) i s := by
  fin_cases i
  all_goals first
    | change derivativeCoefficient 2 L 0 0 ![] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 1 0 ![contactE0] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 1 0 ![contactE1] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 2 0 ![contactE0,contactE0] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 2 0 ![contactE0,contactE1] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 3 0 ![contactE0,contactE0,contactE0] (sourceContactMode s) = _
    | change derivativeCoefficient 2 L 2 0 ![contactE1,contactE1] (sourceContactMode s) = _
  all_goals fin_cases s <;>
    norm_num [sourceContactCoefficient, sourceContactMode, sourceContactAmplitude,
      sourceModeAmplitude, contactOrder, contactE0, contactE1,
      derivativeCoefficient_none, derivativeCoefficient_some, contactWitness,
      phaseCLM_apply, phase, frequency10, frequency01, frequency11, frequency20,
      Fin.prod_univ_succ, Real.iteratedDeriv_add_one_sin, Real.iteratedDeriv_add_one_cos,
      iteratedDeriv_zero] <;> ring_nf <;> simp

/-- Positive definiteness of the actual contact covariance, obtained from its
infinite coefficient Gram sum and seven proved canonical source columns. -/
theorem sourceContactCovariance_posDef {L : ℝ} (hL : L ≠ 0) :
    (sourceContactCovariance L).PosDef := by
  rw [sourceContactCovariance_eq_coefficientGram hL]
  apply contact_coefficientGram_posDef (sourceContactCoefficient L)
    (sourceContactCoefficient_summable hL)
    (show sourceC L ≠ 0 from div_ne_zero
      (mul_ne_zero (by norm_num) (ne_of_gt Real.pi_pos)) hL)
    sourceContactMode (sourceContactAmplitude L)
    (fun s => (sourceContactAmplitude_pos hL s).ne')
  exact sourceContact_selected_columns L

end I4SourceContact
