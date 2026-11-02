import TwoPinAlgebra
import PinMajorant
import SourceContactRank
import Mathlib.LinearAlgebra.Matrix.PosDef

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets I4JetLaw I4SourceGaussian I4PinMajorant
  I4Torus MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix
namespace I4TwoPinRank

def combinedCoefficient (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (z : PinIndex d → ℝ) (m : SourceMode d) : ℝ :=
  ∑ i, z i * pinCoefficient d L sites m i
def twoPinObservable (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (z : PinIndex d → ℝ) (omega : SourceMode d → ℝ) : ℝ :=
  ∑ i, z i * actualPinObservation d L sites omega i
def pinCovariance (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Matrix (PinIndex d) (PinIndex d) ℝ :=
  fun i j => cov[fun omega => actualPinObservation d L sites omega i,
    fun omega => actualPinObservation d L sites omega j;sourceLaw d]

private theorem modeJet_some (d : ℕ) (L : ℝ) (k : ↥(halfLattice d)) (b : Bool)
    (j : ℕ) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    modeJetCoefficient d L (some (k,b)) j x v =
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        ((∏ i, phaseCLM d L k (v i)) *
          iteratedDeriv j (if b then Real.sin else Real.cos) (phase d L k x)) := by
  classical
  let g : ℝ → ℝ := if b then Real.sin else Real.cos
  let ell := phaseCLM d L k
  let amp := Real.sqrt (2 * sourceWeight d L k / sourceZ d L)
  have hg : ContDiff ℝ j g := by cases b <;> simp [g] <;> fun_prop
  have heq : modeCoefficient d L (some (k,b)) = amp • (g ∘ ell) := by
    funext y
    cases b <;> simp [modeCoefficient, amp, g, ell, Function.comp_apply,
      Pi.smul_apply, smul_eq_mul, phaseCLM_apply]
  unfold modeJetCoefficient
  rw [heq, iteratedFDeriv_const_smul_apply (hg.comp_continuousLinearMap).contDiffAt,
    ell.iteratedFDeriv_comp_right hg x le_rfl]
  simp only [smul_apply,
    ContinuousMultilinearMap.compContinuousLinearMap_apply,
    iteratedFDeriv_apply_eq_iteratedDeriv_mul_prod, smul_eq_mul]
  simp only [amp, g, ell, phaseCLM_apply]

private theorem jet_one (d : ℕ) (L : ℝ) (k : ↥(halfLattice d)) (b : Bool)
    (x u : Fin d → ℝ) :
    modeJetCoefficient d L (some (k,b)) 1 x (fun _ => u) =
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) * phaseCLM d L k u *
        (if b then Real.cos (phase d L k x) else -Real.sin (phase d L k x)) := by
  rw [modeJet_some]
  cases b <;> norm_num [Finset.prod_const, Fin.prod_univ_two] <;> ring

private theorem pin_value (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (m : SourceMode d) (s : Fin 2) :
    pinCoefficient d L sites m (s,none) = modeCoefficient d L m (sites s) := by
  exact iteratedFDeriv_zero_apply _

private theorem pin_grad (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (m : SourceMode d) (s : Fin 2) (j : Fin d) :
    pinCoefficient d L sites m (s,some j) =
      modeJetCoefficient d L m 1 (sites s) (fun _ => Pi.single j 1) := rfl

private theorem combine_expand (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (z : PinIndex d → ℝ) (m : SourceMode d) :
    combinedCoefficient d L sites z m =
      z (0,none)*modeCoefficient d L m (sites 0) +
      (∑ j, z (0,some j)*modeJetCoefficient d L m 1 (sites 0) (fun _ => Pi.single j 1)) +
      z (1,none)*modeCoefficient d L m (sites 1) +
      ∑ j, z (1,some j)*modeJetCoefficient d L m 1 (sites 1) (fun _ => Pi.single j 1) := by
  classical
  simp only [combinedCoefficient,Fintype.sum_prod_type,Fin.sum_univ_two,Fintype.sum_option,
    pin_value,pin_grad]
  ring

private theorem combined_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (z : PinIndex d → ℝ) :
    Summable (combinedCoefficient d L sites z) := by
  classical
  apply summable_sum
  intro i _
  exact (modeJetCoefficient_summable d hL (pinOrder d i)
    (by rcases i with ⟨s,j⟩; cases j <;> norm_num [pinOrder])
    (sites i.1) (pinDirections d i)).mul_left (z i)

private theorem observable_series (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (z : PinIndex d → ℝ) :
    twoPinObservable d L sites z =ᵐ[sourceLaw d]
      scalarSeries (combinedCoefficient d L sites z) := by
  classical
  have hs (i : PinIndex d) : Summable (fun m => pinCoefficient d L sites m i) :=
    modeJetCoefficient_summable d hL (pinOrder d i)
      (by rcases i with ⟨s,j⟩; cases j <;> norm_num [pinOrder]) _ _
  filter_upwards [ae_all_iff.2 (fun i : PinIndex d =>
      scalarSeries_hasSum_ae (fun m => pinCoefficient d L sites m i) (hs i)),
    scalarSeries_hasSum_ae (combinedCoefficient d L sites z) (combined_summable d hL sites z),
    pinObservation_eq_actual_ae d hL] with omega hrows hcombined hactual
  have h := hasSum_sum (s := Finset.univ) (fun i _ => (hrows i).mul_left (z i))
  have hh : HasSum (fun m => combinedCoefficient d L sites z m * omega m)
      (twoPinObservable d L sites z omega) := by
    convert h using 1
    · funext m
      simp only [combinedCoefficient,Finset.sum_mul,mul_assoc]
    · unfold twoPinObservable
      rw [← hactual sites]
      rfl
  exact hh.unique hcombined

private theorem actual_row_gaussian (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (i : PinIndex d) :
    HasGaussianLaw (fun omega => actualPinObservation d L sites omega i) (sourceLaw d) := by
  have hs : Summable (fun m => pinCoefficient d L sites m i) :=
    modeJetCoefficient_summable d hL (pinOrder d i)
      (by rcases i with ⟨s,j⟩; cases j <;> norm_num [pinOrder]) _ _
  have hg : HasGaussianLaw (scalarSeries (fun m => pinCoefficient d L sites m i)) (sourceLaw d) := by
    constructor
    · exact (scalarSeries_measurable _).aemeasurable
    · rw [scalarSeries_map_gaussian _ hs]
      infer_instance
  apply hg.congr
  exact (pinObservation_eq_actual_ae d hL).mono (fun _ h => congrFun (h sites) i)

private theorem algebra_phase (d : ℕ) (L : ℝ) (k : Fin d → ℤ) (x : Fin d → ℝ) :
    phase d L k x = I4TwoPinAlgebra.phase d (sourceC L) k x := by
  simp only [phase,I4TwoPinAlgebra.phase,Finset.mul_sum,mul_assoc]

private theorem combined_pair (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (z : PinIndex d → ℝ) (k : ↥(halfLattice d)) (b : Bool) :
    combinedCoefficient d L sites z (some (k,b)) =
      Real.sqrt (2*sourceWeight d L k/sourceZ d L) *
      (if b then I4TwoPinAlgebra.sinSymbol d (sourceC L) (sites 0) (sites 1)
          (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) k
       else I4TwoPinAlgebra.cosSymbol d (sourceC L) (sites 0) (sites 1)
         (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) k) := by
  classical
  let amp := Real.sqrt (2*sourceWeight d L k/sourceZ d L)
  have hgrad (B : Fin d → ℝ) (q : ℝ) :
      (∑ j, B j*(amp*phaseCLM d L k (Pi.single j 1)*q)) = amp*q*phase d L k B := by
    unfold phase
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    have he : phaseCLM d L k (Pi.single j 1) = sourceC L*(k.val j : ℝ) := by
      simp [phaseCLM_apply,phase,Pi.single_apply]
    rw [he]
    ring
  rw [combine_expand]
  simp_rw [jet_one]
  cases b <;> simp only [Bool.false_eq_true,ite_false,ite_true,modeCoefficient]
  all_goals
    change _ = amp*_
    rw [hgrad,hgrad]
    simp only [I4TwoPinAlgebra.cosSymbol,I4TwoPinAlgebra.sinSymbol,algebra_phase]
    ring

private theorem combined_none (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (z : PinIndex d → ℝ) : combinedCoefficient d L sites z none =
      Real.sqrt (1/sourceZ d L)*(z (0,none)+z (1,none)) := by
  classical
  have hn (x u : Fin d → ℝ) : modeJetCoefficient d L none 1 x (fun _ => u) = 0 := by
    unfold modeJetCoefficient
    change (iteratedFDeriv ℝ 1 (fun _ : Fin d → ℝ => Real.sqrt (1/sourceZ d L)) x) _ = 0
    rw [iteratedFDeriv_succ_const]
    rfl
  rw [combine_expand]
  simp only [hn,mul_zero,Finset.sum_const_zero,modeCoefficient,add_zero]
  ring

private theorem algebra_phase_neg (d : ℕ) (c : ℝ) (k : Fin d → ℤ) (x : Fin d → ℝ) :
    I4TwoPinAlgebra.phase d c (-k) x = -I4TwoPinAlgebra.phase d c k x := by
  simp [I4TwoPinAlgebra.phase,Int.cast_neg,Finset.sum_neg_distrib]

private theorem algebra_cos_neg (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ) (k : Fin d → ℤ) :
    I4TwoPinAlgebra.cosSymbol d c x y A C B D (-k) = I4TwoPinAlgebra.cosSymbol d c x y A C B D k := by
  simp [I4TwoPinAlgebra.cosSymbol,algebra_phase_neg]

private theorem algebra_sin_neg (d : ℕ) (c : ℝ) (x y : Fin d → ℝ)
    (A C : ℝ) (B D : Fin d → ℝ) (k : Fin d → ℤ) :
    I4TwoPinAlgebra.sinSymbol d c x y A C B D (-k) = -I4TwoPinAlgebra.sinSymbol d c x y A C B D k := by
  simp only [I4TwoPinAlgebra.sinSymbol,algebra_phase_neg,Real.sin_neg,Real.cos_neg]
  ring

private theorem annihilation_symbols (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ)) (z : PinIndex d → ℝ)
    (hz : ∀ m, combinedCoefficient d L sites z m = 0) (k : Fin d → ℤ) :
    I4TwoPinAlgebra.cosSymbol d (sourceC L) (sites 0) (sites 1)
      (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) k = 0 ∧
    I4TwoPinAlgebra.sinSymbol d (sourceC L) (sites 0) (sites 1)
      (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) k = 0 := by
  have hhalf (l : ↥(halfLattice d)) :
      I4TwoPinAlgebra.cosSymbol d (sourceC L) (sites 0) (sites 1)
        (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) l = 0 ∧
      I4TwoPinAlgebra.sinSymbol d (sourceC L) (sites 0) (sites 1)
        (z (0,none)) (z (1,none)) (fun j => z (0,some j)) (fun j => z (1,some j)) l = 0 := by
    have hamp : Real.sqrt (2*sourceWeight d L l/sourceZ d L) ≠ 0 := by
      apply ne_of_gt
      apply Real.sqrt_pos.2
      apply div_pos
      · rw [sourceWeight_eq_physical]
        positivity
      · exact sourceZ_pos d hL.ne'
    have hc := hz (some (l,false)); have hs := hz (some (l,true))
    rw [combined_pair] at hc hs
    simp only [Bool.false_eq_true,ite_false,ite_true] at hc hs
    exact ⟨(mul_eq_zero.mp hc).resolve_left hamp,(mul_eq_zero.mp hs).resolve_left hamp⟩
  by_cases hk : k=0
  · subst k
    have hn := hz none
    rw [combined_none] at hn
    have hamp : Real.sqrt (1/sourceZ d L) ≠ 0 :=
      ne_of_gt (Real.sqrt_pos.2 (one_div_pos.2 (sourceZ_pos d hL.ne')))
    have hA := (mul_eq_zero.mp hn).resolve_left hamp
    simpa [I4TwoPinAlgebra.cosSymbol,I4TwoPinAlgebra.sinSymbol,I4TwoPinAlgebra.phase] using
      And.intro hA (show (0:ℝ)=0 from rfl)
  · rcases (I4Lattice.halfLattice_partition d hk).1 with hp | hn
    · exact hhalf ⟨k,hp⟩
    · have h := hhalf ⟨-k,hn⟩
      simpa only [algebra_cos_neg,algebra_sin_neg,neg_eq_zero] using h

theorem torus_distinct_phase (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x y : Fin d → ℝ) (hxy : torusProjection d L x ≠ torusProjection d L y) :
    ∃ i, Real.cos (sourceC L * (x i - y i)) ≠ 1 := by
  classical
  by_contra hn
  push Not at hn
  apply hxy
  funext i
  obtain ⟨n,hn⟩ := (Real.cos_eq_one_iff _).mp (hn i)
  have he : (n : ℝ)*L = x i-y i := by
    unfold sourceC at hn
    have hp : 2*Real.pi ≠ 0 := by positivity
    field_simp [hL.ne'] at hn
    nlinarith [Real.pi_pos]
  have hz : ((x i-y i : ℝ) : AddCircle L) = 0 :=
    (AddCircle.coe_eq_zero_iff (p:=L)).2 ⟨n,by simpa [zsmul_eq_mul] using he⟩
  simpa only [torusProjection,AddCircle.coe_sub,sub_eq_zero] using hz

theorem combinedCoefficient_nonzero (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1))
    (z : PinIndex d → ℝ) (hz : z ≠ 0) :
    ∃ m, combinedCoefficient d L sites z m ≠ 0 := by
  classical
  by_contra hn
  push Not at hn
  have hc : sourceC L ≠ 0 := by unfold sourceC; positivity
  obtain ⟨hA,hC,hB,hD⟩ := I4TwoPinAlgebra.trigJet_separation d (sourceC L) hc
    (sites 0) (sites 1) (z (0,none)) (z (1,none))
    (fun j => z (0,some j)) (fun j => z (1,some j))
    (torus_distinct_phase d hL _ _ hxy)
    (fun k => (annihilation_symbols d hL sites z hn k).1)
    (fun k => (annihilation_symbols d hL sites z hn k).2)
  apply hz
  funext i
  rcases i with ⟨s,j⟩
  fin_cases s <;> cases j with
  | none => assumption
  | some j => first | exact congrFun hB j | exact congrFun hD j

theorem twoPinObservable_variance_pos (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1))
    (z : PinIndex d → ℝ) (hz : z ≠ 0) :
    0 < Var[twoPinObservable d L sites z;sourceLaw d] := by
  classical
  have hs := combined_summable d hL.ne' sites z
  rw [variance_congr (observable_series d hL.ne' sites z),scalarSeries_variance _ hs]
  have hsq : Summable (fun m => (combinedCoefficient d L sites z m)^2) := by
    simpa only [Function.comp_def,pow_two] using
      (summable_mul_of_summable_norm hs.norm hs.norm).comp_injective
        (show Function.Injective (fun m : SourceMode d => (m,m)) from
          fun _ _ h => congrArg Prod.fst h)
  obtain ⟨m,hm⟩ := combinedCoefficient_nonzero d hL sites hxy z hz
  exact hsq.tsum_pos (fun _ => sq_nonneg _) m (sq_pos_of_ne_zero hm)

theorem pinCovariance_posDef (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1)) :
    (pinCovariance d L sites).PosDef := by
  classical
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_,?_⟩
  · ext i j
    simp [pinCovariance,Matrix.conjTranspose_apply,covariance_comm]
  · intro z hz
    have hm (i : PinIndex d) : MemLp (fun omega => actualPinObservation d L sites omega i) 2 (sourceLaw d) :=
      (actual_row_gaussian d hL.ne' sites i).memLp_two
    have hv := variance_fun_sum (fun i => (hm i).const_mul (z i))
    change Var[twoPinObservable d L sites z;sourceLaw d] = _ at hv
    simp only [covariance_const_mul_left,covariance_const_mul_right] at hv
    have he : star z ⬝ᵥ (pinCovariance d L sites *ᵥ z) =
        Var[twoPinObservable d L sites z;sourceLaw d] := by
      rw [hv]
      simp only [Matrix.mulVec,dotProduct,Pi.star_apply,star_trivial,Finset.mul_sum,pinCovariance]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [he]
    exact twoPinObservable_variance_pos d hL sites hxy z hz

end I4TwoPinRank

