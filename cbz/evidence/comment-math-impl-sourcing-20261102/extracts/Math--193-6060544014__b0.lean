import SourceContactRank
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Analysis.Normed.Module.Multilinear.Curry

noncomputable section
open I4Source I4Jets I4JetLaw I4ContactRank MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal Matrix
namespace I4MorseRank

def morseObservable (p : ℕ) (L : ℝ) (x u alpha beta : Fin (p+1) → ℝ)
    (sample : SourceMode (p+1) → ℝ) : ℝ :=
  sourceJet (p+1) L 1 sample x (fun _ => alpha) +
  sourceJet (p+1) L 2 sample x (fun k : Fin 2 => if k=0 then u else beta)

abbrev MorseIndex (p : ℕ) := Fin (p+1) ⊕ Fin (p+1)
def observationOrder (p : ℕ) : MorseIndex p → ℕ := Sum.elim (fun _ => 1) (fun _ => 2)
def observationDirections (p : ℕ) (u : Fin (p+1) → ℝ) (i : MorseIndex p) :
    Fin (observationOrder p i) → (Fin (p+1) → ℝ) := by
  cases i with
  | inl j => exact fun _ => Pi.single j 1
  | inr j =>
    change Fin 2 → (Fin (p+1) → ℝ)
    exact fun k => if k=0 then u else Pi.single j 1

def projectedMorseVector (p : ℕ) (L : ℝ) (x u : Fin (p+1) → ℝ)
    (sample : SourceMode (p+1) → ℝ) : MorseIndex p → ℝ :=
  fun i => sourceJet (p+1) L (observationOrder p i) sample x (observationDirections p u i)

def morseCovariance (p : ℕ) (L : ℝ) (x u : Fin (p+1) → ℝ) :
    Matrix (MorseIndex p) (MorseIndex p) ℝ :=
  fun i j => cov[fun sample => projectedMorseVector p L x u sample i,
    fun sample => projectedMorseVector p L x u sample j; sourceLaw (p+1)]

private def columnEquiv (p : ℕ) (u : Fin (p+1) → ℝ) (i : Fin (p+1)) (hi : u i ≠ 0) :
    (Fin (p+1) → ℝ) ≃ₗ[ℝ] (Fin (p+1) → ℝ) where
  toFun z := fun j => if j=i then u i*z i else z j+u j*z i
  invFun z := fun j => if j=i then z i/u i else z j-u j*(z i/u i)
  left_inv z := by
    funext j
    by_cases h : j=i <;> simp [h] <;> field_simp [hi] <;> ring
  right_inv z := by
    funext j
    by_cases h : j=i <;> simp [h] <;> field_simp [hi] <;> ring
  map_add' z w := by
    funext j
    by_cases h : j=i <;> simp [h] <;> ring
  map_smul' c z := by
    funext j
    by_cases h : j=i <;> simp [h,smul_eq_mul] <;> ring

private def swapEquiv (p : ℕ) (i : Fin (p+1)) :
    (Fin (p+1) → ℝ) ≃ₗ[ℝ] (Fin (p+1) → ℝ) where
  toFun z := fun j => z (Equiv.swap 0 i j)
  invFun z := fun j => z (Equiv.swap 0 i j)
  left_inv z := by funext j; simp
  right_inv z := by funext j; simp
  map_add' z w := rfl
  map_smul' c z := rfl

private theorem frame_sum (p : ℕ)
    (T : (Fin (p+1) → ℝ) ≃L[ℝ] (Fin (p+1) → ℝ))
    (ell : (Fin (p+1) → ℝ) →L[ℝ] ℝ) (a : Fin (p+1) → ℝ) :
    (∑ i, a i * ell (frameVector p T i)) = ell (T a) := by
  classical
  symm
  conv_lhs => rw [← LinearMap.sum_single_apply (fun _ : Fin (p+1) => ℝ) a]
  rw [map_sum,map_sum]
  apply Finset.sum_congr rfl
  intro i _
  have hs : Pi.single i (a i) = a i • Pi.single i (1 : ℝ) := by
    rw [← Pi.single_smul,smul_eq_mul,mul_one]
  rw [hs,map_smul,map_smul]
  rfl

private def hessianCLM (p : ℕ) (L : ℝ) (sample : SourceMode (p+1) → ℝ)
    (x u : Fin (p+1) → ℝ) : (Fin (p+1) → ℝ) →L[ℝ] ℝ :=
  (iteratedFDeriv ℝ 2 (sourceField (p+1) L sample) x).curryRight (fun _ : Fin 1 => u)

private theorem jet1_eq (p : ℕ) (L : ℝ) (sample : SourceMode (p+1) → ℝ)
    (x a : Fin (p+1) → ℝ) : sourceJet (p+1) L 1 sample x (fun _ => a) =
      fderiv ℝ (sourceField (p+1) L sample) x a := by
  simp only [sourceJet,iteratedFDeriv_one_apply]

private theorem jet2_eq (p : ℕ) (L : ℝ) (sample : SourceMode (p+1) → ℝ)
    (x u b : Fin (p+1) → ℝ) :
    sourceJet (p+1) L 2 sample x (fun k : Fin 2 => if k=0 then u else b) =
      hessianCLM p L sample x u b := by
  unfold hessianCLM sourceJet
  rw [ContinuousMultilinearMap.curryRight_apply]
  congr 1
  funext k
  fin_cases k <;> rfl

private theorem observable_contact (p : ℕ) (L : ℝ)
    (T : (Fin (p+1) → ℝ) ≃L[ℝ] (Fin (p+1) → ℝ))
    (x a w : Fin (p+1) → ℝ) :
    contactObservable p L T x 0 a (w 0) 0 (fun j => w j.succ) =
      morseObservable p L x (frameVector p T 0) (T a) (T w) := by
  classical
  funext sample
  simp only [contactObservable,morseObservable,zero_mul,zero_add,add_zero,jet1_eq,jet2_eq]
  rw [frame_sum]
  have hsame : sourceJet (p+1) L 2 sample x (fun _ => frameVector p T 0) =
      hessianCLM p L sample x (frameVector p T 0) (frameVector p T 0) := by
    simpa only [ite_self] using jet2_eq p L sample x (frameVector p T 0) (frameVector p T 0)
  rw [hsame]
  have he := frame_sum p T (hessianCLM p L sample x (frameVector p T 0)) w
  rw [Fin.sum_univ_succ] at he
  rw [add_assoc,he]

theorem exists_axial_frame (p : ℕ) (u : Fin (p+1) → ℝ) (hu : u ≠ 0) :
    ∃ T : (Fin (p+1) → ℝ) ≃L[ℝ] (Fin (p+1) → ℝ), T (Pi.single 0 1) = u := by
  classical
  obtain ⟨i,hi⟩ : ∃ i, u i ≠ 0 := by
    by_contra hn
    push Not at hn
    exact hu (funext hn)
  let P := swapEquiv p i
  let S := columnEquiv p u i hi
  have he : P (Pi.single 0 1) = Pi.single i 1 := by
    funext j
    by_cases h : i=j
    · subst j
      simp [P,swapEquiv,Pi.single_apply]
    · have hj : Equiv.swap (0 : Fin (p+1)) i j ≠ 0 := by
        intro hz
        have hji : j=i := by
          simpa only [Equiv.swap_apply_left] using (Equiv.swap_apply_eq_iff.mp hz)
        exact h hji.symm
      simp [P,swapEquiv,Pi.single_apply,h,hj.symm]
  refine ⟨(P.trans S).toContinuousLinearEquiv,?_⟩
  change S (P (Pi.single 0 1)) = u
  rw [he]
  funext j
  by_cases h : j=i <;> simp [S,columnEquiv,h]

theorem morseObservable_variance_pos (p : ℕ) (hp : 1 ≤ p) {L : ℝ} (hL : 0 < L)
    (x u alpha beta : Fin (p+1) → ℝ) (hu : u ≠ 0) (hw : alpha ≠ 0 ∨ beta ≠ 0) :
    0 < Var[morseObservable p L x u alpha beta; sourceLaw (p+1)] := by
  classical
  obtain ⟨T,hT⟩ := exists_axial_frame p u hu
  let a := T.symm alpha
  let w := T.symm beta
  have hw' : a ≠ 0 ∨ w 0 ≠ 0 ∨ (fun j : Fin p => w j.succ) ≠ 0 := by
    by_contra hn
    push Not at hn
    have ha : alpha = 0 := by rw [← T.apply_symm_apply alpha]; change T a = 0; rw [hn.1,map_zero]
    have hw0 : w = 0 := by
      funext i
      exact Fin.cases hn.2.1 (fun j => congrFun hn.2.2 j) i
    have hb : beta = 0 := by rw [← T.apply_symm_apply beta]; change T w = 0; rw [hw0,map_zero]
    exact hw.elim (fun h => h ha) (fun h => h hb)
  have hpv := contactObservable_variance_pos p hp hL T x 0 a (w 0) 0 (fun j => w j.succ)
    (hw'.elim (fun h => Or.inr (Or.inl h)) (fun h => h.elim
      (fun h => Or.inr (Or.inr (Or.inl h)))
      (fun h => Or.inr (Or.inr (Or.inr (Or.inr h))))))
  have he := observable_contact p L T x a w
  have he' : contactObservable p L T x 0 a (w 0) 0 (fun j => w j.succ) =
      morseObservable p L x u alpha beta := by
    simpa only [frameVector,hT,a,w,T.apply_symm_apply] using he
  rw [he'] at hpv
  exact hpv

theorem projectedMorseVector_hasGaussianLaw (p : ℕ) {L : ℝ} (hL : L ≠ 0)
    (x u : Fin (p+1) → ℝ) :
    HasGaussianLaw (projectedMorseVector p L x u) (sourceLaw (p+1)) := by
  classical
  let e := Fintype.equivFin (MorseIndex p)
  let r := fun i => observationOrder p (e.symm i)
  let v := fun i => observationDirections p u (e.symm i)
  have hr : ∀ i, r i ≤ 4 := by
    intro i
    dsimp [r]
    cases e.symm i <;> norm_num [observationOrder]
  have hg := sourceJet_finiteVector_hasGaussianLaw (p+1) hL _ r hr (fun _ => x) v
  let A := (LinearEquiv.piCongrLeft ℝ (fun _ : MorseIndex p => ℝ) e.symm).toContinuousLinearEquiv
  apply (hg.map A.toContinuousLinearMap).congr
  exact Filter.Eventually.of_forall (fun sample => by
    funext i
    simp [A,LinearEquiv.piCongrLeft,LinearEquiv.piCongrLeft',Equiv.piCongrLeft',
      r,v,projectedMorseVector]
    rw [e.symm_apply_apply])

theorem morseCovariance_posDef (p : ℕ) (hp : 1 ≤ p) {L : ℝ} (hL : 0 < L)
    (x u : Fin (p+1) → ℝ) (hu : u ≠ 0) :
    (morseCovariance p L x u).PosDef := by
  classical
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_,?_⟩
  · ext i j
    simp [morseCovariance,Matrix.conjTranspose_apply,covariance_comm]
  · intro z hz
    let alpha : Fin (p+1) → ℝ := fun i => z (Sum.inl i)
    let beta : Fin (p+1) → ℝ := fun i => z (Sum.inr i)
    have hw : alpha ≠ 0 ∨ beta ≠ 0 := by
      by_contra hn
      push Not at hn
      apply hz
      funext i
      cases i with
      | inl j => exact congrFun hn.1 j
      | inr j => exact congrFun hn.2 j
    have hpv := morseObservable_variance_pos p hp hL x u alpha beta hu hw
    have hsum : (fun sample => ∑ i, z i * projectedMorseVector p L x u sample i) =
        morseObservable p L x u alpha beta := by
      funext sample
      simp only [Fintype.sum_sum_type,projectedMorseVector,observationOrder,observationDirections,
        morseObservable,Sum.elim_inl,Sum.elim_inr,id_eq,jet1_eq,jet2_eq]
      have ha := frame_sum p (ContinuousLinearEquiv.refl ℝ (Fin (p+1) → ℝ))
        (fderiv ℝ (sourceField (p+1) L sample) x) alpha
      have hb := frame_sum p (ContinuousLinearEquiv.refl ℝ (Fin (p+1) → ℝ))
        (hessianCLM p L sample x u) beta
      simpa only [frameVector,ContinuousLinearEquiv.refl_apply,alpha,beta] using congrArg₂ (·+·) ha hb
    have hg := projectedMorseVector_hasGaussianLaw p hL.ne' x u
    have hm (i : MorseIndex p) :
        MemLp (fun sample => projectedMorseVector p L x u sample i) 2 (sourceLaw (p+1)) :=
      (hg.map (ContinuousLinearMap.proj i)).memLp_two
    have hv := variance_fun_sum (fun i => (hm i).const_mul (z i))
    rw [hsum] at hv
    simp only [covariance_const_mul_left,covariance_const_mul_right] at hv
    have he : star z ⬝ᵥ (morseCovariance p L x u *ᵥ z) =
        Var[morseObservable p L x u alpha beta;sourceLaw (p+1)] := by
      rw [hv]
      simp only [Matrix.mulVec,dotProduct,Pi.star_apply,star_trivial,Finset.mul_sum,morseCovariance]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [he]
    exact hpv

end I4MorseRank

