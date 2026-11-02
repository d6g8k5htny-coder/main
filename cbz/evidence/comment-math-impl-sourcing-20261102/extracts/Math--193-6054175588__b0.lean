import SourceJetLaw
import SourceLattice
import ContactAlgebra

/-! Positive actual variance of the required physical contact vector.
No assumed functional independence, nonzero Fourier coefficient or pin covariance interface. -/

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets I4JetLaw I4SourceGaussian
  MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace I4ContactRank

def frameVector (d : ℕ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ)) (i : Fin (d+1)) :
    Fin (d+1) → ℝ := T (Pi.single i 1)

def contactObservable (d : ℕ) (L : ℝ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ)
    (D : Fin d → ℝ) (ω : SourceMode (d+1) → ℝ) : ℝ :=
  A * sourceField (d+1) L ω x +
  (∑ i, a i * sourceJet (d+1) L 1 ω x (fun _ => frameVector d T i)) +
  B * sourceJet (d+1) L 2 ω x (fun _ => frameVector d T 0) +
  C * sourceJet (d+1) L 3 ω x (fun _ => frameVector d T 0) +
  ∑ j, D j * sourceJet (d+1) L 2 ω x
    (fun k : Fin 2 => if k = 0 then frameVector d T 0 else frameVector d T j.succ)

private def contactCoefficient (d : ℕ) (L : ℝ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ)
    (D : Fin d → ℝ) (m : SourceMode (d+1)) : ℝ :=
  A * modeCoefficient (d+1) L m x +
  (∑ i, a i * modeJetCoefficient (d+1) L m 1 x (fun _ => frameVector d T i)) +
  B * modeJetCoefficient (d+1) L m 2 x (fun _ => frameVector d T 0) +
  C * modeJetCoefficient (d+1) L m 3 x (fun _ => frameVector d T 0) +
  ∑ j, D j * modeJetCoefficient (d+1) L m 2 x
    (fun k : Fin 2 => if k = 0 then frameVector d T 0 else frameVector d T j.succ)

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

private theorem jet_two (d : ℕ) (L : ℝ) (k : ↥(halfLattice d)) (b : Bool)
    (x : Fin d → ℝ) (v : Fin 2 → (Fin d → ℝ)) :
    modeJetCoefficient d L (some (k,b)) 2 x v =
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        phaseCLM d L k (v 0) * phaseCLM d L k (v 1) *
        (if b then -Real.sin (phase d L k x) else -Real.cos (phase d L k x)) := by
  rw [modeJet_some]
  cases b <;> norm_num [Finset.prod_const, Fin.prod_univ_two] <;> ring

private theorem jet_three (d : ℕ) (L : ℝ) (k : ↥(halfLattice d)) (b : Bool)
    (x u : Fin d → ℝ) :
    modeJetCoefficient d L (some (k,b)) 3 x (fun _ => u) =
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) * (phaseCLM d L k u)^3 *
        (if b then -Real.cos (phase d L k x) else Real.sin (phase d L k x)) := by
  rw [modeJet_some]
  cases b <;> norm_num [Finset.prod_const, Fin.prod_univ_two] <;> ring

private theorem coefficient_none (d : ℕ) (L : ℝ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ) :
    contactCoefficient d L T x A a B C D none = A * Real.sqrt (1 / sourceZ (d+1) L) := by
  have hn (j : ℕ) (v : Fin (j+1) → (Fin (d+1) → ℝ)) :
      modeJetCoefficient (d+1) L none (j+1) x v = 0 := by
    unfold modeJetCoefficient
    change (iteratedFDeriv ℝ (j+1)
      (fun _ : Fin (d+1) → ℝ => Real.sqrt (1 / sourceZ (d+1) L)) x) v = 0
    rw [iteratedFDeriv_succ_const]
    rfl
  simp [contactCoefficient, hn, modeCoefficient]

private theorem frame_phase (d : ℕ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (ell : (Fin (d+1) → ℝ) →L[ℝ] ℝ) (a : Fin (d+1) → ℝ) :
    (∑ i, a i * ell (frameVector d T i)) = ell (T a) := by
  classical
  symm
  conv_lhs => rw [← LinearMap.sum_single_apply (fun _ : Fin (d+1) => ℝ) a]
  rw [map_sum, map_sum]
  apply Finset.sum_congr rfl
  intro i _
  have hs : Pi.single i (a i) = a i • Pi.single i (1 : ℝ) := by
    rw [← Pi.single_smul, smul_eq_mul, mul_one]
  rw [hs, map_smul, map_smul]
  rfl

private def contactW (d : ℕ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ)) (B : ℝ) (D : Fin d → ℝ) :
    Fin (d+1) → ℝ := B • frameVector d T 0 + ∑ j, D j • frameVector d T j.succ

private theorem coefficient_pair (d : ℕ) (L : ℝ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ)
    (k : ↥(halfLattice (d+1))) (b : Bool) :
    contactCoefficient d L T x A a B C D (some (k,b)) =
      Real.sqrt (2 * sourceWeight (d+1) L k / sourceZ (d+1) L) *
      (if b then
        (A - phaseCLM (d+1) L k (frameVector d T 0) * phaseCLM (d+1) L k (contactW d T B D)) *
          Real.sin (phase (d+1) L k x) +
        (phaseCLM (d+1) L k (T a) - C * (phaseCLM (d+1) L k (frameVector d T 0))^3) *
          Real.cos (phase (d+1) L k x)
      else
        (A - phaseCLM (d+1) L k (frameVector d T 0) * phaseCLM (d+1) L k (contactW d T B D)) *
          Real.cos (phase (d+1) L k x) -
        (phaseCLM (d+1) L k (T a) - C * (phaseCLM (d+1) L k (frameVector d T 0))^3) *
          Real.sin (phase (d+1) L k x)) := by
  classical
  let ell := phaseCLM (d+1) L k
  let amp := Real.sqrt (2 * sourceWeight (d+1) L k / sourceZ (d+1) L)
  have hW : ell (contactW d T B D) =
      B * ell (frameVector d T 0) + ∑ j, D j * ell (frameVector d T j.succ) := by
    simp [contactW, map_add, map_sum, map_smul, smul_eq_mul]
  unfold contactCoefficient
  simp_rw [jet_one, jet_two, jet_three]
  simp only [ite_true, show (1 : Fin 2) ≠ 0 from by decide, ite_false]
  cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true, modeCoefficient]
  all_goals
    have hgrad (q : ℝ) :
        (∑ i, a i * (amp * ell (frameVector d T i) * q)) = amp * q * ell (T a) := by
      rw [← frame_phase d T ell a, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    have hmix (q : ℝ) :
        (∑ j, D j * (amp * ell (frameVector d T 0) * ell (frameVector d T j.succ) * q)) =
          amp * ell (frameVector d T 0) * q * ∑ j, D j * ell (frameVector d T j.succ) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    change _ = amp * _
    rw [hgrad, hmix]
    rw [show phaseCLM (d+1) L k = ell from rfl, hW]
    ring

private theorem half_nonneg (n : ℕ) (k : Fin n → ℤ) (hk : k ≠ 0)
    (hn : ∀ i, 0 ≤ k i) : k ∈ halfLattice n := by
  rcases (I4Lattice.halfLattice_partition n hk).1 with h | h
  · exact h
  · rcases h with ⟨i, hi, _⟩
    have hi' : 0 < -k i := by simpa using hi
    have := hn i
    omega

private theorem phase_single (n : ℕ) (L : ℝ) (i : Fin n) (z : ℤ) (v : Fin n → ℝ) :
    phaseCLM n L (Pi.single i z) v = sourceC L * (z : ℝ) * v i := by
  classical
  simp [phaseCLM_apply, phase, Pi.single_apply]

private theorem phase_add (n : ℕ) (L : ℝ) (k l : Fin n → ℤ) (v : Fin n → ℝ) :
    phaseCLM n L (k+l) v = phaseCLM n L k v + phaseCLM n L l v := by
  simp [phaseCLM_apply, phase, mul_add, add_mul, Finset.sum_add_distrib]

private theorem coefficient_zero_symbols (d : ℕ) {L : ℝ} (hL : 0 < L)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ)
    (hz : ∀ m, contactCoefficient d L T x A a B C D m = 0)
    (k : ↥(halfLattice (d+1))) :
    (A - phaseCLM (d+1) L k (frameVector d T 0) * phaseCLM (d+1) L k (contactW d T B D) = 0) ∧
    (phaseCLM (d+1) L k (T a) - C * (phaseCLM (d+1) L k (frameVector d T 0))^3 = 0) := by
  have ha : Real.sqrt (2 * sourceWeight (d+1) L k / sourceZ (d+1) L) ≠ 0 := by
    apply ne_of_gt
    apply Real.sqrt_pos.2
    apply div_pos
    · rw [sourceWeight_eq_physical]
      positivity
    · exact sourceZ_pos (d+1) (ne_of_gt hL)
  have hc := hz (some (k,false))
  have hs := hz (some (k,true))
  rw [coefficient_pair] at hc hs
  simp only [Bool.false_eq_true, ite_false, ite_true] at hc hs
  have hc' := (mul_eq_zero.mp hc).resolve_left ha
  have hs' := (mul_eq_zero.mp hs).resolve_left ha
  exact SourceContactRank.contact_phase_elimination _ _ _ _
    (Real.sin_sq_add_cos_sq _) hc' hs'

private theorem coefficient_exists (d : ℕ) {L : ℝ} (hL : 0 < L)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ)
    (hw : A ≠ 0 ∨ a ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0 ∨ D ≠ 0) :
    ∃ m, contactCoefficient d L T x A a B C D m ≠ 0 := by
  classical
  by_contra hn
  push Not at hn
  have hA : A = 0 := by
    have h := hn none
    rw [coefficient_none] at h
    have hp : Real.sqrt (1 / sourceZ (d+1) L) ≠ 0 :=
      ne_of_gt (Real.sqrt_pos.2 (one_div_pos.2 (sourceZ_pos (d+1) (ne_of_gt hL))))
    exact (mul_eq_zero.mp h).resolve_right hp
  let u := frameVector d T 0
  let V := T a
  let W := contactW d T B D
  let c := sourceC L
  have hc : c ≠ 0 := by dsimp [c, sourceC]; positivity
  have hu : u ≠ 0 := by
    intro h
    change T (Pi.single (0 : Fin (d+1)) (1 : ℝ)) = 0 at h
    have he : (Pi.single (0 : Fin (d+1)) (1 : ℝ)) = 0 :=
      T.injective (h.trans (map_zero T).symm)
    have he0 := congrFun he 0
    norm_num at he0
  have hbase (i : Fin (d+1)) :
      A - (c * u i) * (c * W i) = 0 ∧ c * V i - C * (c * u i)^3 = 0 := by
    let k : Fin (d+1) → ℤ := Pi.single i 1
    have hk : k ≠ 0 := by
      intro h
      have h' := congrFun h i
      simp [k] at h'
    have hn' : ∀ j, 0 ≤ k j := by intro j; simp [k, Pi.single_apply]; split <;> norm_num
    have hh := coefficient_zero_symbols d hL T x A a B C D hn ⟨k, half_nonneg _ k hk hn'⟩
    simpa only [k, phase_single, Int.cast_one, mul_one, c, u, V, W] using hh
  have hdouble (i : Fin (d+1)) : 2*c*V i - C*(2*c*u i)^3 = 0 := by
    let k : Fin (d+1) → ℤ := Pi.single i 2
    have hk : k ≠ 0 := by
      intro h
      have h' := congrFun h i
      simp [k] at h'
    have hn' : ∀ j, 0 ≤ k j := by intro j; simp [k, Pi.single_apply]; split <;> norm_num
    have hh := (coefficient_zero_symbols d hL T x A a B C D hn
      ⟨k, half_nonneg _ k hk hn'⟩).2
    simp only [k, phase_single, Int.cast_ofNat] at hh
    nlinarith [hh]
  have hpair (i j : Fin (d+1)) (hij : i ≠ j) :
      A - (c*(u i+u j)) * (c*(W i+W j)) = 0 := by
    let k : Fin (d+1) → ℤ := Pi.single i 1 + Pi.single j 1
    have hk : k ≠ 0 := by
      intro h
      have h' := congrFun h i
      simp [k, hij] at h'
    have hn' : ∀ t, 0 ≤ k t := by
      intro t
      simp only [k, Pi.add_apply, Pi.single_apply]
      split <;> split <;> norm_num
    have hh := (coefficient_zero_symbols d hL T x A a B C D hn
      ⟨k, half_nonneg _ k hk hn'⟩).1
    simp only [k, phase_add, phase_single, Int.cast_one, mul_one] at hh
    nlinarith [hh]
  obtain ⟨hC, hV, hW⟩ := SourceContactRank.contact_polynomial_separation c hc u V W hu A C hA
    (fun i => (hbase i).2) hdouble (fun i => (hbase i).1) hpair
  have ha : a = 0 := T.injective (by simpa [V] using hV)
  have hrep : W = T (Fin.cons B D : Fin (d+1) → ℝ) := by
    have he : (∑ i : Fin (d+1), (Fin.cons B D : Fin (d+1) → ℝ) i • frameVector d T i) = T (Fin.cons B D : Fin (d+1) → ℝ) := by
      simp only [frameVector, ← map_smul, ← map_sum]
      congr 1
      simpa only [← Pi.single_smul, smul_eq_mul, mul_one] using
        LinearMap.sum_single_apply (fun _ : Fin (d+1) => ℝ) (Fin.cons B D : Fin (d+1) → ℝ)
    rw [← he, Fin.sum_univ_succ]
    simp [W, contactW]
  have hBD : (Fin.cons B D : Fin (d+1) → ℝ) = 0 := T.injective (by rw [← hrep]; simpa using hW)
  have hB : B = 0 := by simpa using congrFun hBD 0
  have hD : D = 0 := by ext j; simpa using congrFun hBD j.succ
  rcases hw with h | h | h | h | h
  · exact h hA
  · exact h ha
  · exact h hB
  · exact h hC
  · exact h hD

private theorem coefficient_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ) :
    Summable (contactCoefficient d L T x A a B C D) := by
  classical
  have h0 : Summable (fun m => modeCoefficient (d+1) L m x) :=
    (modeMajorant_summable (d+1) hL).of_norm_bounded
      (fun m => by simpa only [Real.norm_eq_abs] using modeCoefficient_bound (d+1) hL m x)
  have h1 (i : Fin (d+1)) := modeJetCoefficient_summable (d+1) hL 1 (by norm_num) x
    (fun _ => frameVector d T i)
  have h2 := modeJetCoefficient_summable (d+1) hL 2 (by norm_num) x (fun _ => frameVector d T 0)
  have h3 := modeJetCoefficient_summable (d+1) hL 3 (by norm_num) x (fun _ => frameVector d T 0)
  have hm (j : Fin d) := modeJetCoefficient_summable (d+1) hL 2 (by norm_num) x
    (fun k : Fin 2 => if k = 0 then frameVector d T 0 else frameVector d T j.succ)
  exact ((((h0.mul_left A).add (summable_sum (fun i _ => (h1 i).mul_left (a i)))).add
    (h2.mul_left B)).add (h3.mul_left C)).add (summable_sum (fun j _ => (hm j).mul_left (D j)))

private theorem observable_eq_series (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ) (D : Fin d → ℝ) :
    contactObservable d L T x A a B C D =ᵐ[sourceLaw (d+1)]
      scalarSeries (contactCoefficient d L T x A a B C D) := by
  classical
  let q0 := fun m => modeCoefficient (d+1) L m x
  let q1 := fun (i : Fin (d+1)) m => modeJetCoefficient (d+1) L m 1 x (fun _ => frameVector d T i)
  let q2 := fun m => modeJetCoefficient (d+1) L m 2 x (fun _ => frameVector d T 0)
  let q3 := fun m => modeJetCoefficient (d+1) L m 3 x (fun _ => frameVector d T 0)
  let qm := fun (j : Fin d) m => modeJetCoefficient (d+1) L m 2 x
    (fun k : Fin 2 => if k = 0 then frameVector d T 0 else frameVector d T j.succ)
  have h0 : Summable q0 := (modeMajorant_summable (d+1) hL).of_norm_bounded
    (fun m => by simpa only [q0, Real.norm_eq_abs] using modeCoefficient_bound (d+1) hL m x)
  have h1 (i) : Summable (q1 i) := modeJetCoefficient_summable (d+1) hL 1 (by norm_num) x _
  have h2 : Summable q2 := modeJetCoefficient_summable (d+1) hL 2 (by norm_num) x _
  have h3 : Summable q3 := modeJetCoefficient_summable (d+1) hL 3 (by norm_num) x _
  have hm (j) : Summable (qm j) := modeJetCoefficient_summable (d+1) hL 2 (by norm_num) x _
  let q := contactCoefficient d L T x A a B C D
  filter_upwards [scalarSeries_hasSum_ae q0 h0,
    ae_all_iff.2 (fun i : Fin (d+1) => scalarSeries_hasSum_ae (q1 i) (h1 i)),
    scalarSeries_hasSum_ae q2 h2, scalarSeries_hasSum_ae q3 h3,
    ae_all_iff.2 (fun j : Fin d => scalarSeries_hasSum_ae (qm j) (hm j)),
    scalarSeries_hasSum_ae q (coefficient_summable d hL T x A a B C D),
    sourceJet_eq_series_ae (d+1) hL] with ω hω0 hω1 hω2 hω3 hωm hωq hjet
  have hs := ((((hω0.mul_left A).add (hasSum_sum (s := Finset.univ) (fun i _ => (hω1 i).mul_left (a i)))).add
    (hω2.mul_left B)).add (hω3.mul_left C)).add (hasSum_sum (s := Finset.univ) (fun j _ => (hωm j).mul_left (D j)))
  have hs' : HasSum (fun m => q m * ω m) (contactObservable d L T x A a B C D ω) := by
    convert hs using 1
    · funext m
      simp only [q, contactCoefficient, q0, q1, q2, q3, qm, add_mul, Finset.sum_mul, mul_assoc]
    · unfold contactObservable
      simp only [hjet 1 (by norm_num), hjet 2 (by norm_num), hjet 3 (by norm_num)]
      rfl
  exact (hωq.unique hs').symm

theorem contactObservable_variance_pos (d : ℕ) (hd : 1 ≤ d) {L : ℝ} (hL : 0 < L)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ))
    (x : Fin (d+1) → ℝ) (A : ℝ) (a : Fin (d+1) → ℝ) (B C : ℝ)
    (D : Fin d → ℝ) (hw : A ≠ 0 ∨ a ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0 ∨ D ≠ 0) :
    0 < Var[contactObservable d L T x A a B C D; sourceLaw (d+1)] := by
  classical
  have _hDim : d ≠ 0 := Nat.one_le_iff_ne_zero.mp hd
  have ha := coefficient_summable d (ne_of_gt hL) T x A a B C D
  rw [variance_congr (observable_eq_series d (ne_of_gt hL) T x A a B C D),
    scalarSeries_variance _ ha]
  have hp := summable_mul_of_summable_norm ha.norm ha.norm
  have hsq : Summable (fun m => (contactCoefficient d L T x A a B C D m)^2) := by
    simpa only [Function.comp_def, pow_two] using hp.comp_injective
      (show Function.Injective (fun m : SourceMode (d+1) => (m,m)) from
        fun _ _ h => congrArg Prod.fst h)
  obtain ⟨m, hm⟩ := coefficient_exists d hL T x A a B C D hw
  exact hsq.tsum_pos (fun _ => sq_nonneg _) m (sq_pos_of_ne_zero hm)

end I4ContactRank

