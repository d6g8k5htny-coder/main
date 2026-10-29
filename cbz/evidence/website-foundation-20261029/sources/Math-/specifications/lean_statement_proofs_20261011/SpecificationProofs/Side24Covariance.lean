import Specifications.Side24
import SpecificationProofs.Side24Analysis

/-!
# SIDE24 reference covariance lower bound: `C_ref - I/3` is positive semidefinite
(statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The one theorem outside the `Aux` namespace has as its type exactly the registered `Prop`
`UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound` of `Specifications/Side24.lean`
(statement packet `specifications/lean_statements_20261010`, which is not modified here). A
kernel-checked proof of that `Prop` is evidence about that `Prop` only, for the formal objects
(`E d`, `Kinf`, `derivContraction`, `jetDirs`, `svecScale`, `jointCov`, `Cref`, and Mathlib's
`Matrix.PosSemidef`, which includes `IsHermitian`): it does not make the informal Layer 0 note
(SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) reviewed, accepted or
"verified", it does not identify the formal definitions with the informal objects (that
alignment is PENDING_INDEPENDENT_REVIEW), it does not show that `Cref` is the covariance of any
random vector (no field is constructed), it says nothing about `C_24`, `CovarianceOrdering` or
`EntrywiseCovarianceBound`, it does not enclose the coefficient `c_{d,24}`, and it promotes,
closes, discharges or reclassifies nothing.

Provenance: the round-5 lane for `ReferenceCovarianceLowerBound`, the `Prop` that the round-4
triage (workflow `wf_a3531fd8-891`) classified PROVABLE_WITH_WORK. The contraction values of
orders two, four and six are the round-4 helpers `Side24Analysis.Aux.rb_order_two`,
`rb_order_four`, `rb_order_six`, imported, not re-proved. Helper lemmas live in
`UniversalLaw.Proofs.Side24Covariance.Aux`, tagged `cv_`; the gate inventories only the theorem
outside that namespace. The mathematics:

* Odd orders vanish at `0` (`cv_iteratedFDeriv_odd`, `cv_odd_zero`): `Kinf d` is even, i.e.
  `Kinf d ∘ (-id) = Kinf d`, so `ContinuousLinearMap.iteratedFDeriv_comp_right` gives
  `D^n Kinf 0 [m] = D^n Kinf 0 [-m_1, …, -m_n] = (-1)^n D^n Kinf 0 [m]`
  (`ContinuousMultilinearMap.map_smul_univ`), hence `0` for odd `n`, in arbitrary directions. No
  general-direction evaluation of order three or five is needed.
* The entries of `Cref d u R` (`cv_GG`, `cv_Gt`, `cv_tG`, `cv_tt`, `cv_GH`, `cv_HG`, `cv_tH`,
  `cv_Ht`, `cv_HH`), read off `jointCov` with the sign `(-1)^{|b|}` and the `svec` scaling: gradient
  block `⟪R i, R j⟫`; `Cov(G_i, t) = Cov(t, G_i) = -3 ⟪u, u⟫ ⟪R i, u⟫`; `Var(t) = 15 ⟪u, u⟫^3`;
  every gradient/Hessian and `t`/Hessian entry is an odd-order contraction, hence `0`; the
  Hessian block is `s_p s_q` times the three pairings of order four.
* In an orthonormal frame with `u = R 0` (`cv_decomp`), entrywise,
  `Cref - (1/3) I = diagonal δ + (2/3) v vᵀ + w wᵀ` with `δ = 0` at `G_0`, `2/3` at `G_i`
  (`i ≠ 0`), `7/6` at `t`, `5/3` at every `svec` coordinate; `v = e_{G_0} - (9/2) e_t`; `w` the
  indicator of the diagonal Hessian coordinates. On the odd block this reads as the certificate
  `(2/3) a^2 - 6 a b + (44/3) b^2 = (2/3) (a - (9/2) b)^2 + (7/6) b^2` (an informal reading of
  the entries, not a separate lemma). The minors `2/3` and `7/9` named in the statement-packet
  docstring are quoted for comparison only (`7/6 = (7/9)/(2/3)`): main's
  `UniversalLaw.Side24.odd_block_eigenvalues`, `odd_block_shifted_minors` and
  `smaller_eigenvalue_exceeds_third` are neither imported, cited nor relied on; the entry
  arithmetic (`15 - 1/3 = 7/6 + (2/3) (9/2)^2`, `-3 = (2/3) (-9/2)`) is discharged inline by
  `simp`/`norm_num` in `cv_decomp`, and no minor, no eigenvalue `8 ± sqrt 58` and no comparison
  of an eigenvalue with `1/3` is computed or stated here. On the Hessian block the decomposition
  is `J + 2 I - I/3` on the diagonal coordinates and `2 - 1/3` on the off-diagonal ones (the `√2`
  scaling squares to `2`).
* Each summand is positive semidefinite (`Matrix.PosSemidef.diagonal`,
  `Matrix.posSemidef_vecMulVec_self_star`, `Matrix.PosSemidef.smul`, `Matrix.PosSemidef.add`), so
  `cv_psd` holds for every `d` with `0 < d`; the inventory theorem specialises it, and the
  hypothesis `d = 2 ∨ d = 3` of the `Prop` is not used by the proof (a remark about the formal
  statement, not a change of it). The symmetry the `Prop` requires (the `IsHermitian` half of
  `PosSemidef`, of `Cref d (R ⟨0, hd⟩) R - (1/3) • 1` for orthonormal `R` and `0 < d`) follows
  from the entrywise decomposition `cv_decomp` into symmetric summands, not from any symmetry of
  `iteratedFDeriv`. Symmetry of `Cref d u R` for general `u` and `R` is not stated here.

Remark recorded by the lane, for the maintainers (nothing below is edited here): the proof-packet
README row "Side24 covariance and density" says, of this `Prop`, that "order 5 needs one more
general-direction level of the peeling" and names `Matrix.posSemidef_iff_dotProduct_mulVec` as the
route; neither is load-bearing here (the evenness argument above replaces the order-five level,
and the decomposition into manifestly positive semidefinite summands replaces the quadratic-form
expansion over the index type). This is a remark about an informal reason, not about any claim;
no register label changes.
-/

namespace UniversalLaw.Proofs.Side24Covariance

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

namespace Aux

/-! ## `cv_`: odd orders of `Kinf d` vanish at the origin -/

/-- `Kinf d` is even: `Kinf d ∘ (-id) = Kinf d` (`norm_neg`). -/
theorem cv_kinf_comp_neg {d : ℕ} :
    (Kinf d ∘ (-ContinuousLinearMap.id ℝ (E d))) = Kinf d := by
  funext x
  simp [Kinf]

/-- Every odd-order iterated derivative of `Kinf d` at `0` vanishes, in arbitrary directions:
`ContinuousLinearMap.iteratedFDeriv_comp_right` with the evenness `cv_kinf_comp_neg` gives
`D^n Kinf 0 [m] = (-1)^n D^n Kinf 0 [m]` (`ContinuousMultilinearMap.map_smul_univ`). -/
theorem cv_iteratedFDeriv_odd {d : ℕ} (n : ℕ) (hn : Odd n) (m : Fin n → E d) :
    iteratedFDeriv ℝ n (Kinf d) 0 m = 0 := by
  have h := (-ContinuousLinearMap.id ℝ (E d)).iteratedFDeriv_comp_right
    (Side24Analysis.Aux.rb_kinf_contDiff (d := d)) (0 : E d) (i := n) (mod_cast le_top)
  rw [cv_kinf_comp_neg] at h
  have h2 := congrArg (fun T => T m) h
  simp only [ContinuousMultilinearMap.compContinuousLinearMap_apply, map_zero] at h2
  have h3 : (fun i => (-ContinuousLinearMap.id ℝ (E d)) (m i)) = fun i => (-1 : ℝ) • m i := by
    funext i
    simp
  rw [h3, ContinuousMultilinearMap.map_smul_univ] at h2
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin, hn.neg_one_pow] at h2
  simp at h2
  linarith

/-- A contraction of `Kinf d` at `0` against a list of odd length vanishes. -/
theorem cv_odd_zero {d : ℕ} (l : List (E d)) (hl : Odd l.length) :
    derivContraction d (Kinf d) 0 l = 0 :=
  cv_iteratedFDeriv_odd _ hl _

/-! ## `cv_`: the entries of `Cref d u R` in a general frame -/

/-- Gradient/gradient entry: `(-1)^1 · (-⟪R i, R j⟫) = ⟪R i, R j⟫`. -/
theorem cv_GG {d : ℕ} (u : E d) (R : Fin d → E d) (i j : Fin d) :
    Cref d u R (Sum.inl i) (Sum.inl j) = ⟪R i, R j⟫ := by
  simp [Cref, jointCov, svecScale, jetDirs, Side24Analysis.Aux.rb_order_two]

/-- Gradient/`t` entry: `(-1)^3 · D^4 Kinf 0 [R i, u, u, u] = -3 ⟪u, u⟫ ⟪R i, u⟫`. -/
theorem cv_Gt {d : ℕ} (u : E d) (R : Fin d → E d) (i : Fin d) (t : Unit) :
    Cref d u R (Sum.inl i) (Sum.inr (Sum.inl t)) = -(3 * (⟪u, u⟫ * ⟪R i, u⟫)) := by
  simp [Cref, jointCov, svecScale, jetDirs, Side24Analysis.Aux.rb_order_four]
  ring

/-- `t`/gradient entry: `(-1)^1 · D^4 Kinf 0 [u, u, u, R j] = -3 ⟪u, u⟫ ⟪u, R j⟫`. -/
theorem cv_tG {d : ℕ} (u : E d) (R : Fin d → E d) (j : Fin d) (t : Unit) :
    Cref d u R (Sum.inr (Sum.inl t)) (Sum.inl j) = -(3 * (⟪u, u⟫ * ⟪u, R j⟫)) := by
  simp [Cref, jointCov, svecScale, jetDirs, Side24Analysis.Aux.rb_order_four]
  ring

/-- `t`/`t` entry: `(-1)^3 · D^6 Kinf 0 [u, …, u] = 15 ⟪u, u⟫^3`. -/
theorem cv_tt {d : ℕ} (u : E d) (R : Fin d → E d) (t t' : Unit) :
    Cref d u R (Sum.inr (Sum.inl t)) (Sum.inr (Sum.inl t')) =
      15 * (⟪u, u⟫ * ⟪u, u⟫ * ⟪u, u⟫) := by
  simp [Cref, jointCov, svecScale, jetDirs, Side24Analysis.Aux.rb_order_six]
  ring

/-- Gradient/Hessian entry: an order-three contraction, hence `0`. -/
theorem cv_GH {d : ℕ} (u : E d) (R : Fin d → E d) (i : Fin d) (q : SymIndex d) :
    Cref d u R (Sum.inl i) (Sum.inr (Sum.inr q)) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [cv_odd_zero _ ⟨1, rfl⟩, mul_zero]

/-- Hessian/gradient entry: an order-three contraction, hence `0`. -/
theorem cv_HG {d : ℕ} (u : E d) (R : Fin d → E d) (p : SymIndex d) (j : Fin d) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inl j) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [cv_odd_zero _ ⟨1, rfl⟩, mul_zero]

/-- `t`/Hessian entry: an order-five contraction, hence `0`. -/
theorem cv_tH {d : ℕ} (u : E d) (R : Fin d → E d) (t : Unit) (q : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inl t)) (Sum.inr (Sum.inr q)) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [cv_odd_zero _ ⟨2, rfl⟩, mul_zero]

/-- Hessian/`t` entry: an order-five contraction, hence `0`. -/
theorem cv_Ht {d : ℕ} (u : E d) (R : Fin d → E d) (p : SymIndex d) (t : Unit) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inr (Sum.inl t)) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [cv_odd_zero _ ⟨2, rfl⟩, mul_zero]

/-- Hessian/Hessian entry: `s_p s_q (-1)^2` times the three pairings of
`D^4 Kinf 0 [R p₁, R p₂, R q₁, R q₂]`. -/
theorem cv_HH {d : ℕ} (u : E d) (R : Fin d → E d) (p q : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inr (Sum.inr q)) =
      svecScale d (Sum.inr (Sum.inr p)) * svecScale d (Sum.inr (Sum.inr q)) *
        (⟪R q.1.1, R q.1.2⟫ * ⟪R p.1.1, R p.1.2⟫ + ⟪R p.1.2, R q.1.2⟫ * ⟪R p.1.1, R q.1.1⟫
          + ⟪R p.1.1, R q.1.2⟫ * ⟪R p.1.2, R q.1.1⟫) := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append,
    Side24Analysis.Aux.rb_order_four]
  simp

/-! ## `cv_`: the decomposition into positive semidefinite summands -/

/-- In an orthonormal frame with `u = R 0`, `Cref - (1/3) I` is, entrywise, a nonnegative
diagonal matrix plus `(2/3) v vᵀ` with `v = e_{G_0} - (9/2) e_t` plus `w wᵀ` with `w` the
indicator of the diagonal Hessian coordinates. -/
theorem cv_decomp {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    Cref d (R ⟨0, hd⟩) R - (1 / 3 : ℝ) • (1 : Matrix (JointIndex d) (JointIndex d) ℝ) =
      Matrix.diagonal (Sum.elim (fun i : Fin d => if i = ⟨0, hd⟩ then (0 : ℝ) else 2 / 3)
          (Sum.elim (fun _ : Unit => (7 / 6 : ℝ)) (fun _ : SymIndex d => (5 / 3 : ℝ))))
        + (2 / 3 : ℝ) • Matrix.vecMulVec
            (Sum.elim (fun i : Fin d => if i = ⟨0, hd⟩ then (1 : ℝ) else 0)
              (Sum.elim (fun _ : Unit => (-9 / 2 : ℝ)) (fun _ : SymIndex d => (0 : ℝ))))
            (Sum.elim (fun i : Fin d => if i = ⟨0, hd⟩ then (1 : ℝ) else 0)
              (Sum.elim (fun _ : Unit => (-9 / 2 : ℝ)) (fun _ : SymIndex d => (0 : ℝ))))
        + Matrix.vecMulVec
            (Sum.elim (fun _ : Fin d => (0 : ℝ))
              (Sum.elim (fun _ : Unit => (0 : ℝ))
                (fun p : SymIndex d => if p.1.1 = p.1.2 then (1 : ℝ) else 0)))
            (Sum.elim (fun _ : Fin d => (0 : ℝ))
              (Sum.elim (fun _ : Unit => (0 : ℝ))
                (fun p : SymIndex d => if p.1.1 = p.1.2 then (1 : ℝ) else 0))) := by
  have hite := orthonormal_iff_ite.mp hR
  ext a b
  rcases a with i | t | p <;> rcases b with j | t' | q
  · simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply,
      Matrix.diagonal_apply, Matrix.vecMulVec_apply, cv_GG, hite, Sum.inl.injEq, Sum.elim_inl,
      smul_eq_mul]
    split_ifs <;> simp_all <;> norm_num
  · rw [Matrix.sub_apply, cv_Gt, hite, hite]
    simp [Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply]
    split_ifs <;> norm_num
  · simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply, cv_GH]
  · rw [Matrix.sub_apply, cv_tG, hite, hite]
    simp [Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply]
    by_cases hj : j = ⟨0, hd⟩
    · subst hj
      norm_num
    · simp [hj, Ne.symm hj]
  · rw [Matrix.sub_apply, cv_tt, hite]
    simp [Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply]
    norm_num
  · simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply, cv_tH]
  · simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply, cv_HG]
  · simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply, cv_Ht]
  · obtain ⟨⟨i, j⟩, hij⟩ := p
    obtain ⟨⟨k, l⟩, hkl⟩ := q
    simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply,
      Matrix.diagonal_apply, Matrix.vecMulVec_apply, cv_HH, hite, svecScale]
    rw [Fin.le_def] at hij hkl
    split_ifs <;> simp_all [Fin.ext_iff] <;> first | omega | norm_num

/-- `Cref - (1/3) I` is positive semidefinite in every orthonormal frame with `u = R 0`, for every
`d > 0`: the three summands of `cv_decomp` are positive semidefinite. -/
theorem cv_psd {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    (Cref d (R ⟨0, hd⟩) R -
      (1 / 3 : ℝ) • (1 : Matrix (JointIndex d) (JointIndex d) ℝ)).PosSemidef := by
  rw [cv_decomp hd R hR]
  refine Matrix.PosSemidef.add (Matrix.PosSemidef.add ?_ ?_) ?_
  · refine Matrix.PosSemidef.diagonal fun a => ?_
    rcases a with i | t | p
    · simp only [Sum.elim_inl, Pi.zero_apply]
      split_ifs <;> norm_num
    · simp only [Sum.elim_inr, Sum.elim_inl, Pi.zero_apply]
      norm_num
    · simp only [Sum.elim_inr, Pi.zero_apply]
      norm_num
  · refine Matrix.PosSemidef.smul ?_ (by norm_num)
    simpa using Matrix.posSemidef_vecMulVec_self_star
      (Sum.elim (fun i : Fin d => if i = ⟨0, hd⟩ then (1 : ℝ) else 0)
        (Sum.elim (fun _ : Unit => (-9 / 2 : ℝ)) (fun _ : SymIndex d => (0 : ℝ))))
  · simpa using Matrix.posSemidef_vecMulVec_self_star
      (Sum.elim (fun _ : Fin d => (0 : ℝ))
        (Sum.elim (fun _ : Unit => (0 : ℝ))
          (fun p : SymIndex d => if p.1.1 = p.1.2 then (1 : ℝ) else 0)))

end Aux

/-- Proves: `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound` (claim
`side24-coefficient`), for the formal `Cref` of `Specifications.Side24`.
Method: odd-order contractions of the even kernel `Kinf d` vanish at `0` (`Aux.cv_odd_zero`,
via `ContinuousLinearMap.iteratedFDeriv_comp_right` and `ContinuousMultilinearMap.map_smul_univ`);
the orders two, four and six are the round-4 helpers `Side24Analysis.Aux.rb_order_two`,
`rb_order_four`, `rb_order_six`; in an orthonormal frame with `u = R 0`,
`Cref - I/3 = diagonal (0, 2/3, …, 2/3; 7/6; 5/3, …, 5/3) + (2/3) v vᵀ + w wᵀ` entrywise
(`Aux.cv_decomp`, `v = e_{G_0} - (9/2) e_t`, `w` the indicator of the diagonal Hessian
coordinates), and each summand is positive semidefinite (`Aux.cv_psd`, which holds for every
`d > 0`; the hypothesis `d = 2 ∨ d = 3` of the `Prop` is not used by the proof).
Does not establish: the informal Section 3 display "C_ref >= I/3" of the source as the source's
object (the alignment of `E d`, `Kinf`, `derivContraction`, `jetDirs`, `svecScale`, `jointCov` and
`Cref` with the source's `R^d`, `K_infty`, jet and joint covariance, and of Mathlib's
`Matrix.PosSemidef` (which includes `IsHermitian`) with the source's `>=` order, is
PENDING_INDEPENDENT_REVIEW); that `Cref` is the covariance of any random vector (no field is
constructed); symmetry of `Cref d u R` for general `u` or a general frame (only the `IsHermitian`
half of the `Prop`'s `PosSemidef`, at `u = R 0` in an orthonormal frame with `0 < d`, is proved);
the statement module's own `Does not claim:` text ("the bound (statement only); the odd-block
eigenvalues `8 ± sqrt 58` and the minors `2/3`, `7/9` are kernel-checked on main
(`UniversalLaw.Side24.odd_block_eigenvalues`, `odd_block_shifted_minors`,
`smaller_eigenvalue_exceeds_third`), not here"), which describes the statement packet and is not
edited here; main's `UniversalLaw.Side24.odd_block_eigenvalues`, `odd_block_shifted_minors` and
`smaller_eigenvalue_exceeds_third` are neither imported, cited nor relied on (the certificate's
entry arithmetic is discharged inline by `simp`/`norm_num` in `Aux.cv_decomp`), and no minor
(in particular not `7/9`), no eigenvalue `8 ± sqrt 58` and no comparison of the smaller
eigenvalue with `1/3` is computed or stated here; anything about `C_24`, `CovarianceOrdering` or
`EntrywiseCovarianceBound`; any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here;
nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. It promotes, closes,
discharges or reclassifies nothing. -/
theorem referenceCovarianceLowerBound : UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound := by
  intro d _ hd R hR
  exact Aux.cv_psd hd R hR

end UniversalLaw.Proofs.Side24Covariance
