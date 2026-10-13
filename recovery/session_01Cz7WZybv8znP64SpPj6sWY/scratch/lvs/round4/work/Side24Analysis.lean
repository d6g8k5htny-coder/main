import Specifications.Side24
import SpecificationProofs.Side24Elementary

/-!
# SIDE24 analysis proofs: the theta family, the Section 2 derivative bounds and lattice sum,
the Section 1 reference blocks and pin density, the Section 4 ratio and density comparison
(statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1, register
row `side24-coefficient`) reviewed, accepted or "verified", it does not identify the formal
definitions with the informal objects (that alignment is PENDING_INDEPENDENT_REVIEW), it does
not enclose the coefficient `c_{d,24}`, and it promotes, closes, discharges or reclassifies
nothing. The disposition of `side24-coefficient` (REVIEWED_SCOPED) is transcribed in the
register, not derived here.

Provenance: the nine `Prop`s proved here were selected by the round-4 triage (workflow
`wf_a3531fd8-891`) of the registered `Prop`s still without a theorem; each proof was written and
verified in its own lane (one file per `Prop`) and the lane files were merged into this module,
with the developments the lanes reproduced from one another removed, call sites repointed to the
sibling inventory theorems, and the lanes' helper tags kept. Helper lemmas live in
`UniversalLaw.Proofs.Side24Analysis.Aux`, tagged by the `Prop` they serve (`th_`, `dp_`, `d1_`,
`d0_`, `rb_`, `rp_`, `ir_`, `li_`, `dc_`); the gate inventories only the nine theorems outside
that namespace. The mathematics, in the order of the module:

* `ThetaSummable d` (`th_`): `‖24 n‖^2 = 576 ∑ n_i^2`, the shift bound
  `exp(-‖z + v‖^2/2) ≤ exp(‖z‖^2/2) exp(-‖v‖^2/4)`, `Real.exp_sum`, the one-dimensional
  summability of `m ↦ exp(-c m^2)` on `ℤ` (`HurwitzKernelBounds.summable_f_int`) and a
  `Fintype.piFinset` product bound for `n ↦ ∏ i, g (n i)` over `Fin d → ℤ`;
  `Summable.of_nonneg_of_le` assembles the `Prop` for every `d`.
* `DerivativePairingBound` (`dp_`): no closed form for the iterated Fréchet derivative of the
  Gaussian is used; the operator norms `b_q = ‖iteratedFDeriv ℝ q (Kinf d) x‖` satisfy the
  Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` (`norm_iteratedFDeriv_fderiv`,
  `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`, the Leibniz bound
  `norm_iteratedFDeriv_smul_le`), `b_0 = Kinf d x`, `b_1 = Kinf d x ‖x‖`, and the orders `q ≤ 6`
  match the unfolded `pairingSum`; `ContinuousMultilinearMap.le_opNorm` at a unit tuple.
* `DerivativeBoundAtLeastOne` (`d1_`) and `DerivativeBoundAtZero` (`d0_`): consequences of the
  pairing bound through hypothesis-form helpers, with `pairingSum q r ≤ 76 r^6` for `r ≥ 1` and
  `pairingSum q 0 ≤ 15`.
* `ReferenceOddAndPinBlocks` (`rb_`): the contractions of `Kinf d` at `0` are evaluated by
  peeling one direction at a time (`iteratedFDeriv_succ_apply_right`,
  `iteratedFDeriv_clm_apply_const_apply`) with explicit `HasFDerivAt` closed forms of the
  intermediate functions, giving `-⟪a, b⟫`, the three pairings at order four and `-15 ⟪u, u⟫^3`
  at order six; in an orthonormal frame the five blocks follow.
* `ReferencePinDensityProduct` (`rp_`): the two block identities of the previous `Prop` and the
  density algebra `(2π)^(-d/2) (2π)^(-d/2) 3^(-1/2) = (2π)^(-d) / √3`.
* `IntegrandRatioBounds` (`ir_`): pure real arithmetic in `epsilon = 10^(-108)`; Bernoulli's
  inequality, exponent monotonicity of the real power and two exact rational inequalities of
  degree five decided by `norm_num`.
* `LatticeImageSumBound` (`li_`): a uniform bound on the finite partial sums via a power-mean
  inequality, `Real.exp_sum`, the `Fintype.piFinset` factorisation and one-dimensional sums over
  `ℤ` compared with the geometric series through the registered `SuccessiveTermRatio`
  (`SpecificationProofs.Side24Elementary.successiveTermRatio`, whence the import); the bound
  obtained is `432 exp(-288) ≤ 1458 exp(-288)`.
* `DensityComparison` (`dc_`): real matrix analysis; Loewner antitonicity of the inverse and
  monotonicity of the determinant (through `CFC.sqrt`, `Matrix.PosSemidef.conjTranspose_mul_mul_same`,
  `Matrix.IsHermitian.det_eq_prod_eigenvalues`), determinant scaling and real-power bookkeeping.

Remarks recorded by the lanes, for the maintainers (nothing below is edited here): the
proof-packet README rows "Side24 derivative bounds" ("Mathlib has no closed form for the iterated
Fréchet derivatives of a Gaussian"), "Side24 lattice sums" ("Mathlib's Jacobi-theta results are
one-dimensional") and "Side24 covariance and density" ("the evaluation of the Fréchet derivatives
is not available"; `IntegrandRatioBounds` grouped as "built on the derivative contractions") give
informal reasons that these proofs show are not load-bearing for the `Prop`s proved here; the
statement module's docstring for `ThetaSummable` reads "Does not claim: that the family is summable
(true, but not shown here)". Each is a remark about an informal reason or a statement-packet text,
not about any claim; no register label changes.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped ContDiff
open scoped RealInnerProductSpace
open scoped MatrixOrder
open scoped Pointwise

namespace Aux

/-! ## `th_`: the theta family (`ThetaSummable`) -/

/-! ### (a) Coordinates and squared norm of the lattice vector `24 n` -/

/-- The `i`-th coordinate of `latticeVec d n` is `24 n_i`. -/
theorem th_latticeVec_apply (d : ℕ) (n : Fin d → ℤ) (i : Fin d) :
    latticeVec d n i = 24 * (n i : ℝ) := by
  simp [latticeVec]

/-- `‖24 n‖^2 = 576 * ∑ i, n_i^2` (`EuclideanSpace.norm_sq_eq`, coordinatewise). -/
theorem th_norm_latticeVec_sq (d : ℕ) (n : Fin d → ℤ) :
    ‖latticeVec d n‖ ^ 2 = 576 * intNormSq d n := by
  rw [EuclideanSpace.norm_sq_eq, intNormSq, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [th_latticeVec_apply, Real.norm_eq_abs, sq_abs]
  ring

/-- `exp(-‖24 n‖^2/4) = ∏ i, exp(-144 n_i^2)`, by `Real.exp_sum` and the previous lemma. -/
theorem th_exp_neg_norm_latticeVec_sq (d : ℕ) (n : Fin d → ℤ) :
    Real.exp (-(‖latticeVec d n‖ ^ 2) / 4) = ∏ i, Real.exp (-144 * ((n i : ℤ) : ℝ) ^ 2) := by
  rw [← Real.exp_sum, th_norm_latticeVec_sq, intNormSq, Finset.mul_sum, neg_div,
    Finset.sum_div, ← Finset.sum_neg_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-! ### (b) One-dimensional Gaussian summability on `ℤ` -/

/-- `m ↦ exp(-c m^2)` is summable on `ℤ` for `c > 0`: Mathlib's
`HurwitzKernelBounds.summable_f_int 0 0` at `t = c/π` after `-π m^2 (c/π) = -c m^2`. -/
theorem th_summable_gauss_int (c : ℝ) (hc : 0 < c) :
    Summable (fun m : ℤ => Real.exp (-c * (m : ℝ) ^ 2)) := by
  have h := HurwitzKernelBounds.summable_f_int 0 0 (t := c / Real.pi) (by positivity)
  refine h.congr fun m => ?_
  simp only [HurwitzKernelBounds.f_int, add_zero, pow_zero, one_mul]
  congr 1
  field_simp

/-! ### (c) Products of a summable nonnegative function over `Fin d → ℤ` -/

/-- For a nonnegative summable `g : ℤ → ℝ`, the product `n ↦ ∏ i, g (n i)` is summable over
`Fin d → ℤ`: a finite set `s` of lattice points lies in the box `Fintype.piFinset t` with
`t i = s.image (· i)`, over which the sum factors as `∏ i, ∑ m ∈ t i, g m ≤ (∑' m, g m)^d`. -/
theorem th_summable_pi_prod (d : ℕ) (g : ℤ → ℝ) (hg0 : ∀ m, 0 ≤ g m) (hg : Summable g) :
    Summable (fun n : Fin d → ℤ => ∏ i, g (n i)) := by
  refine summable_of_sum_le (c := (∑' m, g m) ^ d)
    (fun n => Finset.prod_nonneg fun i _ => hg0 _) fun s => ?_
  set t : Fin d → Finset ℤ := fun i => s.image (fun n => n i)
  have hsub : s ⊆ Fintype.piFinset t := fun n hn => by
    rw [Fintype.mem_piFinset]
    intro i
    exact Finset.mem_image_of_mem _ hn
  have hle : ∀ i, ∑ m ∈ t i, g m ≤ ∑' m, g m := fun i =>
    hg.sum_le_tsum (t i) (fun m _ => hg0 m)
  calc ∑ n ∈ s, ∏ i, g (n i) ≤ ∑ n ∈ Fintype.piFinset t, ∏ i, g (n i) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun n _ _ => Finset.prod_nonneg fun i _ => hg0 _
    _ = ∏ i, ∑ m ∈ t i, g m := (Finset.prod_univ_sum t fun _ m => g m).symm
    _ ≤ ∏ _i : Fin d, ∑' m, g m :=
        Finset.prod_le_prod₀ (fun i _ => Finset.sum_nonneg fun m _ => hg0 m) (fun i _ => hle i)
    _ = (∑' m, g m) ^ d := by simp

/-! ### (d) The pointwise shift bound -/

/-- `exp(-‖z + v‖^2/2) ≤ exp(‖z‖^2/2) exp(-‖v‖^2/4)`: from `‖v‖ ≤ ‖z + v‖ + ‖z‖` one has
`‖v‖^2/4 ≤ (‖z + v‖ + ‖z‖)^2/4 ≤ ‖z + v‖^2/2 + ‖z‖^2/2`. -/
theorem th_exp_shift_le (d : ℕ) (z v : E d) :
    Real.exp (-(‖z + v‖ ^ 2) / 2) ≤ Real.exp (‖z‖ ^ 2 / 2) * Real.exp (-(‖v‖ ^ 2) / 4) := by
  rw [← Real.exp_add, Real.exp_le_exp]
  have h1 : ‖v‖ ≤ ‖z + v‖ + ‖z‖ := by
    calc ‖v‖ = ‖(z + v) - z‖ := by rw [add_sub_cancel_left]
      _ ≤ ‖z + v‖ + ‖z‖ := norm_sub_le _ _
  have h2 : 0 ≤ ‖z + v‖ := norm_nonneg _
  have h3 : 0 ≤ ‖z‖ := norm_nonneg _
  have h4 : 0 ≤ ‖v‖ := norm_nonneg _
  nlinarith [sq_nonneg (‖z + v‖ - ‖z‖)]

/-- The dominating family `n ↦ exp(‖z‖^2/2) ∏ i, exp(-144 n_i^2)` is summable over
`Fin d → ℤ`. -/
theorem th_summable_dominant (d : ℕ) (z : E d) :
    Summable (fun n : Fin d → ℤ =>
      Real.exp (‖z‖ ^ 2 / 2) * ∏ i, Real.exp (-144 * ((n i : ℤ) : ℝ) ^ 2)) :=
  (th_summable_pi_prod d _ (fun m => (Real.exp_pos _).le)
    (th_summable_gauss_int 144 (by norm_num))).mul_left _

/-- Each theta term is bounded by the corresponding dominating term. -/
theorem th_term_le_dominant (d : ℕ) (z : E d) (n : Fin d → ℤ) :
    Real.exp (-(‖z + latticeVec d n‖ ^ 2) / 2) ≤
      Real.exp (‖z‖ ^ 2 / 2) * ∏ i, Real.exp (-144 * ((n i : ℤ) : ℝ) ^ 2) := by
  calc Real.exp (-(‖z + latticeVec d n‖ ^ 2) / 2)
      ≤ Real.exp (‖z‖ ^ 2 / 2) * Real.exp (-(‖latticeVec d n‖ ^ 2) / 4) := th_exp_shift_le d z _
    _ = Real.exp (‖z‖ ^ 2 / 2) * ∏ i, Real.exp (-144 * ((n i : ℤ) : ℝ) ^ 2) := by
      rw [th_exp_neg_norm_latticeVec_sq]

/-! ## `dp_`: the derivative pairing bound (`DerivativePairingBound`) -/

/-! ### Smoothness and first derivative of the reference kernel `Kinf` -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` on `E d`. -/
theorem dp_contDiff_Kinf (d : ℕ) : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq ℝ (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

/-- The first derivative of `Kinf d` at `x` is `Kinf d x • (-(innerSL ℝ x))`, i.e.
`v ↦ -⟪x, v⟫ Kinf d x`. -/
theorem dp_hasFDerivAt_Kinf (d : ℕ) (x : E d) :
    HasFDerivAt (Kinf d) (Kinf d x • (-(innerSL ℝ x))) x := by
  have h0 : HasFDerivAt (fun y : E d => ‖y‖ ^ 2) ((2 : ℕ) • innerSL ℝ x) x :=
    (hasStrictFDerivAt_norm_sq x).hasFDerivAt
  have h1 := (h0.const_smul (-(1 / 2 : ℝ))).exp
  convert h1 using 1
  · funext y
    simp only [Kinf, Pi.smul_apply, smul_eq_mul]
    ring_nf
  · ext v
    simp [Kinf]
    left
    ring

/-- The derivative function of `Kinf d`. -/
theorem dp_fderiv_Kinf (d : ℕ) :
    fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (dp_hasFDerivAt_Kinf d y).fderiv

/-! ### The operator-valued factor `g y = -(innerSL ℝ y)` -/

/-- `y ↦ -(innerSL ℝ y)` is `C^∞` (it is a continuous linear map). -/
theorem dp_contDiff_g (d : ℕ) : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) :=
  (ContinuousLinearMap.contDiff (innerSL ℝ)).neg

/-- The factor as the coercion of the continuous linear map `-(innerSL ℝ)`. -/
theorem dp_g_eq (d : ℕ) :
    (fun y : E d => -(innerSL ℝ y)) = ⇑(-(innerSL ℝ (E := E d))) := by
  funext y
  simp

/-- Order zero: `‖g x‖ = ‖x‖` (`innerSL_apply_norm`). -/
theorem dp_norm_g_zero (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

/-- Order one: `‖D g x‖ = ‖innerSL ℝ‖ ≤ 1` (`norm_innerSL_le`). -/
theorem dp_norm_g_one (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 1 (fun y : E d => -(innerSL ℝ y)) x‖ ≤ 1 := by
  rw [norm_iteratedFDeriv_one, dp_g_eq, ContinuousLinearMap.fderiv, norm_neg]
  exact norm_innerSL_le ℝ

/-- Orders `k ≥ 2` of the linear factor vanish. -/
theorem dp_g_higher (d : ℕ) (x : E d) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k (fun y : E d => -(innerSL ℝ y)) x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [dp_g_eq, iteratedFDeriv_succ_eq_comp_right]
  have hconst : fderiv ℝ ⇑(-(innerSL ℝ (E := E d))) = fun _ => -(innerSL ℝ (E := E d)) :=
    funext fun y => (-(innerSL ℝ (E := E d))).fderiv
  simp only [Function.comp_apply, hconst, iteratedFDeriv_const_of_ne (show j ≠ 0 by omega),
    Pi.zero_apply]
  exact LinearIsometryEquiv.map_zero _

/-! ### The Hermite-type recursion for the operator norms -/

/-- Hermite-type recursion for `b_q = ‖D^q Kinf (x)‖`:
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, from `norm_iteratedFDeriv_fderiv`, the explicit first
derivative, and the Leibniz bound `norm_iteratedFDeriv_smul_le` in which only the terms
`i = q` and `i = q + 1` of the binomial sum survive. -/
theorem dp_recursion (d : ℕ) (x : E d) (q : ℕ) :
    ‖iteratedFDeriv ℝ (q + 2) (Kinf d) x‖ ≤
      ‖x‖ * ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ +
        (q + 1) * ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, dp_fderiv_Kinf]
  have h := norm_iteratedFDeriv_smul_le (dp_contDiff_Kinf d) (dp_contDiff_g d) x
    (n := q + 1) (WithTop.coe_le_coe.mpr le_top)
  refine h.trans ?_
  rw [Finset.sum_range_succ, Finset.sum_range_succ]
  have hz : ∑ i ∈ Finset.range q, ((q + 1).choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
      ‖iteratedFDeriv ℝ (q + 1 - i) (fun y : E d => -(innerSL ℝ y)) x‖ = 0 := by
    refine Finset.sum_eq_zero fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [dp_g_higher d x _ (by omega), norm_zero, mul_zero]
  rw [hz, zero_add, Nat.choose_self, Nat.choose_succ_self_right, Nat.add_sub_cancel_left,
    Nat.sub_self, dp_norm_g_zero]
  have h1 := dp_norm_g_one d x
  have hb : 0 ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := norm_nonneg _
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h1 (mul_nonneg (by positivity : (0 : ℝ) ≤ (q : ℝ) + 1) hb)]

/-! ### The seven explicit orders -/

/-- `Kinf d x > 0`. -/
theorem dp_Kinf_pos (d : ℕ) (x : E d) : 0 < Kinf d x := Real.exp_pos _

/-- `b_0 = Kinf d x`. -/
theorem dp_b0 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 0 (Kinf d) x‖ = Kinf d x := by
  rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs, abs_of_pos (dp_Kinf_pos d x)]

/-- `b_1 = Kinf d x * ‖x‖`. -/
theorem dp_b1 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 1 (Kinf d) x‖ = Kinf d x * ‖x‖ := by
  rw [norm_iteratedFDeriv_one, dp_fderiv_Kinf, norm_smul, norm_neg, innerSL_apply_norm,
    Real.norm_eq_abs, abs_of_pos (dp_Kinf_pos d x)]

/-- `b_2 ≤ Kinf d x * (‖x‖^2 + 1)`. -/
theorem dp_b2 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 2 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 2 + 1) := by
  have h := dp_recursion d x 0
  rw [dp_b1, dp_b0] at h
  push_cast at h
  nlinarith [h]

/-- `b_3 ≤ Kinf d x * (‖x‖^3 + 3 ‖x‖)`. -/
theorem dp_b3 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 3 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 3 + 3 * ‖x‖) := by
  have h := dp_recursion d x 1
  have h2 := dp_b2 d x
  have h1 := dp_b1 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h2 hr]

/-- `b_4 ≤ Kinf d x * (‖x‖^4 + 6 ‖x‖^2 + 3)`. -/
theorem dp_b4 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 4 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 4 + 6 * ‖x‖ ^ 2 + 3) := by
  have h := dp_recursion d x 2
  have h3 := dp_b3 d x
  have h2 := dp_b2 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h3 hr]

/-- `b_5 ≤ Kinf d x * (‖x‖^5 + 10 ‖x‖^3 + 15 ‖x‖)`. -/
theorem dp_b5 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 5 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 5 + 10 * ‖x‖ ^ 3 + 15 * ‖x‖) := by
  have h := dp_recursion d x 3
  have h4 := dp_b4 d x
  have h3 := dp_b3 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h4 hr]

/-- `b_6 ≤ Kinf d x * (‖x‖^6 + 15 ‖x‖^4 + 45 ‖x‖^2 + 15)`. -/
theorem dp_b6 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 6 (Kinf d) x‖ ≤
      Kinf d x * (‖x‖ ^ 6 + 15 * ‖x‖ ^ 4 + 45 * ‖x‖ ^ 2 + 15) := by
  have h := dp_recursion d x 4
  have h5 := dp_b5 d x
  have h4 := dp_b4 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h5 hr]

/-- For `q ≤ 6`, `b_q ≤ Kinf d x * pairingSum q ‖x‖`: the explicit bounds `dp_b0` to `dp_b6`
match the unfolded pairing sums (`Finset.sum_range_succ`, `Nat.factorial`). -/
theorem dp_pairing_all (d : ℕ) (x : E d) (q : ℕ) (hq : q ≤ 6) :
    ‖iteratedFDeriv ℝ q (Kinf d) x‖ ≤ Kinf d x * pairingSum q ‖x‖ := by
  interval_cases q
  · rw [dp_b0]
    simp [pairingSum]
  · rw [dp_b1]
    simp [pairingSum]
  · refine (dp_b2 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b3 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b4 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b5 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b6 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring

/-! ### Contraction at a unit tuple -/

/-- At a unit tuple the contraction is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm` with `∏ ‖u i‖ = 1`). -/
theorem dp_abs_le_opNorm (d q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

/-! ## `d1_`: the derivative bound at `|x| ≥ 1` (`DerivativeBoundAtLeastOne`) -/

/-! ### The elementary pairing-sum bound at `r ≥ 1` -/

/-- For `q ≤ 6` and `1 ≤ r`, `pairingSum q r ≤ 76 r^6`. Each pairing sum unfolds
(`Finset.sum_range_succ`, `Nat.factorial`) to a polynomial with nonnegative coefficients summing
to `1, 1, 2, 4, 10, 26, 76` for `q = 0, …, 6`; every monomial `r^k`, `k ≤ 6`, is at most `r^6`
(`pow_le_pow_right₀`, `one_le_pow₀`). -/
theorem d1_pairingSum_le_of_one_le (q : ℕ) (hq : q ≤ 6) (r : ℝ) (hr : 1 ≤ r) :
    pairingSum q r ≤ 76 * r ^ 6 := by
  have h0 : 1 ≤ r ^ 6 := one_le_pow₀ hr
  have h1 : r ≤ r ^ 6 := by
    simpa using pow_le_pow_right₀ hr (show 1 ≤ 6 by norm_num)
  have h2 : r ^ 2 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h3 : r ^ 3 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h4 : r ^ 4 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h5 : r ^ 5 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  interval_cases q <;> simp only [pairingSum, Finset.sum_range_succ, Finset.sum_range_zero,
    Nat.factorial] <;> norm_num <;> nlinarith [h0, h1, h2, h3, h4, h5, hr]

/-! ### Hypothesis form: the Prop from `DerivativePairingBound` -/

/-- `DerivativeBoundAtLeastOne` from `DerivativePairingBound`: at `1 ≤ ‖x‖` the pairing bound
`|D^q Kinf (x) u| ≤ exp(-‖x‖²/2) · pairingSum q ‖x‖` and `pairingSum q ‖x‖ ≤ 76 ‖x‖^6`
(`d1_pairingSum_le_of_one_le`) give the Prop by `mul_le_mul_of_nonneg_left` with
`0 ≤ exp(-‖x‖²/2)` (`Real.exp_pos`). -/
theorem d1_derivativeBoundAtLeastOne_of_derivativePairingBound
    (h : UniversalLaw.Spec.Side24.DerivativePairingBound) :
    UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne := by
  intro d q hq x hx u hu
  have hpair : |iteratedFDeriv ℝ q (Kinf d) x u| ≤
      Real.exp (-(‖x‖ ^ 2) / 2) * pairingSum q ‖x‖ := h d q hq x u hu
  have hps : pairingSum q ‖x‖ ≤ 76 * ‖x‖ ^ 6 := d1_pairingSum_le_of_one_le q hq ‖x‖ hx
  have hexp : 0 ≤ Real.exp (-(‖x‖ ^ 2) / 2) := (Real.exp_pos _).le
  calc |iteratedFDeriv ℝ q (Kinf d) x u|
      ≤ Real.exp (-(‖x‖ ^ 2) / 2) * pairingSum q ‖x‖ := hpair
    _ ≤ Real.exp (-(‖x‖ ^ 2) / 2) * (76 * ‖x‖ ^ 6) := mul_le_mul_of_nonneg_left hps hexp
    _ = 76 * ‖x‖ ^ 6 * Real.exp (-(‖x‖ ^ 2) / 2) := by ring

/-! ## `d0_`: the derivative bound at the origin (`DerivativeBoundAtZero`) -/

/-! ### The pairing sums at the origin -/

/-- At `r = 0` the pairing sums of order `q ≤ 6` are `1, 0, 1, 0, 3, 0, 15` (only the `j = q/2`
term of an even order survives, `0 ^ 0 = 1`), hence at most `15`. -/
theorem d0_pairingSum_zero_le (q : ℕ) (hq : q ≤ 6) : pairingSum q 0 ≤ 15 := by
  interval_cases q <;> simp [pairingSum, Finset.sum_range_succ, Nat.factorial] <;> norm_num

/-! ### The hypothesis form: the bound at the origin from the pairing bound -/

/-- `DerivativeBoundAtZero` from `DerivativePairingBound` taken as a hypothesis: specialise the
pairing bound to `x = 0`, where `‖0‖ = 0` and `exp (-(0 ^ 2) / 2) = 1`, and bound
`pairingSum q 0 ≤ 15` by `d0_pairingSum_zero_le`. -/
theorem d0_derivativeBoundAtZero_of_derivativePairingBound
    (h : UniversalLaw.Spec.Side24.DerivativePairingBound) :
    UniversalLaw.Spec.Side24.DerivativeBoundAtZero := by
  intro d q hq u hu
  have h0 := h d q hq 0 u hu
  have hx : Real.exp (-(‖(0 : E d)‖ ^ 2) / 2) * pairingSum q ‖(0 : E d)‖ = pairingSum q 0 := by
    simp
  rw [hx] at h0
  exact h0.trans (d0_pairingSum_zero_le q hq)

/-! ## `rb_`: the Section 1 reference blocks (`ReferenceOddAndPinBlocks`) -/

/-! ### (a) Smoothness and first derivative of the reference kernel -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` (`contDiff_norm_sq`, `ContDiff.exp`). -/
theorem rb_kinf_contDiff {d : ℕ} : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact ContDiff.exp (ContDiff.div_const (ContDiff.neg (contDiff_norm_sq ℝ)) 2)

/-- The Fréchet derivative of `Kinf d` at `y`, as `HasFDerivAt` (chain rule through
`hasStrictFDerivAt_norm_sq` and `HasFDerivAt.exp`). -/
theorem rb_hasFDerivAt_kinf {d : ℕ} (y : E d) :
    HasFDerivAt (Kinf d)
      (Real.exp (-(‖y‖ ^ 2) / 2) • ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y))) y := by
  have h1 : HasFDerivAt (fun z : E d => -(‖z‖ ^ 2) / 2)
      ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y)) y := by
    have := (hasStrictFDerivAt_norm_sq y).hasFDerivAt.neg.mul_const (2 : ℝ)⁻¹
    refine this.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => ?_)
    simp [div_eq_mul_inv]
  exact h1.exp

/-- `fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y)`. -/
theorem rb_fderiv_kinf {d : ℕ} (y v : E d) :
    fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y) := by
  rw [(rb_hasFDerivAt_kinf y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply]
  ring

/-! ### (b) Peeling one direction off an iterated derivative -/

/-- For a `C^∞` real function `f`, `D^{n+1} f x [m] = D^n (y ↦ D f y (m (Fin.last n))) x [Fin.init m]`
(`iteratedFDeriv_succ_apply_right` with `iteratedFDeriv_clm_apply_const_apply`). -/
theorem rb_peel {d : ℕ} (f : E d → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (n : ℕ) (x : E d)
    (m : Fin (n + 1) → E d) :
    iteratedFDeriv ℝ (n + 1) f x m =
      iteratedFDeriv ℝ n (fun y => fderiv ℝ f y (m (Fin.last n))) x (Fin.init m) := by
  rw [iteratedFDeriv_succ_apply_right]
  rw [iteratedFDeriv_clm_apply_const_apply (contDiff_infty_iff_fderiv.mp hf).2
    (mod_cast le_top)]

/-- `y ↦ ⟪y, v⟫` is `C^∞`. -/
theorem rb_inner_contDiff {d : ℕ} (v : E d) : ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => ⟪y, v⟫) :=
  contDiff_id.inner ℝ contDiff_const

/-! ### (c) Level one: `g₁ y = -(⟪y, v⟫ Kinf d y)` -/

/-- Smoothness of the level-one function. -/
theorem rb_g1_contDiff {d : ℕ} (v : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪y, v⟫ * Kinf d y)) :=
  ((rb_inner_contDiff v).mul rb_kinf_contDiff).neg

/-- Derivative of the level-one function in direction `w`. -/
theorem rb_fderiv_g1 {d : ℕ} (v y w : E d) :
    fderiv ℝ (fun y : E d => -(⟪y, v⟫ * Kinf d y)) y w =
      -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y := by
  have h : HasFDerivAt (fun y : E d => -(⟪y, v⟫ * Kinf d y)) _ y :=
    (((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul (rb_hasFDerivAt_kinf y)).neg
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! ### (d) Level two: `g₂ y = -(⟪w, v⟫ Kinf d y) + ⟪y, v⟫ ⟪y, w⟫ Kinf d y` -/

/-- Smoothness of the level-two function. -/
theorem rb_g2_contDiff {d : ℕ} (v w : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) :=
  ((contDiff_const.mul rb_kinf_contDiff).neg.add
    (((rb_inner_contDiff v).mul (rb_inner_contDiff w)).mul rb_kinf_contDiff))

/-- Derivative of the level-two function in direction `a`. -/
theorem rb_fderiv_g2 {d : ℕ} (v w y a : E d) :
    fderiv ℝ (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) y a =
      ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
        - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y := by
  have h : HasFDerivAt
      (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) _ y :=
    (((rb_hasFDerivAt_kinf y).const_mul _).neg).add
      ((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul
        ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y))).mul (rb_hasFDerivAt_kinf y))
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! ### (e) Level three at the origin, general directions -/

/-- Derivative at `0` of the level-three function in direction `c`: the three pairings. -/
theorem rb_fderiv_g3_zero {d : ℕ} (v w a c : E d) :
    fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  have h : HasFDerivAt (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) _ 0 :=
    (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0)).const_mul _).mul
        (rb_hasFDerivAt_kinf 0)).add
      ((((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0)).const_mul _).add
          (((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul_const _))).mul
        (rb_hasFDerivAt_kinf 0))).sub
      (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0))).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0))).mul
        (rb_hasFDerivAt_kinf 0))
  rw [h.fderiv]
  simp [Kinf, fderivInnerCLM_apply]
  ring

/-- Fourth-order contraction of `Kinf d` at `0`: `D^4 Kinf 0 [c, a, w, v]` is the sum of the
three pairings `⟪w, v⟫⟪c, a⟫ + ⟪a, v⟫⟪c, w⟫ + ⟪c, v⟫⟪a, w⟫`. -/
theorem rb_order_four {d : ℕ} (c a w v : E d) :
    derivContraction d (Kinf d) 0 [c, a, w, v] =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  show iteratedFDeriv ℝ 4 (Kinf d) 0 (fun i => [c, a, w, v].get i) = _
  rw [rb_peel _ rb_kinf_contDiff]
  simp only [rb_fderiv_kinf]
  show iteratedFDeriv ℝ 3 (fun y => -(⟪y, v⟫ * Kinf d y)) 0
    (Fin.init fun i => [c, a, w, v].get i) = _
  rw [rb_peel _ (rb_g1_contDiff v)]
  simp only [rb_fderiv_g1]
  show iteratedFDeriv ℝ 2 (fun y => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) 0
    (Fin.init (Fin.init fun i => [c, a, w, v].get i)) = _
  rw [rb_peel _ (rb_g2_contDiff v w)]
  simp only [rb_fderiv_g2]
  show iteratedFDeriv ℝ 1 (fun y => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0
    (Fin.init (Fin.init (Fin.init fun i => [c, a, w, v].get i))) = _
  rw [iteratedFDeriv_one_apply]
  show fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c = _
  exact rb_fderiv_g3_zero v w a c

/-! ### (f) Levels three to five along one direction `u` (Hermite forms) -/

/-- Smoothness of `(3 ⟪u,u⟫ ⟪y,u⟫ - ⟪y,u⟫^3) Kinf d y`. -/
theorem rb_P3_contDiff {d : ℕ} (u : E d) : ContDiff ℝ (⊤ : ℕ∞)
    (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) :=
  ((contDiff_const.mul (rb_inner_contDiff u)).sub
    (((rb_inner_contDiff u).mul (rb_inner_contDiff u)).mul (rb_inner_contDiff u))).mul
    rb_kinf_contDiff

/-- Derivative along `u` of the level-three Hermite form. -/
theorem rb_fderiv_P3 {d : ℕ} (u y : E d) :
    fderiv ℝ (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u =
      (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ y :=
    (hasFDerivAt_id y).inner ℝ (hasFDerivAt_const u y)
  have h : HasFDerivAt
      (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ y :=
    ((hin.const_mul _).sub ((hin.mul hin).mul hin)).mul (rb_hasFDerivAt_kinf y)
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-- Smoothness of the level-four Hermite form. -/
theorem rb_P4_contDiff {d : ℕ} (u : E d) : ContDiff ℝ (⊤ : ℕ∞)
    (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) :=
  ((contDiff_const.sub
      (contDiff_const.mul ((rb_inner_contDiff u).mul (rb_inner_contDiff u)))).add
    ((((rb_inner_contDiff u).mul (rb_inner_contDiff u)).mul (rb_inner_contDiff u)).mul
      (rb_inner_contDiff u))).mul rb_kinf_contDiff

/-- Derivative along `u` of the level-four Hermite form. -/
theorem rb_fderiv_P4 {d : ℕ} (u y : E d) :
    fderiv ℝ (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u =
      (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫ + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ y :=
    (hasFDerivAt_id y).inner ℝ (hasFDerivAt_const u y)
  have h : HasFDerivAt (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ y :=
    (((hasFDerivAt_const _ y).sub ((hin.mul hin).const_mul _)).add
      (((hin.mul hin).mul hin).mul hin)).mul (rb_hasFDerivAt_kinf y)
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-- Derivative at `0` along `u` of the level-five Hermite form: `-15 ⟪u, u⟫^3`. -/
theorem rb_fderiv_P5_zero {d : ℕ} (u : E d) :
    fderiv ℝ (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 u =
      -15 * (⟪u, u⟫ * ⟪u, u⟫ * ⟪u, u⟫) := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ (0 : E d) :=
    (hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const u 0)
  have h : HasFDerivAt (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ 0 :=
    (((hin.const_mul _).add (((hin.mul hin).mul hin).const_mul _)).sub
      ((((hin.mul hin).mul hin).mul hin).mul hin)).mul (rb_hasFDerivAt_kinf 0)
  rw [h.fderiv]
  simp [Kinf, fderivInnerCLM_apply]
  ring

/-- Sixth-order contraction of `Kinf d` at `0` along one direction: `-15 ⟪u, u⟫^3`. -/
theorem rb_order_six {d : ℕ} (u : E d) :
    derivContraction d (Kinf d) 0 [u, u, u, u, u, u] = -15 * (⟪u, u⟫ * ⟪u, u⟫ * ⟪u, u⟫) := by
  show iteratedFDeriv ℝ 6 (Kinf d) 0 (fun i => [u, u, u, u, u, u].get i) = _
  rw [rb_peel _ rb_kinf_contDiff]
  simp only [rb_fderiv_kinf]
  show iteratedFDeriv ℝ 5 (fun y => -(⟪y, u⟫ * Kinf d y)) 0 _ = _
  rw [rb_peel _ (rb_g1_contDiff u)]
  simp only [rb_fderiv_g1]
  show iteratedFDeriv ℝ 4 (fun y => -(⟪u, u⟫ * Kinf d y) + ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) 0 _ = _
  rw [rb_peel _ (rb_g2_contDiff u u)]
  simp only [rb_fderiv_g2]
  show iteratedFDeriv ℝ 3 (fun y => ⟪u, u⟫ * ⟪y, u⟫ * Kinf d y
      + (⟪u, u⟫ * ⟪y, u⟫ + ⟪y, u⟫ * ⟪u, u⟫) * Kinf d y
      - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) 0 _ = _
  rw [show (fun y : E d => ⟪u, u⟫ * ⟪y, u⟫ * Kinf d y
      + (⟪u, u⟫ * ⟪y, u⟫ + ⟪y, u⟫ * ⟪u, u⟫) * Kinf d y
      - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) =
      fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y from
      funext fun y => by ring]
  rw [rb_peel _ (rb_P3_contDiff u)]
  show iteratedFDeriv ℝ 2 (fun y => fderiv ℝ (fun y : E d =>
      (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u) 0 _ = _
  simp only [rb_fderiv_P3]
  show iteratedFDeriv ℝ 2 (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 _ = _
  rw [rb_peel _ (rb_P4_contDiff u)]
  show iteratedFDeriv ℝ 1 (fun y => fderiv ℝ (fun y : E d =>
      (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u) 0 _ = _
  simp only [rb_fderiv_P4]
  show iteratedFDeriv ℝ 1 (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 _ = _
  rw [iteratedFDeriv_one_apply]
  show fderiv ℝ (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 u = _
  exact rb_fderiv_P5_zero u

/-! ### (g) Second-order contraction and the blocks in an orthonormal frame -/

/-- Second-order contraction of `Kinf d` at `0`: `D^2 Kinf 0 [a, b] = -⟪a, b⟫`. -/
theorem rb_order_two {d : ℕ} (a b : E d) : derivContraction d (Kinf d) 0 [a, b] = -⟪a, b⟫ := by
  show iteratedFDeriv ℝ 2 (Kinf d) 0 (fun i => [a, b].get i) = -⟪a, b⟫
  rw [rb_peel _ rb_kinf_contDiff, iteratedFDeriv_one_apply]
  simp only [rb_fderiv_kinf]
  show fderiv ℝ (fun y : E d => -(⟪y, b⟫ * Kinf d y)) 0 a = -⟪a, b⟫
  rw [rb_fderiv_g1]
  simp [Kinf]

/-- The gradient block of `Kinf d` in an orthonormal frame is the identity. -/
theorem rb_gradCov {d : ℕ} (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    gradCov d (Kinf d) R = 1 := by
  ext i j
  simp only [gradCov, Matrix.of_apply, rb_order_two, neg_neg, Matrix.one_apply]
  exact orthonormal_iff_ite.mp hR i j

/-- The pin block of `Kinf d` with `u = R 0` in an orthonormal frame is `diagonal (3, 1, …, 1)`. -/
theorem rb_pinCov {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    pinCov d (Kinf d) (R ⟨0, hd⟩) R =
      Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1) := by
  have hite := orthonormal_iff_ite.mp hR
  ext i j
  simp only [pinCov, Matrix.of_apply, rb_order_four, hite, Matrix.diagonal_apply]
  by_cases hi : i = ⟨0, hd⟩
  · subst hi
    by_cases hj : j = ⟨0, hd⟩
    · subst hj
      norm_num
    · simp [Ne.symm hj]
  · by_cases hj : j = ⟨0, hd⟩
    · subst hj
      simp [hi, Ne.symm hi]
    · simp [hi, Ne.symm hi, Ne.symm hj]

/-- The third-derivative/gradient covariance vector of `Kinf d` with `u = R 0` is
`(-3, 0, …, 0)`. -/
theorem rb_thirdGradCov {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    thirdGradCov d (Kinf d) (R ⟨0, hd⟩) R = fun j => if j = ⟨0, hd⟩ then -3 else 0 := by
  have hite := orthonormal_iff_ite.mp hR
  funext j
  simp only [thirdGradCov, rb_order_four, hite]
  by_cases hj : j = ⟨0, hd⟩
  · simp [hj]; norm_num
  · have hj' : ¬ ((⟨0, hd⟩ : Fin d) = j) := fun h => hj h.symm
    simp [hj, hj']

/-- The third-derivative variance of `Kinf d` along a unit vector `u = R 0` is `15`. -/
theorem rb_thirdVar {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    thirdVar d (Kinf d) (R ⟨0, hd⟩) = 15 := by
  have hite := orthonormal_iff_ite.mp hR
  simp only [thirdVar, rb_order_six, hite]
  simp

/-- The Schur complement `condThirdVariance` of `Kinf d` in an orthonormal frame is `15 - 9 = 6`. -/
theorem rb_condThirdVariance {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    condThirdVariance d (Kinf d) (R ⟨0, hd⟩) R = 6 := by
  unfold condThirdVariance
  rw [rb_thirdVar hd R hR, rb_gradCov R hR, rb_thirdGradCov hd R hR, inv_one,
    Matrix.one_mulVec]
  simp [dotProduct, ite_mul, Finset.sum_ite_eq']
  norm_num

/-! ## `rp_`: the reference pin-density product (`ReferencePinDensityProduct`) -/

/-! ### The density algebra at the reference blocks -/

/-- `det (diagonal (3, 1, …, 1)) = 3` (the single `3` sits at the coordinate `0`). -/
theorem rp_det_pinDiagonal (d : ℕ) (hd : 0 < d) :
    (Matrix.diagonal fun i : Fin d => if i = ⟨0, hd⟩ then (3 : ℝ) else 1).det = 3 := by
  rw [Matrix.det_diagonal, Finset.prod_ite_eq']
  simp

/-- The product of the two centred Gaussian densities at `0`, for the covariance blocks `1` and
`diagonal (3, 1, …, 1)`: `(2π)^(-d/2) · (2π)^(-d/2) · 3^(-1/2) = (2π)^(-d) / √3`. -/
theorem rp_density_algebra (d : ℕ) (hd : 0 < d) :
    gaussDensity d 1 0 *
        gaussDensity d (Matrix.diagonal fun i : Fin d => if i = ⟨0, hd⟩ then (3 : ℝ) else 1) 0 =
      (2 * Real.pi) ^ (-(d : ℝ)) / Real.sqrt 3 := by
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  simp only [gaussDensity, rp_det_pinDiagonal d hd, Matrix.det_one, Matrix.mulVec_zero,
    dotProduct_zero, neg_zero, zero_div, Real.exp_zero, mul_one, Real.one_rpow]
  rw [← mul_assoc, ← Real.rpow_add hpi, show -(d : ℝ) / 2 + -(d : ℝ) / 2 = -(d : ℝ) by ring,
    show (-1 : ℝ) / 2 = -(1 / 2) by norm_num, Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 3),
    Real.sqrt_eq_rpow]
  exact (div_eq_mul_inv _ _).symm

/-! ### The common core: the Prop from the two block identities -/

/-- `ReferencePinDensityProduct` from the two block identities `gradCov d (Kinf d) R = 1` and
`pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)` (for `d ∈ {2, 3}` and orthonormal `R`), by
rewriting the blocks and `rp_density_algebra`. -/
theorem rp_referencePinDensityProduct_of_blockIdentities
    (hblk : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
      gradCov d (Kinf d) R = 1 ∧
      pinCov d (Kinf d) (R ⟨0, hd⟩) R = Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1)) :
    UniversalLaw.Spec.Side24.ReferencePinDensityProduct := by
  intro d hd23 hd R hR
  obtain ⟨hG, hV⟩ := hblk d hd23 hd R hR
  rw [hG, hV]
  exact rp_density_algebra d hd

/-! ### The hypothesis form: the Prop from `ReferenceOddAndPinBlocks` -/

/-- `ReferencePinDensityProduct` from `ReferenceOddAndPinBlocks` taken as a hypothesis: only its
first conjunct (`gradCov d (Kinf d) R = 1`) and fourth conjunct
(`pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)`) are used. -/
theorem rp_referencePinDensityProduct_of_referenceOddAndPinBlocks
    (h : UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks) :
    UniversalLaw.Spec.Side24.ReferencePinDensityProduct := by
  apply rp_referencePinDensityProduct_of_blockIdentities
  intro d hd23 hd R hR
  obtain ⟨hG, -, -, hV, -⟩ := h d hd23 hd R hR
  exact ⟨hG, hV⟩

/-! ## `ir_`: the Section 4 integrand-ratio bounds (`IntegrandRatioBounds`) -/

/-! ### (a) Elementary facts about `epsilon = 10^(-108)` -/

/-- `0 < epsilon`. -/
theorem ir_epsilon_pos : (0 : ℝ) < epsilon := by
  unfold epsilon
  positivity

/-- `epsilon < 1/100`: a crude upper bound sufficient for every sign condition below. -/
theorem ir_epsilon_lt : epsilon < (1 / 100 : ℝ) := by
  unfold epsilon
  norm_num

/-- `0 < 1 - epsilon`. -/
theorem ir_one_sub_epsilon_pos : (0 : ℝ) < 1 - epsilon := by
  have := ir_epsilon_lt
  linarith

/-- `1 ≤ 1 + epsilon`. -/
theorem ir_one_le_one_add_epsilon : (1 : ℝ) ≤ 1 + epsilon := by
  have := ir_epsilon_pos
  linarith

/-! ### (b) The two one-sided power estimates -/

/-- For a real exponent `b ≤ 5`, `(1 + epsilon)^b ≤ (1 + epsilon)^5`: monotonicity of the real
power in the exponent for a base `≥ 1` (`Real.rpow_le_rpow_of_exponent_le`), with the natural
power on the right via `Real.rpow_natCast`. -/
theorem ir_rpow_le_pow_five (b : ℝ) (hb : b ≤ 5) :
    (1 + epsilon) ^ b ≤ (1 + epsilon) ^ 5 := by
  have h := Real.rpow_le_rpow_of_exponent_le ir_one_le_one_add_epsilon hb
  rwa [show (5 : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast] at h

/-- Bernoulli's inequality at `s = -epsilon`: for a real exponent `a ≥ 1`,
`1 - a epsilon ≤ (1 - epsilon)^a` (`one_add_mul_self_le_rpow_one_add`). -/
theorem ir_bernoulli (a : ℝ) (ha : 1 ≤ a) : 1 - a * epsilon ≤ (1 - epsilon) ^ a := by
  have hε := ir_epsilon_lt
  have h := one_add_mul_self_le_rpow_one_add (show (-1 : ℝ) ≤ -epsilon by linarith) ha
  rw [show (1 : ℝ) + -epsilon = 1 - epsilon by ring] at h
  linarith

/-! ### (c) The two exact rational inequalities (degree five in `epsilon`) -/

/-- `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon`, exact rational arithmetic. -/
theorem ir_lower_rational : (1 - 13 * epsilon) * (1 + epsilon) ^ 5 ≤ 1 - 5 * epsilon := by
  unfold epsilon
  norm_num

/-- `(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)`, exact rational arithmetic. -/
theorem ir_upper_rational : (1 + epsilon) ^ 5 ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by
  unfold epsilon
  norm_num

/-! ### (d) The ratio bounds for arbitrary exponents in `[1, 5]` -/

/-- Lower bound: for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`,
`1 - 13 epsilon ≤ (1 - epsilon)^a / (1 + epsilon)^b`. -/
theorem ir_lower_bound (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b := by
  have hε := ir_epsilon_pos
  have hε1 := ir_epsilon_lt
  have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
  have hae : a * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right ha5 hε.le
  rw [le_div_iff₀ hpb]
  calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
      ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
        mul_le_mul_of_nonneg_left (ir_rpow_le_pow_five b hb5) (by linarith)
    _ ≤ 1 - 5 * epsilon := ir_lower_rational
    _ ≤ 1 - a * epsilon := by linarith
    _ ≤ (1 - epsilon) ^ a := ir_bernoulli a ha1

/-- Upper bound: for `a ≤ 5` and `1 ≤ b ≤ 5`,
`(1 + epsilon)^a / (1 - epsilon)^b ≤ 1 + 28 epsilon`. -/
theorem ir_upper_bound (a b : ℝ) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon := by
  have hε := ir_epsilon_pos
  have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos ir_one_sub_epsilon_pos _
  have hbe : b * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right hb5 hε.le
  rw [div_le_iff₀ hqb]
  calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := ir_rpow_le_pow_five a ha5
    _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := ir_upper_rational
    _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    _ ≤ (1 + 28 * epsilon) * (1 - epsilon) ^ b :=
        mul_le_mul_of_nonneg_left (ir_bernoulli b hb1) (by linarith)

/-- Both ratio bounds together, for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`. -/
theorem ir_ratio_bounds (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b ∧
      (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon :=
  ⟨ir_lower_bound a b ha1 ha5 hb5, ir_upper_bound a b ha5 hb1 hb5⟩

/-! ### (e) The exponents in dimensions two and three -/

/-- `exponentA 2 = 13/6` (`m = 1`, `n = 1`: `1 + 2/3 + 1/2`). -/
theorem ir_exponentA_two : exponentA 2 = 13 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 2 = 5/2` (`d = 2`, `n = 1`: `2 + 1/2`). -/
theorem ir_exponentB_two : exponentB 2 = 5 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

/-- `exponentA 3 = 25/6` (`m = 2`, `n = 3`: `2 + 2/3 + 3/2`). -/
theorem ir_exponentA_three : exponentA 3 = 25 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 3 = 9/2` (`d = 3`, `n = 3`: `3 + 3/2`). -/
theorem ir_exponentB_three : exponentB 3 = 9 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

/-! ## `li_`: the lattice-image sum bound (`LatticeImageSumBound`) -/

/-! ### (a) The geometric tail `(n+1)^9 exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n` -/

/-- Induction on `n` from the registered successive-term ratio: the ratio of consecutive terms
`j^9 exp(-288 j^2)` is at most `512 exp(-864) < 1/2`
(`UniversalLaw.Proofs.Side24Elementary.successiveTermRatio`). -/
theorem li_geom_tail (n : ℕ) :
    ((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    have h1 := (UniversalLaw.Proofs.Side24Elementary.successiveTermRatio).1 (n + 1) (by omega)
    have h2 := UniversalLaw.Proofs.Side24Elementary.Aux.ratio_const_lt_half
    have hnn : 0 ≤ ((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) := by positivity
    push_cast at h1 ⊢
    rw [pow_succ]
    calc _ ≤ 512 * Real.exp (-864) * (((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2)) :=
          h1
      _ ≤ (1 / 2) * (Real.exp (-288) * (1 / 2) ^ n) := mul_le_mul h2.le ih hnn (by norm_num)
      _ = Real.exp (-288) * ((1 / 2) ^ n * (1 / 2)) := by ring

/-- `(n+1)^6 exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n`: the sixth power is at most the ninth. -/
theorem li_pow6_succ_le (n : ℕ) :
    ((n : ℝ) + 1) ^ 6 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  refine le_trans ?_ (li_geom_tail n)
  have h0 : (0 : ℝ) ≤ n := by positivity
  have h1 : (1 : ℝ) ≤ (n : ℝ) + 1 := by linarith
  exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ h1 (by norm_num)) (Real.exp_pos _).le

/-- `exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n`: `1 ≤ (n+1)^9`. -/
theorem li_exp_succ_le (n : ℕ) :
    Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  refine le_trans ?_ (li_geom_tail n)
  have h0 : (0 : ℝ) ≤ n := by positivity
  have h1 : (1 : ℝ) ≤ ((n : ℝ) + 1) ^ 9 := one_le_pow₀ (by linarith)
  exact le_mul_of_one_le_left (Real.exp_pos _).le h1

/-! ### (b) One-dimensional sums over `ℤ` -/

/-- A nonnegative `φ : ℤ → ℝ` whose values at `n+1` and at `-(n+1)` are at most
`exp(-288) (1/2)^n` is summable over `ℤ`, with `∑' φ ≤ φ 0 + 4 exp(-288)`: split `ℤ` into `ℕ`
and the negatives (`tsum_of_nat_of_neg_add_one`), peel off `φ 0` (`Summable.tsum_eq_zero_add`)
and compare both tails with the geometric series (`tsum_geometric_two`). -/
theorem li_summable_int_tsum_le (φ : ℤ → ℝ) (h0 : ∀ m, 0 ≤ φ m)
    (hpos : ∀ n : ℕ, φ ((n : ℤ) + 1) ≤ Real.exp (-288) * (1 / 2) ^ n)
    (hneg : ∀ n : ℕ, φ (-((n : ℤ) + 1)) ≤ Real.exp (-288) * (1 / 2) ^ n) :
    Summable φ ∧ ∑' m : ℤ, φ m ≤ φ 0 + 4 * Real.exp (-288) := by
  have hgeo : Summable (fun n : ℕ => Real.exp (-288) * (1 / 2 : ℝ) ^ n) :=
    summable_geometric_two.mul_left _
  have hgeo_sum : ∑' n : ℕ, Real.exp (-288) * (1 / 2 : ℝ) ^ n = 2 * Real.exp (-288) := by
    rw [tsum_mul_left, tsum_geometric_two]
    ring
  have hs1 : Summable (fun n : ℕ => φ ((n : ℤ) + 1)) :=
    Summable.of_nonneg_of_le (fun n => h0 _) hpos hgeo
  have hs2 : Summable (fun n : ℕ => φ (-((n : ℤ) + 1))) :=
    Summable.of_nonneg_of_le (fun n => h0 _) hneg hgeo
  have hs0 : Summable (fun n : ℕ => φ (n : ℤ)) := by
    refine (summable_nat_add_iff 1).mp ?_
    refine hs1.congr fun n => ?_
    norm_cast
  have hsum0 : ∑' n : ℕ, φ (n : ℤ) ≤ φ 0 + 2 * Real.exp (-288) := by
    rw [hs0.tsum_eq_zero_add, Nat.cast_zero]
    have htail : ∑' n : ℕ, φ (((n + 1 : ℕ) : ℤ)) ≤ 2 * Real.exp (-288) := by
      rw [← hgeo_sum]
      refine ((summable_nat_add_iff 1).mpr hs0).tsum_le_tsum (fun n => ?_) hgeo
      rw [Nat.cast_succ]
      exact hpos n
    linarith
  have hsum2 : ∑' n : ℕ, φ (-((n : ℤ) + 1)) ≤ 2 * Real.exp (-288) := by
    rw [← hgeo_sum]
    exact hs2.tsum_le_tsum hneg hgeo
  refine ⟨Summable.of_nat_of_neg_add_one hs0 hs2, ?_⟩
  rw [tsum_of_nat_of_neg_add_one hs0 hs2]
  linarith

/-- The Gaussian weight `m ↦ exp(-288 m^2)` is summable over `ℤ` with sum `A ≤ 2`
(`A ≤ 1 + 4 exp(-288)` and `exp(-288) < 10^(-125)`). -/
theorem li_gauss_int :
    Summable (fun m : ℤ => Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) ∧
      ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) ≤ 2 := by
  have h := li_summable_int_tsum_le (fun m : ℤ => Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
    (fun m => (Real.exp_pos _).le)
    (fun n => by
      push_cast
      exact li_exp_succ_le n)
    (fun n => by
      push_cast
      rw [neg_sq]
      exact li_exp_succ_le n)
  refine ⟨h.1, h.2.trans ?_⟩
  have he := UniversalLaw.Proofs.Side24Elementary.Aux.exp_neg_lt
  have hp : (1 : ℝ) / 10 ^ 125 ≤ 1 / 4 := by norm_num
  have h0 : Real.exp (-288 * ((0 : ℤ) : ℝ) ^ 2) = 1 := by simp
  simp only [h0]
  linarith

/-- The weight `m ↦ m^6 exp(-288 m^2)` is summable over `ℤ` with sum `B ≤ 4 exp(-288)`. -/
theorem li_pow_gauss_int :
    Summable (fun m : ℤ => ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) ∧
      ∑' m : ℤ, ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) ≤
        4 * Real.exp (-288) := by
  have h := li_summable_int_tsum_le
    (fun m : ℤ => ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
    (fun m => mul_nonneg (by positivity) (Real.exp_pos _).le)
    (fun n => by
      push_cast
      exact li_pow6_succ_le n)
    (fun n => by
      push_cast
      rw [neg_sq, Even.neg_pow (by decide : Even 6)]
      exact li_pow6_succ_le n)
  refine ⟨h.1, h.2.trans ?_⟩
  simp

/-! ### (c) Finite sums of products over `Fin d → ℤ` -/

/-- For `h : Fin d → ℤ → ℝ` nonnegative and summable in each coordinate, the sum of
`∏ k, h k (n k)` over a finite set `s` of lattice points is at most `∏ k, ∑' m, h k m`: `s` lies
in the box `Fintype.piFinset (fun k => s.image (· k))`, over which the sum factors
(`Finset.prod_univ_sum`), and each factor is a partial sum of a nonnegative summable family
(`Summable.sum_le_tsum`). -/
theorem li_sum_prod_le (d : ℕ) (h : Fin d → ℤ → ℝ) (h0 : ∀ k m, 0 ≤ h k m)
    (hs : ∀ k, Summable (h k)) (s : Finset (Fin d → ℤ)) :
    ∑ n ∈ s, ∏ k, h k (n k) ≤ ∏ k, ∑' m, h k m := by
  set t : Fin d → Finset ℤ := fun k => s.image (fun n => n k)
  have hsub : s ⊆ Fintype.piFinset t := fun n hn => by
    rw [Fintype.mem_piFinset]
    intro k
    exact Finset.mem_image_of_mem _ hn
  have hle : ∀ k, ∑ m ∈ t k, h k m ≤ ∑' m, h k m := fun k =>
    (hs k).sum_le_tsum (t k) (fun m _ => h0 k m)
  calc ∑ n ∈ s, ∏ k, h k (n k) ≤ ∑ n ∈ Fintype.piFinset t, ∏ k, h k (n k) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun n _ _ => Finset.prod_nonneg fun k _ => h0 _ _
    _ = ∏ k, ∑ m ∈ t k, h k m := (Finset.prod_univ_sum t fun k m => h k m).symm
    _ ≤ ∏ k, ∑' m, h k m :=
        Finset.prod_le_prod₀ (fun k _ => Finset.sum_nonneg fun m _ => h0 k m) (fun k _ => hle k)

/-- A product over `Fin d` with one distinguished factor:
`∏ k, (if k = i then F k else G k) = F i * ∏ k ∈ univ.erase i, G k`. -/
theorem li_prod_ite_eq (d : ℕ) (i : Fin d) (F G : Fin d → ℝ) :
    (∏ k : Fin d, if k = i then F k else G k) = F i * ∏ k ∈ Finset.univ.erase i, G k := by
  rw [← Finset.mul_prod_erase Finset.univ (fun k => if k = i then F k else G k)
    (Finset.mem_univ i), ite_eq_left rfl]
  congr 1
  exact Finset.prod_congr rfl fun k hk => ite_eq_right (Finset.ne_of_mem_erase hk)

/-- `∏ k, exp(-288 n_k^2) = exp(-288 |n|^2)` (`Real.exp_sum`). -/
theorem li_prod_exp (d : ℕ) (n : Fin d → ℤ) :
    ∏ k : Fin d, Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2) = Real.exp (-288 * intNormSq d n) := by
  rw [← Real.exp_sum, intNormSq, Finset.mul_sum]

/-! ### (d) The pointwise domination for `d ≤ 3` -/

/-- Power mean for at most three nonnegative terms: `(∑ a_i)^3 ≤ 9 ∑ a_i^3` for `d ≤ 3`
(Jensen's `pow_sum_div_card_le_sum_pow` with `d^2 ≤ 9`; the case `d = 0` is `0 ≤ 0`). -/
theorem li_sum_cube_le (d : ℕ) (hd : d ≤ 3) (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i) :
    (∑ i, a i) ^ 3 ≤ 9 * ∑ i, a i ^ 3 := by
  rcases Nat.eq_zero_or_pos d with hd0 | hdpos
  · subst hd0
    simp
  · have hdR : (0 : ℝ) < d := by exact_mod_cast hdpos
    have h := pow_sum_div_card_le_sum_pow (s := Finset.univ) (f := a) (fun i _ => ha i) 2
    rw [Finset.card_univ, Fintype.card_fin, div_le_iff₀ (pow_pos hdR 2)] at h
    have hd' : (d : ℝ) ≤ 3 := by exact_mod_cast hd
    have hsum : 0 ≤ ∑ i, a i ^ 3 := Finset.sum_nonneg fun i _ => pow_nonneg (ha i) 3
    calc (∑ i, a i) ^ 3 ≤ (∑ i, a i ^ 3) * (d : ℝ) ^ 2 := h
      _ ≤ (∑ i, a i ^ 3) * 9 := by
          apply mul_le_mul_of_nonneg_left _ hsum
          nlinarith
      _ = 9 * ∑ i, a i ^ 3 := by ring

/-- `imageTerm d n ≥ 0`. -/
theorem li_imageTerm_nonneg (d : ℕ) (n : Fin d → ℤ) : 0 ≤ imageTerm d n := by
  unfold imageTerm
  split_ifs
  · exact le_rfl
  · exact mul_nonneg (pow_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) 3) (Real.exp_pos _).le

/-- Pointwise domination for `d ≤ 3`: `imageTerm d n ≤ 9 ∑ i, ∏ k, h_i k (n k)`, where
`h_i i m = m^6 exp(-288 m^2)` and `h_i k m = exp(-288 m^2)` for `k ≠ i`. Each product equals
`(n_i^2)^3 exp(-288 |n|^2)` (`li_prod_ite_eq`, `li_prod_exp`), and `(∑ n_k^2)^3 ≤ 9 ∑ (n_i^2)^3`
is `li_sum_cube_le`. -/
theorem li_imageTerm_le (d : ℕ) (hd : d ≤ 3) (n : Fin d → ℤ) :
    imageTerm d n ≤ 9 * ∑ i : Fin d, ∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) := by
  have hprod : ∀ i : Fin d, (∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))) =
      (((n i : ℤ) : ℝ) ^ 2) ^ 3 * Real.exp (-288 * intNormSq d n) := by
    intro i
    rw [li_prod_ite_eq d i (fun k => ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))
      (fun k => Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)), ← li_prod_exp d n,
      ← Finset.mul_prod_erase Finset.univ (fun k => Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))
        (Finset.mem_univ i)]
    ring
  rw [Finset.sum_congr rfl (fun i _ => hprod i), ← Finset.sum_mul]
  unfold imageTerm
  split_ifs
  · positivity
  · have h := li_sum_cube_le d hd (fun i => ((n i : ℤ) : ℝ) ^ 2) (fun i => sq_nonneg _)
    unfold intNormSq
    calc (∑ i, ((n i : ℤ) : ℝ) ^ 2) ^ 3 * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2)
        ≤ (9 * ∑ i, (((n i : ℤ) : ℝ) ^ 2) ^ 3) * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2) :=
          mul_le_mul_of_nonneg_right h (Real.exp_pos _).le
      _ = 9 * ((∑ i, (((n i : ℤ) : ℝ) ^ 2) ^ 3) * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2)) := by
          ring

/-! ### (e) The uniform partial-sum bound -/

/-- For `d ≤ 3` and every finite `s`, `∑ n ∈ s, imageTerm d n ≤ 1458 exp(-288)`: the pointwise
bound `li_imageTerm_le`, the box bound `li_sum_prod_le` for each `i`, the one-dimensional sums
`A ≤ 2` and `B ≤ 4 exp(-288)` (`li_gauss_int`, `li_pow_gauss_int`), and
`9 · d · B · A^(d-1) ≤ 9 · 3 · 4 exp(-288) · 4 = 432 exp(-288)`. -/
theorem li_partial_sum_le (d : ℕ) (hd : d ≤ 3) (s : Finset (Fin d → ℤ)) :
    ∑ n ∈ s, imageTerm d n ≤ 1458 * Real.exp (-288) := by
  obtain ⟨hgs, hgle⟩ := li_gauss_int
  obtain ⟨hfs, hfle⟩ := li_pow_gauss_int
  have hA0 : 0 ≤ ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) :=
    tsum_nonneg fun m => (Real.exp_pos _).le
  have hexp : 0 ≤ Real.exp (-288) := (Real.exp_pos _).le
  set A : ℝ := ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) with hA
  set B : ℝ := ∑' m : ℤ, ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) with hB
  have hidx : ∀ i : Fin d, ∑ n ∈ s, ∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) ≤ 16 * Real.exp (-288) := by
    intro i
    have h1 := li_sum_prod_le d
      (fun k m => if k = i then ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
      (fun k m => by
        split_ifs
        · positivity
        · positivity)
      (fun k => by
        by_cases hk : k = i
        · simp only [hk, ite_true]
          exact hfs
        · simp only [hk, ite_false]
          exact hgs)
      s
    have h2 : ∀ k : Fin d, ∑' m : ℤ,
        (if k = i then ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)
          else Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) = if k = i then B else A := by
      intro k
      by_cases hk : k = i
      · simp only [hk, ite_true, hB]
      · simp only [hk, ite_false, hA]
    calc _ ≤ _ := h1
      _ = ∏ k : Fin d, (if k = i then B else A) := Finset.prod_congr rfl fun k _ => h2 k
      _ = B * ∏ _k ∈ Finset.univ.erase i, A := li_prod_ite_eq d i (fun _ => B) (fun _ => A)
      _ = B * A ^ (d - 1) := by
          rw [Finset.prod_const, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
            Fintype.card_fin]
      _ ≤ (4 * Real.exp (-288)) * 2 ^ (d - 1) :=
          mul_le_mul hfle (pow_le_pow_left₀ hA0 hgle _) (pow_nonneg hA0 _) (by positivity)
      _ ≤ (4 * Real.exp (-288)) * 2 ^ 2 := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          exact pow_le_pow_right₀ (by norm_num) (by omega)
      _ = 16 * Real.exp (-288) := by ring
  calc ∑ n ∈ s, imageTerm d n
      ≤ ∑ n ∈ s, 9 * ∑ i : Fin d, ∏ k : Fin d,
          (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
            else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) :=
        Finset.sum_le_sum fun n _ => li_imageTerm_le d hd n
    _ = 9 * ∑ i : Fin d, ∑ n ∈ s, ∏ k : Fin d,
          (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
            else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) := by
        rw [← Finset.mul_sum, Finset.sum_comm]
    _ ≤ 9 * ∑ _i : Fin d, 16 * Real.exp (-288) :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => hidx i) (by norm_num)
    _ = 9 * ((d : ℝ) * (16 * Real.exp (-288))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ ≤ 9 * (3 * (16 * Real.exp (-288))) := by
        have hd' : (d : ℝ) ≤ 3 := by exact_mod_cast hd
        gcongr
    _ ≤ 1458 * Real.exp (-288) := by linarith

/-! ## `dc_`: the Section 4 Gaussian density comparison (`DensityComparison`) -/

/-! ### (a) Elementary facts about `epsilon = 10^(-108)`

`0 < epsilon` and `0 < 1 - epsilon` are `ir_epsilon_pos` and `ir_one_sub_epsilon_pos` above
(the `dc_` copies were removed at integration); only `0 < 1 + epsilon` is added here. -/

/-- `0 < 1 + epsilon`. -/
theorem dc_one_add_epsilon_pos : (0 : ℝ) < 1 + epsilon := by
  have := ir_epsilon_pos
  linarith

/-! ### (b) Loewner order: antitonicity of the inverse (L1) -/

/-- The expansion identity `(C - A) A⁻¹ (C - A) + (C - A) = C A⁻¹ C - C` for an invertible `A`. -/
theorem dc_expand_identity {k : ℕ} (A C : Matrix (Fin k) (Fin k) ℝ) (hA : IsUnit A.det) :
    (C - A) * A⁻¹ * (C - A) + (C - A) = C * A⁻¹ * C - C := by
  have h1 : A * A⁻¹ = 1 := Matrix.mul_nonsing_inv A hA
  have h2 : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul A hA
  calc (C - A) * A⁻¹ * (C - A) + (C - A)
      = C * A⁻¹ * C - C * (A⁻¹ * A) - (A * A⁻¹) * C + (A * A⁻¹) * A + (C - A) := by
        noncomm_ring
    _ = C * A⁻¹ * C - C := by
        rw [h1, h2]
        simp only [Matrix.mul_one, Matrix.one_mul]
        abel

/-- The conjugation identity `C⁻¹ (C A⁻¹ C - C) C⁻¹ = A⁻¹ - C⁻¹` for an invertible `C`. -/
theorem dc_conj_identity {k : ℕ} (A C : Matrix (Fin k) (Fin k) ℝ) (hC : IsUnit C.det) :
    C⁻¹ * (C * A⁻¹ * C - C) * C⁻¹ = A⁻¹ - C⁻¹ := by
  have h1 : C * C⁻¹ = 1 := Matrix.mul_nonsing_inv C hC
  have h2 : C⁻¹ * C = 1 := Matrix.nonsing_inv_mul C hC
  calc C⁻¹ * (C * A⁻¹ * C - C) * C⁻¹
      = (C⁻¹ * C) * A⁻¹ * (C * C⁻¹) - C⁻¹ * (C * C⁻¹) := by
        noncomm_ring
    _ = A⁻¹ - C⁻¹ := by
        rw [h1, h2]
        simp only [Matrix.mul_one, Matrix.one_mul]

/-- (L1) For real matrices: `A` positive definite, `C` positive definite and `C - A` positive
semidefinite give `A⁻¹ - C⁻¹` positive semidefinite, i.e. `C⁻¹ ≤ A⁻¹` in the Loewner order. -/
theorem dc_inv_sub_inv_posSemidef {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hC : C.PosDef) (hAC : (C - A).PosSemidef) :
    (A⁻¹ - C⁻¹).PosSemidef := by
  have hAu : IsUnit A.det := isUnit_iff_ne_zero.mpr hA.det_pos.ne'
  have hCu : IsUnit C.det := isUnit_iff_ne_zero.mpr hC.det_pos.ne'
  have hP : (C * A⁻¹ * C - C).PosSemidef := by
    have h1 : ((C - A).conjTranspose * A⁻¹ * (C - A)).PosSemidef :=
      hA.inv.posSemidef.conjTranspose_mul_mul_same (C - A)
    rw [hAC.isHermitian.eq] at h1
    have h2 := h1.add hAC
    rwa [dc_expand_identity A C hAu] at h2
  have h3 : ((C⁻¹).conjTranspose * (C * A⁻¹ * C - C) * C⁻¹).PosSemidef :=
    hP.conjTranspose_mul_mul_same C⁻¹
  rw [hC.inv.isHermitian.eq] at h3
  rwa [dc_conj_identity A C hCu] at h3

/-- The quadratic-form consequence of (L1): `x ⬝ C⁻¹ x ≤ x ⬝ A⁻¹ x`. -/
theorem dc_quad_le {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hC : C.PosDef) (hAC : (C - A).PosSemidef) (x : Fin k → ℝ) :
    dotProduct x (Matrix.mulVec C⁻¹ x) ≤ dotProduct x (Matrix.mulVec A⁻¹ x) := by
  have h := (dc_inv_sub_inv_posSemidef hA hC hAC).dotProduct_mulVec_nonneg x
  simp only [star_trivial, Matrix.sub_mulVec, dotProduct_sub] at h
  linarith

/-! ### (c) Loewner order: monotonicity of the determinant (L2) -/

/-- A real matrix `M` with `M - 1` positive semidefinite has determinant at least one: every
eigenvalue of `M` lies in `spectrum ℝ M`, so every eigenvalue minus one lies in
`spectrum ℝ (M - 1) ⊆ [0, ∞)`, and `det M` is the product of the eigenvalues. -/
theorem dc_one_le_det {k : ℕ} {M : Matrix (Fin k) (Fin k) ℝ} (hM : (M - 1).PosSemidef) :
    1 ≤ M.det := by
  have hherm : M.IsHermitian := by
    have h := hM.isHermitian.add (Matrix.isHermitian_one (n := Fin k) (α := ℝ))
    simpa using h
  rw [hherm.det_eq_prod_eigenvalues]
  apply Finset.one_le_prod₀
  intro i _
  have hspec := (Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg.mp hM).2
  have hmem : hherm.eigenvalues i ∈ spectrum ℝ M := hherm.eigenvalues_mem_spectrum_real i
  have hmem' : hherm.eigenvalues i - 1 ∈ spectrum ℝ (M - 1) := by
    have h : hherm.eigenvalues i - 1 ∈ spectrum ℝ M - ({1} : Set ℝ) :=
      Set.sub_mem_sub hmem (Set.mem_singleton (1 : ℝ))
    rw [spectrum.sub_singleton_eq] at h
    simpa using h
  have h0 : (0 : ℝ) ≤ hherm.eigenvalues i - 1 := hspec hmem'
  simp only [RCLike.ofReal_real_eq_id, id]
  linarith

/-- (L2) For real matrices: `A` positive definite and `C - A` positive semidefinite give
`det A ≤ det C`, through the square root `S = CFC.sqrt A` and `M = S⁻¹ C S⁻¹ ≥ 1`. -/
theorem dc_det_le_det {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hAC : (C - A).PosSemidef) : A.det ≤ C.det := by
  obtain ⟨S, hSS, hSpsd⟩ : ∃ S : Matrix (Fin k) (Fin k) ℝ, S * S = A ∧ S.PosSemidef :=
    ⟨CFC.sqrt A, CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg,
      Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg A)⟩
  have hSherm : S.conjTranspose = S := hSpsd.isHermitian.eq
  have hdetA : A.det = S.det * S.det := by
    rw [← hSS, Matrix.det_mul]
  have hSdet : S.det ≠ 0 := by
    intro h
    have h0 : A.det = 0 := by
      rw [hdetA, h, mul_zero]
    exact hA.det_pos.ne' h0
  have hSu : IsUnit S.det := isUnit_iff_ne_zero.mpr hSdet
  have hSinv : (S⁻¹).conjTranspose = S⁻¹ := by
    rw [Matrix.conjTranspose_nonsing_inv, hSherm]
  have h1 : S * S⁻¹ = 1 := Matrix.mul_nonsing_inv S hSu
  have h2 : S⁻¹ * S = 1 := Matrix.nonsing_inv_mul S hSu
  have hM : (S⁻¹ * C * S⁻¹ - 1).PosSemidef := by
    have h3 : ((S⁻¹).conjTranspose * (C - A) * S⁻¹).PosSemidef :=
      hAC.conjTranspose_mul_mul_same S⁻¹
    rw [hSinv] at h3
    have h4 : S⁻¹ * (C - A) * S⁻¹ = S⁻¹ * C * S⁻¹ - 1 := by
      rw [← hSS]
      calc S⁻¹ * (C - S * S) * S⁻¹
          = S⁻¹ * C * S⁻¹ - (S⁻¹ * S) * (S * S⁻¹) := by
            noncomm_ring
        _ = S⁻¹ * C * S⁻¹ - 1 := by
            rw [h1, h2, Matrix.mul_one]
    rwa [h4] at h3
  have hdetM : 1 ≤ (S⁻¹ * C * S⁻¹).det := dc_one_le_det hM
  have hCeq : S * (S⁻¹ * C * S⁻¹) * S = C := by
    calc S * (S⁻¹ * C * S⁻¹) * S = (S * S⁻¹) * C * (S⁻¹ * S) := by
          noncomm_ring
      _ = C := by
          rw [h1, h2, Matrix.one_mul, Matrix.mul_one]
  have hdetC : C.det = S.det * S.det * (S⁻¹ * C * S⁻¹).det := by
    calc C.det = (S * (S⁻¹ * C * S⁻¹) * S).det := by
          rw [hCeq]
      _ = S.det * (S⁻¹ * C * S⁻¹).det * S.det := by
          rw [Matrix.det_mul, Matrix.det_mul]
      _ = S.det * S.det * (S⁻¹ * C * S⁻¹).det := by
          ring
  have hSS2 : 0 ≤ S.det * S.det := mul_self_nonneg _
  calc A.det = S.det * S.det := hdetA
    _ = S.det * S.det * 1 := (mul_one _).symm
    _ ≤ S.det * S.det * (S⁻¹ * C * S⁻¹).det := mul_le_mul_of_nonneg_left hdetM hSS2
    _ = C.det := hdetC.symm

/-! ### (d) Determinant scaling and real-power bookkeeping -/

/-- `det (c • C) = c ^ k * det C` for a `k × k` real matrix. -/
theorem dc_det_smul {k : ℕ} (C : Matrix (Fin k) (Fin k) ℝ) (c : ℝ) :
    (c • C).det = c ^ k * C.det := by
  rw [Matrix.det_smul, Fintype.card_fin]

/-- The scaling identity `(a / b)^(k/2) · (a^k D)^(-1/2) = (b^k D)^(-1/2)` for `a, b, D > 0`. -/
theorem dc_scale_identity (a b D : ℝ) (ha : 0 < a) (hb : 0 < b) (hD : 0 < D) (k : ℕ) :
    (a / b) ^ ((k : ℝ) / 2) * (a ^ k * D) ^ (-(1 : ℝ) / 2) = (b ^ k * D) ^ (-(1 : ℝ) / 2) := by
  have hak : (a ^ k) ^ (-(1 : ℝ) / 2) = (a ^ ((k : ℝ) / 2))⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul ha.le, ← Real.rpow_neg ha.le]
    congr 1
    ring
  have hbk : (b ^ k) ^ (-(1 : ℝ) / 2) = (b ^ ((k : ℝ) / 2))⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hb.le, ← Real.rpow_neg hb.le]
    congr 1
    ring
  have hapos : a ^ ((k : ℝ) / 2) ≠ 0 := (Real.rpow_pos_of_pos ha _).ne'
  have hbpos : b ^ ((k : ℝ) / 2) ≠ 0 := (Real.rpow_pos_of_pos hb _).ne'
  rw [Real.div_rpow ha.le hb.le, Real.mul_rpow (pow_nonneg ha.le k) hD.le,
    Real.mul_rpow (pow_nonneg hb.le k) hD.le, hak, hbk]
  field_simp

/-! ### (e) Assembly of the two pointwise inequalities from abstract real factors -/

/-- Lower assembly: `s * dA = dB`, `dB ≤ dC`, `eA ≤ eC` give `s (P dA eA) ≤ P dC eC`. -/
theorem dc_assemble_lower (P s dA dB dC eA eC : ℝ) (hP : 0 ≤ P) (hs : s * dA = dB)
    (hd : dB ≤ dC) (hdC : 0 ≤ dC) (he : eA ≤ eC) (heA : 0 ≤ eA) :
    s * (P * dA * eA) ≤ P * dC * eC := by
  calc s * (P * dA * eA) = P * (s * dA) * eA := by ring
    _ = P * dB * eA := by rw [hs]
    _ ≤ P * dC * eC := mul_le_mul (mul_le_mul_of_nonneg_left hd hP) he heA (mul_nonneg hP hdC)

/-- Upper assembly: `s * dB = dA`, `dC ≤ dA`, `eC ≤ eB` give `P dC eC ≤ s (P dB eB)`. -/
theorem dc_assemble_upper (P s dA dB dC eB eC : ℝ) (hP : 0 ≤ P) (hs : s * dB = dA)
    (hd : dC ≤ dA) (hdA : 0 ≤ dA) (he : eC ≤ eB) (heC : 0 ≤ eC) :
    P * dC * eC ≤ s * (P * dB * eB) := by
  calc P * dC * eC ≤ P * dA * eB :=
        mul_le_mul (mul_le_mul_of_nonneg_left hd hP) he heC (mul_nonneg hP hdA)
    _ = P * (s * dB) * eB := by rw [hs]
    _ = s * (P * dB * eB) := by ring

end Aux

/-! ## The theta family is summable at every point, in every dimension -/

/-- Proves: `UniversalLaw.Spec.Side24.ThetaSummable` (claim `side24-coefficient`).
Method: comparison `Summable.of_nonneg_of_le` with the dominating family
`exp(‖z‖^2/2) ∏ i, exp(-144 n_i^2)` (`Aux.th_term_le_dominant`, from `‖24 n‖^2 = 576 ∑ n_i^2`
and `Real.exp_sum`); that family is summable (`Aux.th_summable_dominant`) because
`m ↦ exp(-144 m^2)` is summable on `ℤ` (`HurwitzKernelBounds.summable_f_int`) and products of
a nonnegative summable function are summable over `Fin d → ℤ` (`Aux.th_summable_pi_prod`, via
`Fintype.piFinset`, `Finset.prod_univ_sum` and `summable_of_sum_le`).
Does not establish: the informal convergence statement of the source as the source's object
(the alignment of `E d`, `latticeVec` and this family with the source's `R^d`, `24 Z^d` and
`K_24` is PENDING_INDEPENDENT_REVIEW); any value, positivity, smoothness or Poisson-summation
property of `thetaSum24` or `K24`; `ImageBound`, `LatticeImageSumBound` or the derivative
bounds of Section 2; any enclosure of the coefficient; the statement module's own
`Does not claim:` text (the statement packet is not edited here); the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived
here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem thetaSummable (d : ℕ) : UniversalLaw.Spec.Side24.ThetaSummable d := by
  intro z
  exact Summable.of_nonneg_of_le (fun n => (Real.exp_pos _).le)
    (fun n => Aux.th_term_le_dominant d z n) (Aux.th_summable_dominant d z)

/-! ## Inventory theorem -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativePairingBound` (register row `side24-coefficient`,
Section 2 first display of SIDE24-COEFFICIENT-D23-20260924-v1).
Method: the contraction at a unit tuple is at most the operator norm `‖iteratedFDeriv ℝ q (Kinf d) x‖`
(`Aux.dp_abs_le_opNorm`, from `ContinuousMultilinearMap.le_opNorm`); the operator norms satisfy the
Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` (`Aux.dp_recursion`, from
`norm_iteratedFDeriv_fderiv`, the explicit derivative `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`
and the Leibniz bound `norm_iteratedFDeriv_smul_le`, the linear factor's derivatives of order `≥ 2`
vanishing), and with `b_0 = Kinf d x`, `b_1 = Kinf d x ‖x‖` the orders `q ≤ 6` are matched to the
unfolded `pairingSum q ‖x‖` (`Aux.dp_pairing_all`, `interval_cases`, `nlinarith`).
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of the
formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q`) with the source's
(PENDING_INDEPENDENT_REVIEW); validity for `q > 6` (the Prop restricts to `q ≤ 6`, as the source does,
and nothing beyond it is stated or proved here); any enclosure of the coefficient `c_{d,24}`; the
disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the register, not
derived here. -/
theorem derivativePairingBound : UniversalLaw.Spec.Side24.DerivativePairingBound := by
  intro d q hq x u hu
  exact (Aux.dp_abs_le_opNorm d q x u hu).trans (Aux.dp_pairing_all d x q hq)

/-! ## The Section 2 derivative bound at `|x| ≥ 1` -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "At |x|>=1 this is at most 76 |x|^6 e^(-|x|^2/2)").
Method: the Section 2 pairing bound `|D^q Kinf (x) u| ≤ exp(-‖x‖²/2) · pairingSum q ‖x‖` is the
inventory theorem `derivativePairingBound` of this module (the Hermite-type recursion
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` for the operator norms, `Aux.dp_recursion`, with
`ContinuousMultilinearMap.le_opNorm` at a unit tuple); the hypothesis-form helper
`Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound` combines it with the elementary
inequality `pairingSum q r ≤ 76 r^6` for `1 ≤ r`, `q ≤ 6` (`Aux.d1_pairingSum_le_of_one_le`:
`interval_cases`, `Finset.sum_range_succ`, `Nat.factorial`, `norm_num`, and the monomial bounds
`r^k ≤ r^6` from `pow_le_pow_right₀`, `one_le_pow₀`), by `mul_le_mul_of_nonneg_left` with
`0 ≤ exp(-‖x‖²/2)`.
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of the
formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q`) with the source's
(PENDING_INDEPENDENT_REVIEW); any identification with the main-branch targets
`UniversalLaw.Side24.pairing_sum_six` and `pairing_sum_le_six`, which kernel-check the coefficient
arithmetic `1+15+45+15=76` and `76` bounding the smaller orders there, not here; validity for `q > 6` or
for `‖x‖ < 1` (the Prop restricts to `q ≤ 6` and `1 ≤ ‖x‖`, as the source does); any enclosure of the
coefficient `c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed
in the register, not derived here. -/
theorem derivativeBoundAtLeastOne : UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne :=
  Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound derivativePairingBound

/-! ## The Section 2 derivative bound at the origin -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtZero` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "contraction has absolute value at most 15").
Method: the hypothesis-form helper `Aux.d0_derivativeBoundAtZero_of_derivativePairingBound`
specialises the pairing bound `DerivativePairingBound` to `x = 0`, where `‖0‖ = 0`,
`exp (-(0 ^ 2) / 2) = 1` and `pairingSum q 0 ≤ 15` for `q ≤ 6` (`Aux.d0_pairingSum_zero_le`:
the values are `1, 0, 1, 0, 3, 0, 15`, by `interval_cases`, `Finset.sum_range_succ`, `norm_num`);
the pairing bound itself is the inventory theorem `derivativePairingBound` of this module
(`Aux.dp_recursion`, `Aux.dp_pairing_all`, `Aux.dp_abs_le_opNorm`).
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of
the formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q` at `0`) with
the source's (PENDING_INDEPENDENT_REVIEW); the double-factorial identity `5!! = 15` as such, which
is kernel-checked on main (`UniversalLaw.Side24.sixth_moment_double_factorial`) and is neither
re-proved nor cited here (only `pairingSum 6 0 = 15` is computed); validity for `q > 6` (the Prop
restricts to `q ≤ 6`, as the source does); any value of a contraction (only an upper bound on its
absolute value); any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the register, not derived here. -/
theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :=
  Aux.d0_derivativeBoundAtZero_of_derivativePairingBound derivativePairingBound

/-! ## The Section 1 reference blocks of `K_infty` in every orthonormal frame -/

/-- Proves: `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks` (claim `side24-coefficient`).
Method: the derivative contractions of `Kinf d` at `0` are evaluated by peeling one direction
at a time (`Aux.rb_peel`, from `iteratedFDeriv_succ_apply_right` and
`iteratedFDeriv_clm_apply_const_apply`) with explicit `HasFDerivAt` closed forms of the
intermediate functions (`Aux.rb_fderiv_kinf`, `rb_fderiv_g1`, `rb_fderiv_g2`,
`rb_fderiv_g3_zero`, `rb_fderiv_P3`, `rb_fderiv_P4`, `rb_fderiv_P5_zero`), giving
`Aux.rb_order_two` (`-⟪a, b⟫`), `Aux.rb_order_four` (the three pairings) and `Aux.rb_order_six`
(`-15 ⟪u, u⟫^3`); in an orthonormal frame (`orthonormal_iff_ite`) the five blocks follow
(`Aux.rb_gradCov`, `rb_thirdVar`, `rb_thirdGradCov`, `rb_pinCov`, `rb_condThirdVariance`,
the last with `inv_one`, `Matrix.one_mulVec` and `Finset.sum_ite_eq'`).
Does not establish: the informal Section 1 block values of the source as the source's objects
(the alignment of `E d`, `Kinf`, `derivContraction`, `gradCov`, `pinCov`, `thirdGradCov`,
`thirdVar` and `condThirdVariance` with the source's `R^d`, `K_infty`, covariances and
conditional variance is PENDING_INDEPENDENT_REVIEW); that any of these matrices is the
covariance of a random vector or field (none is constructed in the statement module); the
statement module's own `Does not claim:` text ("the identities (statement only); `15 - 9 = 6` is
kernel-checked on main (`UniversalLaw.Side24.conditional_third_derivative_variance`), not
here"), which describes the statement packet, is not edited here, and main's theorem is neither
cited nor relied on; `ReferencePinDensityProduct`, `CovarianceOrdering`,
`EntrywiseCovarianceBound` or any other Prop of the module; any enclosure of the coefficient;
the disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register,
not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem referenceOddAndPinBlocks : UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks := by
  intro d _ hd R hR
  exact ⟨Aux.rb_gradCov R hR, Aux.rb_thirdVar hd R hR, Aux.rb_thirdGradCov hd R hR,
    Aux.rb_pinCov hd R hR, Aux.rb_condThirdVariance hd R hR⟩

/-! ## The Section 1 reference pin-density product -/

/-- Proves: `UniversalLaw.Spec.Side24.ReferencePinDensityProduct` (register row `side24-coefficient`,
Section 1 of SIDE24-COEFFICIENT-D23-20260924-v1, "p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3)").
Method: the hypothesis form `Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks` takes
the inventory theorem `referenceOddAndPinBlocks` of this module and uses only its first conjunct
`gradCov d (Kinf d) R = 1` and fourth conjunct `pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)`
(both from the contractions `Aux.rb_order_two`, `Aux.rb_order_four` in an orthonormal frame); the
common core `Aux.rp_referencePinDensityProduct_of_blockIdentities` rewrites the two blocks and
finishes with the density algebra `Aux.rp_density_algebra` (`Matrix.det_one`, `Matrix.det_diagonal`,
`Finset.prod_ite_eq'`, `Matrix.mulVec_zero`, `Real.rpow_add`, `Real.rpow_neg`, `Real.sqrt_eq_rpow`).
Does not establish: the informal Section 1 display or its review and acceptance; the alignment of
the formal objects (`gaussDensity`, `gradCov`, `pinCov`, `derivContraction` through Mathlib's
`iteratedFDeriv ℝ` at `0`, `Orthonormal ℝ R`) with the source's (PENDING_INDEPENDENT_REVIEW); the
determinant identity `det diag(3,1,…,1) = 3` as the main-branch target
`UniversalLaw.Side24.pin_determinant_and_joint_dimension`, which is kernel-checked on main and is
neither cited nor re-used here (the determinant is recomputed by `Matrix.det_diagonal` in
`Aux.rp_det_pinDiagonal`); that `gaussDensity` is the density of any random vector or that any field
is constructed; the remaining conjuncts of `ReferenceOddAndPinBlocks` (`Var(t) = 15`,
`Cov(t, G) = (-3, 0, …, 0)`, `tau^2 = 6`), which this theorem does not use; the value for
`d ∉ {2, 3}` (the Prop restricts to `d = 2 ∨ d = 3`, as the source does); any enclosure of the
coefficient `c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as
transcribed in the register, not derived here. -/
theorem referencePinDensityProduct : UniversalLaw.Spec.Side24.ReferencePinDensityProduct :=
  Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks referenceOddAndPinBlocks

/-! ## The Section 4 integrand-ratio bounds in dimensions two and three -/

/-- Proves: `UniversalLaw.Spec.Side24.IntegrandRatioBounds` (claim `side24-coefficient`).
Method: case split `d = 2`, `d = 3`, where the exponents evaluate to `a = 13/6, b = 5/2` and
`a = 25/6, b = 9/2` (`Aux.ir_exponentA_two`, `Aux.ir_exponentB_two`, `Aux.ir_exponentA_three`,
`Aux.ir_exponentB_three`); for arbitrary `1 ≤ a, b ≤ 5` both bounds follow
(`Aux.ir_ratio_bounds`) from Bernoulli's inequality `1 - a epsilon ≤ (1 - epsilon)^a`
(`one_add_mul_self_le_rpow_one_add`, `Aux.ir_bernoulli`), exponent monotonicity
`(1 + epsilon)^b ≤ (1 + epsilon)^5` (`Real.rpow_le_rpow_of_exponent_le`,
`Aux.ir_rpow_le_pow_five`), `le_div_iff₀` / `div_le_iff₀`, and the two exact rational
inequalities `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon` and
`(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)` decided by `norm_num`
(`Aux.ir_lower_rational`, `Aux.ir_upper_rational`).
Does not establish: the informal Section 4 display of the source as the source's object (the
alignment of `epsilon`, `exponentA`, `exponentB` and these power quotients with the source's
`epsilon`, `a`, `b` and integrand ratio is PENDING_INDEPENDENT_REVIEW); that the integrand ratio
of the source equals these power quotients (not specified in the statement packet); the
arithmetic `13 ≤ 32`, `28 ≤ 32` and `32*10^(-108) < 10^(-106)` (kernel-checked on main in
`UniversalLaw.Side24.ratio_constant_ordering` and `density_comparison_below_reported`, not
here); `DensityComparison` or any covariance ordering; any enclosure of the coefficient; the
statement module's own `Does not claim:` text (the statement packet is not edited here); the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register,
not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
  intro d hd
  rcases hd with rfl | rfl
  · rw [Aux.ir_exponentA_two, Aux.ir_exponentB_two]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rw [Aux.ir_exponentA_three, Aux.ir_exponentB_three]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ## The lattice-image sum bound for `d ≤ 3` -/

/-- Proves: `UniversalLaw.Spec.Side24.LatticeImageSumBound` (claim `side24-coefficient`).
Method: both conjuncts from the uniform partial-sum bound `Aux.li_partial_sum_le` via
`summable_of_sum_le` and `Real.tsum_le_of_sum_le` (terms nonnegative, `Aux.li_imageTerm_nonneg`);
the partial sums are dominated pointwise by `9 ∑ i, ∏ k, h_i k (n k)` (`Aux.li_imageTerm_le`,
from the power mean `pow_sum_div_card_le_sum_pow` and `Real.exp_sum`), each finite sum of
products sits in a `Fintype.piFinset` box and factors (`Aux.li_sum_prod_le`, via
`Finset.prod_univ_sum` and `Summable.sum_le_tsum`), and the one-dimensional sums over `ℤ` are
bounded by `2` and `4 exp(-288)` (`Aux.li_gauss_int`, `Aux.li_pow_gauss_int`) through
`tsum_of_nat_of_neg_add_one`, the geometric series `tsum_geometric_two` and the registered
`SuccessiveTermRatio` (`SpecificationProofs.Side24Elementary.successiveTermRatio`), giving
`432 exp(-288) ≤ 1458 exp(-288)`.
Does not establish: the informal Section 2 lattice display of the source as the source's object
(the alignment of `intNormSq`, `imageTerm` and this `tsum` over `Fin d → ℤ` with the source's
`sum_{n ≠ 0} |n|^6 e^(-288|n|^2)` is PENDING_INDEPENDENT_REVIEW); the intermediate shell count
`729 ∑_j j^9 e^(-288 j^2)` of the source, which the Prop does not specify and this proof does not
reproduce; the arithmetic `27·27 = 729`, `729·2 = 1458` kernel-checked on main, not here;
`ImageBound`, the derivative bounds of Section 2 or any enclosure of the coefficient; the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not
derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem latticeImageSumBound : UniversalLaw.Spec.Side24.LatticeImageSumBound := by
  intro d hd
  exact ⟨summable_of_sum_le (fun n => Aux.li_imageTerm_nonneg d n) (Aux.li_partial_sum_le d hd),
    Real.tsum_le_of_sum_le (fun n => Aux.li_imageTerm_nonneg d n) (Aux.li_partial_sum_le d hd)⟩

/-! ## The Section 4 Gaussian density comparison -/

/-- Proves: `UniversalLaw.Spec.Side24.DensityComparison` (claim `side24-coefficient`).
Method: with `A = (1 - epsilon) • C0` and `B = (1 + epsilon) • C0` (positive definite by
`Matrix.PosDef.smul`), the Loewner lemmas `Aux.dc_det_le_det` (`det A ≤ det C ≤ det B`, via
`CFC.sqrt`, `Matrix.PosSemidef.conjTranspose_mul_mul_same`,
`Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg` and
`Matrix.IsHermitian.det_eq_prod_eigenvalues`) and `Aux.dc_quad_le`
(`x ⬝ B⁻¹ x ≤ x ⬝ C⁻¹ x ≤ x ⬝ A⁻¹ x`, via `Aux.dc_inv_sub_inv_posSemidef`) give the two
determinant and exponent comparisons; `Aux.dc_scale_identity` with `Matrix.det_smul` turns
`((1 ∓ epsilon)/(1 ± epsilon))^(n/2) · det((1 ∓ epsilon) • C0)^(-1/2)` into
`det((1 ± epsilon) • C0)^(-1/2)`; `Real.rpow_le_rpow_of_nonpos`, `Real.exp_le_exp` and
`Aux.dc_assemble_lower` / `Aux.dc_assemble_upper` finish after unfolding `gaussDensity`.
Does not establish: the informal Section 4 display of the source as the source's object (the
alignment of `gaussDensity`, `symCoordDim`, `epsilon` and the positive-semidefinite hypotheses
with the source's densities, `n = m(m+1)/2`, `epsilon` and ordering is
PENDING_INDEPENDENT_REVIEW); the cone-moment consequence `(1 ± epsilon)^m D0` and the scaling
`a^m` of `h(A) = det(A)^2 1{A<0}` (not specified in the statement packet); that
`(1 ± epsilon) • C0` or `C` are covariances of any random object; `CovarianceOrdering` or any
ordering of `C24` against `Cref`; any enclosure of the coefficient; the statement module's own
`Does not claim:` text (the statement packet is not edited here); the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here;
nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem densityComparison : UniversalLaw.Spec.Side24.DensityComparison := by
  intro m C0 C hC0 hC hlow hup x
  have ha : (0 : ℝ) < 1 - epsilon := Aux.ir_one_sub_epsilon_pos
  have hb : (0 : ℝ) < 1 + epsilon := Aux.dc_one_add_epsilon_pos
  have hA : ((1 - epsilon) • C0).PosDef := hC0.smul ha
  have hB : ((1 + epsilon) • C0).PosDef := hC0.smul hb
  have hD : 0 < C0.det := hC0.det_pos
  have hdetC : 0 < C.det := hC.det_pos
  have hdetApos : 0 < ((1 - epsilon) • C0).det := hA.det_pos
  have hdetAC : ((1 - epsilon) • C0).det ≤ C.det := Aux.dc_det_le_det hA hlow
  have hdetCB : C.det ≤ ((1 + epsilon) • C0).det := Aux.dc_det_le_det hC hup
  have hqAC : dotProduct x (Matrix.mulVec C⁻¹ x) ≤
      dotProduct x (Matrix.mulVec ((1 - epsilon) • C0)⁻¹ x) := Aux.dc_quad_le hA hC hlow x
  have hqCB : dotProduct x (Matrix.mulVec ((1 + epsilon) • C0)⁻¹ x) ≤
      dotProduct x (Matrix.mulVec C⁻¹ x) := Aux.dc_quad_le hC hB hup x
  have hdetA : ((1 - epsilon) • C0).det = (1 - epsilon) ^ symCoordDim m * C0.det :=
    Aux.dc_det_smul C0 _
  have hdetB : ((1 + epsilon) • C0).det = (1 + epsilon) ^ symCoordDim m * C0.det :=
    Aux.dc_det_smul C0 _
  have hscale1 : ((1 - epsilon) / (1 + epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
      ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) = ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) := by
    rw [hdetA, hdetB]
    exact Aux.dc_scale_identity _ _ _ ha hb hD _
  have hscale2 : ((1 + epsilon) / (1 - epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
      ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) = ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) := by
    rw [hdetA, hdetB]
    exact Aux.dc_scale_identity _ _ _ hb ha hD _
  have hP : (0 : ℝ) ≤ (2 * Real.pi) ^ (-(symCoordDim m : ℝ) / 2) :=
    (Real.rpow_pos_of_pos (by positivity) _).le
  have hrB : ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) ≤ C.det ^ (-(1 : ℝ) / 2) :=
    Real.rpow_le_rpow_of_nonpos hdetC hdetCB (by norm_num)
  have hrA : C.det ^ (-(1 : ℝ) / 2) ≤ ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) :=
    Real.rpow_le_rpow_of_nonpos hdetApos hdetAC (by norm_num)
  have heA : Real.exp (-(dotProduct x (Matrix.mulVec ((1 - epsilon) • C0)⁻¹ x)) / 2) ≤
      Real.exp (-(dotProduct x (Matrix.mulVec C⁻¹ x)) / 2) :=
    Real.exp_le_exp.mpr (by linarith)
  have heB : Real.exp (-(dotProduct x (Matrix.mulVec C⁻¹ x)) / 2) ≤
      Real.exp (-(dotProduct x (Matrix.mulVec ((1 + epsilon) • C0)⁻¹ x)) / 2) :=
    Real.exp_le_exp.mpr (by linarith)
  unfold gaussDensity
  constructor
  · exact Aux.dc_assemble_lower _ _ _ _ _ _ _ hP hscale1 hrB (Real.rpow_nonneg hdetC.le _) heA
      (Real.exp_pos _).le
  · exact Aux.dc_assemble_upper _ _ _ _ _ _ _ hP hscale2 hrA (Real.rpow_nonneg hdetApos.le _) heB
      (Real.exp_pos _).le

end UniversalLaw.Proofs.Side24Analysis
