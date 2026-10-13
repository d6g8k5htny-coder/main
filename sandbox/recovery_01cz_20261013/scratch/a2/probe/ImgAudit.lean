import Specifications.Side24
import SpecificationProofs.Side24Elementary
import SpecificationProofs.Side24Analysis

/-!
# SIDE24 lattice-image bound (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean`, `ImageBound`. A kernel-checked proof of it is evidence about that
`Prop` only: it does not make the informal note reviewed or accepted, it does not identify the
formal objects with the informal ones (alignment is PENDING_INDEPENDENT_REVIEW), it does not
enclose the coefficient `c_{d,24}`, and it promotes, closes, discharges or reclassifies nothing.

Method. Mathlib's `iteratedFDeriv_tsum` needs summable bounds on the derivatives of the summands
uniform over the whole space, which the lattice translates `z ↦ Kinf d (z + 24 n)` do not have;
`ib_iteratedFDeriv_tsum_of_isOpen` is its local version on an open preconnected set, proved by
the same induction from `hasFDerivAt_tsum_of_isPreconnected`. On the unit ball the translates
have summable bounds (the operator-norm pairing bound `dp_pairing_all`, `pairingSum q r ≤
76 (1 + r)^6`, `(1 + r)^6 ≤ 720 e^(1 + r)`, `r - r^2/4 ≤ 1` and the product form of the Gaussian
over `Fin d → ℤ`). So `D^q K_24(0) = θ(0)⁻¹ ∑_n D^q K_∞(24 n)`, and the difference with
`D^q K_∞(0)` is `θ(0)⁻¹ (∑_{n ≠ 0} D^q K_∞(24 n) - (θ(0) - 1) D^q K_∞(0))`, bounded through the
registered `DerivativeBoundAtLeastOne` (`76`), `DerivativeBoundAtZero` (`15`),
`LatticeImageSumBound` (`1458 e^(-288)`) and `ExponentialBounds` (`e^(-288) < 10^(-125)`), all
with theorems in this packet, by `1458 (76·24^6 + 15) e^(-288) < imageConstE`.
-/

namespace UniversalLaw.Proofs.Side24Images

open UniversalLaw.Spec.Side24
open scoped ContDiff Topology

namespace Aux

/-! ## `ib_`: helpers for `ImageBound` -/

/-- Local version of Mathlib's `iteratedFDeriv_tsum`: on an open preconnected set `U`, summable
bounds on the successive derivatives over `U` give the term-wise iterated derivative at every
point of `U`. Same induction as Mathlib's global lemma, with the local
`hasFDerivAt_tsum_of_isPreconnected` and `Filter.EventuallyEq.fderiv_eq` on `U`. -/
theorem ib_iteratedFDeriv_tsum_of_isOpen {α E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [CompleteSpace F] [NormedSpace ℝ F]
    {f : α → E → F} {v : ℕ → α → ℝ} {N : ℕ∞} {U : Set E}
    (hU : IsOpen U) (hU' : IsPreconnected U)
    (hf : ∀ i, ContDiff ℝ N (f i))
    (hv : ∀ k : ℕ, (k : ℕ∞) ≤ N → Summable (v k))
    (h'f : ∀ (k : ℕ) (i : α) (x : E), (k : ℕ∞) ≤ N → x ∈ U →
      ‖iteratedFDeriv ℝ k (f i) x‖ ≤ v k i)
    {k : ℕ} (hk : (k : ℕ∞) ≤ N) {x : E} (hx : x ∈ U) :
    iteratedFDeriv ℝ k (fun y => ∑' n, f n y) x = ∑' n, iteratedFDeriv ℝ k (f n) x := by
  induction k generalizing x with
  | zero =>
    simp_rw [iteratedFDeriv_zero_eq_comp]
    exact (continuousMultilinearCurryFin0 ℝ E F).symm.toContinuousLinearEquiv.map_tsum
  | succ k IH =>
    have h'k : (k : ℕ∞) < N := lt_of_lt_of_le (WithTop.coe_lt_coe.2 (Nat.lt_succ_self _)) hk
    have hEq : (iteratedFDeriv ℝ k fun y => ∑' n, f n y) =ᶠ[𝓝 x]
        fun y => ∑' n, iteratedFDeriv ℝ k (f n) y :=
      Filter.eventually_of_mem (hU.mem_nhds hx) fun y hy => IH h'k.le hy
    have hD : HasFDerivAt (fun y => ∑' n, iteratedFDeriv ℝ k (f n) y)
        (∑' n, fderiv ℝ (iteratedFDeriv ℝ k (f n)) x) x := by
      refine hasFDerivAt_tsum_of_isPreconnected (hv (k + 1) hk) hU hU' (fun n y _ => ?_)
        (fun n y hy => ?_) hx ?_ hx
      · exact ((hf n).differentiable_iteratedFDeriv (mod_cast h'k) y).hasFDerivAt
      · rw [norm_fderiv_iteratedFDeriv]
        exact h'f (k + 1) n y hk hy
      · exact .of_norm_bounded (hv k h'k.le) fun n => h'f k n x h'k.le hx
    rw [iteratedFDeriv_succ_eq_comp_left, Function.comp_apply, hEq.fderiv_eq, hD.fderiv]
    simp_rw [iteratedFDeriv_succ_eq_comp_left, Function.comp_apply]
    exact (continuousMultilinearCurryLeftEquiv ℝ
      (fun _ : Fin (k + 1) => E) F).symm.toContinuousLinearEquiv.map_tsum

/-- The pairing sums are monotone in `r ≥ 0`: every term is a nonnegative multiple of `r^m`. -/
theorem ib_pairingSum_mono (q : ℕ) {r s : ℝ} (hr : 0 ≤ r) (hrs : r ≤ s) :
    pairingSum q r ≤ pairingSum q s := by
  unfold pairingSum
  refine Finset.sum_le_sum fun j _ => ?_
  have hden : 0 ≤ (2 : ℝ) ^ j * (Nat.factorial j : ℝ) * (Nat.factorial (q - 2 * j) : ℝ) := by
    positivity
  exact div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hr hrs _) (by positivity)) hden

/-- For `q ≤ 6` and `r ≥ 0`, `pairingSum q r ≤ 76 (1 + r)^6`. -/
theorem ib_pairingSum_le (q : ℕ) (hq : q ≤ 6) {r : ℝ} (hr : 0 ≤ r) :
    pairingSum q r ≤ 76 * (1 + r) ^ 6 := by
  have h1 : (1 : ℝ) ≤ 1 + r := by linarith
  calc pairingSum q r ≤ pairingSum q (1 + r) := ib_pairingSum_mono q hr (by linarith)
    _ ≤ 76 * (1 + r) ^ 6 := Side24Analysis.Aux.d1_pairingSum_le_of_one_le q hq _ h1

/-- `(1 + r)^6 e^(-r^2/2) ≤ 720 e^2 e^(-r^2/4)` for `r ≥ 0`: `(1 + r)^6 ≤ 6! e^(1 + r)` and
`1 + r - r^2/2 ≤ 2 - r^2/4`, that is `(r - 2)^2 ≥ 0`. -/
theorem ib_poly_gauss_le {r : ℝ} (hr : 0 ≤ r) :
    (1 + r) ^ 6 * Real.exp (-(r ^ 2) / 2) ≤ 720 * Real.exp 2 * Real.exp (-(r ^ 2) / 4) := by
  have h1 : (1 + r) ^ 6 / (Nat.factorial 6 : ℝ) ≤ Real.exp (1 + r) :=
    Real.pow_div_factorial_le_exp (1 + r) (by linarith) 6
  have h6 : (Nat.factorial 6 : ℝ) = 720 := by norm_num [Nat.factorial]
  rw [h6, div_le_iff₀ (by norm_num)] at h1
  have h2 : Real.exp (1 + r) * Real.exp (-(r ^ 2) / 2) ≤ Real.exp 2 * Real.exp (-(r ^ 2) / 4) := by
    rw [← Real.exp_add, ← Real.exp_add, Real.exp_le_exp]
    nlinarith [sq_nonneg (r - 2)]
  calc (1 + r) ^ 6 * Real.exp (-(r ^ 2) / 2)
      ≤ (Real.exp (1 + r) * 720) * Real.exp (-(r ^ 2) / 2) :=
        mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
    _ = 720 * (Real.exp (1 + r) * Real.exp (-(r ^ 2) / 2)) := by ring
    _ ≤ 720 * (Real.exp 2 * Real.exp (-(r ^ 2) / 4)) := by gcongr
    _ = 720 * Real.exp 2 * Real.exp (-(r ^ 2) / 4) := by ring

/-- `exp(-‖z + v‖^2/4) ≤ exp(‖z‖^2/4) exp(-‖v‖^2/8)`, from `‖v‖ ≤ ‖z + v‖ + ‖z‖`. -/
theorem ib_exp_shift_le {d : ℕ} (z v : E d) :
    Real.exp (-(‖z + v‖ ^ 2) / 4) ≤ Real.exp (‖z‖ ^ 2 / 4) * Real.exp (-(‖v‖ ^ 2) / 8) := by
  rw [← Real.exp_add, Real.exp_le_exp]
  have h1 : ‖v‖ ≤ ‖z + v‖ + ‖z‖ := by
    calc ‖v‖ = ‖(z + v) - z‖ := by rw [add_sub_cancel_left]
      _ ≤ ‖z + v‖ + ‖z‖ := norm_sub_le _ _
  have h2 : 0 ≤ ‖z + v‖ := norm_nonneg _
  have h3 : 0 ≤ ‖z‖ := norm_nonneg _
  have h4 : 0 ≤ ‖v‖ := norm_nonneg _
  nlinarith [sq_nonneg (‖z + v‖ - ‖z‖)]

/-- `exp(-‖24 n‖^2/8) = ∏ i, exp(-72 n_i^2)`. -/
theorem ib_exp_lattice_eighth (d : ℕ) (n : Fin d → ℤ) :
    Real.exp (-(‖latticeVec d n‖ ^ 2) / 8) = ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2) := by
  rw [← Real.exp_sum, Side24Analysis.Aux.th_norm_latticeVec_sq, intNormSq, Finset.mul_sum, neg_div,
    Finset.sum_div, ← Finset.sum_neg_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- The constant of the summable bound. -/
theorem ib_const_pos : 0 < 76 * (720 * Real.exp 2) * Real.exp (1 / 4) := by positivity

/-- On the unit ball, every derivative of order `k ≤ 6` of a lattice translate of `Kinf d` is
bounded by the summable family `C ∏ i, exp(-72 n_i^2)`. -/
theorem ib_translate_bound {d : ℕ} (k : ℕ) (hk : k ≤ 6) (n : Fin d → ℤ) {z : E d}
    (hz : z ∈ Metric.ball (0 : E d) 1) :
    ‖iteratedFDeriv ℝ k (Kinf d) (z + latticeVec d n)‖ ≤
      76 * (720 * Real.exp 2) * Real.exp (1 / 4) * ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2) := by
  set y := z + latticeVec d n
  have hr : 0 ≤ ‖y‖ := norm_nonneg _
  have hz1 : ‖z‖ < 1 := by simpa using hz
  have hb := Side24Analysis.Aux.dp_pairing_all d y k hk
  have hp := ib_pairingSum_le k hk hr
  have hg := ib_poly_gauss_le hr
  have hs := ib_exp_shift_le z (latticeVec d n)
  have hK : Kinf d y = Real.exp (-(‖y‖ ^ 2) / 2) := rfl
  have hzz : Real.exp (‖z‖ ^ 2 / 4) ≤ Real.exp (1 / 4) := by
    rw [Real.exp_le_exp]
    have : ‖z‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg z]
    linarith
  rw [ib_exp_lattice_eighth] at hs
  calc ‖iteratedFDeriv ℝ k (Kinf d) y‖ ≤ Kinf d y * pairingSum k ‖y‖ := hb
    _ ≤ Real.exp (-(‖y‖ ^ 2) / 2) * (76 * (1 + ‖y‖) ^ 6) := by
        rw [hK]; exact mul_le_mul_of_nonneg_left hp (Real.exp_pos _).le
    _ = 76 * ((1 + ‖y‖) ^ 6 * Real.exp (-(‖y‖ ^ 2) / 2)) := by ring
    _ ≤ 76 * (720 * Real.exp 2 * Real.exp (-(‖y‖ ^ 2) / 4)) := by gcongr
    _ ≤ 76 * (720 * Real.exp 2 * (Real.exp (‖z‖ ^ 2 / 4) *
          ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2))) := by gcongr
    _ ≤ 76 * (720 * Real.exp 2 * (Real.exp (1 / 4) *
          ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2))) := by
        gcongr
    _ = 76 * (720 * Real.exp 2) * Real.exp (1 / 4) *
          ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2) := by ring

/-- `latticeVec d 0 = 0`. -/
theorem ib_latticeVec_zero (d : ℕ) : latticeVec d 0 = 0 := by
  ext i
  simp [latticeVec]

/-- `θ(0) = ∑_n exp(-‖24 n‖^2/2) ≥ 1`: the summand at `n = 0` is `1` and the others are positive. -/
theorem ib_theta0_ge_one (d : ℕ) : 1 ≤ thetaSum24 d 0 := by
  have hs := Side24Analysis.thetaSummable d 0
  have h := hs.le_tsum 0 (fun j _ => (Real.exp_pos _).le)
  simpa [thetaSum24, ib_latticeVec_zero] using h

/-- `K_24` as a sum of scaled lattice translates of `Kinf d`. -/
theorem ib_K24_eq (d : ℕ) :
    K24 d = fun z => ∑' n : Fin d → ℤ, (thetaSum24 d 0)⁻¹ • Kinf d (z + latticeVec d n) := by
  funext z
  simp only [smul_eq_mul]
  rw [tsum_mul_left, K24, div_eq_inv_mul]
  rfl

/-- Each scaled translate is `C^∞`. -/
theorem ib_contDiff_term (d : ℕ) (c : ℝ) (n : Fin d → ℤ) :
    ContDiff ℝ ∞ (fun z : E d => c • Kinf d (z + latticeVec d n)) :=
  ((Side24Analysis.Aux.dp_contDiff_Kinf d).comp (contDiff_id.add contDiff_const)).const_smul c

/-- The iterated derivative of a scaled translate. -/
theorem ib_iteratedFDeriv_term (d k : ℕ) (c : ℝ) (n : Fin d → ℤ) (z : E d) :
    iteratedFDeriv ℝ k (fun z : E d => c • Kinf d (z + latticeVec d n)) z =
      c • iteratedFDeriv ℝ k (Kinf d) (z + latticeVec d n) := by
  have hg : ContDiff ℝ ∞ (fun z : E d => Kinf d (z + latticeVec d n)) :=
    (Side24Analysis.Aux.dp_contDiff_Kinf d).comp (contDiff_id.add contDiff_const)
  have h := iteratedFDeriv_const_smul_apply (a := c) (i := k) (x := z)
    ((hg.of_le (by exact_mod_cast le_top)).contDiffAt)
  have he : (fun z : E d => c • Kinf d (z + latticeVec d n)) =
      c • (fun z : E d => Kinf d (z + latticeVec d n)) := rfl
  rw [he, h, iteratedFDeriv_comp_add_right]

/-- The summable bound family is summable. -/
theorem ib_summable_bound (d : ℕ) :
    Summable (fun n : Fin d → ℤ =>
      76 * (720 * Real.exp 2) * Real.exp (1 / 4) * ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2)) :=
  (Side24Analysis.Aux.th_summable_pi_prod d _ (fun m => (Real.exp_pos _).le)
    (Side24Analysis.Aux.th_summable_gauss_int 72 (by norm_num))).mul_left _

/-- The derivative bound of a scaled translate on the unit ball, for a scale `0 ≤ c ≤ 1`. -/
theorem ib_term_bound {d : ℕ} {c : ℝ} (hc0 : 0 ≤ c) (hc1 : c ≤ 1) (k : ℕ) (hk : k ≤ 6)
    (n : Fin d → ℤ) {z : E d} (hz : z ∈ Metric.ball (0 : E d) 1) :
    ‖iteratedFDeriv ℝ k (fun z : E d => c • Kinf d (z + latticeVec d n)) z‖ ≤
      76 * (720 * Real.exp 2) * Real.exp (1 / 4) * ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2) := by
  rw [ib_iteratedFDeriv_term, norm_smul, Real.norm_eq_abs, abs_of_nonneg hc0]
  calc c * ‖iteratedFDeriv ℝ k (Kinf d) (z + latticeVec d n)‖
      ≤ 1 * ‖iteratedFDeriv ℝ k (Kinf d) (z + latticeVec d n)‖ :=
        mul_le_mul_of_nonneg_right hc1 (norm_nonneg _)
    _ = ‖iteratedFDeriv ℝ k (Kinf d) (z + latticeVec d n)‖ := one_mul _
    _ ≤ _ := ib_translate_bound k hk n hz

/-- The term-wise iterated derivative of `K_24` at `0`, as continuous multilinear maps. -/
theorem ib_K24_deriv (d q : ℕ) (hq : q ≤ 6) :
    iteratedFDeriv ℝ q (K24 d) 0 =
      ∑' n : Fin d → ℤ, (thetaSum24 d 0)⁻¹ • iteratedFDeriv ℝ q (Kinf d) (latticeVec d n) := by
  have hΘ := ib_theta0_ge_one d
  have hc0 : 0 ≤ (thetaSum24 d 0)⁻¹ := inv_nonneg.2 (by linarith)
  have hc1 : (thetaSum24 d 0)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hΘ
  have h6 : ((6 : ℕ∞) : WithTop ℕ∞) ≤ ∞ := WithTop.coe_le_coe.2 le_top
  have hq' : ((q : ℕ) : ℕ∞) ≤ 6 := by exact_mod_cast hq
  have key := ib_iteratedFDeriv_tsum_of_isOpen (α := Fin d → ℤ) (E := E d) (F := ℝ)
    (f := fun n z => (thetaSum24 d 0)⁻¹ • Kinf d (z + latticeVec d n)) (N := 6)
    (U := Metric.ball (0 : E d) 1)
    (v := fun _ n => 76 * (720 * Real.exp 2) * Real.exp (1 / 4) *
      ∏ i, Real.exp (-72 * ((n i : ℤ) : ℝ) ^ 2))
    Metric.isOpen_ball (convex_ball (0 : E d) 1).isPreconnected
    (fun n => (ib_contDiff_term d _ n).of_le h6)
    (fun k _ => ib_summable_bound d)
    (fun k n z hk hz => ib_term_bound hc0 hc1 k (by exact_mod_cast hk) n hz)
    hq' (Metric.mem_ball_self one_pos)
  rw [ib_K24_eq, key]
  refine tsum_congr fun n => ?_
  rw [ib_iteratedFDeriv_term, zero_add]

/-- Nonzero integer vectors have `intNormSq d n ≥ 1`. -/
theorem ib_intNormSq_ge_one {d : ℕ} {n : Fin d → ℤ} (hn : n ≠ 0) : 1 ≤ intNormSq d n := by
  obtain ⟨i, hi⟩ := Function.ne_iff.1 hn
  have h1 : (1 : ℝ) ≤ ((n i : ℤ) : ℝ) ^ 2 := by
    have : (1 : ℤ) ≤ |n i| := Int.one_le_abs hi
    have h' : (1 : ℝ) ≤ |((n i : ℤ) : ℝ)| := by exact_mod_cast this
    nlinarith [abs_nonneg ((n i : ℤ) : ℝ), sq_abs ((n i : ℤ) : ℝ)]
  unfold intNormSq
  calc (1 : ℝ) ≤ ((n i : ℤ) : ℝ) ^ 2 := h1
    _ ≤ ∑ j, ((n j : ℤ) : ℝ) ^ 2 :=
        Finset.single_le_sum (f := fun j => ((n j : ℤ) : ℝ) ^ 2) (fun j _ => sq_nonneg _)
          (Finset.mem_univ i)

/-- For `n ≠ 0`: `‖24 n‖^6 exp(-‖24 n‖^2/2) = 24^6 imageTerm d n`. -/
theorem ib_lattice_term (d : ℕ) {n : Fin d → ℤ} (hn : n ≠ 0) :
    ‖latticeVec d n‖ ^ 6 * Real.exp (-(‖latticeVec d n‖ ^ 2) / 2) = 24 ^ 6 * imageTerm d n := by
  have h2 := Side24Analysis.Aux.th_norm_latticeVec_sq d n
  have h6 : ‖latticeVec d n‖ ^ 6 = (‖latticeVec d n‖ ^ 2) ^ 3 := by ring
  rw [h6, h2, imageTerm, ite_eq_right hn]
  have he : -(576 * intNormSq d n) / 2 = -288 * intNormSq d n := by ring
  rw [he]
  ring

/-- For `n ≠ 0`: `exp(-‖24 n‖^2/2) ≤ imageTerm d n` (the factor `|n|^6 ≥ 1`). -/
theorem ib_gauss_le_image (d : ℕ) {n : Fin d → ℤ} (hn : n ≠ 0) :
    Real.exp (-(‖(0 : E d) + latticeVec d n‖ ^ 2) / 2) ≤ imageTerm d n := by
  have h1 := ib_intNormSq_ge_one hn
  have h2 := Side24Analysis.Aux.th_norm_latticeVec_sq d n
  rw [zero_add, h2, imageTerm, ite_eq_right hn]
  have he : -(576 * intNormSq d n) / 2 = -288 * intNormSq d n := by ring
  rw [he]
  have h3 : (1 : ℝ) ≤ intNormSq d n ^ 3 := one_le_pow₀ h1
  calc Real.exp (-288 * intNormSq d n) = 1 * Real.exp (-288 * intNormSq d n) := (one_mul _).symm
    _ ≤ intNormSq d n ^ 3 * Real.exp (-288 * intNormSq d n) :=
        mul_le_mul_of_nonneg_right h3 (Real.exp_pos _).le

/-- The image bound's constant: `1458 (76 · 24^6 + 15) e^(-288) < imageConstE`. -/
theorem ib_final_constant :
    1458 * (76 * 24 ^ 6 + 15) * Real.exp (-288) < imageConstE := by
  have h := (UniversalLaw.Proofs.Side24Elementary.exponentialBounds).2
  have hpos : (0 : ℝ) < 1458 * (76 * 24 ^ 6 + 15) := by norm_num
  calc 1458 * (76 * 24 ^ 6 + 15) * Real.exp (-288)
      < 1458 * (76 * 24 ^ 6 + 15) * (1 / 10 ^ 125) := mul_lt_mul_of_pos_left h hpos
    _ = imageConstE := by norm_num [imageConstE]

end Aux

open Aux in
/-- Proves: `ImageBound`, for `d ∈ {2, 3}` and every unit tuple of order `q ≤ 6`,
`|D^q K_24(0) u - D^q K_∞(0) u| < imageConstE`.
Method: `D^q K_24(0) = θ(0)⁻¹ ∑_n D^q K_∞(24 n)` by the local term-wise derivative on the unit
ball; splitting off `n = 0` in both sums gives `θ(0)⁻¹ (∑_{n ≠ 0} D^q K_∞(24 n) u - (θ(0) - 1)
D^q K_∞(0) u)`, bounded by `76 · 24^6 · 1458 e^(-288) + 15 · 1458 e^(-288)` through
`DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero` and `LatticeImageSumBound`, and
`1458 (76 · 24^6 + 15) e^(-288) < imageConstE` through `ExponentialBounds`.
Does not establish: any enclosure of the coefficient `c_{d,24}`; any bound for `q > 6` or
`d ∉ {2, 3}`; anything about the informal note (alignment PENDING_INDEPENDENT_REVIEW); the
hypothesis `ThetaSummable d` is not used (the packet proves it for every `d`). -/
theorem imageBound : UniversalLaw.Spec.Side24.ImageBound := by
  intro d hd _ q hq u hu
  have hd3 : d ≤ 3 := by rcases hd with rfl | rfl <;> norm_num
  set Θ := thetaSum24 d 0 with hΘdef
  have hΘ : 1 ≤ Θ := ib_theta0_ge_one d
  have hΘpos : 0 < Θ := by linarith
  -- the evaluation at `u` and the term-wise derivative
  set A : (Fin d → ℤ) → ContinuousMultilinearMap ℝ (fun _ : Fin q => E d) ℝ :=
    fun n => iteratedFDeriv ℝ q (Kinf d) (latticeVec d n) with hAdef
  set a : (Fin d → ℤ) → ℝ := fun n => A n u with hadef
  have hAs : Summable (fun n => Θ⁻¹ • A n) := by
    refine .of_norm_bounded (ib_summable_bound d) fun n => ?_
    have h := ib_term_bound (inv_nonneg.2 hΘpos.le) (inv_le_one_of_one_le₀ hΘ) q hq n
      (Metric.mem_ball_self one_pos)
    rwa [ib_iteratedFDeriv_term, zero_add] at h
  have hK : iteratedFDeriv ℝ q (K24 d) 0 u = Θ⁻¹ * ∑' n, a n := by
    rw [ib_K24_deriv d q hq]
    have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin q => E d) ℝ u).map_tsum hAs
    simp only [ContinuousMultilinearMap.apply_apply] at h
    rw [h, show (∑' n, (Θ⁻¹ • A n) u) = ∑' n, Θ⁻¹ * a n from tsum_congr fun n => rfl,
      tsum_mul_left]
  have has : Summable a := by
    have h := (hAs.mapL (ContinuousMultilinearMap.apply ℝ (fun _ : Fin q => E d) ℝ u)).mul_left Θ
    refine h.congr fun n => ?_
    simp [hadef, hAdef, smul_eq_mul, ← mul_assoc, mul_inv_cancel₀ hΘpos.ne']
  -- the theta family
  set g : (Fin d → ℤ) → ℝ := fun n => Real.exp (-(‖(0 : E d) + latticeVec d n‖ ^ 2) / 2) with hgdef
  have hgs : Summable g := Side24Analysis.thetaSummable d 0
  have hΘg : Θ = ∑' n, g n := rfl
  -- split off `n = 0`
  have ha0 : a 0 = iteratedFDeriv ℝ q (Kinf d) 0 u := by
    simp [hadef, hAdef, ib_latticeVec_zero]
  have hg0 : g 0 = 1 := by simp [hgdef, ib_latticeVec_zero]
  have hsplit_a := has.tsum_eq_add_tsum_ite 0
  have hsplit_g := hgs.tsum_eq_add_tsum_ite 0
  set A' := ∑' n, (if n = 0 then 0 else a n) with hA'def
  set G' := ∑' n, (if n = 0 then 0 else g n) with hG'def
  -- the lattice-image family
  obtain ⟨hTs, hT⟩ := Side24Analysis.latticeImageSumBound d hd3
  set T := ∑' n, imageTerm d n with hTdef
  have hT0 : 0 ≤ T := tsum_nonneg fun n => by
    unfold imageTerm; split_ifs
    · exact le_rfl
    · exact mul_nonneg (pow_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) 3) (Real.exp_pos _).le
  have himg0 : ∀ n, 0 ≤ imageTerm d n := fun n => by
    unfold imageTerm; split_ifs
    · exact le_rfl
    · exact mul_nonneg (pow_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) 3) (Real.exp_pos _).le
  -- |A'| ≤ 76 · 24^6 · T
  have hbound_a : ∀ n, ‖(if n = 0 then 0 else a n)‖ ≤ 76 * 24 ^ 6 * imageTerm d n := fun n => by
    split_ifs with hn
    · simp only [norm_zero]; exact mul_nonneg (by norm_num) (himg0 n)
    · have h1 : 1 ≤ ‖latticeVec d n‖ := by
        have h2 := Side24Analysis.Aux.th_norm_latticeVec_sq d n
        have h3 := ib_intNormSq_ge_one hn
        nlinarith [norm_nonneg (latticeVec d n)]
      have h := Side24Analysis.derivativeBoundAtLeastOne d q hq (latticeVec d n) h1 u hu
      rw [Real.norm_eq_abs]
      calc |a n| ≤ 76 * ‖latticeVec d n‖ ^ 6 * Real.exp (-(‖latticeVec d n‖ ^ 2) / 2) := h
        _ = 76 * (‖latticeVec d n‖ ^ 6 * Real.exp (-(‖latticeVec d n‖ ^ 2) / 2)) := by ring
        _ = 76 * (24 ^ 6 * imageTerm d n) := by rw [ib_lattice_term d hn]
        _ = 76 * 24 ^ 6 * imageTerm d n := by ring
  have hA' : |A'| ≤ 76 * 24 ^ 6 * T := by
    have hs : Summable (fun n => ‖(if n = 0 then 0 else a n)‖) :=
      .of_nonneg_of_le (fun n => norm_nonneg _) hbound_a (hTs.mul_left _)
    calc |A'| = ‖A'‖ := (Real.norm_eq_abs _).symm
      _ ≤ ∑' n, ‖(if n = 0 then 0 else a n)‖ := norm_tsum_le_tsum_norm hs
      _ ≤ ∑' n, 76 * 24 ^ 6 * imageTerm d n := hs.tsum_le_tsum hbound_a (hTs.mul_left _)
      _ = 76 * 24 ^ 6 * T := tsum_mul_left
  -- 0 ≤ G' ≤ T
  have hbound_g : ∀ n, (if n = 0 then 0 else g n) ≤ imageTerm d n := fun n => by
    split_ifs with hn
    · exact himg0 n
    · exact ib_gauss_le_image d hn
  have hG'0 : 0 ≤ G' := tsum_nonneg fun n => by
    split_ifs
    · exact le_rfl
    · exact (Real.exp_pos _).le
  have hG' : G' ≤ T := by
    have hs : Summable (fun n => (if n = 0 then 0 else g n)) :=
      .of_nonneg_of_le (fun n => by split_ifs; exacts [le_rfl, (Real.exp_pos _).le]) hbound_g hTs
    exact hs.tsum_le_tsum hbound_g hTs
  have ha0b : |a 0| ≤ 15 := by
    rw [ha0]; exact Side24Analysis.derivativeBoundAtZero d q hq u hu
  -- assemble
  have hdiff : iteratedFDeriv ℝ q (K24 d) 0 u - iteratedFDeriv ℝ q (Kinf d) 0 u =
      Θ⁻¹ * (A' - G' * a 0) := by
    rw [hK, ← ha0, hsplit_a]
    have hΘ' : Θ = 1 + G' := by rw [hΘg, hsplit_g, hg0]
    field_simp
    rw [hΘ']
    ring
  rw [hdiff, abs_mul, abs_of_pos (inv_pos.2 hΘpos)]
  have hinv : Θ⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hΘ
  have hnum : |A' - G' * a 0| ≤ 76 * 24 ^ 6 * T + T * 15 := by
    calc |A' - G' * a 0| ≤ |A'| + |G' * a 0| := abs_sub _ _
      _ = |A'| + G' * |a 0| := by rw [abs_mul, abs_of_nonneg hG'0]
      _ ≤ 76 * 24 ^ 6 * T + T * 15 := by
          gcongr
  have hTb : 76 * 24 ^ 6 * T + T * 15 ≤ 1458 * (76 * 24 ^ 6 + 15) * Real.exp (-288) := by
    nlinarith [hT, hT0]
  calc Θ⁻¹ * |A' - G' * a 0| ≤ 1 * |A' - G' * a 0| :=
        mul_le_mul_of_nonneg_right hinv (abs_nonneg _)
    _ = |A' - G' * a 0| := one_mul _
    _ ≤ 1458 * (76 * 24 ^ 6 + 15) * Real.exp (-288) := hnum.trans hTb
    _ < imageConstE := ib_final_constant

end UniversalLaw.Proofs.Side24Images

#print axioms UniversalLaw.Proofs.Side24Images.imageBound
#print axioms UniversalLaw.Proofs.Side24Images.Aux.ib_iteratedFDeriv_tsum_of_isOpen
example : UniversalLaw.Spec.Side24.ImageBound := UniversalLaw.Proofs.Side24Images.imageBound
#check (UniversalLaw.Proofs.Side24Images.imageBound : UniversalLaw.Spec.Side24.ImageBound)
