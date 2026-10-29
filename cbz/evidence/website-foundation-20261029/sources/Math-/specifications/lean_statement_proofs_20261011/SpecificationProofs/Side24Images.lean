import Specifications.Side24
import SpecificationProofs.Side24Elementary
import SpecificationProofs.Side24Analysis
import SpecificationProofs.Side24Covariance

/-!
# SIDE24 lattice images and the covariance ordering (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Each of the three theorems outside the `Aux` namespace has as its type exactly one registered
`Prop` of `Specifications/Side24.lean`: `ImageBound` (display (2)), `EntrywiseCovarianceBound`
(the Section 3 entrywise sentence) and `CovarianceOrdering` (display (3)). A kernel-checked proof
of one of them is evidence about that `Prop`, for the formal objects of the statement module,
only: it does not make the informal note reviewed or accepted, it does not identify the formal
objects with the informal ones (alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the
coefficient `c_{d,24}`, and it promotes, closes, discharges or reclassifies nothing.

Method. Mathlib's `iteratedFDeriv_tsum` needs summable bounds on the derivatives of the summands
uniform over the whole space, which the lattice translates `z ↦ Kinf d (z + 24 n)` do not have;
`Aux.ib_iteratedFDeriv_tsum_of_isOpen` is its local version on an open preconnected set, proved by
the same induction from `hasFDerivAt_tsum_of_isPreconnected` (present at the packet's Mathlib pin
`d13f23b7`; first order). On the unit ball the translates have summable bounds (the operator-norm
pairing bound of round 4, `pairingSum q r ≤ 76 (1 + r)^6`, `(1 + r)^6 ≤ 720 e^(1 + r)`,
`r - r^2/4 ≤ 1`, the shift bound `exp(-‖z + v‖^2/4) ≤ exp(‖z‖^2/4) exp(-‖v‖^2/8)` and the product
form of the Gaussian over `Fin d → ℤ`). So, for `q ≤ 6`, `D^q K_24(0) = θ(0)⁻¹ ∑_n D^q K_∞(24 n)`,
and the difference with `D^q K_∞(0)` is `θ(0)⁻¹ (∑_{n ≠ 0} D^q K_∞(24 n) - (θ(0) - 1) D^q K_∞(0))`, bounded
through the packet's theorems of `DerivativeBoundAtLeastOne` (`76`), `DerivativeBoundAtZero`
(`15`), `LatticeImageSumBound` (`1458 e^(-288)`) and `ExponentialBounds` (`e^(-288) < 10^(-125)`)
by `1458 (76·24^6 + 15) e^(-288) < imageConstE`. The entrywise bound follows with the factor
`|svecScale a · svecScale b · (-1)^|b|| ≤ 2`. For the ordering, `C_24` is symmetric: a
contraction of `K_24` at `0` of total order at most `6` (each joint coordinate carries at most three
directions) is `θ(0)⁻¹ ∑_n` of contractions of the analytic `Kinf d` at `24 n`, which are invariant
under the block swap (`ContDiffAt.iteratedFDeriv_comp_perm`,
`Mathlib/Analysis/Analytic/IteratedFDeriv.lean`, present at Mathlib `d13f23b7`; it needs
`ContDiffAt ℝ ω`, i.e. analyticity), and odd total orders vanish because
`D^q Kinf(-y) = (-1)^q D^q Kinf(y)` (`Aux.co_kinf_neg_apply`, from the evenness of `Kinf d`, the
round-5 helper `Side24Covariance.Aux.cv_kinf_comp_neg`) and the lattice sum is invariant under
`n ↦ -n`; `C_ref` is symmetric with `C_ref - I/3 ≥ 0` by the round-5 theorem; Cauchy-Schwarz over
at most `13` coordinates gives `|x ⬝ (C_24 - C_ref) x| ≤ 26 imageConstE (x ⬝ x)`, and
`78 imageConstE ≤ epsilon`.

Provenance: the round-6 lane for `ImageBound`, `EntrywiseCovarianceBound` and `CovarianceOrdering`,
the three `Prop`s the round-4 triage (workflow `wf_a3531fd8-891`) classified BLOCKED_MISSING_MATHLIB;
that verdict is the historical record, transcribed in the register and not altered here. Imported,
not re-proved: the theorems `Side24Analysis.thetaSummable`, `derivativeBoundAtLeastOne`,
`derivativeBoundAtZero`, `latticeImageSumBound` (round 4), `Side24Elementary.exponentialBounds`
(round 1) and `Side24Covariance.referenceCovarianceLowerBound` (round 5); the round-4 helpers
`Side24Analysis.Aux.dp_pairing_all`, `d1_pairingSum_le_of_one_le`, `dp_contDiff_Kinf`,
`rb_kinf_contDiff`, `th_norm_latticeVec_sq`, `th_summable_pi_prod`, `th_summable_gauss_int`; the
round-5 helper `Side24Covariance.Aux.cv_kinf_comp_neg` (evenness of `Kinf d`). Helper lemmas live in
`UniversalLaw.Proofs.Side24Images.Aux`, tagged `ib_` (`ImageBound`), `ec_`
(`EntrywiseCovarianceBound`) and `co_` (`CovarianceOrdering`), with one auxiliary definition, the
block rotation `Aux.co_rot`.

Remark recorded by the lane, for the maintainers (nothing below is edited here): the proof-packet
README rows "Side24 lattice sums" and "Side24 covariance and density" gave, for these three `Prop`s,
the reasons that local term-wise iterated differentiation of the lattice sum is missing from Mathlib
`d13f23b7` (its only local result, `hasFDerivAt_tsum_of_isPreconnected`, being first order) and,
for `CovarianceOrdering`, (ii) that symmetry of `iteratedFDeriv` beyond order two is missing
("Mathlib has `isSymmSndFDerivAt_iff_iteratedFDeriv` only"). Neither is load-bearing here: the
first is supplied inside this module by `Aux.ib_iteratedFDeriv_tsum_of_isOpen`, by induction from
that first-order result; for the second, the analytic case, which is all the proof needs (`Kinf d`
is analytic), is present at the pin as `ContDiffAt.iteratedFDeriv_comp_perm`, while the general
version for finitely smooth functions beyond order two is indeed absent at `d13f23b7`. This is a
remark about informal reasons, not a statement defect and not about any claim.
-/

namespace UniversalLaw.Proofs.Side24Images

open UniversalLaw.Spec.Side24
open scoped ContDiff Topology Matrix

namespace Aux

/-! ## `ib_`: helpers for `ImageBound` -/

/-- Local version of Mathlib's `iteratedFDeriv_tsum`: for summands that are each `C^N` on the whole
space, summable bounds over an open preconnected set `U` on their derivatives of every order `≤ N`
give, for every `k ≤ N`, the term-wise `k`-th iterated derivative at every point of `U`. Same
induction as Mathlib's global lemma, with the local `hasFDerivAt_tsum_of_isPreconnected` and
`Filter.EventuallyEq.fderiv_eq` on `U`. -/
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

/-! ## `ec_`: helpers for `EntrywiseCovarianceBound` (`ec_length_jetDirs_le` also serves `co_C24_symm`) -/

/-- When `u` is itself a frame vector (`hu : ∃ j, u = R j`), every direction attached to a joint
coordinate is a frame vector `R j`. -/
theorem ec_mem_jetDirs {d : ℕ} {u : E d} {R : Fin d → E d} (hu : ∃ j, u = R j)
    (x : JointIndex d) {v : E d} (hv : v ∈ jetDirs d u R x) : ∃ j, v = R j := by
  rcases x with i | (_ | p)
  · simp only [jetDirs, List.mem_singleton] at hv
    exact ⟨i, hv⟩
  · simp only [jetDirs, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with h | h | h <;> (subst h; exact hu)
  · simp only [jetDirs, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with h | h
    · exact ⟨p.1.1, h⟩
    · exact ⟨p.1.2, h⟩

/-- Each joint coordinate carries at most three directions. -/
theorem ec_length_jetDirs_le {d : ℕ} (u : E d) (R : Fin d → E d) (x : JointIndex d) :
    (jetDirs d u R x).length ≤ 3 := by
  rcases x with i | (_ | p) <;> simp [jetDirs]

/-- The `svec` scale is `1` or `√2`: nonnegative and at most `√2`. -/
theorem ec_svecScale_bounds {d : ℕ} (x : JointIndex d) :
    0 ≤ svecScale d x ∧ svecScale d x ≤ Real.sqrt 2 := by
  have h1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  rcases x with i | (_ | p)
  · simp only [svecScale]; exact ⟨zero_le_one, h1⟩
  · simp only [svecScale]; exact ⟨zero_le_one, h1⟩
  · simp only [svecScale]
    split_ifs
    · exact ⟨zero_le_one, h1⟩
    · exact ⟨Real.sqrt_nonneg _, le_rfl⟩

/-- `|svecScale a · svecScale b · (-1)^n| ≤ 2`. -/
theorem ec_scale_le {d : ℕ} (a b : JointIndex d) (n : ℕ) :
    |svecScale d a * svecScale d b * (-1 : ℝ) ^ n| ≤ 2 := by
  obtain ⟨ha0, ha⟩ := ec_svecScale_bounds a
  obtain ⟨hb0, hb⟩ := ec_svecScale_bounds b
  rw [abs_mul, abs_mul, abs_pow, abs_neg, abs_one, one_pow, mul_one, abs_of_nonneg ha0,
    abs_of_nonneg hb0]
  calc svecScale d a * svecScale d b ≤ Real.sqrt 2 * Real.sqrt 2 := mul_le_mul ha hb hb0 (Real.sqrt_nonneg _)
    _ = 2 := Real.mul_self_sqrt (by norm_num)

/-! ## `co_`: helpers for `CovarianceOrdering` -/

/-- `Kinf d` is analytic (`C^ω`): the exponential of a polynomial. -/
theorem co_kinf_analytic (d : ℕ) : ContDiff ℝ ω (Kinf d) := by
  unfold Kinf
  exact Real.contDiff_exp.comp ((contDiff_norm_sq ℝ).neg.div_const 2)

/-- The iterated derivatives of `Kinf d` are symmetric at every point (analyticity and
`ContDiffAt.iteratedFDeriv_comp_perm`). -/
theorem co_kinf_perm {d n : ℕ} (y : E d) (v : Fin n → E d) (σ : Equiv.Perm (Fin n)) :
    iteratedFDeriv ℝ n (Kinf d) y (v ∘ σ) = iteratedFDeriv ℝ n (Kinf d) y v :=
  (co_kinf_analytic d).contDiffAt.iteratedFDeriv_comp_perm v σ

/-- Evenness of `Kinf d` at a general point: `D^q Kinf (-y) v = (-1)^q D^q Kinf y v`. -/
theorem co_kinf_neg_apply {d : ℕ} (q : ℕ) (y : E d) (v : Fin q → E d) :
    iteratedFDeriv ℝ q (Kinf d) (-y) v = (-1 : ℝ) ^ q * iteratedFDeriv ℝ q (Kinf d) y v := by
  have h := (-ContinuousLinearMap.id ℝ (E d)).iteratedFDeriv_comp_right
    (Side24Analysis.Aux.rb_kinf_contDiff (d := d)) y (i := q) (mod_cast le_top)
  rw [Side24Covariance.Aux.cv_kinf_comp_neg] at h
  have h2 := congrArg (fun T => T v) h
  simp only [ContinuousMultilinearMap.compContinuousLinearMap_apply] at h2
  have h3 : (fun i => (-ContinuousLinearMap.id ℝ (E d)) (v i)) = fun i => (-1 : ℝ) • v i := by
    funext i
    simp
  rw [h3, ContinuousMultilinearMap.map_smul_univ] at h2
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at h2
  have hy : (-ContinuousLinearMap.id ℝ (E d)) y = -y := by simp
  rw [hy] at h2
  rw [h2, ← mul_assoc, ← pow_add, ← two_mul, pow_mul]
  simp

/-- `latticeVec d (-n) = -latticeVec d n`. -/
theorem co_latticeVec_neg (d : ℕ) (n : Fin d → ℤ) : latticeVec d (-n) = -latticeVec d n := by
  ext i
  simp [latticeVec]

/-- The term-wise derivative of `K_24` at `0`, evaluated at a tuple. -/
theorem co_K24_apply (d q : ℕ) (hq : q ≤ 6) (v : Fin q → E d) :
    iteratedFDeriv ℝ q (K24 d) 0 v =
      (thetaSum24 d 0)⁻¹ * ∑' n : Fin d → ℤ, iteratedFDeriv ℝ q (Kinf d) (latticeVec d n) v := by
  have hΘ := ib_theta0_ge_one d
  have hΘpos : 0 < thetaSum24 d 0 := by linarith
  have hAs : Summable (fun n : Fin d → ℤ =>
      (thetaSum24 d 0)⁻¹ • iteratedFDeriv ℝ q (Kinf d) (latticeVec d n)) := by
    refine .of_norm_bounded (ib_summable_bound d) fun n => ?_
    have h := ib_term_bound (inv_nonneg.2 hΘpos.le) (inv_le_one_of_one_le₀ hΘ) q hq n
      (Metric.mem_ball_self one_pos)
    rwa [ib_iteratedFDeriv_term, zero_add] at h
  rw [ib_K24_deriv d q hq]
  have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin q => E d) ℝ v).map_tsum hAs
  simp only [ContinuousMultilinearMap.apply_apply] at h
  rw [h, show (∑' n : Fin d → ℤ, ((thetaSum24 d 0)⁻¹ •
      iteratedFDeriv ℝ q (Kinf d) (latticeVec d n)) v) =
      ∑' n : Fin d → ℤ, (thetaSum24 d 0)⁻¹ * iteratedFDeriv ℝ q (Kinf d) (latticeVec d n) v
      from tsum_congr fun n => rfl, tsum_mul_left]

/-- Odd-order derivatives of `K_24` at `0` of order at most `6` vanish (reindex the lattice sum by
`n ↦ -n`, with `co_kinf_neg_apply`). -/
theorem co_K24_odd (d q : ℕ) (hq : q ≤ 6) (hodd : Odd q) (v : Fin q → E d) :
    iteratedFDeriv ℝ q (K24 d) 0 v = 0 := by
  rw [co_K24_apply d q hq]
  set f : (Fin d → ℤ) → ℝ := fun n => iteratedFDeriv ℝ q (Kinf d) (latticeVec d n) v with hf
  have hneg : ∀ n, f (-n) = -f n := fun n => by
    simp only [hf, co_latticeVec_neg, co_kinf_neg_apply, hodd.neg_one_pow]
    ring
  have hS : ∑' n, f n = -∑' n, f n := by
    calc ∑' n, f n = ∑' n, f ((Equiv.neg (Fin d → ℤ)) n) := ((Equiv.neg _).tsum_eq f).symm
      _ = ∑' n, -f n := tsum_congr fun n => hneg n
      _ = -∑' n, f n := tsum_neg
  have h0 : ∑' n, f n = 0 := by linarith
  rw [h0, mul_zero]

/-- `derivContraction` through a length cast. -/
theorem co_derivContraction_cast {d : ℕ} (K : E d → ℝ) (x : E d) (l : List (E d)) (n : ℕ)
    (h : l.length = n) :
    derivContraction d K x l = iteratedFDeriv ℝ n K x (fun i => l.get (Fin.cast h.symm i)) := by
  subst h
  rfl

/-- The block rotation `i ↦ i + m` (mod `m + k`) of `Fin (m + k)`. -/
def co_rot (m k : ℕ) : Equiv.Perm (Fin (m + k)) where
  toFun i := if h : i.val + m < m + k then ⟨i.val + m, h⟩ else ⟨i.val + m - (m + k), by omega⟩
  invFun j := if h : j.val + k < m + k then ⟨j.val + k, h⟩ else ⟨j.val + k - (m + k), by omega⟩
  left_inv i := by
    apply Fin.ext
    by_cases h1 : i.val + m < m + k
    · simp only [h1, dite_true]
      split_ifs with h2 <;> simp <;> omega
    · simp only [h1, dite_false]
      split_ifs with h2 <;> simp <;> omega
  right_inv j := by
    apply Fin.ext
    by_cases h1 : j.val + k < m + k
    · simp only [h1, dite_true]
      split_ifs with h2 <;> simp <;> omega
    · simp only [h1, dite_false]
      split_ifs with h2 <;> simp <;> omega

/-- Swapping the two blocks of a contraction list does not change a contraction of `Kinf d`. -/
theorem co_kinf_swap {d : ℕ} (x : E d) (l₁ l₂ : List (E d)) :
    derivContraction d (Kinf d) x (l₁ ++ l₂) = derivContraction d (Kinf d) x (l₂ ++ l₁) := by
  set m := l₁.length
  set k := l₂.length
  have h₁ : (l₁ ++ l₂).length = m + k := List.length_append
  have h₂ : (l₂ ++ l₁).length = m + k := by rw [List.length_append]; omega
  rw [co_derivContraction_cast _ x _ (m + k) h₁, co_derivContraction_cast _ x _ (m + k) h₂]
  rw [← co_kinf_perm x (fun i => (l₁ ++ l₂).get (Fin.cast h₁.symm i)) (co_rot m k)]
  congr 1
  funext j
  simp only [Function.comp_apply, co_rot, Equiv.coe_fn_mk, List.get_eq_getElem, Fin.val_cast]
  by_cases hj : j.val + m < m + k
  · simp only [hj, dite_true]
    rw [List.getElem_append_right (by omega), List.getElem_append_left (by omega)]
    congr 1
    omega
  · simp only [hj, dite_false]
    rw [List.getElem_append_left (by omega), List.getElem_append_right (by omega)]
    congr 1
    omega

/-- The same block swap for `K_24` at `0`, for lists of total length at most `6`. -/
theorem co_K24_swap {d : ℕ} (l₁ l₂ : List (E d)) (hl : (l₁ ++ l₂).length ≤ 6) :
    derivContraction d (K24 d) 0 (l₁ ++ l₂) = derivContraction d (K24 d) 0 (l₂ ++ l₁) := by
  have hl' : (l₂ ++ l₁).length ≤ 6 := by
    rw [List.length_append] at hl ⊢; omega
  unfold derivContraction
  rw [co_K24_apply d _ hl, co_K24_apply d _ hl']
  congr 1
  refine tsum_congr fun n => ?_
  exact co_kinf_swap (latticeVec d n) l₁ l₂

/-- A contraction of `K_24` at `0` against a list of odd length at most `6` vanishes. -/
theorem co_K24_odd_list {d : ℕ} (l : List (E d)) (hl : l.length ≤ 6) (hodd : Odd l.length) :
    derivContraction d (K24 d) 0 l = 0 :=
  co_K24_odd d _ hl hodd _

/-- `C_24` is symmetric, for every `u` and every `R`. -/
theorem co_C24_symm {d : ℕ} (u : E d) (R : Fin d → E d) (a b : JointIndex d) :
    C24 d u R a b = C24 d u R b a := by
  have ha := ec_length_jetDirs_le u R a
  have hb := ec_length_jetDirs_le u R b
  have hab : (jetDirs d u R a ++ jetDirs d u R b).length ≤ 6 := by
    rw [List.length_append]; omega
  have hba : (jetDirs d u R b ++ jetDirs d u R a).length ≤ 6 := by
    rw [List.length_append]; omega
  simp only [C24, jointCov, Matrix.of_apply]
  rw [co_K24_swap _ _ hab]
  rcases Nat.even_or_odd ((jetDirs d u R a).length + (jetDirs d u R b).length) with he | ho
  · have hpar : (-1 : ℝ) ^ (jetDirs d u R a).length = (-1 : ℝ) ^ (jetDirs d u R b).length := by
      rcases Nat.even_or_odd (jetDirs d u R a).length with h1 | h1
      · have h2 : Even (jetDirs d u R b).length := by
          have := (Nat.even_add.1 he).1 h1; exact this
        rw [h1.neg_one_pow, h2.neg_one_pow]
      · have h2 : Odd (jetDirs d u R b).length := by
          rcases Nat.even_or_odd (jetDirs d u R b).length with h3 | h3
          · exact absurd ((Nat.even_add.1 he).2 h3) (Nat.not_even_iff_odd.2 h1)
          · exact h3
        rw [h1.neg_one_pow, h2.neg_one_pow]
    rw [hpar]
    ring
  · have hodd : Odd (jetDirs d u R b ++ jetDirs d u R a).length := by
      rw [List.length_append, add_comm]; exact ho
    rw [co_K24_odd_list _ hba hodd]
    ring

/-- `C_24` is Hermitian (real symmetric). -/
theorem co_C24_hermitian {d : ℕ} (u : E d) (R : Fin d → E d) : (C24 d u R).IsHermitian :=
  Matrix.IsHermitian.ext fun i j => by simp [co_C24_symm u R i j]

/-- An entrywise bound `|A i j| ≤ c` bounds the quadratic form by `c · card · (x ⬝ᵥ x)`. -/
theorem co_quad_le {ι : Type*} [Fintype ι] (A : Matrix ι ι ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hA : ∀ i j, |A i j| ≤ c) (x : ι → ℝ) :
    |x ⬝ᵥ (A *ᵥ x)| ≤ c * Fintype.card ι * (x ⬝ᵥ x) := by
  have h1 : |x ⬝ᵥ (A *ᵥ x)| ≤ ∑ i, |x i| * ∑ j, |A i j| * |x j| := by
    unfold dotProduct Matrix.mulVec
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    unfold dotProduct
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
    exact Finset.sum_congr rfl fun j _ => abs_mul _ _
  have h2 : ∑ i, |x i| * ∑ j, |A i j| * |x j| ≤ c * (∑ i, |x i|) ^ 2 := by
    calc ∑ i, |x i| * ∑ j, |A i j| * |x j| ≤ ∑ i, |x i| * ∑ j, c * |x j| :=
          Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left
            (Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hA i j) (abs_nonneg _))
            (abs_nonneg _)
      _ = c * (∑ i, |x i|) ^ 2 := by
          have hS : ∀ i, |x i| * ∑ j, c * |x j| = |x i| * (c * ∑ j, |x j|) := fun i => by
            simp only [Finset.mul_sum]
          rw [Finset.sum_congr rfl fun i _ => hS i, ← Finset.sum_mul]
          ring
  have h3 : (∑ i, |x i|) ^ 2 ≤ Fintype.card ι * ∑ i, |x i| ^ 2 := by
    have := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun i => |x i|)
    simpa using this
  have h4 : ∑ i, |x i| ^ 2 = x ⬝ᵥ x := by
    unfold dotProduct
    exact Finset.sum_congr rfl fun i _ => by rw [sq_abs, sq]
  calc |x ⬝ᵥ (A *ᵥ x)| ≤ c * (∑ i, |x i|) ^ 2 := h1.trans h2
    _ ≤ c * (Fintype.card ι * ∑ i, |x i| ^ 2) := mul_le_mul_of_nonneg_left h3 hc
    _ = c * Fintype.card ι * (x ⬝ᵥ x) := by rw [h4]; ring

/-- `card (JointIndex d) ≤ 13` for `d ≤ 3`. -/
theorem co_card_le {d : ℕ} (hd : d ≤ 3) : Fintype.card (JointIndex d) ≤ 13 := by
  have h1 : Fintype.card (SymIndex d) ≤ d * d := by
    have := Fintype.card_subtype_le (fun p : Fin d × Fin d => p.1 ≤ p.2)
    simpa [Fintype.card_prod] using this
  have h2 : Fintype.card (JointIndex d) = d + (1 + Fintype.card (SymIndex d)) := by
    simp [JointIndex, Fintype.card_sum]
  rw [h2]
  have : d * d ≤ 9 := by nlinarith
  omega

/-- The margin: `2 imageConstE · 13 ≤ epsilon / 3`. -/
theorem co_margin : 2 * imageConstE * 13 ≤ epsilon / 3 := by
  norm_num [imageConstE, epsilon]

end Aux

open Aux

/-- Proves: `UniversalLaw.Spec.Side24.ImageBound` (claim `side24-coefficient`), for the formal
`K24`, `Kinf` and `thetaSum24` of `Specifications.Side24`: for `d ∈ {2, 3}`, every order `q ≤ 6`
and every unit tuple `u`, `|D^q K_24(0) u - D^q K_∞(0) u| < imageConstE`.
Method: `D^q K_24(0) = θ(0)⁻¹ ∑_n D^q K_∞(24 n)` by the local term-wise derivative on the unit
ball (`Aux.ib_K24_deriv`, from `Aux.ib_iteratedFDeriv_tsum_of_isOpen`); splitting off `n = 0` in
both sums gives `θ(0)⁻¹ (∑_{n ≠ 0} D^q K_∞(24 n) u - (θ(0) - 1) D^q K_∞(0) u)`, bounded, with
`θ(0) ≥ 1` (`Aux.ib_theta0_ge_one`) and `0 ≤ θ(0) - 1 ≤ ∑_{n ≠ 0} imageTerm d n` (`|n|^6 ≥ 1`,
`Aux.ib_gauss_le_image`), by `76 · 24^6 · 1458 e^(-288) + 15 · 1458 e^(-288)` through the packet's
theorems
`Side24Analysis.derivativeBoundAtLeastOne`, `derivativeBoundAtZero` and `latticeImageSumBound`,
and `1458 (76 · 24^6 + 15) e^(-288) < imageConstE` through `Side24Elementary.exponentialBounds`.
Does not establish: display (2) of the source as the source's object (the alignment of `E d`,
`Kinf`, `K24`, `thetaSum24`, `IsUnitTuple` and Mathlib's `iteratedFDeriv` with the source's
`R^d`, `K_infty`, `K_24`, unit directions and derivatives is PENDING_INDEPENDENT_REVIEW); any
bound for `q > 6` or `d ∉ {2, 3}`; differentiability of `K24` as a statement (it is used inside the
proof, on the unit ball, through the helper `Aux.ib_iteratedFDeriv_tsum_of_isOpen`, which is a
helper of this module, not an inventory theorem and not a Mathlib result); the Prop's hypothesis
`ThetaSummable d` is not used (the packet's `thetaSummable` proves it for every `d`); the
statement module's own `Does not claim:` text, which describes the statement packet and is not
edited here; the round-4 triage verdict on `ImageBound` (BLOCKED_MISSING_MATHLIB), which is
transcribed in the register and not altered here; any enclosure of the coefficient `c_{d,24}`;
the disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register,
not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. It
promotes, closes, discharges or reclassifies nothing. -/
theorem imageBound : UniversalLaw.Spec.Side24.ImageBound := by
  intro d hd _ q hq u hu
  have hd3 : d ≤ 3 := by rcases hd with rfl | rfl <;> norm_num
  set Θ := thetaSum24 d 0 with hΘeq
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

/-- Proves: `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound` (claim `side24-coefficient`),
for the formal `C24` and `Cref` of `Specifications.Side24`: for `d ∈ {2, 3}` and every
orthonormal frame `R` with `u = R 0`, `|C_24 a b - C_ref a b| ≤ 2 imageConstE` for every pair of
joint coordinates.
Method: the two entries share the factor `svecScale a · svecScale b · (-1)^|b|`, of absolute value
at most `2`, and their derivative contractions are along the list `jetDirs a ++ jetDirs b` of at
most six frame vectors, each of norm one; `imageBound` bounds the difference of the contractions
by `imageConstE` (strictly).
Does not establish: the source's Section 3 sentence as the source's object (the alignment of
`jetDirs`, `svecScale`, `jointCov`, `C24` and `Cref` with the source's jet, `svec` scaling and
joint covariances is PENDING_INDEPENDENT_REVIEW); that `C24` or `Cref` is the covariance of any
random vector (no field is constructed); the spectral-norm bound `20E` (not specified); the
ordering `CovarianceOrdering` (a separate theorem of this module); the Prop's hypothesis
`ThetaSummable d` is passed to `imageBound`, which does not use it; the round-4 triage verdict on
`EntrywiseCovarianceBound` (BLOCKED_MISSING_MATHLIB), which is transcribed in the register and not
altered here; any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient` (REVIEWED_SCOPED); nonauthor review or acceptance of
SIDE24-COEFFICIENT-D23-20260924-v1. It promotes, closes, discharges or reclassifies nothing. -/
theorem entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound := by
  intro d hd hθ hd0 R hR a b
  set u := R ⟨0, hd0⟩
  set l := jetDirs d u R a ++ jetDirs d u R b with hldef
  have hu : ∃ j, u = R j := ⟨⟨0, hd0⟩, rfl⟩
  have hl : l.length ≤ 6 := by
    rw [hldef, List.length_append]
    have := ec_length_jetDirs_le u R a
    have := ec_length_jetDirs_le u R b
    omega
  have hunit : IsUnitTuple d l.length (fun i => l.get i) := by
    intro i
    have hmem : l.get i ∈ jetDirs d u R a ++ jetDirs d u R b := List.get_mem l i
    obtain ⟨j, hj⟩ : ∃ j, l.get i = R j := by
      rcases List.mem_append.1 hmem with h | h
      · exact ec_mem_jetDirs hu a h
      · exact ec_mem_jetDirs hu b h
    show ‖l.get i‖ = 1
    rw [hj]
    exact hR.1 j
  have himg := imageBound d hd hθ l.length hl (fun i => l.get i) hunit
  simp only [C24, Cref, jointCov, Matrix.of_apply, derivContraction]
  rw [← mul_sub, abs_mul]
  calc |svecScale d a * svecScale d b * (-1 : ℝ) ^ (jetDirs d u R b).length| *
        |iteratedFDeriv ℝ l.length (K24 d) 0 (fun i => l.get i) -
          iteratedFDeriv ℝ l.length (Kinf d) 0 (fun i => l.get i)|
      ≤ 2 * imageConstE :=
        mul_le_mul (ec_scale_le a b _) himg.le (abs_nonneg _) (by norm_num)

/-- Proves: `UniversalLaw.Spec.Side24.CovarianceOrdering` (claim `side24-coefficient`), for the
formal `C24` and `Cref` of `Specifications.Side24`: for `d ∈ {2, 3}` and every orthonormal frame
`R` with `u = R 0`, `C_24 - (1 - epsilon) C_ref` and `(1 + epsilon) C_ref - C_24` are positive
semidefinite in Mathlib's sense (`Matrix.PosSemidef`, which includes `IsHermitian`).
Method: `C_24` is symmetric (`Aux.co_C24_hermitian`): by the term-wise derivative a contraction of
`K_24` at `0` of total order at most `6` is `θ(0)⁻¹ ∑_n` of contractions of the analytic `Kinf d`
at `24 n`, which are
invariant under the block swap (`ContDiffAt.iteratedFDeriv_comp_perm`), and odd total orders
vanish by reindexing `n ↦ -n` together with `D^q Kinf(-y) = (-1)^q D^q Kinf(y)`
(`Aux.co_kinf_neg_apply`, from the round-5 helper `Side24Covariance.Aux.cv_kinf_comp_neg`), so the
sign `(-1)^|b|` may be replaced by `(-1)^|a|`. `C_ref` is
symmetric and `C_ref - I/3` positive semidefinite by `Side24Covariance.referenceCovarianceLowerBound`.
With `Δ = C_24 - C_ref`, `|Δ a b| ≤ 2 imageConstE` (`entrywiseCovarianceBound`), so
`|x ⬝ Δ x| ≤ 2 imageConstE · card · (x ⬝ x)` (Cauchy-Schwarz), the cardinality of the joint index
is at most `d + 1 + d^2 ≤ 13` for `d ≤ 3` (its exact value `6` or `10` is not used), and
`26 imageConstE (x ⬝ x) ≤ (epsilon / 3) (x ⬝ x) ≤ epsilon (x ⬝ C_ref x)` (`Aux.co_margin` and
`C_ref - I/3 ≥ 0`), which gives both orderings.
Does not establish: display (3) of the source as the source's object (the alignment of `C24`,
`Cref` and Mathlib's `Matrix.PosSemidef`, `IsHermitian` included, with the source's covariances and
positive-semidefinite order is PENDING_INDEPENDENT_REVIEW); that `C24` or `Cref` is the
covariance of any random vector (no field is constructed); the passage to marginal and
conditional covariances (not specified); the source's route through the spectral-norm bound `20E`
and `60E < epsilon` (not reproduced: the proof uses the cruder `2E` times at most `13` coordinates
and `26E ≤ epsilon / 3`, that is `78E ≤ epsilon`, decided by `norm_num` in the helper
`Aux.co_margin`; main's `UniversalLaw.Side24.covariance_relative_bound` is neither imported, cited
nor relied on);
symmetry of `Cref d u R` for a general `u` or frame (it is taken from the round-5 theorem at
`u = R 0` in an orthonormal frame; the helper `Aux.co_C24_hermitian`, stated for every `d`, `u`
and `R`, is a helper, not an inventory theorem); the Prop's hypothesis `ThetaSummable d` is only
forwarded, through `entrywiseCovarianceBound`, to `imageBound`, which does not use it; the round-4
triage verdict on `CovarianceOrdering` (BLOCKED_MISSING_MATHLIB), which is transcribed in the
register and not altered here; any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient` (REVIEWED_SCOPED); nonauthor review or acceptance of
SIDE24-COEFFICIENT-D23-20260924-v1. It promotes, closes, discharges or reclassifies nothing. -/
theorem covarianceOrdering : UniversalLaw.Spec.Side24.CovarianceOrdering := by
  intro d hd hθ hd0 R hR
  have hd3 : d ≤ 3 := by rcases hd with rfl | rfl <;> norm_num
  set u := R ⟨0, hd0⟩
  set C := C24 d u R
  set M := Cref d u R
  have hMpsd := Side24Covariance.referenceCovarianceLowerBound d hd hd0 R hR
  have hMh : M.IsHermitian := by
    have h1 := hMpsd.isHermitian
    have h2 : ((1 / 3 : ℝ) • (1 : Matrix (JointIndex d) (JointIndex d) ℝ)).IsHermitian :=
      Matrix.isHermitian_one.smul (by simp [IsSelfAdjoint])
    simpa using h1.add h2
  have hCh : C.IsHermitian := co_C24_hermitian u R
  have hΔ : ∀ a b, |(C - M) a b| ≤ 2 * imageConstE := fun a b => by
    simpa [Matrix.sub_apply] using entrywiseCovarianceBound d hd hθ hd0 R hR a b
  have hcard := co_card_le hd3
  have hM3 : ∀ x : JointIndex d → ℝ, (1 / 3) * (x ⬝ᵥ x) ≤ x ⬝ᵥ (M *ᵥ x) := fun x => by
    have h := hMpsd.dotProduct_mulVec_nonneg x
    simp only [star_trivial, Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
      dotProduct_sub, dotProduct_smul, smul_eq_mul] at h
    linarith
  have hquad : ∀ x : JointIndex d → ℝ,
      |x ⬝ᵥ ((C - M) *ᵥ x)| ≤ epsilon * (1 / 3) * (x ⬝ᵥ x) := fun x => by
    have hE : 0 ≤ 2 * imageConstE := by norm_num [imageConstE]
    have h1 := co_quad_le (C - M) hE hΔ x
    have hxx : 0 ≤ x ⬝ᵥ x := by
      unfold dotProduct; exact Finset.sum_nonneg fun i _ => mul_self_nonneg _
    have h2 : 2 * imageConstE * (Fintype.card (JointIndex d) : ℝ) ≤ 2 * imageConstE * 13 :=
      mul_le_mul_of_nonneg_left (by exact_mod_cast hcard) hE
    calc |x ⬝ᵥ ((C - M) *ᵥ x)| ≤ 2 * imageConstE * Fintype.card (JointIndex d) * (x ⬝ᵥ x) := h1
      _ ≤ 2 * imageConstE * 13 * (x ⬝ᵥ x) := mul_le_mul_of_nonneg_right h2 hxx
      _ ≤ epsilon / 3 * (x ⬝ᵥ x) := mul_le_mul_of_nonneg_right co_margin hxx
      _ = epsilon * (1 / 3) * (x ⬝ᵥ x) := by ring
  have heps : 0 ≤ epsilon := by norm_num [epsilon]
  refine ⟨Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
      (hCh.sub (hMh.smul (by simp [IsSelfAdjoint]))) fun x => ?_,
    Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
      ((hMh.smul (by simp [IsSelfAdjoint])).sub hCh) fun x => ?_⟩
  · have hq := hquad x
    have hm := hM3 x
    have hsplit : star x ⬝ᵥ ((C - (1 - epsilon) • M) *ᵥ x) =
        x ⬝ᵥ ((C - M) *ᵥ x) + epsilon * (x ⬝ᵥ (M *ᵥ x)) := by
      simp only [star_trivial, Matrix.sub_mulVec, Matrix.smul_mulVec, dotProduct_sub,
        dotProduct_smul, smul_eq_mul]
      ring
    rw [hsplit]
    have := neg_abs_le (x ⬝ᵥ ((C - M) *ᵥ x))
    nlinarith
  · have hq := hquad x
    have hm := hM3 x
    have hsplit : star x ⬝ᵥ (((1 + epsilon) • M - C) *ᵥ x) =
        epsilon * (x ⬝ᵥ (M *ᵥ x)) - x ⬝ᵥ ((C - M) *ᵥ x) := by
      simp only [star_trivial, Matrix.sub_mulVec, Matrix.smul_mulVec, dotProduct_sub,
        dotProduct_smul, smul_eq_mul]
      ring
    rw [hsplit]
    have := le_abs_self (x ⬝ᵥ ((C - M) *ᵥ x))
    nlinarith

end UniversalLaw.Proofs.Side24Images
