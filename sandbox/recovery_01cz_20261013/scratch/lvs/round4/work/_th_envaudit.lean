import Specifications.Side24

/-!
# SIDE24 analysis proofs: summability of the Gaussian theta family (`ThetaSummable d`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` (prefix `th_`). The route: the
lattice vector `24 n` has coordinates `24 n_i`, so `‖24 n‖^2 = 576 ∑ n_i^2`; the pointwise bound
`exp(-‖z + v‖^2/2) ≤ exp(‖z‖^2/2) exp(-‖v‖^2/4)` (from `‖v‖ ≤ ‖z + v‖ + ‖z‖`); `Real.exp_sum`
turns `exp(-‖24 n‖^2/4)` into the product `∏ i, exp(-144 n_i^2)`; the one-dimensional family
`m ↦ exp(-c m^2)` on `ℤ` is summable by Mathlib's `HurwitzKernelBounds.summable_f_int`; and a
product `n ↦ ∏ i, g (n i)` of a nonnegative summable `g` is summable over `Fin d → ℤ` because
every finite partial sum sits inside a `Fintype.piFinset` and is bounded by `(∑' g)^d`.
`Summable.of_nonneg_of_le` with `Summable.mul_left` assembles the Prop for every `d : ℕ`.

The statement module's docstring for `ThetaSummable` reads "Does not claim: that the family is
summable (true, but not shown here)"; that text describes the statement packet, which this
module does not edit. Nothing here changes any register label.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24

namespace Aux

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

end UniversalLaw.Proofs.Side24Analysis

-- SCRATCH COPY ONLY: list every constant this module adds, with kind and Prop-ness.
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut rows : Array String := #[]
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp ci.type)
    rows := rows.push s!"{kind} prop={isP} unsafe={ci.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"TOTAL {rows.size}"
