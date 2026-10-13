import Lean
import Specifications.Side24

/-!
# SIDE24: the Section 1 reference pin-density product (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The inventory theorem below has as its TYPE the registered `Prop` constant
`UniversalLaw.Spec.Side24.ReferencePinDensityProduct` of the statement packet
`specifications/lean_statements_20261010`. A kernel-checked proof of that `Prop` is evidence
about that `Prop` only: it does not make the informal Layer 0 claim
(SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) accepted, reviewed or
"verified", does not change its transcribed disposition (REVIEWED_SCOPED), does not enclose the
coefficient `c_{d,24}`, and does not touch the statement register (the statement module is not
edited here). Helper lemmas live in the `Aux` namespace (prefix `rp_`) and are not inventory
items.

Route. In an orthonormal frame `R` with `u = R 0`, the two derivative-rule blocks of the
reference kernel `Kinf d z = exp (-‖z‖^2 / 2)` are `gradCov d (Kinf d) R = 1` and
`pinCov d (Kinf d) u R = diagonal (3, 1, …, 1)`; both are the first and fourth conjuncts of the
registered `Prop` `ReferenceOddAndPinBlocks`. Two forms are delivered: the hypothesis-form helper
`Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks`, which takes that `Prop` as a
hypothesis, and the inventory theorem itself, which discharges the two block identities with a
reproduction under the `rp_` tag (`Aux.rp_blockIdentities`). Both go through the common core
`Aux.rp_referencePinDensityProduct_of_blockIdentities`, so that the integrator may replace the
`rp_` reproduction by the development of the sibling `ReferenceOddAndPinBlocks` file by changing
only the final line of the inventory theorem.

The reproduction evaluates `iteratedFDeriv ℝ n (Kinf d) 0` in the needed directions through
order four without any closed form for the iterated Fréchet derivative: one order is peeled at a
time (`Aux.rp_peel`, from `iteratedFDeriv_succ_apply_right` and
`iteratedFDeriv_clm_apply_const_apply`, for `C^∞` functions), the first derivative is
`fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y)` (`Aux.rp_fderiv_Kinf`), and the intermediate
functions after each peel have explicit derivatives (`Aux.rp_fderiv_g1`, `Aux.rp_fderiv_g2`,
`Aux.rp_fderiv_g3_zero`). This gives `D^2 K(0)[a, b] = -⟪a, b⟫` (`Aux.rp_order_two`) and the
three-pairing formula `D^4 K(0)[c, a, w, v] = ⟪w,v⟫⟪c,a⟫ + ⟪a,v⟫⟪c,w⟫ + ⟪c,v⟫⟪a,w⟫`
(`Aux.rp_order_four`); `orthonormal_iff_ite` then evaluates the blocks. The density algebra
(`Aux.rp_density_algebra`) is `det 1 = 1`, `det (diagonal (3,1,…,1)) = 3`, `x = 0` killing the
exponential, and `(2π)^(-d/2) (2π)^(-d/2) 3^(-1/2) = (2π)^(-d) / √3`.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

namespace Aux

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

/-! ### Reproduction of the two block identities: smoothness and first derivative of `Kinf` -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` on `E d`. -/
theorem rp_contDiff_Kinf (d : ℕ) : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact ContDiff.exp (ContDiff.div_const (ContDiff.neg (contDiff_norm_sq ℝ)) 2)

/-- The first derivative of `Kinf d` at `y`, in the form produced by `hasStrictFDerivAt_norm_sq`
and the chain rule for `exp`. -/
theorem rp_hasFDerivAt_Kinf {d : ℕ} (y : E d) :
    HasFDerivAt (Kinf d)
      (Real.exp (-(‖y‖ ^ 2) / 2) • ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y))) y := by
  have h1 : HasFDerivAt (fun z : E d => -(‖z‖ ^ 2) / 2)
      ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y)) y := by
    have := (hasStrictFDerivAt_norm_sq y).hasFDerivAt.neg.mul_const (2 : ℝ)⁻¹
    refine this.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => ?_)
    simp [div_eq_mul_inv]
  exact h1.exp

/-- `fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y)`. -/
theorem rp_fderiv_Kinf {d : ℕ} (y v : E d) :
    fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y) := by
  rw [(rp_hasFDerivAt_Kinf y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply]
  ring

/-- Peeling one order off an iterated derivative of a `C^∞` real function: the last direction
is fed to `fderiv` and the remaining `n` directions to the `n`-th derivative of the result. -/
theorem rp_peel {d : ℕ} (f : E d → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (n : ℕ) (x : E d)
    (m : Fin (n + 1) → E d) :
    iteratedFDeriv ℝ (n + 1) f x m =
      iteratedFDeriv ℝ n (fun y => fderiv ℝ f y (m (Fin.last n))) x (Fin.init m) := by
  rw [iteratedFDeriv_succ_apply_right]
  rw [iteratedFDeriv_clm_apply_const_apply (contDiff_infty_iff_fderiv.mp hf).2
    (mod_cast le_top)]

/-- `y ↦ ⟪y, v⟫` is `C^∞`. -/
theorem rp_contDiff_inner {d : ℕ} (v : E d) : ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => ⟪y, v⟫) :=
  contDiff_id.inner ℝ contDiff_const

/-! ### Level one: `g₁ y = -(⟪y, v⟫ K y)` -/

/-- Level one is `C^∞`. -/
theorem rp_contDiff_g1 {d : ℕ} (v : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪y, v⟫ * Kinf d y)) :=
  ((rp_contDiff_inner v).mul (rp_contDiff_Kinf d)).neg

/-- The derivative of level one in the direction `w`. -/
theorem rp_fderiv_g1 {d : ℕ} (v y w : E d) :
    fderiv ℝ (fun y : E d => -(⟪y, v⟫ * Kinf d y)) y w =
      -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y := by
  have h : HasFDerivAt (fun y : E d => -(⟪y, v⟫ * Kinf d y)) _ y :=
    (((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul (rp_hasFDerivAt_Kinf y)).neg
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! ### Level two: `g₂ y = -(⟪w, v⟫ K y) + ⟪y, v⟫ ⟪y, w⟫ K y` -/

/-- Level two is `C^∞`. -/
theorem rp_contDiff_g2 {d : ℕ} (v w : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) :=
  ((contDiff_const.mul (rp_contDiff_Kinf d)).neg.add
    (((rp_contDiff_inner v).mul (rp_contDiff_inner w)).mul (rp_contDiff_Kinf d)))

/-- The derivative of level two in the direction `a`. -/
theorem rp_fderiv_g2 {d : ℕ} (v w y a : E d) :
    fderiv ℝ (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) y a =
      ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
        - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y := by
  have h : HasFDerivAt
      (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) _ y :=
    (((rp_hasFDerivAt_Kinf y).const_mul _).neg).add
      ((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul
        ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y))).mul (rp_hasFDerivAt_Kinf y))
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! ### Level three at the origin, general directions -/

/-- The derivative of level three at `0` in the direction `c`: the three pairings. -/
theorem rp_fderiv_g3_zero {d : ℕ} (v w a c : E d) :
    fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  have h : HasFDerivAt (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) _ 0 :=
    (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0)).const_mul _).mul
        (rp_hasFDerivAt_Kinf 0)).add
      ((((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0)).const_mul _).add
          (((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul_const _))).mul
        (rp_hasFDerivAt_Kinf 0))).sub
      (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0))).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0))).mul
        (rp_hasFDerivAt_Kinf 0))
  rw [h.fderiv]
  simp [Kinf, fderivInnerCLM_apply]
  ring

/-! ### The contractions of orders two and four at the origin -/

/-- Second-order contraction at zero: `D^2 K(0)[a, b] = -⟪a, b⟫`. -/
theorem rp_order_two {d : ℕ} (a b : E d) :
    derivContraction d (Kinf d) 0 [a, b] = -⟪a, b⟫ := by
  show iteratedFDeriv ℝ 2 (Kinf d) 0 (fun i => [a, b].get i) = -⟪a, b⟫
  rw [rp_peel _ (rp_contDiff_Kinf d), iteratedFDeriv_one_apply]
  simp only [rp_fderiv_Kinf]
  show fderiv ℝ (fun y : E d => -(⟪y, b⟫ * Kinf d y)) 0 a = -⟪a, b⟫
  rw [rp_fderiv_g1]
  simp [Kinf]

/-- Fourth-order contraction at zero: the three pairings
`D^4 K(0)[c, a, w, v] = ⟪w, v⟫⟪c, a⟫ + ⟪a, v⟫⟪c, w⟫ + ⟪c, v⟫⟪a, w⟫`. -/
theorem rp_order_four {d : ℕ} (c a w v : E d) :
    derivContraction d (Kinf d) 0 [c, a, w, v] =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  show iteratedFDeriv ℝ 4 (Kinf d) 0 (fun i => [c, a, w, v].get i) = _
  rw [rp_peel _ (rp_contDiff_Kinf d)]
  simp only [rp_fderiv_Kinf]
  show iteratedFDeriv ℝ 3 (fun y => -(⟪y, v⟫ * Kinf d y)) 0
    (Fin.init fun i => [c, a, w, v].get i) = _
  rw [rp_peel _ (rp_contDiff_g1 v)]
  simp only [rp_fderiv_g1]
  show iteratedFDeriv ℝ 2 (fun y => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) 0
    (Fin.init (Fin.init fun i => [c, a, w, v].get i)) = _
  rw [rp_peel _ (rp_contDiff_g2 v w)]
  simp only [rp_fderiv_g2]
  show iteratedFDeriv ℝ 1 (fun y => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0
    (Fin.init (Fin.init (Fin.init fun i => [c, a, w, v].get i))) = _
  rw [iteratedFDeriv_one_apply]
  show fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c = _
  exact rp_fderiv_g3_zero v w a c

/-! ### The two blocks in an orthonormal frame -/

/-- `gradCov d (Kinf d) R = 1` for orthonormal `R`. -/
theorem rp_gradCov {d : ℕ} (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    gradCov d (Kinf d) R = 1 := by
  ext i j
  simp only [gradCov, Matrix.of_apply, rp_order_two, neg_neg, Matrix.one_apply]
  exact orthonormal_iff_ite.mp hR i j

/-- `pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)` for orthonormal `R`: the entry `(i, j)`
is `[0 = j][0 = i] + [i = j] + [0 = j][i = 0]`. -/
theorem rp_pinCov {d : ℕ} (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    pinCov d (Kinf d) (R ⟨0, hd⟩) R =
      Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1) := by
  have hite := orthonormal_iff_ite.mp hR
  ext i j
  simp only [pinCov, Matrix.of_apply, rp_order_four, hite, Matrix.diagonal_apply]
  rcases eq_or_ne i j with hij | hij
  · subst hij
    by_cases hi : i = ⟨0, hd⟩
    · subst hi
      norm_num
    · simp [hi, Ne.symm hi]
  · by_cases hj : j = ⟨0, hd⟩
    · subst hj
      simp [hij, Ne.symm hij]
    · simp [hij, Ne.symm hj]

/-- The two block identities of `ReferenceOddAndPinBlocks` used by the density product, in the
hypothesis shape of `rp_referencePinDensityProduct_of_blockIdentities` (the restriction
`d = 2 ∨ d = 3` is not used by the proof). -/
theorem rp_blockIdentities :
    ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
      gradCov d (Kinf d) R = 1 ∧
      pinCov d (Kinf d) (R ⟨0, hd⟩) R = Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1) := by
  intro d _ hd R hR
  exact ⟨rp_gradCov R hR, rp_pinCov hd R hR⟩

end Aux

/-! ## Inventory theorem -/

/-- Proves: `UniversalLaw.Spec.Side24.ReferencePinDensityProduct` (register row `side24-coefficient`,
Section 1 of SIDE24-COEFFICIENT-D23-20260924-v1, "p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3)").
Method: the common core `Aux.rp_referencePinDensityProduct_of_blockIdentities` rewrites the two
blocks `gradCov d (Kinf d) R = 1` and `pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)` and
finishes with the density algebra `Aux.rp_density_algebra` (`Matrix.det_one`, `Matrix.det_diagonal`,
`Finset.prod_ite_eq'`, `Matrix.mulVec_zero`, `Real.rpow_add`, `Real.rpow_neg`, `Real.sqrt_eq_rpow`);
the two blocks are reproduced as `Aux.rp_blockIdentities` from the contractions
`D^2 K(0)[a, b] = -⟪a, b⟫` (`Aux.rp_order_two`) and the three-pairing formula for `D^4 K(0)`
(`Aux.rp_order_four`), obtained by peeling one order at a time (`Aux.rp_peel`, from
`iteratedFDeriv_succ_apply_right` and `iteratedFDeriv_clm_apply_const_apply`) with the explicit
derivatives `Aux.rp_fderiv_Kinf`, `Aux.rp_fderiv_g1`, `Aux.rp_fderiv_g2`, `Aux.rp_fderiv_g3_zero`,
and evaluated in the frame by `orthonormal_iff_ite`; the hypothesis form
`Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks` derives the same Prop from
`ReferenceOddAndPinBlocks` taken as a hypothesis.
Does not establish: the informal Section 1 display or its review and acceptance; the alignment of
the formal objects (`gaussDensity`, `gradCov`, `pinCov`, `derivContraction` through Mathlib's
`iteratedFDeriv ℝ` at `0`, `Orthonormal ℝ R`) with the source's (PENDING_INDEPENDENT_REVIEW); the
determinant identity `det diag(3,1,…,1) = 3` as the main-branch target
`UniversalLaw.Side24.pin_determinant_and_joint_dimension`, which is kernel-checked on main and is
neither cited nor re-used here (the determinant is recomputed by `Matrix.det_diagonal` in
`Aux.rp_det_pinDiagonal`); that `gaussDensity` is the density of any random vector or that any field
is constructed; the remaining conjuncts of `ReferenceOddAndPinBlocks` (`Var(t) = 15`,
`Cov(t, G) = (-3, 0, …, 0)`, `tau^2 = 6`), which are not reproduced here; the value for `d ∉ {2, 3}`
(the Prop restricts to `d = 2 ∨ d = 3`, as the source does); any enclosure of the coefficient
`c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the
register, not derived here. -/
theorem referencePinDensityProduct : UniversalLaw.Spec.Side24.ReferencePinDensityProduct :=
  Aux.rp_referencePinDensityProduct_of_blockIdentities Aux.rp_blockIdentities

end UniversalLaw.Proofs.Side24Analysis


-- VERIFIER SCRATCH TAIL (not part of the deliverable) ------------------------------
#check @UniversalLaw.Proofs.Side24Analysis.referencePinDensityProduct
#print UniversalLaw.Spec.Side24.ReferencePinDensityProduct
#check @UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_det_pinDiagonal
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_density_algebra
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_blockIdentities
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_contDiff_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_hasFDerivAt_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_fderiv_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_peel
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_contDiff_inner
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_contDiff_g1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_fderiv_g1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_contDiff_g2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_fderiv_g2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_fderiv_g3_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_order_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_order_four
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_gradCov
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_pinCov
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rp_blockIdentities
#print axioms UniversalLaw.Proofs.Side24Analysis.referencePinDensityProduct
open Lean Meta Elab Command
def auditModules : Array Name := #[]
def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis
def registeredProps : Array Name := #[`UniversalLaw.Spec.Field.HessianHasInertia, `UniversalLaw.Spec.Field.IsCandidatePair, `UniversalLaw.Spec.Field.IsCriticalPoint, `UniversalLaw.Spec.Field.IsFiniteBar, `UniversalLaw.Spec.Field.IsIndexSaddle, `UniversalLaw.Spec.Field.IsLPeriodic, `UniversalLaw.Spec.Field.IsLifetimeDensityVersion, `UniversalLaw.Spec.Field.IsMorseDistinct, `UniversalLaw.Spec.Field.IsNondegMax, `UniversalLaw.Spec.Field.IsSmooth, `UniversalLaw.Spec.Lifetime.HessianHasInertia, `UniversalLaw.Spec.Lifetime.IsCandidatePair, `UniversalLaw.Spec.Lifetime.IsCriticalPoint, `UniversalLaw.Spec.Lifetime.IsFiniteBar, `UniversalLaw.Spec.Lifetime.IsIndexSaddle, `UniversalLaw.Spec.Lifetime.IsLPeriodic, `UniversalLaw.Spec.Lifetime.IsLifetimeDensityVersion, `UniversalLaw.Spec.Lifetime.IsMorseDistinct, `UniversalLaw.Spec.Lifetime.IsNondegMax, `UniversalLaw.Spec.Lifetime.IsSmooth, `UniversalLaw.Spec.Lifetime.IsUnrestrictedLeadingCoefficient, `UniversalLaw.Spec.Lifetime.IsUnrestrictedPair, `UniversalLaw.Spec.Lifetime.IsWindowElderPair, `UniversalLaw.Spec.Lifetime.IsWindowPair, `UniversalLaw.Spec.Lifetime.ParentCompactWindowDensityB, `UniversalLaw.Spec.Lifetime.ParentUniformSelectionA, `UniversalLaw.Spec.Lifetime.ParentUnrestrictedLeadingDensityC, `UniversalLaw.Spec.Lifetime.PlanarDensityGapE11, `UniversalLaw.Spec.Lifetime.PlanarElderLowerE2, `UniversalLaw.Spec.Lifetime.PlanarTwoSidedRateE4, `UniversalLaw.Spec.Lifetime.RemainderBoundedR1, `UniversalLaw.Spec.P15.Benchmark816Cover, `UniversalLaw.Spec.P15.Benchmark816IsRealized, `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts, `UniversalLaw.Spec.P15.Benchmark816Palette, `UniversalLaw.Spec.P15.Benchmark816Price, `UniversalLaw.Spec.P15.Benchmark816RationalPrice, `UniversalLaw.Spec.P15.Benchmark816TransformedRange, `UniversalLaw.Spec.P15.Benchmark816UnionBound, `UniversalLaw.Spec.P15.Benchmark816WholeGround, `UniversalLaw.Spec.P15.CapTestAtTwoThirds, `UniversalLaw.Spec.P15.CapacityHazardIsBinomial, `UniversalLaw.Spec.P15.CapacityWorstCase, `UniversalLaw.Spec.P15.CoordinatewiseConcavity, `UniversalLaw.Spec.P15.CoverCostHazardBound, `UniversalLaw.Spec.P15.DemandOneBoundary, `UniversalLaw.Spec.P15.DemandOneRefutation, `UniversalLaw.Spec.P15.ElementaryPaletteFormula, `UniversalLaw.Spec.P15.FullBlockCoverIff, `UniversalLaw.Spec.P15.FullGroundBudgetIff, `UniversalLaw.Spec.P15.FullGroundPriceBound, `UniversalLaw.Spec.P15.FullTransformedPriceBudget, `UniversalLaw.Spec.P15.GlobalAssemblyHazard, `UniversalLaw.Spec.P15.HStarExceedsOne, `UniversalLaw.Spec.P15.HazardPhiBound, `UniversalLaw.Spec.P15.HazardProductBound, `UniversalLaw.Spec.P15.HazardStarPositive, `UniversalLaw.Spec.P15.IndependentBlockHazard, `UniversalLaw.Spec.P15.IsGeneratorCover, `UniversalLaw.Spec.P15.IsGood, `UniversalLaw.Spec.P15.IsKDecomposable, `UniversalLaw.Spec.P15.IsMinimalForbidden, `UniversalLaw.Spec.P15.IsPaletteFeasible, `UniversalLaw.Spec.P15.IsProperDownset, `UniversalLaw.Spec.P15.IsRealizedFamily, `UniversalLaw.Spec.P15.IsTransversal, `UniversalLaw.Spec.P15.LocalDecomposability, `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock, `UniversalLaw.Spec.P15.LocalPriceBoundFull, `UniversalLaw.Spec.P15.LocalPriceChain, `UniversalLaw.Spec.P15.LocalRestriction, `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity, `UniversalLaw.Spec.P15.LogRatioBound, `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation, `UniversalLaw.Spec.P15.OddMajorityIdentity, `UniversalLaw.Spec.P15.PaletteNumberAttained, `UniversalLaw.Spec.P15.PhiAtPStar, `UniversalLaw.Spec.P15.RatioExponentBound, `UniversalLaw.Spec.P15.RestrictedLocalBudget, `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths, `UniversalLaw.Spec.P15.RhoStarEnclosure, `UniversalLaw.Spec.P15.RhoTestSufficient, `UniversalLaw.Spec.P15.SharpnessOfRhoStar, `UniversalLaw.Spec.P15.SixSeventhsCertificate, `UniversalLaw.Spec.P15.TriangleNeedsThree, `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices, `UniversalLaw.Spec.P15.WholeGroundChromaticNumber, `UniversalLaw.Spec.RN.AnnulusBridgeAllHeight2, `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3, `UniversalLaw.Spec.RN.D5aPuncturedPinDisks, `UniversalLaw.Spec.RN.D5bMicrodisk, `UniversalLaw.Spec.RN.D5cCollarC1, `UniversalLaw.Spec.RN.D5dSummedPinNeighbourhoodC2, `UniversalLaw.Spec.RN.D5eIntermediateShellsI3I4, `UniversalLaw.Spec.RN.D5fGlobalWindowFirstMomentI5, `UniversalLaw.Spec.RN.DimensionLiftCompactCollarC, `UniversalLaw.Spec.RN.DimensionLiftGlobalWindowG, `UniversalLaw.Spec.RN.DimensionLiftIntermediateShellsI, `UniversalLaw.Spec.RN.DimensionLiftPuncturedPinBallP, `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5, `UniversalLaw.Spec.RN.FixedAnnulusWindowCandidateA2, `UniversalLaw.Spec.RN.FixedRemoteCountBoundA3, `UniversalLaw.Spec.RN.FixedRemoteMeanMeasureA2, `UniversalLaw.Spec.RN.FourierCutoffPlanarSecondFactorialF4, `UniversalLaw.Spec.RN.HessianHasInertia, `UniversalLaw.Spec.RN.HolderConsequenceN2, `UniversalLaw.Spec.RN.HolderCountBoundN1, `UniversalLaw.Spec.RN.IsCriticalPoint, `UniversalLaw.Spec.RN.PalmRouteFactorialMomentsQ, `UniversalLaw.Spec.RN.PalmRouteSecondFactorialTheta, `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO, `UniversalLaw.Spec.RN.RemoteCollisionSecondFactorialD, `UniversalLaw.Spec.RN.UniformCounterexampleN4, `UniversalLaw.Spec.RN.WitnessCollisionNearPinPairsTarget, `UniversalLaw.Spec.Side24.CoefficientEnclosure, `UniversalLaw.Spec.Side24.ConeMomentD1Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Algebra, `UniversalLaw.Spec.Side24.ConeMomentD2Identity, `UniversalLaw.Spec.Side24.CovarianceOrdering, `UniversalLaw.Spec.Side24.DensityComparison, `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativeBoundAtZero, `UniversalLaw.Spec.Side24.DerivativePairingBound, `UniversalLaw.Spec.Side24.ElementaryConeIntegral, `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound, `UniversalLaw.Spec.Side24.ExponentialBounds, `UniversalLaw.Spec.Side24.ImageBound, `UniversalLaw.Spec.Side24.IntegrandRatioBounds, `UniversalLaw.Spec.Side24.IsUnitTuple, `UniversalLaw.Spec.Side24.LatticeImageSumBound, `UniversalLaw.Spec.Side24.NegDefCone, `UniversalLaw.Spec.Side24.RayleighSquareLaw, `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity, `UniversalLaw.Spec.Side24.ReferenceComparison, `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound, `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks, `UniversalLaw.Spec.Side24.ReferencePinDensityProduct, `UniversalLaw.Spec.Side24.SuccessiveTermRatio, `UniversalLaw.Spec.Side24.ThetaSummable, `UniversalLaw.Spec.Side24.TraceMoments]

run_cmd do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.referencePinDensityProduct
    | throwError "EXACT REFUSED: inventory theorem missing"
  let target : Expr := .const ``UniversalLaw.Spec.Side24.ReferencePinDensityProduct []
  unless ci.type == target do
    throwError "EXACT REFUSED: inventory type is not syntactically the Prop constant: {ci.type}"
  unless ci matches .thmInfo _ do throwError "EXACT REFUSED: inventory is not a theorem"
  logInfo m!"EXACT OK inventory type == Expr.const ReferencePinDensityProduct [] ; kind thm"
  let some hi := env.find? ``UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks
    | throwError "EXACT REFUSED: hypothesis-form helper missing"
  match hi.type with
  | .forallE _ dom body _ =>
    unless dom == (.const ``UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks []) do
      throwError "EXACT REFUSED: hypothesis is not the upstream Prop: {dom}"
    unless body == target do
      throwError "EXACT REFUSED: hypothesis-form conclusion is not the Prop: {body}"
    unless !body.hasLooseBVars do throwError "EXACT REFUSED: dependent conclusion"
    logInfo m!"EXACT OK hypothesis-form helper : ReferenceOddAndPinBlocks → ReferencePinDensityProduct (single hypothesis, exact constants)"
  | t => throwError "EXACT REFUSED: hypothesis-form helper has unexpected shape: {t}"


/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`, an equation lemma `eq_*` of an unfolded statement definition, which may
be generated in several modules and is audited once), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  -- an equation lemma of an imported definition (`foo.eq_1`) is generated on demand in every module
  -- that unfolds it and is one constant of the environment: audit each name once
  let mut seen : NameSet := {}
  let mut unique : Array ConstantInfo := #[]
  for ci in constants do
    unless seen.contains ci.name do
      seen := seen.insert ci.name
      unique := unique.push ci
  constants := unique
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"

