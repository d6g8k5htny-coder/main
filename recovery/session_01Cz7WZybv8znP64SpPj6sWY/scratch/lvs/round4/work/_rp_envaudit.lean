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
