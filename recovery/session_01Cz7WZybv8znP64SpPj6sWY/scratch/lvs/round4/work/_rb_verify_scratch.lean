import Specifications.Side24

/-!
# SIDE24 analysis proofs: the Section 1 reference blocks (`ReferenceOddAndPinBlocks`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` with the tag `rb_`. The route:
`Kinf d` is `C^∞` with `fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ Kinf d y)`; one direction of a
contraction `iteratedFDeriv ℝ (n+1) f x m` peels off through `iteratedFDeriv_succ_apply_right`
and `iteratedFDeriv_clm_apply_const_apply`, leaving the `n`-th derivative of
`y ↦ fderiv ℝ f y (m (Fin.last n))` in the remaining directions; the intermediate functions
after each peel have closed forms (polynomials in inner products times `Kinf d`), each proved by
`HasFDerivAt` calculus and `ring`. At the origin this gives the second-order contraction
`-⟪a, b⟫`, the fourth-order contraction as the three pairings
`⟪w, v⟫⟪c, a⟫ + ⟪a, v⟫⟪c, w⟫ + ⟪c, v⟫⟪a, w⟫`, and the sixth-order contraction along one direction
`-15 ⟪u, u⟫^3`. In an orthonormal frame (`orthonormal_iff_ite`) the blocks are then
`gradCov = 1`, `thirdVar = 15`, `thirdGradCov = (-3, 0, …, 0)`, `pinCov = diagonal (3, 1, …, 1)`
and `condThirdVariance = 15 - 9 = 6` (with `inv_one`, `Matrix.one_mulVec`,
`Finset.sum_ite_eq'`).

The proof-packet README row "Side24 covariance and density" records as its informal reason
that "the evaluation of the Fréchet derivatives is not available"; for `Kinf` at `0` in frame
directions through order six that evaluation is carried out here with existing Mathlib API.
This is a remark about an informal reason in a README, not about any claim; nothing here changes
any register label.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

namespace Aux

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

end Aux

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

end UniversalLaw.Proofs.Side24Analysis

-- VERIFIER SCRATCH COPY ONLY (not the deliverable).
#print UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks
#check @UniversalLaw.Proofs.Side24Analysis.referenceOddAndPinBlocks
#print UniversalLaw.Proofs.Side24Analysis.referenceOddAndPinBlocks
#print axioms UniversalLaw.Proofs.Side24Analysis.referenceOddAndPinBlocks
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_kinf_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_hasFDerivAt_kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_peel
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_inner_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_g1_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_g1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_g2_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_g2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_g3_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_order_four
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_P3_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_P3
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_P4_contDiff
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_P4
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_fderiv_P5_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_order_six
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_order_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_gradCov
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_pinCov
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_thirdGradCov
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_thirdVar
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.rb_condThirdVariance

-- Exact-type check: the theorem's type must be syntactically the registered Prop constant.
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.referenceOddAndPinBlocks
    | throwError "inventory theorem missing"
  let ty := ci.type
  logInfo m!"TYPE_EXPR {ty}"
  match ty with
  | .const n ls => logInfo m!"TYPE_IS_CONST {n} levels={ls.length}"
  | _ => logInfo m!"TYPE_IS_NOT_A_BARE_CONST"
  logInfo m!"TYPE_EQ_REGISTERED {ty == Expr.const ``UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks []}"
  logInfo m!"KIND {match ci with | .thmInfo _ => "thm" | .defnInfo _ => "def" | _ => "other"}"

-- Environment audit: every constant this module adds, with kind, Prop-ness, unsafe flag.
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
