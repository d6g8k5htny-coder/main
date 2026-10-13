import Mathlib
import Specifications.RN

/-!
Feasibility probe (scratch only, not a packet file), second part of the RN successor design
(main#307 6136538069 §1, field 5): can the Kac-Rice contact kernel be DEFINED from the regression
law's finite-dimensional Gaussian parameters, and the Kac-Rice identities stated as fields, against the
pinned Mathlib `d13f23b7`? The kernel `Λ'_j(x; b, k, R)(y) = Z⁻¹ · p_{(f(x),∇f(x)) | pins}(y, 0) ·
E_Q[W_r F_j(H_x) | pins, f(x) = y, ∇f(x) = 0]` is written with two Schur steps (jet given pins; the
three Hessians given pins and jet) and Mathlib's `multivariateGaussian`. `#check`s only; no theorem.
-/

namespace Probe.RN2b

open UniversalLaw.Spec.RN MeasureTheory

noncomputable def thetaSum (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)

noncomputable def periodicCovariance (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  thetaSum d L z / thetaSum d L 0

noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)

/-- `Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K(x - y)` for the stationary field. -/
noncomputable def jetCovL (d : ℕ) (L : ℝ) (x : E d) (l1 : List (E d)) (y : E d) (l2 : List (E d)) : ℝ :=
  (-1 : ℝ) ^ l2.length * dK d L (x - y) (l1 ++ l2)

abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)
abbrev JetIdx (d : ℕ) : Type := Option (Fin d)
abbrev SymIdx (d : ℕ) : Type := {p : Fin d × Fin d // p.1 ≤ p.2}
abbrev HessIdx (d : ℕ) : Type := Fin 3 × SymIdx d
abbrev CondIdx (d : ℕ) : Type := PinIdx d ⊕ JetIdx d

/-- An observation: a point and the list of differentiation directions. -/
abbrev Obs (d : ℕ) : Type := E d × List (E d)

noncomputable def site3 (d : ℕ) (r : ℝ) (u x : E d) : Fin 3 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u
  | 2 => x

noncomputable def pinDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : PinIdx d → Obs d
  | Sum.inl a => (site3 d r (R 0) 0 (Fin.castLE (by norm_num) a), [])
  | Sum.inr (a, i) => (site3 d r (R 0) 0 (Fin.castLE (by norm_num) a), [R i])

noncomputable def jetDesc (d : ℕ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) : JetIdx d → Obs d
  | none => (x, [])
  | some i => (x, [R i])

noncomputable def condDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : CondIdx d → Obs d
  | Sum.inl a => pinDesc d r R a
  | Sum.inr a => jetDesc d x R a

noncomputable def hessDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : HessIdx d → Obs d
  | (a, p) => (site3 d r (R 0) x a, [R p.1.1, R p.1.2])

/-- Covariance matrix between two families of observations. -/
noncomputable def covOf (d : ℕ) (L : ℝ) {ι κ : Type} (f : ι → Obs d) (g : κ → Obs d) : Matrix ι κ ℝ :=
  Matrix.of fun a b => jetCovL d L (f a).1 (f a).2 (g b).1 (g b).2

noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

noncomputable def jetTarget (d : ℕ) (y : ℝ) : JetIdx d → ℝ
  | none => y
  | some _ => 0

noncomputable def condTarget (d : ℕ) (b k r y : ℝ) : CondIdx d → ℝ
  | Sum.inl a => pinTarget d b k r a
  | Sum.inr a => jetTarget d y a

/-- Gaussian density with covariance `C` and mean `m` at `v`, on a finite index type. -/
noncomputable def gaussDensityOn {ι : Type} [Fintype ι] [DecidableEq ι] (C : Matrix ι ι ℝ) (m v : ι → ℝ) : ℝ :=
  (2 * Real.pi) ^ (-(Fintype.card ι : ℝ) / 2) * C.det ^ (-(1 : ℝ) / 2) *
    Real.exp (-(dotProduct (v - m) (Matrix.mulVec C⁻¹ (v - m))) / 2)

/-- First Schur step: the `(d+1)`-jet at `x` given the pins. -/
noncomputable def jetCovGivenPins (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (JetIdx d) (JetIdx d) ℝ :=
  covOf d L (jetDesc d x R) (jetDesc d x R) -
    covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹ *
      covOf d L (pinDesc d r R) (jetDesc d x R)

noncomputable def jetMeanGivenPins (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    JetIdx d → ℝ :=
  Matrix.mulVec (covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹) (pinTarget d b k r)

/-- `p_{(f(x), ∇f(x)) | pins}(y, 0)`. -/
noncomputable def jetDensityAt (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) : ℝ :=
  gaussDensityOn (jetCovGivenPins d L r R x) (jetMeanGivenPins d L r b k R x) (jetTarget d y)

/-- Second Schur step: the three Hessians (at `M`, `S`, `x`) given the pins and the `x`-jet. -/
noncomputable def hessCovGivenCond (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (HessIdx d) (HessIdx d) ℝ :=
  covOf d L (hessDesc d r R x) (hessDesc d r R x) -
    covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹ *
      covOf d L (condDesc d r R x) (hessDesc d r R x)

noncomputable def hessMeanGivenCond (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) :
    HessIdx d → ℝ :=
  Matrix.mulVec (covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹)
    (condTarget d b k r y)

/-- The symmetric matrix at site `a` read off the Hessian coordinates. -/
noncomputable def symOfH (d : ℕ) (h : HessIdx d → ℝ) (a : Fin 3) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => if hij : i ≤ j then h (a, ⟨(i, j), hij⟩) else h (a, ⟨(j, i), le_of_not_ge hij⟩)

/-- Sylvester-inertia encoding for a matrix (the matrix analogue of `HessianHasInertia`). -/
def MatHasInertia (d : ℕ) (H : Matrix (Fin d) (Fin d) ℝ) (j : ℕ) : Prop :=
  ∃ V W : Submodule ℝ (Fin d → ℝ), IsCompl V W ∧
    Module.finrank ℝ V = j ∧ Module.finrank ℝ W = d - j ∧
    (∀ v ∈ V, v ≠ 0 → dotProduct v (H.mulVec v) < 0) ∧
    (∀ w ∈ W, w ≠ 0 → 0 < dotProduct w (H.mulVec w))

/-- `F_j(H) = |det H| 1{index H = j}`. -/
noncomputable def typedF (d : ℕ) (H : Matrix (Fin d) (Fin d) ℝ) (j : ℕ) : ℝ :=
  Set.indicator {H : Matrix (Fin d) (Fin d) ℝ | MatHasInertia d H j} (fun H => |H.det|) H

/-- The Kac-Rice integrand `W_r · F_j(H_x) = F_d(H_M) F_(d-1)(H_S) F_j(H_x)` on the Hessian coordinates. -/
noncomputable def kacRiceWeight (d : ℕ) (j : ℕ) (h : HessIdx d → ℝ) : ℝ :=
  typedF d (symOfH d h 0) d * typedF d (symOfH d h 1) (d - 1) * typedF d (symOfH d h 2) j

/-- The contact kernel `Λ'_j(x; b, k, R)(y)` for normalizer `Z`: the jet density times the conditional
expectation of the Kac-Rice weight under the Gaussian law of the three Hessians given pins and jet. -/
noncomputable def contactKernel (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (j : ℕ)
    (x : E d) (y : ℝ) (Z : ℝ) : ℝ :=
  Z⁻¹ * jetDensityAt d L r b k R x y *
    ∫ h, kacRiceWeight d j (fun a => h a)
      ∂(ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (hessMeanGivenCond d L r b k R x y)) (hessCovGivenCond d L r R x))

/-- Successor fields of part 5: the Kac-Rice identities for the window and all-height mean counts. -/
structure PinnedLawInterface3 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    extends PinnedLawInterface d L Ω where
  /-- Kac-Rice (d5-dimension-lift (6.8); rn-fixed-remote-window (13)): the tilted mean index-`j` window count of a
  Borel region of the fundamental domain is the integral of the contact kernel over the region and the height window. -/
  kacRice_window : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (windowCount d (F ω) b k r j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y in Set.Ioo (b - k * r ^ 3) b,
          contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))
  /-- The all-height analogue: the height integral runs over all of `ℝ`. -/
  kacRice_allHeight : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (allHeightCount d (F ω) j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y, contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))

#check @contactKernel
#check @PinnedLawInterface3.kacRice_window
#check @PinnedLawInterface3.kacRice_allHeight

end Probe.RN2b
