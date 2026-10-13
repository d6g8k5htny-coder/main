import Mathlib
import Specifications.RN

/-!
Feasibility probe (scratch only, not a packet file): can the RN successor interface of the design note
(main#307 6136538069 §1) be stated against the pinned Mathlib `d13f23b7`, fields 1–4 and 6 (regression
law, nondegenerate observation covariance, defined typed weight, normalizer floor on compact windows,
a.s. finiteness and measurability of the counts)? Field 5 (the Kac-Rice identities and the defined
contact kernel `Λ'`) is left to a second probe: it needs conditional densities of the `(d+1)`-jet given
the pins. `#check`s only; no theorem, no sorry.
-/

namespace Probe.RN2

open UniversalLaw.Spec.RN MeasureTheory

/-- The theta sum and the periodized covariance `K_L`, as Lifetime.lean has them (RN.lean does not
define the kernel; the packet's convention is duplication per module). -/
noncomputable def thetaSum (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)

noncomputable def periodicCovariance (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  thetaSum d L z / thetaSum d L 0

/-- The `2(d+1)` pin observations: two values and the `2d` frame gradient components. -/
abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)

noncomputable def pinAt (d : ℕ) (r : ℝ) (u : E d) : Fin 2 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u

noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)

/-- Jet covariances of the stationary field with covariance `K_L` (sign conventions as in the
Lifetime probe: `Cov(f(x), ∂_v f(y)) = -∂_v K(x-y)`, `Cov(∂_u f(x), ∂_v f(y)) = -∂_u ∂_v K(x-y)`). -/
noncomputable def jetCov (d : ℕ) (L : ℝ) (x : E d) (u : Option (E d)) (y : E d) (v : Option (E d)) : ℝ :=
  match u, v with
  | none, none => dK d L (x - y) []
  | some u, none => dK d L (x - y) [u]
  | none, some v => - dK d L (x - y) [v]
  | some u, some v => - dK d L (x - y) [u, v]

noncomputable def obsPoint (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    PinIdx d → E d × Option (E d)
  | Sum.inl a => (pinAt d r (R 0) a, none)
  | Sum.inr (a, i) => (pinAt d r (R 0) a, some (R i))

noncomputable def pinObsCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (PinIdx d) (PinIdx d) ℝ :=
  Matrix.of fun a b =>
    jetCov d L (obsPoint d r R a).1 (obsPoint d r R a).2 (obsPoint d r R b).1 (obsPoint d r R b).2

noncomputable def pinCross (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    PinIdx d → ℝ :=
  fun a => jetCov d L x none (obsPoint d r R a).1 (obsPoint d r R a).2

noncomputable def regMean (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : ℝ :=
  dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinTarget d b k r))

noncomputable def regCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x y : E d) : ℝ :=
  periodicCovariance d L (x - y) -
    dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinCross d L r R y))

noncomputable def hessMat (d : ℕ) (g : E d → ℝ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => hessian d g x (R i) (R j)

/-- `W_r = F_d(H_M) F_(d-1)(H_S)` with `F_j(H) = |det H| 1{index H = j}` (zero on singular matrices,
through `HessianHasInertia`). -/
noncomputable def typedWeight (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (g : E d → ℝ) : ℝ :=
  Set.indicator {g : E d → ℝ | HessianHasInertia d g (pinM d r (R 0)) d ∧ HessianHasInertia d g (pinS d r (R 0)) (d - 1)}
    (fun g => |(hessMat d g (pinM d r (R 0)) R).det| * |(hessMat d g (pinS d r (R 0)) R).det|) g

/-- Successor of `PinnedLawInterface`: the given law `Q` is characterised as the Gaussian regression of
the `K_L` field on the pin observations; the given weight `W` is tied to the typed determinant weight;
the normalizer floor of the source's (5.5) on compact windows, a.s. finiteness of the critical set and
measurability of the counts are fields (assumptions, to be named as such). -/
structure PinnedLawInterface2 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    extends PinnedLawInterface d L Ω where
  /-- Section 3: the observation covariance is positive definite in the pin regime. -/
  pinObsCov_posDef : ∀ (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    (pinObsCov d L r R).PosDef
  /-- Section 1 clause "Q_r denotes continuous Gaussian regression on these 2(d+1) observations": under
  `Q`, every finite vector of point evaluations is the multivariate Gaussian with the regression mean
  and the Schur-complement covariance. -/
  Q_regression : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (m : ℕ) (x : Fin m → E d),
      Measure.map (fun ω => WithLp.toLp 2 (fun i => F ω (x i))) (Q r b k R) =
        ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (fun i => regMean d L r b k R (x i)))
          (Matrix.of fun i j => regCov d L r R (x i) (x j))
  /-- Section 1 clause: the given weight IS the typed determinant weight of the sample path. -/
  W_det : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), W r b k R ω = typedWeight d r R (F ω)
  /-- [LP] (5.5) `Z_r ≥ z_* r^2` on compact windows `b ∈ B`, `k ∈ K` with `k > 0`, for `0 < r ≤ r_*`
  (a theorem of the source, assumed here as a field). -/
  Z_floor : ∀ bLo bHi kLo kHi : ℝ, 0 < kLo → kLo ≤ kHi → bLo ≤ bHi →
    ∃ zStar : ℝ, 0 < zStar ∧ ∃ rStar : ℝ, 0 < rStar ∧ rStar < L ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), zStar * r ^ 2 ≤ ∫ ω, W r b k R ω ∂(Q r b k R)
  /-- The critical set of a sample path in the fundamental domain is finite, `Q^W`-almost surely, in
  the pin regime (lifetime-parent: "their determinants are nonzero almost surely"; a.s. Morse). -/
  critical_finite : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(QW r b k R), {x : E d | x ∈ FundDomain d L ∧ IsCriticalPoint d (F ω) x}.Finite
  /-- The window and all-height counts are measurable functions of the sample for every Borel region. -/
  count_measurable : ∀ (b k r : ℝ) (j : ℕ) (A : Set (E d)), MeasurableSet A →
    Measurable (fun ω => (windowCount d (F ω) b k r j A : ℝ)) ∧
    Measurable (fun ω => (allHeightCount d (F ω) j A : ℝ))

/-- Two of the 19 successor Props, as the shape every successor would take (structure projection). -/
def FixedRemoteCountBoundA3v2 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  FixedRemoteCountBoundA3 d L Ω I.toPinnedLawInterface ρ bLo bHi kLo kHi

def AnnulusBridgeAllHeight2v2 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi A B : ℝ) : Prop :=
  AnnulusBridgeAllHeight2 L Ω I.toPinnedLawInterface bLo bHi kLo kHi A B

#check @pinObsCov
#check @typedWeight
#check @PinnedLawInterface2.Q_regression
#check @PinnedLawInterface2.W_det
#check @PinnedLawInterface2.Z_floor
#check @FixedRemoteCountBoundA3v2
#check @AnnulusBridgeAllHeight2v2

end Probe.RN2
