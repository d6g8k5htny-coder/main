import Mathlib
import Specifications.Lifetime

/-!
Feasibility probe (scratch only, not a packet file): can the Lifetime successor interface of the
design note (main#307 6136538069 §2) be stated against the pinned Mathlib `d13f23b7`? The free
probability `p` of `PairLawInterface` is tied to a pinned Gaussian regression law `Q` of `G`'s field
on the `2(d+1)` pin observations, the typed determinant weight `W`, the tilted law `Q^W` (a
DEFINITION from `Q` and `W`), and the elder event of `I`. `#check`s only; no theorem, no sorry.
-/

namespace Probe.Lifetime2

open UniversalLaw.Spec.Lifetime MeasureTheory

/-- The `2(d+1)` pin observations: the two values and the `2d` gradient components in the frame. -/
abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)

/-- The two pins as a function of the index `0 ↦ M`, `1 ↦ S`. -/
noncomputable def pinAt (d : ℕ) (r : ℝ) (u : E d) : Fin 2 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u

/-- The observation functionals `f ↦ f(pin)` and `f ↦ ∂_{R i} f(pin)`. -/
noncomputable def pinObs (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    PinIdx d → (E d → ℝ) → ℝ
  | Sum.inl a => fun f => f (pinAt d r (R 0) a)
  | Sum.inr (a, i) => fun f => fderiv ℝ f (pinAt d r (R 0) a) (R i)

/-- The imposed targets: `f(M) = b`, `f(S) = b - k r^3`, all pin gradients `0`. -/
noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

/-- Derivative of `K_L` at `z` contracted against a list of directions (as `derivContraction` in Side24). -/
noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)

/-- Jet covariance of the stationary field with covariance `K_L`: `Cov(∂_u f(x), ∂_v f(y))` with
`u`, `v` optional directions; `Cov(f(x), f(y)) = K(x-y)`, `Cov(∂_u f(x), f(y)) = ∂_u K(x-y)`,
`Cov(f(x), ∂_v f(y)) = -∂_v K(x-y)`, `Cov(∂_u f(x), ∂_v f(y)) = -∂_u ∂_v K(x-y)`. -/
noncomputable def jetCov (d : ℕ) (L : ℝ) (x : E d) (u : Option (E d)) (y : E d) (v : Option (E d)) : ℝ :=
  match u, v with
  | none, none => dK d L (x - y) []
  | some u, none => dK d L (x - y) [u]
  | none, some v => - dK d L (x - y) [v]
  | some u, some v => - dK d L (x - y) [u, v]

/-- Position and optional direction of an observation. -/
noncomputable def obsPoint (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : PinIdx d → E d × Option (E d)
  | Sum.inl a => (pinAt d r (R 0) a, none)
  | Sum.inr (a, i) => (pinAt d r (R 0) a, some (R i))

/-- Covariance matrix of the `2(d+1)` observations under the field law. -/
noncomputable def pinObsCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (PinIdx d) (PinIdx d) ℝ :=
  Matrix.of fun a b =>
    jetCov d L (obsPoint d r R a).1 (obsPoint d r R a).2 (obsPoint d r R b).1 (obsPoint d r R b).2

/-- Cross-covariance of `f(x)` with the observations. -/
noncomputable def pinCross (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : PinIdx d → ℝ :=
  fun a => jetCov d L x none (obsPoint d r R a).1 (obsPoint d r R a).2

/-- Regression mean of `f(x)` given the observations at their targets (ordinary inverse; the
successor interface carries invertibility of `pinObsCov` as a field). -/
noncomputable def regMean (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : ℝ :=
  dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinTarget d b k r))

/-- Regression covariance of `f(x)`, `f(y)` given the observations: the Schur complement. -/
noncomputable def regCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x y : E d) : ℝ :=
  periodicCovariance d L (x - y) -
    dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinCross d L r R y))

/-- The Hessian matrix of `g` at `x` in the frame `R`. -/
noncomputable def hessMat (d : ℕ) (g : E d → ℝ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => hessian d g x (R i) (R j)

/-- The typed weight `W_r = |det H_M det H_S| 1{H_M < 0, index(H_S) = d-1}` of a sample path. -/
noncomputable def typedWeight (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (g : E d → ℝ) : ℝ :=
  Set.indicator {g : E d → ℝ | HessianHasInertia d g (pinM d r (R 0)) d ∧ HessianHasInertia d g (pinS d r (R 0)) (d - 1)}
    (fun g => |(hessMat d g (pinM d r (R 0)) R).det| * |(hessMat d g (pinS d r (R 0)) R).det|) g

/-- Successor of `PairLawInterface`: the regression law `Q` on `G`'s sample space is GIVEN with its
Gaussian regression characterisation as a hypothesis field; the weight is the typed determinant
weight of the sample; the tilted law is DEFINED; `p` is pinned to the `Q^W` probability of the
elder event of `I`. -/
structure PairLawInterface2 (d : ℕ) [NeZero d] (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) extends PairLawInterface d L G I where
  /-- Section 1 clause: "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1)
  observations"; a measure on `G`'s sample space for each `r`, `b`, `k`, `R`. -/
  Q : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → @Measure G.Ω G.mΩ
  /-- `Q` is a probability measure. -/
  Q_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), @IsProbabilityMeasure G.Ω G.mΩ (Q r b k R)
  /-- The observation covariance is positive definite in the pin regime (the source's Section 3
  "uniform inverse covariance"), so the ordinary-inverse regression formulas are the regression. -/
  pinObsCov_posDef : ∀ (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    (pinObsCov d L r R).PosDef
  /-- Gaussian regression: under `Q`, every finite vector of point evaluations of the field is the
  multivariate Gaussian with the regression mean and the Schur-complement covariance. -/
  Q_regression : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (m : ℕ) (x : Fin m → E d),
      @Measure.map G.Ω (EuclideanSpace ℝ (Fin m)) G.mΩ _ (fun ω => WithLp.toLp 2 (fun i => G.F ω (x i))) (Q r b k R) =
        ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (fun i => regMean d L r b k R (x i)))
          (Matrix.of fun i j => regCov d L r R (x i) (x j))
  /-- Section 1 clause: the pins hold `Q`-almost surely in the pin regime. -/
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(Q r b k R), G.F ω (pinM d r (R 0)) = b ∧ G.F ω (pinS d r (R 0)) = b - k * r ^ 3 ∧
      IsCriticalPoint d (G.F ω) (pinM d r (R 0)) ∧ IsCriticalPoint d (G.F ω) (pinS d r (R 0))
  /-- Section 1 clause: the full normalizer `Z_r = E_Q W_r` is positive in the pin regime (the
  source's (5.5) floor is a theorem there; here only positivity is assumed). -/
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    0 < ∫ ω, typedWeight d r R (G.F ω) ∂(Q r b k R)
  /-- Section 1 clause: `p_r(b,k,R)` is the `Q^W` probability that the elder death partner of `M`
  is `S`, with `dQ^W = (W_r/Z_r) dQ`. -/
  p_def : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    p r b k R =
      (((Q r b k R).withDensity (fun ω => ENNReal.ofReal
          (typedWeight d r R (G.F ω) / ∫ ω', typedWeight d r R (G.F ω') ∂(Q r b k R))))
        {ω | I.ElderPartner (G.F ω) (pinM d r (R 0)) (pinS d r (R 0))}).toReal

/-- Successor Props over the determined interface. -/
def ParentUniformSelectionA2 (d : ℕ) [NeZero d] (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L)
    (PL : PairLawInterface2 d L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA d L G I PL.toPairLawInterface bLo bHi kLo kHi

def PlanarElderLowerE2v2 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface2 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  PlanarElderLowerE2 L G I PL.toPairLawInterface bLo bHi kLo kHi

def PlanarTwoSidedRateE4v2 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface2 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  PlanarTwoSidedRateE4 L G I PL.toPairLawInterface bLo bHi kLo kHi

#check @pinObsCov
#check @regCov
#check @typedWeight
#check @PairLawInterface2.Q_regression
#check @PairLawInterface2.p_def
#check @ParentUniformSelectionA2
#check @PlanarElderLowerE2v2
#check @PlanarTwoSidedRateE4v2

end Probe.Lifetime2
