
/-! ## Round 7, second slice (DRAFT for the design read; not activated): the successor pair-law interface

Everything above this line is byte-identical to the module at Math- `8acc4856`. The declarations below transcribe the
parent's Section 1 model of the typed pair law for the field of a `GaussianFieldInterface`: the `2(d+1)` pin
observations and their jet covariances from `periodicCovariance`, the regression mean and Schur-complement covariance,
the typed determinant weight `W_r`, and a successor `PairLawInterface2` whose fields give the regression law `Q` with its
finite-dimensional Gaussian characterisation, the pins almost surely, positivity of the normalizer, and `p` pinned to the
`Q^W` probability of the elder event of `I`, with `Q^W` DEFINED as `Q.withDensity (W_r / E_Q W_r)`. The three successor
Props restate `ParentUniformSelectionA`, `PlanarElderLowerE2` and `PlanarTwoSidedRateE4` over it. Formal progress of every
declaration here: `specified`. Nothing is proved; the predecessor interface and Props are unchanged; no inhabitant is
constructed. What the successor does NOT pin (for the design read): `Q_regression` characterises `Q` through point
evaluations only, not sample-path jets, so `Q_pinned` stays a separate field; the jet-covariance sign conventions are
hand-written; `pinObsCov_posDef` and `Z_pos` are the source's theorems assumed as fields; measurability of the elder event
and integrability of the weight are not stated; the axial direction `R 0` needs `[NeZero d]`. -/

/-- The `2(d+1)` pin observations: the two values and the `2d` frame gradient components.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations"
Interface: none; an index type.
Does not claim: anything. -/
abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)

/-- The two pins as a function of the index: `0 ↦ M = -(r/2) u`, `1 ↦ S = (r/2) u`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "M=-(r/2)u,     S=(r/2)u,"
Interface: none.
Does not claim: anything. -/
noncomputable def pinAt (d : ℕ) (r : ℝ) (u : E d) : Fin 2 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u

/-- The observation functionals `f ↦ f(pin)` and `f ↦ ∂_{R i} f(pin)` in the frame `R` with `u = R 0`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "grad f(M)=grad f(S)=0."
Interface: none.
Does not claim: anything. -/
noncomputable def pinObs (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    PinIdx d → (E d → ℝ) → ℝ
  | Sum.inl a => fun f => f (pinAt d r (R 0) a)
  | Sum.inr (a, i) => fun f => fderiv ℝ f (pinAt d r (R 0) a) (R i)

/-- The imposed targets `f(M) = b`, `f(S) = b - k r^3`, all pin gradients `0`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "f(M)=b,        f(S)=s=b-k r^3,"
Interface: none.
Does not claim: anything. -/
noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

/-- The derivative of `K_L` at `z` contracted against a list of directions, through Mathlib's `iteratedFDeriv`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "the centered stationary Gaussian field of variance one with covariance"
Interface: none.
Does not claim: differentiability of `K_L` (if absent the derivative is `0` in Lean). -/
noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)

/-- Jet covariance of the stationary field with covariance `K_L`, `u`, `v` optional directions: `Cov(f(x), f(y)) = K(x-y)`,
`Cov(∂_u f(x), f(y)) = ∂_u K(x-y)`, `Cov(f(x), ∂_v f(y)) = -∂_v K(x-y)`, `Cov(∂_u f(x), ∂_v f(y)) = -∂_u ∂_v K(x-y)`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "the centered stationary Gaussian field of variance one with covariance"
Interface: none; the covariances are DEFINED by the derivative rule, no field is constructed.
Does not claim: that these are the covariances of any random variables; the sign convention is the author's reading of the stationary rule and is a reader item. -/
noncomputable def jetCov (d : ℕ) (L : ℝ) (x : E d) (u : Option (E d)) (y : E d) (v : Option (E d)) : ℝ :=
  match u, v with
  | none, none => dK d L (x - y) []
  | some u, none => dK d L (x - y) [u]
  | none, some v => - dK d L (x - y) [v]
  | some u, some v => - dK d L (x - y) [u, v]

/-- Position and optional direction of a pin observation.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations"
Interface: none.
Does not claim: anything. -/
noncomputable def obsPoint (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : PinIdx d → E d × Option (E d)
  | Sum.inl a => (pinAt d r (R 0) a, none)
  | Sum.inr (a, i) => (pinAt d r (R 0) a, some (R i))

/-- The covariance matrix `Sigma_r` of the `2(d+1)` pin observations under the field law, by the derivative rule.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Because Sigma_r^{-1} and v_r are uniformly bounded"
Interface: none; defined by the derivative rule.
Does not claim: that it is a covariance; its positive definiteness (a field of the successor interface, `pinObsCov_posDef`). -/
noncomputable def pinObsCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (PinIdx d) (PinIdx d) ℝ :=
  Matrix.of fun a b =>
    jetCov d L (obsPoint d r R a).1 (obsPoint d r R a).2 (obsPoint d r R b).1 (obsPoint d r R b).2

/-- The cross-covariance `v_r(x)` of `f(x)` with the pin observations.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Because Sigma_r^{-1} and v_r are uniformly bounded"
Interface: none.
Does not claim: that it is a covariance. -/
noncomputable def pinCross (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : PinIdx d → ℝ :=
  fun a => jetCov d L x none (obsPoint d r R a).1 (obsPoint d r R a).2

/-- The regression mean of `f(x)` given the observations at their targets (ordinary inverse; the successor interface carries
invertibility of `pinObsCov` as the field `pinObsCov_posDef`).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations"
Interface: none; the Gaussian regression mean is written with the ordinary matrix inverse.
Does not claim: invertibility of `pinObsCov` (Mathlib's inverse is `0` on a singular matrix). -/
noncomputable def regMean (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : ℝ :=
  dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinTarget d b k r))

/-- The regression covariance of `f(x)`, `f(y)` given the observations: the Schur complement `K(x-y) - v(x)^T Sigma^(-1) v(y)`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations"
Interface: none; written with the ordinary matrix inverse.
Does not claim: invertibility of `pinObsCov`; positive semidefiniteness of the resulting kernel. -/
noncomputable def regCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x y : E d) : ℝ :=
  periodicCovariance d L (x - y) -
    dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinCross d L r R y))

/-- The Hessian matrix of `g` at `x` in the frame `R`, from the module's bilinear `hessian`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1},"
Interface: none.
Does not claim: symmetry (true for `C^2` functions, not stated). -/
noncomputable def hessMat (d : ℕ) (g : E d → ℝ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => hessian d g x (R i) (R j)

/-- The typed weight `W_r = |det H_M det H_S| 1{H_M < 0, index(H_S) = d-1}` of a sample path, the type indicator through
`HessianHasInertia` (zero on singular matrices).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1},"
Interface: none.
Does not claim: that the frame-dependent determinants agree with the source's (they do for orthonormal frames; not stated). -/
noncomputable def typedWeight (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (g : E d → ℝ) : ℝ :=
  Set.indicator {g : E d → ℝ | HessianHasInertia d g (pinM d r (R 0)) d ∧ HessianHasInertia d g (pinS d r (R 0)) (d - 1)}
    (fun g => |(hessMat d g (pinM d r (R 0)) R).det| * |(hessMat d g (pinS d r (R 0)) R).det|) g

/-- Successor of `PairLawInterface` (numbered, the predecessor unchanged): the regression law `Q_{r,b,k,R}` on `G`'s sample
space is GIVEN with its Gaussian regression characterisation as a hypothesis field; the pin observations hold almost surely;
the observation covariance is positive definite; the full normalizer `Z_r = E_Q W_r` is positive; and `p_r(b,k,R)` is the
`Q^W` probability that the elder death partner of `M` is `S`, with `dQ^W = (W_r/Z_r) dQ` DEFINED from `Q` and the typed weight.
With these fields `p` is no longer free, so the inhabitants `p ≡ 0`, `p ≡ 1` of the predecessor are excluded.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Z_r = E_Q W_r,       dQ^W = (W_r/Z_r)dQ."
Interface: every field is a given object or a hypothesis; nothing is derived. `Q_regression` characterises `Q` through finite vectors of point evaluations only (not sample-path jets), so `Q_pinned` is a separate field; `pinObsCov_posDef` (source Section 3) and `Z_pos` (source (5.5)) are the source's theorems assumed as fields; measurability of the elder event and integrability of the weight are not stated (Mathlib's measure and integral are total); the axial direction is `R 0`, which needs `[NeZero d]`.
Does not claim: existence of an object satisfying this interface; that `Q` is the regression law of any field or that `Q^W` is the source's typed law (no field is constructed); the normalizer floor `Z_r >= z_* r^2` (only positivity is a field); anything about the predecessor interface, which is unchanged. -/
structure PairLawInterface2 (d : ℕ) [NeZero d] (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) extends PairLawInterface d L G I where
  /-- Section 1 clause "Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations": a measure on
  `G`'s sample space for each radius, birth, gap mark and frame. -/
  Q : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → @Measure G.Ω G.mΩ
  /-- `Q` is a probability measure. -/
  Q_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), @IsProbabilityMeasure G.Ω G.mΩ (Q r b k R)
  /-- Section 3 clause "Because Sigma_r^{-1} and v_r are uniformly bounded": the observation covariance is positive definite
  in the pin regime, so the ordinary-inverse regression formulas are the regression. -/
  pinObsCov_posDef : ∀ (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L → (pinObsCov d L r R).PosDef
  /-- Gaussian regression: under `Q`, every finite vector of point evaluations of the field is the multivariate Gaussian
  with the regression mean and the Schur-complement covariance. -/
  Q_regression : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (m : ℕ) (x : Fin m → E d),
      @Measure.map G.Ω (EuclideanSpace ℝ (Fin m)) G.mΩ _ (fun ω => WithLp.toLp 2 (fun i => G.F ω (x i))) (Q r b k R) =
        ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (fun i => regMean d L r b k R (x i)))
          (Matrix.of fun i j => regCov d L r R (x i) (x j))
  /-- Section 1 clause: the pins `f(M) = b`, `f(S) = b - k r^3`, `grad f(M) = grad f(S) = 0` hold `Q`-almost surely in the
  pin regime `0 < r < L`. -/
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(Q r b k R), G.F ω (pinM d r (R 0)) = b ∧ G.F ω (pinS d r (R 0)) = b - k * r ^ 3 ∧
      IsCriticalPoint d (G.F ω) (pinM d r (R 0)) ∧ IsCriticalPoint d (G.F ω) (pinS d r (R 0))
  /-- Section 1 clause "Z_r = E_Q W_r": the full normalizer is positive in the pin regime (the source's (5.5) floor is a
  theorem there; here only positivity is assumed). -/
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    0 < ∫ ω, typedWeight d r R (G.F ω) ∂(Q r b k R)
  /-- Section 1 clause "Let p_r(b,k,R) be the Q^W probability that the global ordinary superlevel elder death partner of M
  is S", with `dQ^W = (W_r/Z_r) dQ`. -/
  p_def : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    p r b k R =
      (((Q r b k R).withDensity (fun ω => ENNReal.ofReal
          (typedWeight d r R (G.F ω) / ∫ ω', typedWeight d r R (G.F ω') ∂(Q r b k R))))
        {ω | I.ElderPartner (G.F ω) (pinM d r (R 0)) (pinS d r (R 0))}).toReal

/-- Successor of `ParentUniformSelectionA` over the successor interface: display (1.1), `0 <= 1 - p_r(b,k,R) <= C r^3`
uniformly on compact windows and frames, for any object satisfying `PairLawInterface2`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "0 <= 1-p_r(b,k,R) <= C r^3.                         (1.1)"
Interface: `PL : PairLawInterface2 d L G I` is given; `p` is pinned by `p_def` to the `Q^W` probability of the elder event, the law `Q` by `Q_regression`; the Prop is a predicate on the interface object, stated for every inhabitant.
Does not claim: the result (statement only); existence of an inhabitant; that the formal objects are the source's; anything about `ParentUniformSelectionA`, which is unchanged; the parent's proof (normalizer floor, boundary layer); nonauthor review or acceptance. -/
def ParentUniformSelectionA2 (d : ℕ) [NeZero d] (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L)
    (PL : PairLawInterface2 d L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA d L G I PL.toPairLawInterface bLo bHi kLo kHi

/-- Successor of `PlanarElderLowerE2` over the successor interface: display (E2), `1 - p_r(b,k,R) >= c r^3` on the planar
compact model, for any object satisfying `PairLawInterface2`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "1-p_r(b,k,R) >= c r^3, 0<r<=r_*.                      (E2)"
Interface: `PL : PairLawInterface2 2 L G I` is given, with `p` pinned as above; the Prop is a predicate on the interface object.
Does not claim: the result (statement only); existence of an inhabitant; anything about `PlanarElderLowerE2`, which is unchanged; the source's proof (polygonal path, tilted jet-box probability); nonauthor review or acceptance. -/
def PlanarElderLowerE2B (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface2 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  PlanarElderLowerE2 L G I PL.toPairLawInterface bLo bHi kLo kHi

/-- Successor of `PlanarTwoSidedRateE4` over the successor interface: display (E4), `c r^3 <= 1 - p_r(b,k,R) <= C r^3` on
the planar compact model given the consumed upper input, for any object satisfying `PairLawInterface2`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "c r^3 <= 1-p_r(b,k,R) <= C r^3.                       (E4)"
Interface: `PL : PairLawInterface2 2 L G I` is given, with `p` pinned as above; the hypothesis `ParentUniformSelectionA 2 L G I PL.toPairLawInterface …` is the consumed upper input, a statement shape.
Does not claim: the result (statement only); the upper half independently of the parent (1.1); existence of an inhabitant; anything about `PlanarTwoSidedRateE4`, which is unchanged; nonauthor review or acceptance. -/
def PlanarTwoSidedRateE4B (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface2 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  PlanarTwoSidedRateE4 L G I PL.toPairLawInterface bLo bHi kLo kHi
