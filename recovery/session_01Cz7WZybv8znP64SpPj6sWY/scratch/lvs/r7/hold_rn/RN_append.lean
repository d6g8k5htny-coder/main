
/-! ## Round 7, third slice (DRAFT for the design read; not activated): the successor pinned-law interface

Everything above this line is byte-identical to the module at Math- `7f5f1c82` (unchanged since `8acc4856`). The
declarations below transcribe what `PinnedLawInterface` deliberately left informal (its docstring and the scope note before
it): the periodized covariance `K_L` of the field, the `2(d+1)` pin observations and the jet covariances of the stationary
field, the Gaussian regression mean and Schur-complement covariance, the typed determinant weight
`W_r = F_d(H_M) F_(d-1)(H_S)` of a sample path, and the Kac-Rice contact kernel `Λ'_j(x; b, k, R)(y)` DEFINED from the
regression law's Gaussian parameters with two Schur steps (the `(d+1)`-jet at `x` given the pins; the three Hessians at
`M`, `S`, `x` given the pins and that jet). The successor `PinnedLawInterface2 extends PinnedLawInterface` adds the fields
that tie the given objects to these definitions: `pinObsCov_posDef`, `Q_regression`, `W_det`, `Z_floor`, `critical_finite`,
`count_measurable`, `kacRice_window`, `kacRice_allHeight`. The nineteen successor Props restate the nineteen
interface-blocked Props of the Frontier section over it. Formal progress of every declaration here: `specified`. Nothing is
proved; the predecessor interface and Props are unchanged; no inhabitant is constructed. What the successor does NOT pin
(for the design read): the parent's free kernel `Λ` (Theorem A's limit object as `r → 0`; the mean measures themselves
become determined instead); the sign and scale conventions of the derivative covariances are hand-written
(`Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K(x - y)`); the Hessian coordinates are the raw upper-triangular
entries; `Z_floor` and the two Kac-Rice identities are theorems of the sources assumed as fields (Mathlib `d13f23b7` has no
Kac-Rice formula); measurability and integrability beyond the stated fields are not stated. -/
/-- The unnormalised periodized Gaussian (theta) sum `sum_{n in Z^d} exp(-|z+Ln|^2/2)`, as `Lifetime.lean` has it (the packet's convention is duplication per module; this module had no covariance kernel because its interface left the regression informal).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)"
Interface: none.
Does not claim: summability (Mathlib's `tsum` is `0` for a non-summable family); anything about the field beyond its covariance. -/
noncomputable def thetaSum (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)
/-- The periodized Gaussian covariance `K_L(z) / K_L(0)` of the field, normalised to `1` at `0`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)"
Interface: none.
Does not claim: positive definiteness of the kernel; that this normalisation is the sources' (the regression formulas below are invariant under a constant factor). -/
noncomputable def periodicCovariance (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  thetaSum d L z / thetaSum d L 0
/-- The iterated directional derivative `∂_{l_1} ⋯ ∂_{l_m} K_L (z)` of the covariance along a list of directions (`iteratedFDeriv`, which is `0` where the function is not differentiable enough).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none.
Does not claim: differentiability of `K_L`; convergence of the differentiated theta sum. -/
noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)
/-- The jet covariance of the stationary field, `Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K_L(x - y)` (hand-written convention: derivatives in the second argument of a stationary covariance pick up one sign each).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none; the sign and scale convention is this module's and is a reader item.
Does not claim: that this is the covariance of any field; symmetry of the resulting matrices. -/
noncomputable def jetCovL (d : ℕ) (L : ℝ) (x : E d) (l1 : List (E d)) (y : E d) (l2 : List (E d)) : ℝ :=
  (-1 : ℝ) ^ l2.length * dK d L (x - y) (l1 ++ l2)
/-- An observation of the field: a point and the list of differentiation directions (`[]` the value, `[v]` the derivative along `v`, `[v, w]` a second derivative).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none.
Does not claim: anything; bookkeeping. -/
abbrev Obs (d : ℕ) : Type := E d × List (E d)

/-- The `2(d+1)` pin observations: the two values and the `2d` frame gradient components.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none.
Does not claim: anything; bookkeeping. -/
abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)

/-- The `(d+1)`-jet at a witness point: the value (`none`) and the `d` frame gradient components (`some i`). -/
abbrev JetIdx (d : ℕ) : Type := Option (Fin d)

/-- Upper-triangular index pairs, the coordinates of a symmetric `d × d` matrix. -/
abbrev SymIdx (d : ℕ) : Type := {p : Fin d × Fin d // p.1 ≤ p.2}

/-- Hessian coordinates at the three sites `M`, `S`, `x` (in that order). -/
abbrev HessIdx (d : ℕ) : Type := Fin 3 × SymIdx d

/-- The conditioning observations of the second Schur step: the pins and the witness jet. -/
abbrev CondIdx (d : ℕ) : Type := PinIdx d ⊕ JetIdx d

/-- The two pins `M = -(r/2) u`, `S = (r/2) u` as a `Fin 2`-indexed family. -/
noncomputable def pinSite (d : ℕ) (r : ℝ) (u : E d) : Fin 2 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u

/-- The three sites `M`, `S`, `x` of the Hessian coordinates. -/
noncomputable def hessSite (d : ℕ) (r : ℝ) (u x : E d) : Fin 3 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u
  | 2 => x

/-- The pin observations as `Obs`: values at `M`, `S`; gradient components along the frame `R` at `M`, `S`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none; the axial direction is `R 0`, which needs `[NeZero d]`.
Does not claim: anything; bookkeeping. -/
noncomputable def pinDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : PinIdx d → Obs d
  | Sum.inl a => (pinSite d r (R 0) a, [])
  | Sum.inr (a, i) => (pinSite d r (R 0) a, [R i])

/-- The witness `(d+1)`-jet at `x` as `Obs`. -/
noncomputable def jetDesc (d : ℕ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) : JetIdx d → Obs d
  | none => (x, [])
  | some i => (x, [R i])

/-- Pins and witness jet together. -/
noncomputable def condDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : CondIdx d → Obs d
  | Sum.inl a => pinDesc d r R a
  | Sum.inr a => jetDesc d x R a

/-- The Hessian coordinates at `M`, `S`, `x` as `Obs`: second derivatives along frame pairs. -/
noncomputable def hessDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : HessIdx d → Obs d
  | (a, p) => (hessSite d r (R 0) x a, [R p.1.1, R p.1.2])

/-- The covariance matrix between two families of observations, by the stationary rule `jetCovL`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none.
Does not claim: positive semidefiniteness; that it is the covariance of any random vector. -/
noncomputable def covOf (d : ℕ) (L : ℝ) {ι κ : Type} (f : ι → Obs d) (g : κ → Obs d) : Matrix ι κ ℝ :=
  Matrix.of fun a b => jetCovL d L (f a).1 (f a).2 (g b).1 (g b).2

/-- The pin targets: `f(M) = b`, `f(S) = b - k r^3`, gradients `0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none.
Does not claim: anything; the source's observations. -/
noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

/-- The witness-jet target `(y, 0)`: height `y`, gradient `0`. -/
noncomputable def jetTarget (d : ℕ) (y : ℝ) : JetIdx d → ℝ
  | none => y
  | some _ => 0

/-- Pin and witness-jet targets together. -/
noncomputable def condTarget (d : ℕ) (b k r y : ℝ) : CondIdx d → ℝ
  | Sum.inl a => pinTarget d b k r a
  | Sum.inr a => jetTarget d y a

/-- The observation covariance `Sigma_r` of the `2(d+1)` pin observations.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Uniform convergence shows that the covariances of (U_r,Y_x), their inverses, and the relevant Schur complements are uniformly bounded and positive for r<=r_*."
Interface: none.
Does not claim: invertibility (a field of the successor interface, in the pin regime). -/
noncomputable def pinObsCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : Matrix (PinIdx d) (PinIdx d) ℝ :=
  covOf d L (pinDesc d r R) (pinDesc d r R)

/-- The cross-covariance `v_r(x)` between the value at `x` and the pin observations. -/
noncomputable def pinCross (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : PinIdx d → ℝ :=
  fun a => jetCovL d L x [] (pinDesc d r R a).1 (pinDesc d r R a).2

/-- The Gaussian regression mean of `f(x)` given the pin observations, `v_r(x)^T Sigma_r^{-1} (targets)`, with the ordinary matrix inverse.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none; written with the ordinary inverse, the regression when `Sigma_r` is invertible (the field `pinObsCov_posDef`).
Does not claim: invertibility of `pinObsCov`; that this is the mean of any law. -/
noncomputable def regMean (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : ℝ :=
  dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinTarget d b k r))

/-- The Schur-complement covariance `K_L(x - y) - v_r(x)^T Sigma_r^{-1} v_r(y)` of the regression.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: none; written with the ordinary matrix inverse.
Does not claim: invertibility of `pinObsCov`; positive semidefiniteness of the resulting kernel. -/
noncomputable def regCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x y : E d) : ℝ :=
  periodicCovariance d L (x - y) -
    dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinCross d L r R y))

/-- The Hessian matrix of `g` at `x` in the frame `R`, from the module's bilinear `hessian`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "W_r=F_d(H_M) F_(d-1)(H_S),"
Interface: none.
Does not claim: symmetry (true for `C^2` functions, not stated). -/
noncomputable def hessMat (d : ℕ) (g : E d → ℝ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => hessian d g x (R i) (R j)

/-- The typed endpoint weight `W_r = F_d(H_M) F_(d-1)(H_S)` of a sample path: `|det H_M| |det H_S|` when `H_M` is negative
definite and `H_S` has index `d - 1` (through `HessianHasInertia`), `0` otherwise, hence `0` on singular matrices.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "W_r=F_d(H_M) F_(d-1)(H_S),"
Interface: none.
Does not claim: that the frame-dependent determinants agree with the source's (they do for orthonormal frames; not stated). -/
noncomputable def typedWeight (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (g : E d → ℝ) : ℝ :=
  Set.indicator {g : E d → ℝ | HessianHasInertia d g (pinM d r (R 0)) d ∧ HessianHasInertia d g (pinS d r (R 0)) (d - 1)}
    (fun g => |(hessMat d g (pinM d r (R 0)) R).det| * |(hessMat d g (pinS d r (R 0)) R).det|) g

/-- Gaussian density with covariance `C` and mean `m` at `v` on a finite index type, by the ordinary inverse and determinant (not a probability density where `C` is singular).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "The conditional density p_(Y_x|U_r=v_r)(0,t)"
Interface: none.
Does not claim: integrability or normalisation; invertibility of `C`. -/
noncomputable def gaussDensityOn {ι : Type} [Fintype ι] [DecidableEq ι] (C : Matrix ι ι ℝ) (m v : ι → ℝ) : ℝ :=
  (2 * Real.pi) ^ (-(Fintype.card ι : ℝ) / 2) * C.det ^ (-(1 : ℝ) / 2) *
    Real.exp (-(dotProduct (v - m) (Matrix.mulVec C⁻¹ (v - m))) / 2)

/-- First Schur step: the covariance of the witness `(d+1)`-jet at `x` given the pins. -/
noncomputable def jetCovGivenPins (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (JetIdx d) (JetIdx d) ℝ :=
  covOf d L (jetDesc d x R) (jetDesc d x R) -
    covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹ *
      covOf d L (pinDesc d r R) (jetDesc d x R)

/-- First Schur step: the mean of the witness jet given the pin targets. -/
noncomputable def jetMeanGivenPins (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : JetIdx d → ℝ :=
  Matrix.mulVec (covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹) (pinTarget d b k r)

/-- The conditional density `p_(Y_x | U_r = v_r)(0, y)` of the witness jet at `(y, 0)` given the pins.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "The conditional density p_(Y_x|U_r=v_r)(0,t)"
Interface: none.
Does not claim: that it is a density of any law (it is the Gaussian formula at the regression parameters). -/
noncomputable def jetDensityAt (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) : ℝ :=
  gaussDensityOn (jetCovGivenPins d L r R x) (jetMeanGivenPins d L r b k R x) (jetTarget d y)

/-- Second Schur step: the covariance of the three Hessians (at `M`, `S`, `x`) given the pins and the witness jet. -/
noncomputable def hessCovGivenCond (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (HessIdx d) (HessIdx d) ℝ :=
  covOf d L (hessDesc d r R x) (hessDesc d r R x) -
    covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹ *
      covOf d L (condDesc d r R x) (hessDesc d r R x)

/-- Second Schur step: the mean of the three Hessians given the pin and witness-jet targets. -/
noncomputable def hessMeanGivenCond (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) : HessIdx d → ℝ :=
  Matrix.mulVec (covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹)
    (condTarget d b k r y)

/-- The symmetric matrix at site `a` read off the Hessian coordinates (raw upper-triangular entries, no `sqrt 2` scaling). -/
noncomputable def symOfH (d : ℕ) (h : HessIdx d → ℝ) (a : Fin 3) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => if hij : i ≤ j then h (a, ⟨(i, j), hij⟩) else h (a, ⟨(j, i), le_of_not_ge hij⟩)

/-- `F_j(H) = |det H| 1{H has j negative eigenvalues}` for a matrix, the inertia read through a Sylvester decomposition
(a complementary pair of subspaces of dimensions `j`, `d - j` on which the form is negative, respectively positive, definite),
hence `0` on singular matrices.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "F_j(H)=|det H| 1{H has j negative eigenvalues},"
Interface: none.
Does not claim: that the decomposition characterises the inertia of non-symmetric matrices. -/
noncomputable def typedF (d : ℕ) (H : Matrix (Fin d) (Fin d) ℝ) (j : ℕ) : ℝ :=
  Set.indicator {H : Matrix (Fin d) (Fin d) ℝ | ∃ V W : Submodule ℝ (Fin d → ℝ), IsCompl V W ∧
      Module.finrank ℝ V = j ∧ Module.finrank ℝ W = d - j ∧
      (∀ v ∈ V, v ≠ 0 → dotProduct v (H.mulVec v) < 0) ∧ (∀ w ∈ W, w ≠ 0 → 0 < dotProduct w (H.mulVec w))}
    (fun H => |H.det|) H

/-- The Kac-Rice integrand `W_r F_j(H_x) = F_d(H_M) F_(d-1)(H_S) F_j(H_x)` on the Hessian coordinates.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "E[(W_r/r^2) F_j(H_x)|U_r=v_r,Y_x=(0,t)] dt dx. (12)"
Interface: none.
Does not claim: anything; the source's integrand on coordinates. -/
noncomputable def kacRiceWeight (d : ℕ) (j : ℕ) (h : HessIdx d → ℝ) : ℝ :=
  typedF d (symOfH d h 0) d * typedF d (symOfH d h 1) (d - 1) * typedF d (symOfH d h 2) j

/-- The contact kernel `Λ'_j(x; b, k, R)(y)` for normalizer `Z`: `Z^{-1}` times the conditional jet density at `(y, 0)` times
the expectation of the Kac-Rice weight under the Gaussian law (`multivariateGaussian`, Mathlib) of the three Hessians given pins
and jet. This is the integrand of the source's weighted Kac-Rice identity (12), DEFINED here; the parent's free `Λ` (13) is its
`r → 0` limit object and is not tied to it.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "E[(W_r/r^2) F_j(H_x)|U_r=v_r,Y_x=(0,t)] dt dx. (12)"
Interface: none; `multivariateGaussian` is the Dirac mass at `0` for a covariance that is not positive semidefinite and a degenerate Gaussian on a singular one.
Does not claim: integrability; that it equals the source's kernel for any field; positivity or continuity. -/
noncomputable def contactKernel (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (j : ℕ) (x : E d) (y : ℝ) (Z : ℝ) : ℝ :=
  Z⁻¹ * jetDensityAt d L r b k R x y *
    ∫ h, kacRiceWeight d j (fun a => h a)
      ∂(ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (hessMeanGivenCond d L r b k R x y)) (hessCovGivenCond d L r R x))

/-- Successor of `PinnedLawInterface` (numbered, the predecessor unchanged): the given law `Q_r` is characterised as the
Gaussian regression of the `K_L` field on the pin observations, the observation covariance is positive definite in the pin
regime, the given weight `W_r` IS the typed determinant weight of the sample path, the full normalizer has the source's
`z_* r^2` floor on compact windows, the critical set is almost surely finite and the counts are measurable, and the mean window
and all-height counts of a Borel region are given by the weighted Kac-Rice identity with the DEFINED contact kernel. With
these fields the laws, the weight and the mean counts are determined, so the round-4 refuting inhabitants (Dirac laws at a
pinned path with extra critical points) are excluded.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Apply Kac-Rice under Q_r to grad f(x) on the compact remote domain."
Interface: every field is a given object or a hypothesis; nothing is derived. `Q_regression` characterises `Q` through finite vectors of point evaluations only (not sample-path jets), so `Q_pinned` stays a separate field; `pinObsCov_posDef` (source Section 4), `Z_floor` (lifetime-parent (5.5)), `critical_finite` (lifetime-parent Section 9) and the two Kac-Rice identities (source (12); d5-dimension-lift Section 4.7) are the sources' theorems assumed as fields, Mathlib `d13f23b7` having no Kac-Rice formula; the parent's kernel `Λ` stays free; the axial direction is `R 0`, which needs `[NeZero d]`.
Does not claim: existence of an object satisfying this interface; that `Q` is the regression law of any field or that `Q^W` is the sources' typed law (no field is constructed); the sign and scale conventions of `jetCovL`; integrability of the weight or of the counts; anything about the predecessor interface, which is unchanged. -/
structure PinnedLawInterface2 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    extends PinnedLawInterface d L Ω where
  /-- Section 4 clause "the covariances of (U_r,Y_x), their inverses, and the relevant Schur complements are uniformly bounded
  and positive": the observation covariance is positive definite in the pin regime, so the ordinary-inverse formulas are the
  regression. -/
  pinObsCov_posDef : ∀ (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L → (pinObsCov d L r R).PosDef
  /-- Section 1 clause "Q_r denotes continuous Gaussian regression on these 2(d+1) observations": under `Q`, every finite
  vector of point evaluations of the field is the multivariate Gaussian with the regression mean and the Schur-complement
  covariance. -/
  Q_regression : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (m : ℕ) (x : Fin m → E d),
      Measure.map (fun ω => WithLp.toLp 2 (fun i => F ω (x i))) (Q r b k R) =
        ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (fun i => regMean d L r b k R (x i)))
          (Matrix.of fun i j => regCov d L r R (x i) (x j))
  /-- Section 1 clause `W_r=F_d(H_M) F_(d-1)(H_S)`: the given weight is the typed determinant weight of the sample path. -/
  W_det : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), W r b k R ω = typedWeight d r R (F ω)
  /-- lifetime-parent (5.5) `0 < z_* <= Z_r / r^2`: the full normalizer has a `z_* r^2` floor on compact windows `b ∈ B`,
  `k ∈ K` with `k > 0`, for `0 < r ≤ r_*` (a theorem of the source, assumed as a field). -/
  Z_floor : ∀ bLo bHi kLo kHi : ℝ, 0 < kLo → kLo ≤ kHi → bLo ≤ bHi →
    ∃ zStar : ℝ, 0 < zStar ∧ ∃ rStar : ℝ, 0 < rStar ∧ rStar < L ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), zStar * r ^ 2 ≤ ∫ ω, W r b k R ω ∂(Q r b k R)
  /-- lifetime-parent Section 9 "Q is almost surely Morse": the critical set of a sample path in the fundamental domain is
  finite, `Q^W`-almost surely, in the pin regime. -/
  critical_finite : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(QW r b k R), {x : E d | x ∈ FundDomain d L ∧ IsCriticalPoint d (F ω) x}.Finite
  /-- The window and all-height counts are measurable functions of the sample for every Borel region. -/
  count_measurable : ∀ (b k r : ℝ) (j : ℕ) (A : Set (E d)), MeasurableSet A →
    Measurable (fun ω => (windowCount d (F ω) b k r j A : ℝ)) ∧ Measurable (fun ω => (allHeightCount d (F ω) j A : ℝ))
  /-- The weighted Kac-Rice identity (12): the tilted mean index-`j` window count of a Borel region of the fundamental domain
  is the integral of the contact kernel over the region and the height window (a theorem of the sources, assumed as a field). -/
  kacRice_window : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (windowCount d (F ω) b k r j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y in Set.Ioo (b - k * r ^ 3) b, contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))
  /-- The all-height analogue (d5-dimension-lift Section 4.7): the height integral runs over all of `ℝ`. -/
  kacRice_allHeight : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (allHeightCount d (F ω) j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y, contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))

/-- Successor of `FixedRemoteMeanMeasureA2` over the successor interface: shape of Theorem A, display (2) (fixed-remote mean measure), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "- integral_E Lambda_j(x;b,k,u) dx |"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `FixedRemoteMeanMeasureA2` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` (sample paths, laws `Q_r`, `Q_r^W`, weight, kernel `Λ`) is a given object; the statement is about any `Λ` and any laws satisfying `PinnedLawInterface`; the Kac-Rice representation (12) and the contact formula (13) are not transcribed.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `FixedRemoteMeanMeasureA2`, which is unchanged; the predecessor's own exclusions: the result or its proof (uniform remote nondegeneracy, coupling (5), weight expansion (10), normalizer (11)); identification of `I.Λ` with the kernel (13); the total-variation convergence remark; the multi-witness Section 6; any numerical `C` or `r_*`; uniformity as `ρ → 0`; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN is transcribed by the register, not here). -/
def FixedRemoteMeanMeasureA2B (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  FixedRemoteMeanMeasureA2 d L Ω I.toPinnedLawInterface ρ bLo bHi kLo kHi

/-- Successor of `FixedRemoteCountBoundA3` over the successor interface: shape of Theorem A, display (3) (fixed-remote count and event bounds), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "P_(Q_r^W){N_(r,j)(E)>=1} <= C k r^3 |E|."
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `FixedRemoteCountBoundA3` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object as in `FixedRemoteMeanMeasureA2`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `FixedRemoteCountBoundA3`, which is unchanged; the predecessor's own exclusions: a matching lower bound on the event probability (the source says this is NOT one); the unnormalized numerator bound (14); the result or its proof; any numerical constant; nonauthor review or acceptance. -/
def FixedRemoteCountBoundA3B (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  FixedRemoteCountBoundA3 d L Ω I.toPinnedLawInterface ρ bLo bHi kLo kHi

/-- Successor of `FixedAnnulusWindowCandidateA2` over the successor interface: shape of the fixed-annulus height-window candidate statement (A2), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md sha256 1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b
Anchor: "**Candidate theorem.** There are C,r_*>0, uniform in the frame, compact marks, E0 and j=0,1,2, such that"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `FixedAnnulusWindowCandidateA2` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `r E0` is embedded in the torus through the frame by `scaledAt`; the six-pin regression (A3)-(A5), the normalizer (A8) and the weighted Kac-Rice formula (A21) are not transcribed.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `FixedAnnulusWindowCandidateA2`, which is unchanged; the predecessor's own exclusions: the result or its proof (contact vector (A10), eigenfloor (A12), cutoff (A13), three-Hessian estimate (A18), inner-strip bound (A26)); the all-index sum or the Markov event bound; an all-height assertion; uniformity in `L`, as `k → 0`, as `B0 → ∞`, or for `d ≥ 3`; any numerical constant; nonauthor review or acceptance (the landing disposition REVIEWED_SCOPED is transcribed by the register, not here). -/
def FixedAnnulusWindowCandidateA2B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi A0 B0 : ℝ) : Prop :=
  FixedAnnulusWindowCandidateA2 L Ω I.toPinnedLawInterface bLo bHi kLo kHi A0 B0

/-- Successor of `AnnulusBridgeAllHeight2` over the successor interface: shape of the all-height fixed-annulus bridge candidate statement, display (2), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_annulus_bridge_20260925/PROOF.md sha256 d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05
Anchor: "**Candidate theorem.** There are C,r_*>0, uniform in the stated marks, frames, Borel E and j=0,1,2, such that"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `AnnulusBridgeAllHeight2` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `rE` is embedded through the frame by `scaledAt`; the intensity formula (16) and the normalizer (5) are not transcribed.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `AnnulusBridgeAllHeight2`, which is unchanged; the predecessor's own exclusions: the result or its proof (R1 subtraction and remainders, R2 full-rank compactification (12), R3 conditional moments (14), R4 three-Hessian bound (6)-(8), R5 original `Z`, R6 weighted Kac-Rice, R7 crossover and cover); the rewriting with a `k` factor through `k_-`; uniformity as `k → 0`; pin neighbourhoods, `d ≥ 3`, intermediate distances or collisions; any numerical constant; nonauthor review or acceptance (the R1-R7 ACCEPT record is transcribed by the register, not here). -/
def AnnulusBridgeAllHeight2B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi A B : ℝ) : Prop :=
  AnnulusBridgeAllHeight2 L Ω I.toPinnedLawInterface bLo bHi kLo kHi A B

/-- Successor of `D5aPuncturedPinDisks` over the successor interface: shape of row D5-a (P2, punctured pin disks), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "for Borel `E ⊂ {0 < p² + q² ≤ 1/16}`, and the same at `S`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5aPuncturedPinDisks` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the chart `M + rE` is `scaledAt` based at the pin.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5aPuncturedPinDisks`, which is unchanged; the predecessor's own exclusions: the P2 source (PUNCTURED_PIN_PROOF.md) or its proof; the P2 continuum review or the proposed register transitions of the reconciliation; any numerical constant; nonauthor review or acceptance. -/
def D5aPuncturedPinDisksB (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5aPuncturedPinDisks L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `D5bMicrodisk` over the successor interface: shape of row D5-b (microdisk), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "**D5-b** (microdisk) | the nested disks of radius `κr²` about each pin, for fixed `κ`: `O(r⁵)`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5bMicrodisk` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the disk is punctured at the pin (the pin itself is a critical point under the law), following the dimension-lift reading "nested ball" inside the punctured ball `D`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5bMicrodisk`, which is unchanged; the predecessor's own exclusions: the P2 source or its nested-disk corollary; any numerical constant; uniformity in `κ`; nonauthor review or acceptance. -/
def D5bMicrodiskB (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5bMicrodisk L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `D5cCollarC1` over the successor interface: shape of row D5-c (C1, compact collar), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "**D5-c** (C1) | `E_{Q_r^W} N_j(rE) ≤ C r³|E|` for Borel `E ⊂ C_{η,R}`, `η ≤ 1/4`, fixed `R ≥ 1`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5cCollarC1` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `C_{η,R}` is `collar 2 η R` (definition taken from the dimension-lift source, since RECONCILIATION.md does not define it).)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5cCollarC1`, which is unchanged; the predecessor's own exclusions: the C1 source (COLLAR_PROOF.md) or its proof; the normalizer (P14) it imports; any numerical constant; nonauthor review or acceptance. -/
def D5cCollarC1B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5cCollarC1 L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `D5dSummedPinNeighbourhoodC2` over the successor interface: shape of row D5-d (C2, the summed pin-neighbourhood bound), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "the same on `{|(u,v)| ≤ R} ∖ {(±1/2, 0)}`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5dSummedPinNeighbourhoodC2` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5dSummedPinNeighbourhoodC2`, which is unchanged; the predecessor's own exclusions: the composition `D5-a + D5-c at η = 1/4` as a proof; the "collar to the reviewed annulus" remark; any numerical constant; nonauthor review or acceptance. -/
def D5dSummedPinNeighbourhoodC2B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5dSummedPinNeighbourhoodC2 L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `D5eIntermediateShellsI3I4` over the successor interface: shape of row D5-e (I3/I4, intermediate shells), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "the dyadic shells and `{A_0 r ≤ |X| ≤ ρ}`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5eIntermediateShellsI3I4` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the side conditions `A_0 ≥ 4`, `A_0 r < ρ ≤ s_0` are taken from the dimension-lift display (1.6), since RECONCILIATION.md states the row without them.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5eIntermediateShellsI3I4`, which is unchanged; the predecessor's own exclusions: the single-shell display (1.5) (stated in `DimensionLiftIntermediateShellsI`); the I3/I4 source (intermediate_window_20260928) or its proof; any numerical constant; nonauthor review or acceptance. -/
def D5eIntermediateShellsI3I4B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5eIntermediateShellsI3I4 L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `D5fGlobalWindowFirstMomentI5` over the successor interface: shape of row D5-f (I5, global window first moment and planar event order), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "planar `c r³ ≤ P(A_r) ≤ C r³`"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `D5fGlobalWindowFirstMomentI5` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `A_r` is the event `{N ≥ 1}` for the all-index torus-wide window count.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `D5fGlobalWindowFirstMomentI5`, which is unchanged; the predecessor's own exclusions: the tiling composition `D5-d (R = 4) + I4 + remote Theorem A` as a proof; the lower bound's source (remote_collision Corollary F-) or the I5 source; any numerical constant; nonauthor review or acceptance; any register transition proposed by the reconciliation; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def D5fGlobalWindowFirstMomentI5B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  D5fGlobalWindowFirstMomentI5 L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `DimensionLiftPuncturedPinBallP` over the successor interface: shape of Theorem P_d (all-height punctured pin ball), display (1.2), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "**Theorem P_d (all-height punctured pin ball).**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `DimensionLiftPuncturedPinBallP` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the chart is `scaledAt` based at the pin; the weight is `omegaD`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `DimensionLiftPuncturedPinBallP`, which is unchanged; the predecessor's own exclusions: the result or its proof (devices D1-D5, stable gradient frame, density and conditional moments, weighted Kac-Rice (4.10), crossover); the nested-ball `O(r^5)` corollary (stated for `d = 2` as `D5bMicrodisk`); the planar P2 identification; any numerical constant; uniformity in `d`; nonauthor review or acceptance. -/
def DimensionLiftPuncturedPinBallPB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  DimensionLiftPuncturedPinBallP d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `DimensionLiftCompactCollarC` over the successor interface: shape of Theorem C_d (compact collar), display (1.3), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "**Theorem C_d (compact collar).**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `DimensionLiftCompactCollarC` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `C(η, R)` is `collar d η R`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `DimensionLiftCompactCollarC`, which is unchanged; the predecessor's own exclusions: the consequence (1.4) for the scaled ball with pins removed (stated for `d = 2` as `D5dSummedPinNeighbourhoodC2`); the result or its proof; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftCompactCollarCB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  DimensionLiftCompactCollarC d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `DimensionLiftIntermediateShellsI` over the successor interface: shape of Theorem I_d (intermediate height-window shells), displays (1.5) and (1.6), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "**Theorem I_d (intermediate height-window shells).**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `DimensionLiftIntermediateShellsI` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `DimensionLiftIntermediateShellsI`, which is unchanged; the predecessor's own exclusions: the result or its proof (polynomial row rank, Euler suppression, shell ledger, dyadic sum); the planar I3/I4 identification; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftIntermediateShellsIB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  DimensionLiftIntermediateShellsI d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `DimensionLiftGlobalWindowG` over the successor interface: shape of Theorem G_d (global height-window first moment), display (1.7), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "**Theorem G_d (global height-window first moment).**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `DimensionLiftGlobalWindowG` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `DimensionLiftGlobalWindowG`, which is unchanged; the predecessor's own exclusions: the tiling proof (P_d, C_d, I_d and the fixed-remote Theorem A); Corollaries F_d and M_d; the planar I5 identification; the `d = 3` SIDE24 application; any regional supersession of historical carriers; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftGlobalWindowGB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  DimensionLiftGlobalWindowG d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `FourierCutoffPlanarSecondFactorialF4` over the successor interface: shape of the planar Fourier-cutoff consequence (F4), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_fourier_cutoff_20260929/PROOF.md sha256 c1692379a3a066589bd2522aaa5b4d480e736b737c8c39a474d1735185793733
Anchor: "For d=2, G_2 is the reviewed planar I5 composition under the complete D5"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `FourierCutoffPlanarSecondFactorialF4` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `N` is `torusWindowCount`; `(N)_2 = N(N-1)` is `Nat.descFactorial N 2`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `FourierCutoffPlanarSecondFactorialF4`, which is unchanged; the predecessor's own exclusions: Theorem F (the cap `Psi_F` and its tail (F1)-(F2)); Theorem W for general `d` and `p` (F3); that `G_2` holds (the source takes it from the reviewed planar I5 composition); Proposition B; that the logarithm is sharp or removable; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def FourierCutoffPlanarSecondFactorialF4B (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  FourierCutoffPlanarSecondFactorialF4 L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `PalmRouteFactorialMomentsQ` over the successor interface: shape of Theorem Q (Palm route, factorial moments in every fixed dimension), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "**Theorem Q (factorial moments, every fixed dimension).**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `PalmRouteFactorialMomentsQ` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `N` is `torusWindowCount`; `(N)_q` is `Nat.descFactorial N q`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `PalmRouteFactorialMomentsQ`, which is unchanged; the predecessor's own exclusions: the result or its proof (the cap `Psi` in `d` dimensions, the parametric Rouché tail, the marked Kac-Rice formula with Hölder insertion, the regimes R1-R4); the dependence on [C6L] and [DL] at their bytes; a lower bound for `q ≥ 3`; Corollary P; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def PalmRouteFactorialMomentsQB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  PalmRouteFactorialMomentsQ d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `PalmRouteSecondFactorialTheta` over the successor interface: shape of Corollary Theta (the C6 order is `r^3`), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "**Corollary Theta.** In every fixed `d >= 2`,"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `PalmRouteSecondFactorialTheta` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; `N` is `torusWindowCount`.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `PalmRouteSecondFactorialTheta`, which is unchanged; the predecessor's own exclusions: the lower bounds' source ([EDL] (A4)) or the upper bounds' sources (Theorem Q, [DL] Corollary M_d); that the logarithm of the Fourier cutoff is removed as a reviewed fact; any statement about the positions of the pairs; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def PalmRouteSecondFactorialThetaB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  PalmRouteSecondFactorialTheta d L Ω I.toPinnedLawInterface bLo bHi kLo kHi

/-- Successor of `RemoteCollisionSecondFactorialD` over the successor interface: shape of Corollary D (second factorial moment in the fixed remote region), for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the weight and the mean counts and exclude the round-4 refuting inhabitant.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/remote_collision_20260928/PROOF.md sha256 b9b8b58fd8266db7ffe6537078003888ef138b445d3289ec5588f116af9050c2
Anchor: "Q_r^W{ N(E) >= 2 } <= C r^5 |E|."
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `RemoteCollisionSecondFactorialD` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the pair Kac-Rice formula (3.1) and the divided-difference coordinates (4.1) are not transcribed.)
Does not claim: the result (statement only); existence of an inhabitant; anything about `RemoteCollisionSecondFactorialD`, which is unchanged; the predecessor's own exclusions: Theorem C (the ordered near-pair count) or its proof (Lemmas 1-5); Corollaries E and F; a torus-wide second factorial moment or `math.rn-region.witness-collision` in its full sense; pairs with a witness near the pins; equality of `N(E)` with the all-critical-point count `windowCountAll` (they agree only off the degenerate locus); any numerical constant; nonauthor review or acceptance. -/
def RemoteCollisionSecondFactorialDB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  RemoteCollisionSecondFactorialD d L Ω I.toPinnedLawInterface ρ bLo bHi kLo kHi

/-- Successor of `WitnessCollisionNearPinPairsTarget` over the successor interface: the OPEN target (stated as a target and not as a result) over any object satisfying `PinnedLawInterface2`. The source marks this target OPEN; no field set makes it provable, and the successor is registered `specified` with that said.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "**D5-open (catalog C6): shrinking multiple-witness collision.**"
Interface: `I : PinnedLawInterface2 …` is given; the Prop is `WitnessCollisionNearPinPairsTarget` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor's reading is the successor's (predecessor: `I` is a given object; the pair count is `nearPinPairCount`.)
Does not claim: that this target holds, is attainable, or is implied by any reviewed source or by any field of the successor interface (the source marks it OPEN); the result (statement only); existence of an inhabitant; anything about `WitnessCollisionNearPinPairsTarget`, which is unchanged; the predecessor's own exclusions: that this target holds, is attainable, or is implied by any reviewed source (the Palm route's Theorem Q is an author-side candidate whose `q = 2` case would imply it; that candidate is stated separately as `PalmRouteFactorialMomentsQ` and is not consumed here); that the lower obstruction `E N(N−1) ≥ 2c r³` is stated here; the optimal order; any status transition of the node. -/
def WitnessCollisionNearPinPairsTargetB (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface2 d L Ω) (η bLo bHi kLo kHi : ℝ) : Prop :=
  WitnessCollisionNearPinPairsTarget d L Ω I.toPinnedLawInterface η bLo bHi kLo kHi
