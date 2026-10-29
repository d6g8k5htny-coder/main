import Mathlib

/-!
# Shared objects of the periodized Gaussian lifetime track (statements only)

Scientific effect: NONE. Formal progress of every declaration here: `specified`.
Organizational-independence credit of this transcription: 0 (Anthropic / Claude).

This module declares, as definitions and interface structures, the objects that the
Layer 0 texts of the lifetime track share: the periodized Gaussian covariance `K_L`
(as a quotient of theta sums written with `tsum`), the centred stationary Gaussian field
on the torus `R^d/(L Z^d)` (an interface: probability space, smooth `L`-periodic sample
paths, Gaussian law of every finite real linear combination of point evaluations, and the
covariance identity), critical points, the Hessian form through `iteratedFDeriv`, the
inertia predicates for nondegenerate maxima and index-(d-1) saddles, ordered
maximum/saddle pairs with their height gap and scaled gap, the superlevel H0 bar and
elder-partner interface, and the per-unit-volume expected lifetime density "versions".

What this module does not establish. It consists only of definitions and structures;
it states nothing as a result and proves nothing. It does not construct the Gaussian
field (the existence of a process with these finite-dimensional laws and smooth periodic
sample paths is a hypothesis field), does not prove summability of the theta sums (a
hypothesis field), does not identify the elder-partner relation with any Kac-Rice or
persistence-module construction, does not show that lifetime density versions exist,
does not assert almost-sure Morse genericity, and does not assert that an object
satisfying an interface here coincides with the object of the source texts. Nothing here
promotes, reclassifies or discharges any claim, premise or obligation of the registers;
alignment of these statements with the sources is PENDING_INDEPENDENT_REVIEW.

Primary source (Layer 0, byte-pinned): `d6g8k5htny-coder/Math-` commit
`c39164254ad1529ed49543743598d5dcb24bab98`, path
`imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`,
sha256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`
(source id `lifetime-parent`).
-/

namespace UniversalLaw.Spec.Field

open MeasureTheory

/-- The ambient Euclidean space `R^d` in which the torus `X = R^d/(L Z^d)` is realised
by `L`-periodic functions.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "On the flat torus X = R^d/(L Z^d), let f be the centered stationary Gaussian field of variance one with covariance"
Interface: none; this is the ambient space `EuclideanSpace ℝ (Fin d)`.
Does not claim: that the torus is represented by any quotient type; periodicity is a hypothesis on functions. -/
abbrev E (d : ℕ) : Type := EuclideanSpace ℝ (Fin d)

/-- The lattice vector `L n` for `n ∈ Z^d`, written in the standard coordinate vectors.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)"
Interface: none.
Does not claim: anything beyond the definition of the vector `L n`. -/
noncomputable def latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) : E d :=
  ∑ i, EuclideanSpace.single i (L * (n i : ℝ))

/-- The unnormalised periodized Gaussian (theta) sum `sum_{n in Z^d} exp(-|z+Ln|^2/2)`,
written as a `tsum` over `Z^d`. If the family were not summable the `tsum` would be `0`;
summability is a hypothesis field of `GaussianFieldInterface`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)"
Interface: summability of the family is the field `GaussianFieldInterface.summable`, not derived here.
Does not claim: convergence of the sum. -/
noncomputable def thetaSum (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)

/-- The periodized Gaussian covariance
`K_L(z) = sum_n exp(-|z+Ln|^2/2) / sum_n exp(-|Ln|^2/2)`; the denominator is the theta
sum at `z = 0`, so `K_L(0) = 1` (variance one) whenever that denominator is nonzero (the
summability field is meant to provide this; not shown here).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)"
Interface: summability of both theta sums is the field `GaussianFieldInterface.summable`.
Does not claim: positivity of the Fourier coefficients (Section 2 of the source) or any property of `K_L` beyond its definition. -/
noncomputable def periodicCovariance (d : ℕ) (L : ℝ) (z : E d) : ℝ :=
  thetaSum d L z / thetaSum d L 0

/-- The `K_L`-quadratic form `sum_{i,j} a_i a_j K_L(x_i - x_j)` of a finite real linear
combination `sum_i a_i f(x_i)` of point evaluations: its variance under a centred
stationary field with covariance `K_L`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "let f be the centered stationary Gaussian field of variance one with covariance"
Interface: nonnegativity of this form is the field `GaussianFieldInterface.quad_nonneg`.
Does not claim: positive definiteness (Section 2 of the source). -/
noncomputable def covQuadForm (d : ℕ) (L : ℝ) {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) : ℝ :=
  ∑ i, ∑ j, a i * a j * periodicCovariance d L (x i - x j)

/-- `L`-periodicity of a function on `R^d`: invariance under every lattice translation.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "On the flat torus X = R^d/(L Z^d), let f be the centered stationary Gaussian field of variance one with covariance"
Interface: none; this is the defining predicate of a function on the torus.
Does not claim: anything about the field. -/
def IsLPeriodic (d : ℕ) (L : ℝ) (g : E d → ℝ) : Prop :=
  ∀ (x : E d) (n : Fin d → ℤ), g (x + latticeVec d L n) = g x

/-- Smoothness of a sample path: `C^n` for every finite `n` (written without `⊤`).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "In particular the field has a smooth version and all finite moments of its global derivative suprema exist."
Interface: smoothness of every sample path is the field `GaussianFieldInterface.smooth`.
Does not claim: existence of a smooth version, or moment bounds on derivative suprema. -/
def IsSmooth (d : ℕ) (g : E d → ℝ) : Prop :=
  ∀ n : ℕ, ContDiff ℝ n g

/-- Interface for the centred stationary Gaussian field of variance one on the torus
`R^d/(L Z^d)` with covariance `K_L`. The field is GIVEN: its probability space, its sample
paths, their smoothness and periodicity, the Gaussian law of every finite real linear
combination of point evaluations (law `gaussianReal 0 v` with `v` the `K_L`-quadratic
form) and the covariance identity are hypothesis fields, as is summability of the theta
sums. Existence of such an object is not asserted.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "On the flat torus X = R^d/(L Z^d), let f be the centered stationary Gaussian field of variance one with covariance"
Interface: every field of this structure is a given property of the field, never derived.
Does not claim: existence or uniqueness in law of the field; the Fourier representation and positive spectrum of Section 2; almost-sure Morse genericity of Section 8. -/
structure GaussianFieldInterface (d : ℕ) (L : ℝ) where
  /-- The underlying probability space (sample space). -/
  Ω : Type
  -- Its sigma-algebra (square-bracket field, as in Mathlib's bundled categories). It is not
  -- registered for typeclass resolution after this declaration; consumers bring it into
  -- scope with `haveI := G.mΩ`.
  [mΩ : MeasurableSpace Ω]
  /-- The probability measure `P`. -/
  P : Measure Ω
  -- `P` is a probability measure (square-bracket field).
  [isProb : IsProbabilityMeasure P]
  /-- The sample paths `ω ↦ f_ω : R^d → ℝ`. -/
  F : Ω → E d → ℝ
  /-- Section 2 clause: "the field has a smooth version"; every sample path is smooth. -/
  smooth : ∀ ω, IsSmooth d (F ω)
  /-- Section 1 clause: the field lives "On the flat torus X = R^d/(L Z^d)"; every sample path
  is `L`-periodic. -/
  periodic : ∀ ω, IsLPeriodic d L (F ω)
  /-- Section 1 clause: the theta sums defining `K_L` converge (taken as a hypothesis). -/
  summable : ∀ z : E d, Summable (fun n : Fin d → ℤ => Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2))
  /-- The `K_L`-quadratic form is nonnegative (so it is a variance). -/
  quad_nonneg : ∀ {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ), 0 ≤ covQuadForm d L x a
  /-- Section 1 clause: "the centered stationary Gaussian field of variance one with covariance"
  `K_L`; every finite real linear combination of point evaluations has the centred Gaussian
  law whose variance is the `K_L`-quadratic form. -/
  gaussian : ∀ {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ),
    Measure.map (fun ω => ∑ i, a i * F ω (x i)) P =
      ProbabilityTheory.gaussianReal 0 ⟨covQuadForm d L x a, quad_nonneg x a⟩
  /-- Covariance identity `E[f(x) f(y)] = K_L(x - y)` (centred, stationary). -/
  covariance : ∀ x y : E d, ∫ ω, F ω x * F ω y ∂P = periodicCovariance d L (x - y)

/-- The Hessian of `g` at `x` as a bilinear form, through the second iterated Fréchet
derivative: `hessian d g x v w = D^2 g(x)[v, w]`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "A maximum has d negative Hessian eigenvalues; the relevant saddle has d-1 negative and one positive eigenvalue."
Interface: none.
Does not claim: symmetry of the form (true for `C^2` functions, not stated here). -/
noncomputable def hessian (d : ℕ) (g : E d → ℝ) (x v w : E d) : ℝ :=
  iteratedFDeriv ℝ 2 g x ![v, w]

/-- Critical point: vanishing Fréchet derivative (`grad f = 0`).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "grad f(M)=grad f(S)=0."
Interface: none.
Does not claim: differentiability of `g` at `x` (if `g` is not differentiable, `fderiv` is `0`). -/
def IsCriticalPoint (d : ℕ) (g : E d → ℝ) (x : E d) : Prop :=
  fderiv ℝ g x = 0

/-- Sylvester-inertia encoding of "the Hessian at `x` is nondegenerate with exactly `j`
negative and `d - j` positive eigenvalues": there are complementary subspaces `V`, `W` of
dimensions `j` and `d - j` on which the Hessian form is negative definite, respectively
positive definite. Intended to agree with the eigenvalue count for symmetric (`C^2`)
Hessians; not shown here.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "The index counts negative eigenvalues."
Interface: none.
Does not claim: equivalence with an eigenvalue count for a non-symmetric form. -/
def HessianHasInertia (d : ℕ) (g : E d → ℝ) (x : E d) (j : ℕ) : Prop :=
  ∃ V W : Submodule ℝ (E d), IsCompl V W ∧
    Module.finrank ℝ V = j ∧ Module.finrank ℝ W = d - j ∧
    (∀ v ∈ V, v ≠ 0 → hessian d g x v v < 0) ∧
    (∀ w ∈ W, w ≠ 0 → 0 < hessian d g x w w)

/-- Nondegenerate local maximum: a critical point whose Hessian has `d` negative
eigenvalues (negative definite).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "A maximum has d negative Hessian eigenvalues; the relevant saddle has d-1 negative and one positive eigenvalue."
Interface: none.
Does not claim: that every local maximum of a sample path is nondegenerate (Morse genericity is Section 8 of the source). -/
def IsNondegMax (d : ℕ) (g : E d → ℝ) (x : E d) : Prop :=
  IsCriticalPoint d g x ∧ HessianHasInertia d g x d

/-- Index-(d-1) saddle: a critical point whose Hessian has `d - 1` negative and one
positive eigenvalue.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "A maximum has d negative Hessian eigenvalues; the relevant saddle has d-1 negative and one positive eigenvalue."
Interface: none.
Does not claim: that such a saddle is a merging saddle of any superlevel bar. -/
def IsIndexSaddle (d : ℕ) (g : E d → ℝ) (x : E d) : Prop :=
  IsCriticalPoint d g x ∧ HessianHasInertia d g x (d - 1)

/-- The pin `M = -(r/2) u` of the embedded local cylinder with axial unit vector `u`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "M=-(r/2)u,     S=(r/2)u,"
Interface: none.
Does not claim: that `u` is a unit vector; this pin is an auxiliary definition consumed by no statement of this packet (the typed pair law `p` is a primitive), so it is attached to no claim. -/
noncomputable def pinM (d : ℕ) (r : ℝ) (u : E d) : E d :=
  -((r / 2) • u)

/-- The pin `S = (r/2) u` of the embedded local cylinder with axial unit vector `u`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "M=-(r/2)u,     S=(r/2)u,"
Interface: none.
Does not claim: that `u` is a unit vector; this pin is an auxiliary definition consumed by no statement of this packet (the typed pair law `p` is a primitive), so it is attached to no claim. -/
noncomputable def pinS (d : ℕ) (r : ℝ) (u : E d) : E d :=
  (r / 2) • u

/-- Torus distance between two points of `R^d`: the infimum over the lattice of
`|x - y + L n|`. For `L > 0` it is intended to vanish exactly when `x` and `y` represent the
same torus point; not shown here.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Count ordered maximum/saddle pairs with 0<distance<=r_*, birth in B, and scaled gap (f(M)-f(S))/distance^3 in K."
Interface: none.
Does not claim: anything about the injectivity radius of the torus. -/
noncomputable def torusDist (d : ℕ) (L : ℝ) (x y : E d) : ℝ :=
  ⨅ n : Fin d → ℤ, ‖x - y + latticeVec d L n‖

/-- The fundamental domain `[0, L)^d`: each torus point has exactly one representative in
it, so counting objects with their base point in it counts objects on the torus.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Stationarity converts the spatial integral into a per-unit-volume density."
Interface: none.
Does not claim: anything about stationarity; "per unit volume" is encoded by dividing a count over this domain by `L^d`. -/
def FundDomain (d : ℕ) (L : ℝ) : Set (E d) :=
  {x | ∀ i, 0 ≤ x i ∧ x i < L}

/-- Height gap (lifetime) `f(M) - f(S)` of an ordered pair.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "let nu_cand^all(ell) count ALL ordered maximum/index-(d-1)-saddle pairs with height difference ell>0, and let nu_eld^all(ell) count all finite ordinary superlevel H0 persistence bars, both in expectation per unit volume."
Interface: none.
Does not claim: that the height gap of an elder pair is the persistence of a bar in any persistence module. -/
def heightGap (d : ℕ) (g : E d → ℝ) (M S : E d) : ℝ :=
  g M - g S

/-- Scaled gap `(f(M) - f(S)) / distance^3` of an ordered pair, with the torus distance.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Count ordered maximum/saddle pairs with 0<distance<=r_*, birth in B, and scaled gap (f(M)-f(S))/distance^3 in K."
Interface: none.
Does not claim: anything when the distance is `0` (division by zero gives `0` in Lean). -/
noncomputable def scaledGap (d : ℕ) (L : ℝ) (g : E d → ℝ) (M S : E d) : ℝ :=
  heightGap d g M S / torusDist d L M S ^ 3

/-- Ordered maximum / index-(d-1)-saddle pair at positive torus distance (distinct torus
points): a nondegenerate local maximum `M` and an index-(d-1) saddle `S`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "let nu_cand^all(ell) count ALL ordered maximum/index-(d-1)-saddle pairs with height difference ell>0, and let nu_eld^all(ell) count all finite ordinary superlevel H0 persistence bars, both in expectation per unit volume."
Interface: none.
Does not claim: positivity of the height gap (added separately where the source requires it). -/
def IsCandidatePair (d : ℕ) (L : ℝ) (g : E d → ℝ) (M S : E d) : Prop :=
  IsNondegMax d g M ∧ IsIndexSaddle d g S ∧ 0 < torusDist d L M S

/-- Morse genericity with distinct critical values on the torus: every critical point is
nondegenerate (has some inertia) and critical points at distinct torus positions have
distinct critical values.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Countable exhaustion now proves that Q is almost surely Morse with distinct critical values."
Interface: none; used only as a hypothesis of the elder-partner interface clause.
Does not claim: that sample paths are almost surely Morse (Section 8 of the source). -/
def IsMorseDistinct (d : ℕ) (L : ℝ) (g : E d → ℝ) : Prop :=
  (∀ x : E d, IsCriticalPoint d g x → ∃ j : ℕ, HessianHasInertia d g x j) ∧
  (∀ x y : E d, IsCriticalPoint d g x → IsCriticalPoint d g y →
    0 < torusDist d L x y → g x ≠ g y)

/-- Interface for the superlevel-set H0 persistence objects used by the lifetime track:
the maximin death height `d_f(M)`, the essential (global-maximum) class, and the ordinary
elder death partner relation. These are GIVEN relations; the fields transcribe the source
clauses that characterise them on smooth periodic functions, and the elder relation is
pinned only on the Morse distinct-value locus, exactly as the source does.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "On the Morse distinct-value locus it expresses ordinary elder death at S; an essential maximum has no older endpoint."
Interface: `deathHeight`, `IsEssential`, `ElderPartner` are given; `deathHeight_maximin`, `essential_iff`, `elderPartner_iff` are their hypothesis clauses.
Does not claim: Borel measurability of the elder event (Section 8 / the Section 9 replacement), identification with a persistence-module elder rule, or any behaviour off the Morse distinct-value locus. -/
structure SuperlevelBarInterface (d : ℕ) (L : ℝ) where
  /-- Section 8 clause: the maximin connection level
  `d_f(M)=sup_{gamma(0)=M, f(gamma(1))>f(M)} min_t f(gamma(t))`. -/
  deathHeight : (E d → ℝ) → E d → ℝ
  /-- The essential class: the global maximum, which "has no older endpoint" and whose
  superlevel H0 bar is not finite. -/
  IsEssential : (E d → ℝ) → E d → Prop
  /-- Section 1 clause: "the global ordinary superlevel elder death partner of M is S". -/
  ElderPartner : (E d → ℝ) → E d → E d → Prop
  /-- `M` is essential exactly when no point is strictly higher (so the supremum in the
  maximin is over an empty set, "equal to -infinity"). -/
  essential_iff : ∀ (g : E d → ℝ), IsSmooth d g → IsLPeriodic d L g →
    ∀ M : E d, IsEssential g M ↔ ∀ y : E d, g y ≤ g M
  /-- Section 8 maximin clause, in the form `h < d_f(M)` iff some path from `M` to a strictly
  higher point stays strictly above `h`. -/
  deathHeight_maximin : ∀ (g : E d → ℝ), IsSmooth d g → IsLPeriodic d L g →
    ∀ M : E d, ¬ IsEssential g M → ∀ h : ℝ,
      h < deathHeight g M ↔
        ∃ y : E d, g M < g y ∧ ∃ γ : Path M y, ∀ t : unitInterval, h < g (γ t)
  /-- Section 8 / Section 14 clause: on the Morse distinct-value locus, "equality
  d_f(M)=f(S)" with `M` a nondegenerate maximum and `S` an index-(d-1) saddle "expresses
  ordinary elder death at S". -/
  elderPartner_iff : ∀ (g : E d → ℝ), IsSmooth d g → IsLPeriodic d L g → IsMorseDistinct d L g →
    ∀ M S : E d, ElderPartner g M S ↔
      (IsNondegMax d g M ∧ IsIndexSaddle d g S ∧ ¬ IsEssential g M ∧ deathHeight g M = g S)

/-- Finite ordinary superlevel H0 bar, as an ordered pair: a nondegenerate maximum `M`
(birth) with its ordinary elder death partner `S`, an index-(d-1) saddle at positive torus
distance. The essential global-maximum class is excluded because `ElderPartner` requires
`¬ IsEssential` on the Morse locus.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "each finite superlevel H0 bar has exactly one local-maximum birth and one index-(d-1) merging-saddle death."
Interface: `I : SuperlevelBarInterface d L` is given.
Does not claim: the identification of these pairs with bars of a persistence module; that the count of such pairs equals the finite-bar count off the Morse locus. -/
def IsFiniteBar (d : ℕ) (L : ℝ) (I : SuperlevelBarInterface d L) (g : E d → ℝ) (M S : E d) : Prop :=
  IsCandidatePair d L g M S ∧ 0 < heightGap d g M S ∧ I.ElderPartner g M S

/-- The set of pairs of a population `Pair` with both points in the fundamental domain and
lifetime (height gap) in `A`; `Pair g M S` is the membership predicate of the population.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Let nu_cand(ell) be their expected lifetime density per unit volume; let nu_eld(ell) count only pairs which are actual ordinary elder partners."
Interface: the population predicate `Pair` is a parameter.
Does not claim: finiteness of this set (a field of the density-version predicate). -/
def pairSet (d : ℕ) (L : ℝ) (Pair : (E d → ℝ) → E d → E d → Prop) (g : E d → ℝ) (A : Set ℝ) :
    Set (E d × E d) :=
  {p | p.1 ∈ FundDomain d L ∧ p.2 ∈ FundDomain d L ∧ Pair g p.1 p.2 ∧ heightGap d g p.1 p.2 ∈ A}

/-- Number of pairs of the population with lifetime in `A` and base points in the
fundamental domain (`Set.ncard`, which is `0` for an infinite set).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Let nu_cand(ell) be their expected lifetime density per unit volume; let nu_eld(ell) count only pairs which are actual ordinary elder partners."
Interface: the population predicate `Pair` is a parameter.
Does not claim: finiteness of the counted set. -/
noncomputable def pairCount (d : ℕ) (L : ℝ) (Pair : (E d → ℝ) → E d → E d → Prop)
    (g : E d → ℝ) (A : Set ℝ) : ℕ :=
  Set.ncard (pairSet d L Pair g A)

/-- `ν` is a VERSION of the expected per-unit-volume lifetime density of the population
`Pair` under the field `G`: it is nonnegative and measurable, the pair sets are almost
surely finite with integrable counts, `ν` is integrable on every Borel `A ⊆ (0, ∞)`, and
`∫_A ν = E[#pairs with lifetime in A and base point in [0,L)^d] / L^d` for every such `A`.
The source's densities are only defined up to such versions. The sigma-algebra of the sample
space is brought into scope by the inlined term-level `haveI := G.mΩ` (a square-bracket
field of a bundled record is not registered for typeclass resolution after the declaration).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "These densities admit versions with"
Interface: the field `G` and the population predicate `Pair` are given; the Kac-Rice representation of the expectation (Sections 9-10 of the source) is NOT used; the density is characterised only by the integral identity.
Does not claim: existence of a version; the weighted Kac-Rice identity; the radial ledger (10.2). Measurability of the pair count as a function of the sample is REQUIRED by the predicate (through `Integrable`, hence `AEStronglyMeasurable`), not established here. -/
def IsLifetimeDensityVersion (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (Pair : (E d → ℝ) → E d → E d → Prop) (ν : ℝ → ℝ) : Prop :=
  haveI := G.mΩ
  (∀ ℓ : ℝ, 0 ≤ ν ℓ) ∧ Measurable ν ∧
  (∀ A : Set ℝ, MeasurableSet A → A ⊆ Set.Ioi 0 →
    (∀ᵐ ω ∂G.P, (pairSet d L Pair (G.F ω) A).Finite) ∧
    Integrable (fun ω => ((pairCount d L Pair (G.F ω) A : ℕ) : ℝ)) G.P ∧
    IntegrableOn ν A ∧
    ∫ ℓ in A, ν ℓ = (∫ ω, ((pairCount d L Pair (G.F ω) A : ℕ) : ℝ) ∂G.P) / L ^ d)

end UniversalLaw.Spec.Field
