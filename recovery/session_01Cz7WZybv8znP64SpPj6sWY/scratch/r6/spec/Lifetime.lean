import Mathlib

/-!
# Lifetime-track statements: parent Theorems A, B, C, the bounded remainder R and the
# planar elder lower bound / density gap (statements only)

Scientific effect: NONE. Formal progress of every declaration here: `specified`.
Organizational-independence credit of this transcription: 0 (Anthropic / Claude).

This module states, as `Prop`-valued definitions, the shapes of the parent lifetime
results (1.1), (1.2), (1.3) of UNIFORM_MATRIX_CAP_AND_LIFETIME.md, of the bounded
remainder (R1) of LIFETIME_REMAINDER.md, and of the planar lower bound (E2), the
two-sided rate (E4) and the compact-window density gap (E11) of
ELDER_LOWER_AND_DENSITY_GAP.md. Every statement is parametrised by interface objects
(the Gaussian field, the superlevel-bar / elder-partner relation, and the typed pair law
probability `p_r(b,k,R)`); it is a statement about whatever objects satisfy those
interfaces, and the identification of those objects with the objects of the source texts
is not part of the statement.

This module is self-contained: it imports only Mathlib and re-declares, verbatim up to
the namespace, the shared objects of `Specifications/Field.lean` (covariance, field
interface, critical points, Hessian inertia, pairs, superlevel bars, density versions).
The duplication is deliberate at this layer and is recorded as such.

What this module does not establish. It consists only of definitions and structures;
it states nothing as a result and proves nothing: none of the parent results, the
remainder bound or the planar bounds is shown here, and no kernel receipt about them
exists. It does not transcribe the explicit Gaussian/angular integral (15.2) for the
leading coefficient `c_{d,L}` (the coefficient is characterised only through the
leading-term asymptotics of (1.3)); it does not transcribe the regression law
`Q_{r,b,k,R}`, the typed weight `W_r`, the full normalizer `Z_r` or the marked Kac-Rice
representation (all informal in the source and taken as the interface field `p`); it
does not transcribe the reading-rule amendments bound to the parent (congruence erratum,
Section 9 replacement v1.1, wording W1, embedding radius `r < L/(4 sqrt 2)`), nor the
marked-cylinder cap support; it asserts no numerical `C`, `r_*`, `z_*`, `ell_*` or
coefficient value; it does not state Theorem A as consumed by Theorem R; and it does
not state the compact-window `O(ell^(2/3))` difference for the unrestricted densities.
The planar statements (E4) and (E11) carry the parent upper bound (1.1), for the same
typed pair law at `d = 2`, as an explicit hypothesis, as the source does ("If the precise
Theorem A upper bound in [LP] is consumed"; its Section 4 says the upper half of (E11)
consumes the parent upper bound already identified there); the direct lower bound (E2) is
unconditional, as in the source.
Nothing here promotes, reclassifies or discharges any claim, premise or obligation of
the registers; the landing disposition `REVIEWED_SCOPED` (lifetime-remainder) and the
graph classification and review disposition recorded for
`math.uniform-matrix-cap-lifetime` are transcribed from the registers by the statement
register, never re-derived here; alignment of these statements with the sources is
PENDING_INDEPENDENT_REVIEW.

Sources (Layer 0, byte-pinned, `d6g8k5htny-coder/Math-`):
* `lifetime-parent`: commit `c39164254ad1529ed49543743598d5dcb24bab98`, path
  `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`,
  sha256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`.
* `lifetime-remainder`: commit `760340e921ac4ceda296b8118da936f1133e956e`, path
  `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md`,
  sha256 `380b7d0abdb0fe2de5a6564565af9560d3f1b1ce22fd5538927db0c52f4f3a4a`.
* `elder-lower-density-gap`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md`,
  sha256 `8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c`.
-/

namespace UniversalLaw.Spec.Lifetime

open MeasureTheory

/-! ### Shared objects, re-declared from `Specifications/Field.lean` (same text, this
namespace). The duplication is deliberate: modules of this packet import only Mathlib. -/

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

/-! ### Populations of pairs -/

/-- Compact-window population of Theorem B: ordered maximum / index-(d-1)-saddle pairs with
`0 < distance <= ρ`, birth `f(M)` in `B = [bLo, bHi]` and scaled gap in `K = [kLo, kHi]`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Count ordered maximum/saddle pairs with 0<distance<=r_*, birth in B, and scaled gap (f(M)-f(S))/distance^3 in K."
Interface: none beyond the shared predicates.
Does not claim: that the distance cutoff `ρ` equals the `r_*` of (1.1); it is quantified existentially in the statements. -/
def IsWindowPair (d : ℕ) (L ρ bLo bHi kLo kHi : ℝ) (g : E d → ℝ) (M S : E d) : Prop :=
  IsCandidatePair d L g M S ∧ torusDist d L M S ≤ ρ ∧
    g M ∈ Set.Icc bLo bHi ∧ scaledGap d L g M S ∈ Set.Icc kLo kHi

/-- Compact-window pairs which are actual ordinary elder partners (the population of
`nu_eld` in Theorem B).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "let nu_eld(ell) count only pairs which are actual ordinary elder partners."
Interface: `I : SuperlevelBarInterface d L` supplies the elder-partner relation.
Does not claim: identification of `ElderPartner` with the source's elder rule off the Morse locus. -/
def IsWindowElderPair (d : ℕ) (L ρ bLo bHi kLo kHi : ℝ) (I : SuperlevelBarInterface d L)
    (g : E d → ℝ) (M S : E d) : Prop :=
  IsWindowPair d L ρ bLo bHi kLo kHi g M S ∧ I.ElderPartner g M S

/-- Unrestricted population of Theorem C and Theorem R: ALL ordered
maximum / index-(d-1)-saddle pairs with positive height difference, every birth height,
every positive gap mark and every separation on the torus.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "let nu_cand^all(ell) count ALL ordered maximum/index-(d-1)-saddle pairs with height difference ell>0, and let nu_eld^all(ell) count all finite ordinary superlevel H0 persistence bars, both in expectation per unit volume."
Interface: none beyond the shared predicates.
Does not claim: anything about the off-diagonal (far) population separately (Section 14 of the source). -/
def IsUnrestrictedPair (d : ℕ) (L : ℝ) (g : E d → ℝ) (M S : E d) : Prop :=
  IsCandidatePair d L g M S ∧ 0 < heightGap d g M S

/-! ### The typed pair law as an interface -/

/-- Interface for the weighted regression (typed pair) law: the only object consumed by the
statements is the probability `p_r(b,k,R)` that the global ordinary superlevel elder death
partner of the pinned maximum `M = -(r/2)u` is the pinned saddle `S = (r/2)u`, under the
law `Q^W = (W_r/Z_r) Q_{r,b,k,R}` with `W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1}`.
That law, its weight, its normalizer and the regression construction are informal in the
source and are NOT transcribed; `p` is a GIVEN function of radius `r`, birth `b`, gap mark
`k` and orthonormal frame `R` with values in `[0, 1]`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Let p_r(b,k,R) be the Q^W probability that the **global ordinary superlevel elder death partner** of M is S."
Interface: `p` is given; its only hypothesis fields are the bounds `0 ≤ p ≤ 1`. The parameters `G` and `I` record which field and which elder relation the law refers to; no field ties `p` to them.
Does not claim: existence of the regression law, the typed weight, the full normalizer `Z_r > 0`, or that `p` is the probability of any event under any measure declared here; that `p` depends on `G` or `I` in any way (no field mentions them; they only record which field and elder relation the law is meant to refer to). -/
structure PairLawInterface (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) where
  /-- Section 1 clause: "Let p_r(b,k,R) be the Q^W probability that the global ordinary
  superlevel elder death partner of M is S." -/
  p : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → ℝ
  /-- `p` is a probability: nonnegative. -/
  p_nonneg : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 ≤ p r b k R
  /-- `p` is a probability: at most one. -/
  p_le_one : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), p r b k R ≤ 1

/-! ### The parent statements (1.1), (1.2), (1.3) -/

/-- Shape of Theorem A (1.1), uniform selection candidate: for fixed `d ≥ 2`, `L > 0`,
compact `B = [bLo, bHi]` and `K = [kLo, kHi]` with `0 < kLo ≤ kHi`, there exist `r_* > 0`
and `C` (quantified OUTSIDE the universal quantifiers over `r`, `b`, `k` and the frame)
such that `0 <= 1 - p_r(b,k,R) <= C r^3` for all `0 < r ≤ r_*`, `b ∈ B`, `k ∈ K` and all
orthonormal frames `R`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "There exist r_*>0 and C<infinity such that, for all 0<r<=r_*, b in B, k in K, and all orthonormal frames,"
Interface: `G` (field), `I` (elder relation) and `PL` (typed pair law probability `p`) are given objects; the statement is about any `p` satisfying `PairLawInterface`. The interface does not determine `p`: the Prop is a predicate on the interface object, not a closed statement; its universal closure over all inhabitants is false whenever the interfaces are inhabited (take `p ≡ 0`), and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the parent result itself or its proof (full-normalizer floor `Z_r ≥ z_* r^2`, boundary-layer integration); identification of `PL.p` with the source's `p_r`; the reading-rule amendments (congruence erratum, Section 9 replacement, embedding radius); any numerical `C` or `r_*`; nonauthor review or acceptance. -/
def ParentUniformSelectionA (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) (PL : PairLawInterface d L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ rStar : ℝ, 0 < rStar ∧ ∃ C : ℝ,
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          0 ≤ 1 - PL.p r b k R ∧ 1 - PL.p r b k R ≤ C * r ^ 3

/-- Shape of Theorem B (1.2), compact-mark density candidate: for fixed `d ≥ 2`, `L > 0` and
windows `B`, `K` of positive length with `kLo > 0`, there are a distance cutoff `r_* > 0`, a
constant `c_{B,K} > 0` and a constant `C_{B,K}` such that the compact-window candidate and
elder lifetime densities admit versions `νc`, `νe` with `νc(ℓ) ~ c_{B,K} ℓ^(-1/3)`,
`νe(ℓ) ~ c_{B,K} ℓ^(-1/3)` as `ℓ ↓ 0` (filter `nhdsWithin 0 (Set.Ioi 0)`) and
`0 ≤ νc(ℓ) - νe(ℓ) ≤ C_{B,K} ℓ^(2/3)` for all sufficiently small `ℓ > 0`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "0 <= nu_cand(ell)-nu_eld(ell) <= C_{B,K} ell^(2/3),"
Interface: `G` and `I` are given; "versions" are any `νc`, `νe` satisfying `IsLifetimeDensityVersion` (integral identity), not a Kac-Rice representation; the coefficient `c_{B,K}` is existential, not the Gaussian integral (11.3).
Does not claim: the parent result or its proof; the exact coefficient formula (11.3); existence of the versions; that the cutoff `r_*` is the one of (1.1); any numerical constant; nonauthor review or acceptance. -/
def ParentCompactWindowDensityB (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo < bHi → 0 < kLo → kLo < kHi →
    ∃ rStar : ℝ, 0 < rStar ∧ ∃ c : ℝ, 0 < c ∧ ∃ C : ℝ, ∃ νc νe : ℝ → ℝ,
      IsLifetimeDensityVersion d L G (IsWindowPair d L rStar bLo bHi kLo kHi) νc ∧
      IsLifetimeDensityVersion d L G (IsWindowElderPair d L rStar bLo bHi kLo kHi I) νe ∧
      Asymptotics.IsEquivalent (nhdsWithin (0 : ℝ) (Set.Ioi 0)) νc
        (fun ℓ : ℝ => c * ℓ ^ ((-1 : ℝ) / 3)) ∧
      Asymptotics.IsEquivalent (nhdsWithin (0 : ℝ) (Set.Ioi 0)) νe
        (fun ℓ : ℝ => c * ℓ ^ ((-1 : ℝ) / 3)) ∧
      ∃ ℓStar : ℝ, 0 < ℓStar ∧ ∀ ℓ : ℝ, 0 < ℓ → ℓ ≤ ℓStar →
        0 ≤ νc ℓ - νe ℓ ∧ νc ℓ - νe ℓ ≤ C * ℓ ^ ((2 : ℝ) / 3)

/-- `c` is a leading coefficient of the unrestricted finite-lifetime densities: `c > 0` and
versions `νc`, `νe` of the unrestricted candidate density and of the finite ordinary
superlevel H0 bar density satisfy `νc(ℓ) ~ c ℓ^(-1/3)` and `νe(ℓ) ~ c ℓ^(-1/3)` as `ℓ ↓ 0`.
The essential global-maximum class is excluded from `νe` through `IsFiniteBar`. This is the
conclusion of (1.3) for a given `c`; it characterises `c_{d,L}` by its leading-term role,
not by the integral (13.6)/(15.2).

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "There is a positive finite constant c_{d,L} such that"
Interface: `G` and `I` are given; versions are characterised by the integral identity only.
Does not claim: the explicit Gaussian/angular representation (15.2) of `c_{d,L}`; uniqueness of `c` (which follows from the asymptotics but is not stated); existence of versions. -/
def IsUnrestrictedLeadingCoefficient (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) (c : ℝ) : Prop :=
  0 < c ∧ ∃ νc νe : ℝ → ℝ,
    IsLifetimeDensityVersion d L G (IsUnrestrictedPair d L) νc ∧
    IsLifetimeDensityVersion d L G (IsFiniteBar d L I) νe ∧
    Asymptotics.IsEquivalent (nhdsWithin (0 : ℝ) (Set.Ioi 0)) νc
      (fun ℓ : ℝ => c * ℓ ^ ((-1 : ℝ) / 3)) ∧
    Asymptotics.IsEquivalent (nhdsWithin (0 : ℝ) (Set.Ioi 0)) νe
      (fun ℓ : ℝ => c * ℓ ^ ((-1 : ℝ) / 3))

/-- Shape of Theorem C (1.3), unrestricted finite-lifetime leading density candidate: for
fixed `d ≥ 2` and `L > 0` there is a positive finite constant `c_{d,L}` such that the
unrestricted candidate density and the finite ordinary superlevel H0 bar density (all birth
heights, all positive gap marks, all separations; essential global maximum excluded) admit
versions with `ν(ℓ) ~ c_{d,L} ℓ^(-1/3)` as `ℓ ↓ 0`. Leading term only.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "nu_cand^all(ell) ~ c_{d,L} ell^(-1/3),"
Interface: `G` and `I` are given; versions are characterised by the integral identity only.
Does not claim: the parent result or its proof (dominated convergence of Section 13, off-diagonal bound (14.1)); the representation (15.2); the compact-window `O(ell^(2/3))` difference for the unrestricted densities; any numerical enclosure of `c_{d,L}` (the SIDE24 coefficient track is separate); nonauthor review or acceptance. -/
def ParentUnrestrictedLeadingDensityC (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) : Prop :=
  2 ≤ d → 0 < L → ∃ c : ℝ, IsUnrestrictedLeadingCoefficient d L G I c

/-! ### The bounded remainder (R1) -/

/-- Shape of Theorem R (R1), bounded unrestricted short-lifetime remainder: for fixed
`d ≥ 2`, `L > 0` and with `c = c_{d,L}` the parent leading coefficient (characterised here
by `IsUnrestrictedLeadingCoefficient`), there are `ell_* > 0` and `C` such that versions of
the unrestricted candidate and finite-bar densities satisfy, for `0 < ℓ ≤ ell_*`,
`|νc(ℓ) - c ℓ^(-1/3)| ≤ C`, `0 ≤ νc(ℓ) - νe(ℓ) ≤ C` and `|νe(ℓ) - c ℓ^(-1/3)| ≤ C`.
The `O(1)` remainder is stated with an explicit absolute-value bound.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md sha256 380b7d0abdb0fe2de5a6564565af9560d3f1b1ce22fd5538927db0c52f4f3a4a
Anchor: "|nu_cand(ell)-c ell^(-1/3)| <= C,"
Interface: `G` and `I` are given; `c` is any real satisfying the parent leading-coefficient property (the source says the coefficient "is exactly the parent's equation (15.2)"; that integral is not transcribed, so `c` is pinned through (1.3) instead); versions are characterised by the integral identity only. The statement does not mention the typed pair law: Theorem R does not consume (1.1).
Does not claim: the result or its proof (quadratic determinant cancellation (R7), coupling (R4), cap-loss bound (R12)); convergence of the remainder; a second coefficient; numerical `C` or `ell_*`; the compact-window `O(ell^(2/3))` difference without cutoffs; the parent interfaces it imports (D1-A-E); nonauthor review or acceptance. If no `c` satisfies the parent property the statement holds vacuously; that is a transcription of its dependency on (1.3), not a claim. -/
def RemainderBoundedR1 (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) : Prop :=
  2 ≤ d → 0 < L → ∀ c : ℝ, IsUnrestrictedLeadingCoefficient d L G I c →
    ∃ ℓStar : ℝ, 0 < ℓStar ∧ ∃ C : ℝ, ∃ νc νe : ℝ → ℝ,
      IsLifetimeDensityVersion d L G (IsUnrestrictedPair d L) νc ∧
      IsLifetimeDensityVersion d L G (IsFiniteBar d L I) νe ∧
      ∀ ℓ : ℝ, 0 < ℓ → ℓ ≤ ℓStar →
        |νc ℓ - c * ℓ ^ ((-1 : ℝ) / 3)| ≤ C ∧
        0 ≤ νc ℓ - νe ℓ ∧ νc ℓ - νe ℓ ≤ C ∧
        |νe ℓ - c * ℓ ^ ((-1 : ℝ) / 3)| ≤ C

/-! ### Planar elder lower bound and density gap (d = 2) -/

/-- Shape of the planar direct lower bound (E2): on the fixed planar torus (`d = 2`), with
birth `b` in a fixed compact interval, `k ∈ [kLo, kHi]` with `kLo > 0` and frames ranging
over `O(2)`, there are `c > 0` and `r_* > 0` such that `1 - p_r(b,k,R) ≥ c r^3` for all
`0 < r ≤ r_*`, uniformly in `b`, `k` and the frame.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "1-p_r(b,k,R) >= c r^3, 0<r<=r_*."
Interface: `G`, `I` and `PL` (the same typed pair law probability `p_r(b,k,R)` as in (1.1), which the source says is exactly the definition of [LP, Section 1]) are given objects. The interface does not determine `p`: the Prop is a predicate on the interface object, not a closed statement; its universal closure over all inhabitants is false whenever the interfaces are inhabited (take `p ≡ 1`), and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the result or its proof (polygonal path, tilted jet-box probability, extra-conditioned `C^4` control); any lower bound for `d ≥ 3`; identification of `PL.p` with the source's `p_r` or with any historical `q(r,b)`; numerical `c` or `r_*`; nonauthor review or acceptance. -/
def PlanarElderLowerE2 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ c : ℝ, 0 < c ∧ ∃ rStar : ℝ, 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), c * r ^ 3 ≤ 1 - PL.p r b k R

/-- Shape of the planar two-sided selection rate (E4), Corollary E2 of the source: if (1.1)
holds for the same typed pair law at `d = 2` (the source: "If the precise Theorem A upper
bound in [LP] is consumed"), then `c r^3 ≤ 1 - p_r(b,k,R) ≤ C r^3` for `0 < r ≤ r_*`,
uniformly on the compact planar model. The parent upper input is an explicit hypothesis of
the statement, not folded into its conclusion.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "c r^3 <= 1-p_r(b,k,R) <= C r^3."
Interface: `G`, `I`, `PL` are given objects as in `PlanarElderLowerE2`; the hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi` is the consumed upper input (a statement shape, not the parent result). The interface does not determine `p`: the Prop is a predicate on the interface object, not a closed statement; its universal closure over all inhabitants is false whenever the interfaces are inhabited (take `p ≡ 1`, for which the hypothesis holds and the lower half fails), and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the upper half independently of the parent (1.1) (the source says the upper half consumes it, and the statement is conditional on it); the parent (1.1) itself; the result or its proof; numerical constants; nonauthor review or acceptance. -/
def PlanarTwoSidedRateE4 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →
    (0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
      ∃ c : ℝ, 0 < c ∧ ∃ C : ℝ, ∃ rStar : ℝ, 0 < rStar ∧
        ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
          ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2),
            c * r ^ 3 ≤ 1 - PL.p r b k R ∧ 1 - PL.p r b k R ≤ C * r ^ 3)

/-- Shape of the planar compact-window density gap (E11): in `d = 2`, if (1.1) holds for the
typed pair law on the windows `B`, `K` (the source says the upper half consumes the parent
upper bound already identified in its Corollary E2 as "the precise Theorem A upper bound in
[LP]"), then for windows of positive length with `kLo > 0` there are a distance cutoff
`r_* > 0`, constants `c_{B,K} > 0`, `C_{B,K}` and `ell_* > 0` such that versions of the
compact-window candidate and ordinary elder densities satisfy
`c_{B,K} ℓ^(2/3) ≤ νc(ℓ) - νe(ℓ) ≤ C_{B,K} ℓ^(2/3)` for all `0 < ℓ ≤ ell_*`. The parent upper
input is an explicit hypothesis, not folded into the conclusion. The lower half, which the
source derives from (E2) alone, is stated under the same hypothesis because the source displays
(E11) as one two-sided inequality; that is a hypothesis the source does not need for the lower
half, recorded here as such.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "       <= C_{B,K} ell^(2/3).                             (E11)"
Interface: `G`, `I` and `PL` are given; the hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi` is the consumed parent upper input (a statement shape, not the parent result); versions are characterised by the integral identity only; the weighted Kac-Rice / radial ledger of [LP, Sections 9-11] that the source consumes is not transcribed. The interface does not determine `p` or the field: the Prop is a predicate on the interface objects, not a closed statement about the source's objects; nothing is asserted about its universal closure over all inhabitants, and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the result or its proof; the upper half independently of the parent (1.1); the parent (1.1) itself; the lower half without the hypothesis (the source has it from (E2) alone); sharpness of the `O(1)` remainder of (R1); removal of the compact-mark restriction; that the cutoff `r_*` is the one of (1.1); any numerical constant; nonauthor review or acceptance. -/
def PlanarDensityGapE11 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →
    (0 < L → bLo < bHi → 0 < kLo → kLo < kHi →
      ∃ rStar : ℝ, 0 < rStar ∧ ∃ c : ℝ, 0 < c ∧ ∃ C : ℝ, ∃ ℓStar : ℝ, 0 < ℓStar ∧
        ∃ νc νe : ℝ → ℝ,
          IsLifetimeDensityVersion 2 L G (IsWindowPair 2 L rStar bLo bHi kLo kHi) νc ∧
          IsLifetimeDensityVersion 2 L G (IsWindowElderPair 2 L rStar bLo bHi kLo kHi I) νe ∧
          ∀ ℓ : ℝ, 0 < ℓ → ℓ ≤ ℓStar →
            c * ℓ ^ ((2 : ℝ) / 3) ≤ νc ℓ - νe ℓ ∧ νc ℓ - νe ℓ ≤ C * ℓ ^ ((2 : ℝ) / 3))

end UniversalLaw.Spec.Lifetime
