import Mathlib.Probability.ConditionalProbability
import Mathlib.MeasureTheory.Measure.WithDensity
import Mathlib.MeasureTheory.Measure.Haar.OfBasis
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Set.Card
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.LinearAlgebra.Dimension.Finrank
import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Data.Finset.Range


/-!
# RN-track statements: count interface (Theorem N), fixed-remote and fixed-annulus
# height windows, the D5 reconciliation rows, the dimension lift, the C6 moment
# bounds, the rare-cluster Poisson obstruction and the open witness-collision target
# (statements only)

Scientific effect: NONE. Formal progress of every declaration here: `specified`.
Organizational-independence credit of this transcription: 0 (Anthropic / Claude).

This module states, as `Prop`-valued definitions, the shapes of the RN-track results:
the abstract probability-to-count interface (N1)-(N5) of RN_COUNT_INTERFACE.md, the
fixed-remote Theorem A displays (2) and (3) of remote_window_20260924/PROOF.md, the
fixed-annulus height-window candidate (A2) of FIXED_ANNULUS_CANDIDATE.md, the all-height
annulus bridge display (2) of rn_annulus_bridge_20260925/PROOF.md, the six reconciled
rows D5-a ... D5-f of RECONCILIATION.md, the headline displays of Theorems P_d, C_d,
I_d, G_d of d5_dimension_lift_20260929/PROOF.md, the planar (F4) display of
c6_fourier_cutoff_20260929/PROOF.md, Theorem Q and Corollary Theta of
c6_palm_route_20260929/PROOF.md, Theorem O of c6_rare_cluster_laws_20260929/PROOF.md,
Corollary D of remote_collision_20260928/PROOF.md, and, labelled as an OPEN obligation
whose Prop is a target and not a result, the regional near-pin witness-collision
second-moment target of GRAPH node `math.rn-region.witness-collision`.

Every statement about the pinned Gaussian model is parametrised by an interface object
`PinnedLawInterface`: the sample space, the sample paths, the Gaussian regression law
`Q_r`, the typed endpoint weight `W_r`, the tilted law `Q_r^W = (W_r/Z_r) Q_r` and the
contact kernel `Lambda_j` are GIVEN; their Gaussian-regression construction, the
determinant form of the weight and the Kac-Rice representation of the kernel are informal
in the sources and are not transcribed. The pin equations, the positivity of the normalizer
and the exact tilt are required only in the pin regime `0 < r < L`, this module's reading of
the sources' "For small r>0 the pins are" (for `0 < r < L` the two pins are distinct modulo
`L Z^d`, so the pin equations are consistent with `L`-periodicity); outside that regime the
laws are unconstrained given probability measures. The statements are about whatever objects satisfy
the interface; the identification of those objects with the objects of the source texts is
not part of any statement.

This module is self-contained: it imports only Mathlib and re-declares, up to the
namespace, the shared objects of `Specifications/Field.lean` that it needs (ambient space,
lattice vectors, Hessian form, critical points, inertia, pins, torus distance, fundamental
domain): the nine bodies are byte-identical to Field.lean's, and so are the docstrings except
the `Does not claim:` lines of `pinM` and `pinS`, which are module-specific (Field.lean
records that no statement there consumes the pins; here the interface and the window
statements consume them). The duplication is deliberate at this layer and is recorded as
such.

What this module does not establish. It consists only of definitions and structures; it
states nothing as a result and proves nothing: none of Theorem N, Theorem A, the annulus
candidates, the D5 rows, the dimension lift, the C6 bounds, the rare-cluster Poisson
obstruction or the collision target is shown here, and no kernel receipt about them exists. It does not
construct the Gaussian field or the regression law `Q_r`; it does not transcribe the
determinant/inertia form of `W_r` (only its sign and typing), the normalizer floor
`Z_r >= z_* r^2`, the observation transform `U_r`, the coupling estimates, the weighted
Kac-Rice formulas (N6), (12), (A21), (16), (3.1) or the contact-kernel integral (13); it
asserts no numerical constant, radius or cutoff; it does not state the two-scale, thin-tube,
microdisk, collar, intermediate-window or elder sources that the D5 rows consume; it does
not state Theorem F (F1)-(F2) or Proposition B of the Fourier cutoff note, Corollary P of
the Palm route, Theorems C and S of the rare-cluster note, or Theorem C / Corollaries E, F
of the remote-collision note; and it does not assert that the open target is attainable.
Nothing here promotes, reclassifies or discharges any claim, premise or obligation of the
registers: the landing dispositions `HOLD_WITH_DOMAIN` (rn-count-interface,
rn-fixed-remote-window) and `REVIEWED_SCOPED` (rn-fixed-annulus-window), and the GRAPH
classifications of the `math.rn-region.*` nodes (including `OPEN_ACTIVE` for
`witness-collision`), are transcribed from the registers by the statement register, never
re-derived here; alignment of these statements with the sources is
PENDING_INDEPENDENT_REVIEW.

Sources (Layer 0, byte-pinned, `d6g8k5htny-coder/Math-`):
* `lifetime-parent` (shared objects): commit `c39164254ad1529ed49543743598d5dcb24bab98`,
  path `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`,
  sha256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`.
* `rn-count-interface`: commit `760340e921ac4ceda296b8118da936f1133e956e`, path
  `frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md`,
  sha256 `aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab`.
* `rn-fixed-remote-window`: commit `760340e921ac4ceda296b8118da936f1133e956e`, path
  `frontiers/remote_window_20260924/PROOF.md`,
  sha256 `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7`.
* `rn-fixed-annulus-window`: commit `760340e921ac4ceda296b8118da936f1133e956e`, path
  `frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md`,
  sha256 `1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b`.
* `rn-annulus-bridge`: commit `760340e921ac4ceda296b8118da936f1133e956e`, path
  `frontiers/rn_annulus_bridge_20260925/PROOF.md`,
  sha256 `d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05`.
* `d5-reconciliation`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `reviews/d5_reconciliation_20260929/RECONCILIATION.md`,
  sha256 `14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432`.
* `d5-dimension-lift`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/d5_dimension_lift_20260929/PROOF.md`,
  sha256 `6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80`.
* `c6-fourier-cutoff`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/c6_fourier_cutoff_20260929/PROOF.md`,
  sha256 `c1692379a3a066589bd2522aaa5b4d480e736b737c8c39a474d1735185793733`.
* `c6-palm-route`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/c6_palm_route_20260929/PROOF.md`,
  sha256 `aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b`.
* `c6-rare-cluster`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/c6_rare_cluster_laws_20260929/PROOF.md`,
  sha256 `c52a3cf197b5cf26071a8cc951e15ef3ac564b3d7453d41647f91af540a1b8eb`.
* `remote-collision`: commit `6c020d6a72936632f2055122e71a7818a4ff497e`, path
  `frontiers/remote_collision_20260928/PROOF.md`,
  sha256 `b9b8b58fd8266db7ffe6537078003888ef138b445d3289ec5588f116af9050c2`.
-/

namespace UniversalLaw.Spec.RN

open MeasureTheory

/-! ### Shared objects, re-declared from `Specifications/Field.lean` (same bodies, this
namespace; see the module header for the two module-specific `Does not claim:` lines). The
duplication is deliberate: modules of this packet import only Mathlib. -/

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

/-- The pin `M = -(r/2) u` of the embedded local cylinder with axial unit vector `u`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "M=-(r/2)u,     S=(r/2)u,"
Interface: none.
Does not claim: that `u` is a unit vector (a hypothesis where used). -/
noncomputable def pinM (d : ℕ) (r : ℝ) (u : E d) : E d :=
  -((r / 2) • u)

/-- The pin `S = (r/2) u` of the embedded local cylinder with axial unit vector `u`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "M=-(r/2)u,     S=(r/2)u,"
Interface: none.
Does not claim: that `u` is a unit vector (a hypothesis where used). -/
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

/-! ### Theorem N: abstract probability-to-count interface (N1)-(N5)

These are statements about an arbitrary probability space, an event and an integer-valued
count; the source stresses that they are abstract and are NOT asserted to be realizations
of the Gaussian field. -/

/-- Shape of (N1), the Hölder count bound: on any probability space `(Ω, P)`, for an event
`E`, an integer-valued count `N ≥ 0` with `N^p` integrable and `p > 1`,
`E[N 1_E] ≤ (E N^p)^(1/p) P(E)^(1-1/p)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "E[N_r 1_Er] <= (E N_r^p)^(1/p) P(E_r)^(1-1/p)."
Interface: none; the probability space, event and count are universally quantified, integrability of `N^p` is a hypothesis.
Does not claim: that any count of the Gaussian model satisfies the moment hypothesis; the result or its proof; any rate in `r`; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/
def HolderCountBoundN1 : Prop :=
  ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Ev : Set Ω) (N : Ω → ℕ) (p : ℝ),
    MeasurableSet Ev → 1 < p → Integrable (fun ω => (N ω : ℝ) ^ p) P →
      ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P ≤
        (∫ ω, (N ω : ℝ) ^ p ∂P) ^ (1 / p) * (P Ev).toReal ^ (1 - 1 / p)

/-- Shape of (N2), the consequence of (N1) with a power-law moment bound: if
`E N^p ≤ A r^(-β)` and `P(E) ≤ C r^3` then
`E[N 1_E] ≤ A^(1/p) C^(1-1/p) r^((3(p-1)-β)/p)`. The powers refer to the `p`-th MOMENT
itself, as the source insists.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "E[N_r 1_Er] <= A^(1/p) C^(1-1/p)"
Interface: none; `A`, `β`, `C`, `p`, `r` and the probability space are universally quantified, the two bounds are hypotheses.
Does not claim: that the exponent can be rounded up to `3`; that any Gaussian count satisfies the hypotheses; the result or its proof; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/
def HolderConsequenceN2 : Prop :=
  ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Ev : Set Ω) (N : Ω → ℕ) (p A β C r : ℝ),
    MeasurableSet Ev → 1 < p → 0 ≤ A → 0 ≤ C → 0 < r →
    Integrable (fun ω => (N ω : ℝ) ^ p) P →
    ∫ ω, (N ω : ℝ) ^ p ∂P ≤ A * r ^ (-β) →
    (P Ev).toReal ≤ C * r ^ 3 →
      ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P ≤
        A ^ (1 / p) * C ^ (1 - 1 / p) * r ^ ((3 * (p - 1) - β) / p)

/-- Shape of (N3), the exact identity `E[N 1_E] = q E[N | E]` with `q = P(E) > 0`, stated
with Mathlib's conditional measure `P[|E] = P(E)⁻¹ P|_E` for the conditional expectation.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "E[N_r 1_Er]=q E[N_r | E_r]."
Interface: none; the probability space, event and count are universally quantified, `P(E) > 0` and integrability of `N` are hypotheses.
Does not claim: the equivalence with a bounded conditional mean, the lower comparison `q ≥ c r^3`, or anything about the Gaussian model; the result or its proof; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/
def ConditionalMeanIdentityN3 : Prop :=
  ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Ev : Set Ω) (N : Ω → ℕ),
    MeasurableSet Ev → 0 < (P Ev).toReal → Integrable (fun ω => (N ω : ℝ)) P →
      ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P =
        (P Ev).toReal * ∫ ω, (N ω : ℝ) ∂(ProbabilityTheory.cond P Ev)

/-- The law of `U` uniform on `[0,1]`: Lebesgue measure restricted to `[0, 1]`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "On one space take U uniform on [0,1]."
Interface: none; an explicit construction.
Does not claim: that this measure is a probability measure (true, not stated as a result here). -/
noncomputable def uniformLaw : Measure ℝ := volume.restrict (Set.Icc (0 : ℝ) 1)

/-- The counterexample count `N_r = ceil(log(1/r)) 1_{E_r}` with `E_r = {U ≤ r^3}`, as a
function of the uniform variable `u`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "N_r=ceil(log(1/r)) 1_Er."
Interface: none; an explicit construction.
Does not claim: that this count is a realization of any Gaussian field quantity (the source says it is NOT). -/
noncomputable def counterCount (r : ℝ) (u : ℝ) : ℝ :=
  (⌈Real.log (1 / r)⌉₊ : ℝ) * (Set.Iic (r ^ 3)).indicator (fun _ => (1 : ℝ)) u

/-- Shape of the sharp counterexample (N4): for `0 < r ≤ e^(-1)`, `P(E_r) = r^3` and
`E N_r / r^3 = ceil(log(1/r))`, which tends to infinity as `r ↓ 0`, while for EACH finite
`p ≥ 1` the `p`-th moments `E N_r^p` are bounded uniformly in `r` (by a constant that may
depend on `p`). So every finite moment does not preserve the cubic exponent.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "E N_r/r^3=ceil(log(1/r)) -> infinity."
Interface: none; the construction is explicit (`uniformLaw`, `counterCount`).
Does not claim: the explicit bound `sup_(t>=1) e^(-3t)(t+1)^p` (only existence of a `p`-dependent bound); the one-`p` sharpness construction `N_r=ceil(r^(-3/p))1_Er`; the exponential-tail remark after (N5); that the construction says anything about the Gaussian field; the result or its proof; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/
def UniformCounterexampleN4 : Prop :=
  (∀ r : ℝ, 0 < r → r ≤ Real.exp (-1) →
    (uniformLaw (Set.Iic (r ^ 3))).toReal = r ^ 3 ∧
    (∫ u, counterCount r u ∂uniformLaw) / r ^ 3 = (⌈Real.log (1 / r)⌉₊ : ℝ)) ∧
  Filter.Tendsto (fun r : ℝ => (∫ u, counterCount r u ∂uniformLaw) / r ^ 3)
    (nhdsWithin (0 : ℝ) (Set.Ioi 0)) Filter.atTop ∧
  (∀ p : ℝ, 1 ≤ p → ∃ A : ℝ, ∀ r : ℝ, 0 < r → r ≤ Real.exp (-1) →
    ∫ u, counterCount r u ^ p ∂uniformLaw ≤ A)

/-- Shape of (N5), the exponential-tail layer-cake bound: if `P(N > t) ≤ A exp(-t/B)` for
all `t ≥ 0` with `A ≥ 1`, `B > 0`, and `q = P(E) > 0`, then
`E[N 1_E] ≤ B q [1 + log(A/q)]`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md sha256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab
Anchor: "= B q[1+log(A/q)]."
Interface: none; the probability space, event, count and tail constants are universally quantified, the tail bound is a hypothesis.
Does not claim: the intermediate integral identity `integral_0^infinity min(q,Ae^(-t/B))dt`; that the logarithmic loss is attained (the source's remark about (N4)); anything about the Gaussian model; the result or its proof; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/
def ExponentialTailLayerCakeN5 : Prop :=
  ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Ev : Set Ω) (N : Ω → ℕ) (A B : ℝ),
    MeasurableSet Ev → 1 ≤ A → 0 < B → 0 < (P Ev).toReal →
    Integrable (fun ω => (N ω : ℝ)) P →
    (∀ t : ℝ, 0 ≤ t → (P {ω | t < (N ω : ℝ)}).toReal ≤ A * Real.exp (-t / B)) →
      ∫ ω, (N ω : ℝ) * Ev.indicator (fun _ => (1 : ℝ)) ω ∂P ≤
        B * (P Ev).toReal * (1 + Real.log (A / (P Ev).toReal))

/-! ### The pinned weighted law as an interface -/

/-- Interface for the pinned Gaussian regression law and its determinant tilt. The sample
space `Ω` and its sigma-algebra are parameters; the sample paths `F`, the regression law
`Q_r = Q r b k R` on the `2(d+1)` pin observations at `M = -(r/2)u`, `S = (r/2)u`
(`u = R e_1`), the typed endpoint weight `W_r`, the tilted law `Q_r^W` and the contact
kernel `Lambda_j` are GIVEN. The hypothesis fields transcribe: smooth `L`-periodic sample
paths; `Q_r` is a probability measure under which, in the pin regime `0 < r < L`, the
`2(d+1)` pin equations hold almost surely; `W_r ≥ 0` vanishes unless `H_M` is negative
definite and `H_S` has index `d-1`; in the same regime `Z_r = E_(Q_r) W_r > 0` and
`dQ_r^W = (W_r/Z_r) dQ_r` exactly (`Measure.withDensity`); and `Q_r^W` is a probability
measure. The regime `0 < r < L` is this module's reading of the source clause "For small r>0
the pins are": for `0 < r < L` the pins `M`, `S` are distinct modulo `L Z^d` (`‖S - M‖ = r`
while every nonzero lattice vector has norm at least `L`), so the pin equations are
consistent with `L`-periodicity; without the gate the structure would have no inhabitant for
`L > 0` (at `r = L` along a coordinate frame the two pins coincide on the torus, so the pin
values `b` and `b - k r^3` would be prescribed at one point). The Gaussian-regression
construction of `Q_r`, the determinant factors `|det H_M| |det H_S|` of `W_r` and the
Kac-Rice contact representation (13) of `Lambda_j` are informal in the sources and are NOT
transcribed. Existence of such an object is not asserted (a Dirac mass at one smooth
`L`-periodic sample path with the prescribed pin values, a nondegenerate maximum at `M` and a
nondegenerate index-`(d-1)` saddle at `S`, with `W = 1` on that path, would satisfy the gated
fields; this consistency remark is not a statement of this module).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Q_r denotes continuous Gaussian regression on these 2(d+1) observations."
Interface: every field is a given property of the law, never derived; in particular `Λ` is only a given kernel (its continuity and positivity are part of the Theorem A statement, not of this interface); the pin regime `0 < r < L` gating `Q_pinned`, `Z_pos` and `QW_eq` is part of the hypothesis fields, not a derived fact.
Does not claim: that `Q_r` is the Gaussian regression of any field; that `W_r` equals `F_d(H_M) F_(d-1)(H_S)`; the normalizer floor `Z_r >= z_* r^2`; that `Λ` is given by (13); existence or uniqueness of an object satisfying the interface (an inhabitant is not asserted; the consistency remark above is not stated); that `0 < r < L` is the sources' unspecified `small r` threshold. -/
structure PinnedLawInterface (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω] where
  /-- The sample paths `ω ↦ f_ω : R^d → ℝ`. -/
  F : Ω → E d → ℝ
  /-- Section 2 clause: "The real Fourier series therefore has smooth versions"; every
  sample path is `C^n` for every finite `n`. -/
  smooth : ∀ (ω : Ω) (n : ℕ), ContDiff ℝ n (F ω)
  /-- Section 1 clause: the field lives "On X=R^d/(L Z^d)"; every sample path is `L`-periodic. -/
  periodic : ∀ (ω : Ω) (x : E d) (n : Fin d → ℤ), F ω (x + latticeVec d L n) = F ω x
  /-- Section 1 clause: "Q_r denotes continuous Gaussian regression on these 2(d+1)
  observations", indexed by radius `r`, birth `b`, gap mark `k` and frame `R`. -/
  Q : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Measure Ω
  /-- `Q_r` is a probability measure. -/
  Q_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), IsProbabilityMeasure (Q r b k R)
  /-- Section 1 clause: the pins "f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0" hold
  `Q_r`-almost surely in the pin regime `0 < r < L` (source: "For small r>0 the pins are"). -/
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(Q r b k R),
      F ω (pinM d r (R 0)) = b ∧ F ω (pinS d r (R 0)) = b - k * r ^ 3 ∧
      IsCriticalPoint d (F ω) (pinM d r (R 0)) ∧ IsCriticalPoint d (F ω) (pinS d r (R 0))
  /-- Section 1 clause: the "actual endpoint weight" `W_r=F_d(H_M) F_(d-1)(H_S)`, as a
  given functional of the sample. -/
  W : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Ω → ℝ
  /-- `W_r ≥ 0` (a product of filtered absolute determinants). -/
  W_nonneg : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), 0 ≤ W r b k R ω
  /-- Section 1 clause: `F_j(H)=|det H| 1{H has j negative eigenvalues}`, "with value zero
  on singular matrices"; so `W_r ≠ 0` forces `H_M` negative definite and `H_S` of index
  `d-1`. -/
  W_typed : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), W r b k R ω ≠ 0 →
    HessianHasInertia d (F ω) (pinM d r (R 0)) d ∧ HessianHasInertia d (F ω) (pinS d r (R 0)) (d - 1)
  /-- Section 1 clause: the "FULL normalizer" `Z_r=E_(Q_r) W_r` is positive in the pin
  regime `0 < r < L`. -/
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    0 < ∫ ω, W r b k R ω ∂(Q r b k R)
  /-- The tilted law `Q_r^W`. -/
  QW : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Measure Ω
  /-- Section 1 clause: `dQ_r^W=(W_r/Z_r)dQ_r`, exactly, in the pin regime `0 < r < L`. -/
  QW_eq : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    QW r b k R = (Q r b k R).withDensity
      (fun ω => ENNReal.ofReal (W r b k R ω / ∫ ω', W r b k R ω' ∂(Q r b k R)))
  /-- `Q_r^W` is a probability measure. -/
  QW_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), IsProbabilityMeasure (QW r b k R)
  /-- Section 5 clause (13): "The contact kernel is Lambda_j(x;b,k,u)", a given function of
  the index `j`, birth `b`, gap mark `k`, axial direction `u` and position `x`. -/
  Λ : ℕ → ℝ → ℝ → E d → E d → ℝ

/-! ### Counts and regions -/

/-- `N_(r,j)(E)`: the number of index-`j` critical points of `g` in `A` with height STRICTLY
between `b - k r^3` and `b` (`Set.ncard`, which is `0` for an infinite set).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "N_(r,j)(E)=#{x in E: grad f(x)=0, index H_x=j,"
Interface: none.
Does not claim: finiteness of the counted set; that boundary heights may be included (the source's remark about zero expected boundary counts is not stated). -/
noncomputable def windowCount (d : ℕ) (g : E d → ℝ) (b k r : ℝ) (j : ℕ) (A : Set (E d)) : ℕ :=
  Set.ncard {x : E d | x ∈ A ∧ IsCriticalPoint d g x ∧ HessianHasInertia d g x j ∧
    b - k * r ^ 3 < g x ∧ g x < b}

/-- All-index window count: the number of critical points of `g` in `A` (degenerate ones
included) with height strictly between `b - k r^3` and `b`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "`N` is the number of critical points of `f` in `X minus {M, S}` with height in `I_r`, all"
Interface: none.
Does not claim: finiteness of the counted set; equality with the sum of the index counts (true off the degenerate locus, not stated). -/
noncomputable def windowCountAll (d : ℕ) (g : E d → ℝ) (b k r : ℝ) (A : Set (E d)) : ℕ :=
  Set.ncard {x : E d | x ∈ A ∧ IsCriticalPoint d g x ∧ b - k * r ^ 3 < g x ∧ g x < b}

/-- `N(E) = sum_j N_j(E)`: the all-index window count of the remote-collision note, the sum
over `j = 0, ..., d` of the index-`j` window counts (critical points with nonsingular Hessian
of index `j`; an index above `d` does not occur in `R^d`).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/remote_collision_20260928/PROOF.md sha256 b9b8b58fd8266db7ffe6537078003888ef138b445d3289ec5588f116af9050c2
Anchor: "N(E)   = sum_j N_j(E)."
Interface: none.
Does not claim: finiteness of the counted sets; equality with `windowCountAll` (the two agree only off the degenerate locus). -/
noncomputable def windowCountIndexSum (d : ℕ) (g : E d → ℝ) (b k r : ℝ) (A : Set (E d)) : ℕ :=
  ∑ j ∈ Finset.range (d + 1), windowCount d g b k r j A

/-- `N_j(E)`: the number of index-`j` critical points of `g` in `A` at all heights.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_annulus_bridge_20260925/PROOF.md sha256 d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05
Anchor: "let N_j(rE) count index-j critical points in rE at **all heights**."
Interface: none.
Does not claim: finiteness of the counted set. -/
noncomputable def allHeightCount (d : ℕ) (g : E d → ℝ) (j : ℕ) (A : Set (E d)) : ℕ :=
  Set.ncard {x : E d | x ∈ A ∧ IsCriticalPoint d g x ∧ HessianHasInertia d g x j}

/-- The remote region `D_rho = {x : dist_X(x,0) >= rho}`, represented in the fundamental
domain so that a Borel subset of it carries one representative per torus point.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "Fix 0<rho<L/4 and D_rho={x:dist_X(x,0)>=rho}."
Interface: none.
Does not claim: anything about the representation of torus Borel sets beyond this choice. -/
def remoteRegion (d : ℕ) (L ρ : ℝ) : Set (E d) :=
  {x : E d | x ∈ FundDomain d L ∧ ρ ≤ torusDist d L x 0}

/-- The torus with the two pins removed, `X minus {M, S}`, represented in the fundamental
domain: points at positive torus distance from both `M = -(r/2)u` and `S = (r/2)u`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "X minus {M, S}"
Interface: none.
Does not claim: anything beyond the definition. -/
def torusMinusPins (d : ℕ) (L r : ℝ) (u : E d) : Set (E d) :=
  {x : E d | x ∈ FundDomain d L ∧ 0 < torusDist d L x (pinM d r u) ∧ 0 < torusDist d L x (pinS d r u)}

/-- The torus-wide all-index window count `N` of the C6 notes: critical points other than
`M, S` with height in `I_r = (b - k r^3, b)`, under the frame `R` (axial direction `R 0`).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "`N` is the number of critical points of `f` in `X minus {M, S}` with height in `I_r`, all"
Interface: `I : PinnedLawInterface` supplies the sample paths.
Does not claim: finiteness or measurability of the count as a function of the sample. -/
noncomputable def torusWindowCount (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω) : ℕ :=
  windowCountAll d (I.F ω) b k r (torusMinusPins d L r (R 0))

/-- Ordered pairs of distinct window critical points (both in `X minus {M, S}`, both with
height in `I_r`, at positive torus distance from each other) such that the FIRST witness is
within torus distance `η` of a pin. All mutual separations are included, down to collision.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "one witness is within fixed `η` of a pin."
Interface: none.
Does not claim: finiteness of the counted set; that ordered pairs with the near-pin witness second are counted (they are counted by symmetry of the roles only when both witnesses are near a pin; the pair with the near-pin witness in first position is always counted). -/
noncomputable def nearPinPairCount (d : ℕ) (L : ℝ) (g : E d → ℝ) (b k r η : ℝ) (u : E d) : ℕ :=
  Set.ncard {q : E d × E d |
    q.1 ∈ torusMinusPins d L r u ∧ q.2 ∈ torusMinusPins d L r u ∧ 0 < torusDist d L q.1 q.2 ∧
    IsCriticalPoint d g q.1 ∧ IsCriticalPoint d g q.2 ∧
    b - k * r ^ 3 < g q.1 ∧ g q.1 < b ∧ b - k * r ^ 3 < g q.2 ∧ g q.2 < b ∧
    (torusDist d L q.1 (pinM d r u) < η ∨ torusDist d L q.1 (pinS d r u) < η)}

/-- The scaled chart `x0 + r E0` read through the frame `R`: the image of a coordinate set
`E0` (coordinates `p` with `p 0` axial) under `p ↦ x0 + r ∑ p_i R_i`. With `x0 = 0` these
are the midpoint scaled coordinates `X = r(u, v)`, in which the pins are `(±1/2, 0)`; with
`x0 = M` they are the endpoint chart `M + r(p, q)`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "Scaled coordinates are `X = r(u, v)` from the midpoint, so the pins are `(±1/2, 0)`."
Interface: none.
Does not claim: that the image lies in one fundamental domain (the sources take `r` small enough for the chart to embed in the torus). -/
noncomputable def scaledAt (d : ℕ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (r : ℝ) (x0 : E d)
    (E0 : Set (E d)) : Set (E d) :=
  (fun p : E d => x0 + r • ∑ i, p i • R i) '' E0

/-- The compact collar `C(eta, R)` in midpoint scaled coordinates: the ball of radius `R`
(boundary included) with the two open balls of radius `eta` about the scaled pins
`(-1/2, 0)` and `(1/2, 0)` removed.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "with the two open balls of radius `eta` about `(-1/2, 0)` and `(1/2, 0)` removed"
Interface: none.
Does not claim: anything beyond the definition; RECONCILIATION.md writes the same set `C_{η,R}` without defining it, and this definition is taken from the dimension-lift source. -/
def collar (d : ℕ) [NeZero d] (η R0 : ℝ) : Set (E d) :=
  {p : E d | ‖p‖ ≤ R0 ∧ η ≤ ‖p - EuclideanSpace.single (0 : Fin d) (-(1 : ℝ) / 2)‖ ∧
    η ≤ ‖p - EuclideanSpace.single (0 : Fin d) ((1 : ℝ) / 2)‖}

/-- The weight `omega_d(p, q) = max(|q|, r |p|)^(2-d)` of the endpoint chart, where `p` is
the axial coordinate (`p 0`) and `q` the transverse part; `omega_2 = 1`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "omega_d(p, q) = max(|q|, r |p|)^(2-d)"
Interface: none.
Does not claim: integrability of the weight over the punctured ball (part of the Theorem P_d statement). -/
noncomputable def omegaD (d : ℕ) [NeZero d] (r : ℝ) (p : E d) : ℝ :=
  (max ‖p - EuclideanSpace.single (0 : Fin d) (p 0)‖ (r * |p 0|)) ^ ((2 : ℝ) - d)

/-- `E_(Q_r^W) N_(r,j)(A)`: the expected index-`j` window count in `A` under the tilted law.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "E_(Q_r^W) N_(r,j)(E)"
Interface: `I : PinnedLawInterface` supplies the sample paths and the tilted law; the Kac-Rice representation (12) of this expectation is NOT used.
Does not claim: integrability of the count (if it is not integrable the Bochner integral is `0`). -/
noncomputable def meanWindowCount (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (j : ℕ)
    (A : Set (E d)) : ℝ :=
  ∫ ω, (windowCount d (I.F ω) b k r j A : ℝ) ∂(I.QW r b k R)

/-- `E_(Q_r^W) N_j(A)`: the expected index-`j` all-height count in `A` under the tilted law.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_annulus_bridge_20260925/PROOF.md sha256 d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05
Anchor: "E_(Q_r^W) N_j(rE)"
Interface: `I : PinnedLawInterface` supplies the sample paths and the tilted law; the weighted Kac-Rice intensity (16) is NOT used.
Does not claim: integrability of the count. -/
noncomputable def meanAllHeightCount (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (j : ℕ)
    (A : Set (E d)) : ℝ :=
  ∫ ω, (allHeightCount d (I.F ω) j A : ℝ) ∂(I.QW r b k R)

/-! ### Fixed-remote Theorem A, displays (2) and (3) -/

/-- Shape of Theorem A, display (2) (fixed-remote mean measure): for fixed `d ≥ 2`, `L > 0`,
`0 < ρ < L/4`, compact `B = [bLo, bHi]` and `K = [kLo, kHi]` with `kLo > 0`, there are
`r_* > 0` and `C` (quantified OUTSIDE the universal quantifiers) such that for all
`0 < r ≤ r_*`, `b ∈ B`, `k ∈ K`, frames `R`, indices `j ≤ d` and Borel `E ⊆ D_ρ`,
`| E_(Q_r^W) N_(r,j)(E)/(k r^3) - ∫_E Lambda_j(x;b,k,u) dx | ≤ C r |E|`; and the kernel
`Lambda_j` is continuous on `D_ρ`, bounded above and away from zero uniformly on the
compact parameter sets.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "| E_(Q_r^W) N_(r,j)(E)/(k r^3)"
Interface: `I` (sample paths, laws `Q_r`, `Q_r^W`, weight, kernel `Λ`) is a given object; the statement is about any `Λ` and any laws satisfying `PinnedLawInterface`; the Kac-Rice representation (12) and the contact formula (13) are not transcribed.
Does not claim: the result or its proof (uniform remote nondegeneracy, coupling (5), weight expansion (10), normalizer (11)); identification of `I.Λ` with the kernel (13); the total-variation convergence remark; the multi-witness Section 6; any numerical `C` or `r_*`; uniformity as `ρ → 0`; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN is transcribed by the register, not here). -/
def FixedRemoteMeanMeasureA2 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → 0 < ρ → ρ < L / 4 → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ rStar : ℝ, 0 < rStar ∧ ∃ C : ℝ,
      (∃ cΛ : ℝ, 0 < cΛ ∧ ∃ CΛ : ℝ, ∀ j : ℕ, j ≤ d →
        ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi, ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          ContinuousOn (I.Λ j b k (R 0)) (remoteRegion d L ρ) ∧
          ∀ x ∈ remoteRegion d L ρ, cΛ ≤ I.Λ j b k (R 0) x ∧ I.Λ j b k (R 0) x ≤ CΛ) ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
          ∀ A : Set (E d), MeasurableSet A → A ⊆ remoteRegion d L ρ →
            |meanWindowCount d L Ω I r b k R j A / (k * r ^ 3) - ∫ x in A, I.Λ j b k (R 0) x| ≤
              C * r * (volume A).toReal

/-- Shape of Theorem A, display (3) (fixed-remote count and event bounds): with the same
fixed data and existential `r_* > 0`, `C`, for all `0 < r ≤ r_*`, `b ∈ B`, `k ∈ K`, frames,
indices and Borel `E ⊆ D_ρ`, `E_(Q_r^W) N_(r,j)(E) ≤ C k r^3 |E|` and
`P_(Q_r^W){N_(r,j)(E) ≥ 1} ≤ C k r^3 |E|`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/remote_window_20260924/PROOF.md sha256 a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7
Anchor: "E_(Q_r^W) N_(r,j)(E) <= C k r^3 |E|,"
Interface: `I` is a given object as in `FixedRemoteMeanMeasureA2`.
Does not claim: a matching lower bound on the event probability (the source says this is NOT one); the unnormalized numerator bound (14); the result or its proof; any numerical constant; nonauthor review or acceptance. -/
def FixedRemoteCountBoundA3 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → 0 < ρ → ρ < L / 4 → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ rStar : ℝ, 0 < rStar ∧ ∃ C : ℝ,
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
          ∀ A : Set (E d), MeasurableSet A → A ⊆ remoteRegion d L ρ →
            meanWindowCount d L Ω I r b k R j A ≤ C * k * r ^ 3 * (volume A).toReal ∧
            ((I.QW r b k R) {ω | 1 ≤ windowCount d (I.F ω) b k r j A}).toReal ≤
              C * k * r ^ 3 * (volume A).toReal

/-! ### Fixed scaled annulus (d = 2): the height-window candidate (A2) and the all-height
bridge (2) -/

/-- Shape of the fixed-annulus height-window candidate statement (A2): `d = 2`, fixed `L > 0`,
compact marks (`b ∈ [bLo, bHi]`, `0 < kLo ≤ k ≤ kHi`), fixed scaled annulus
`K = {A0 ≤ |(u,t)| ≤ B0}` with `1 < A0 < B0`; there are `C, r_* > 0`, uniform in the frame,
the marks, the Borel `E0 ⊆ K` and `j ∈ {0,1,2}`, such that
`E_(Q_r^W) N_j(r E0) ≤ C k r^3 area(E0)` for `0 < r ≤ r_*`, where `N_j(r E0)` counts
index-`j` critical points in `r E0` with height STRICTLY between `b - k r^3` and `b`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md sha256 1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b
Anchor: "E_Qr^W N_j(r E0) <= C*k*r^3*area(E0),"
Interface: `I` is a given object; `r E0` is embedded in the torus through the frame by `scaledAt`; the six-pin regression (A3)-(A5), the normalizer (A8) and the weighted Kac-Rice formula (A21) are not transcribed.
Does not claim: the result or its proof (contact vector (A10), eigenfloor (A12), cutoff (A13), three-Hessian estimate (A18), inner-strip bound (A26)); the all-index sum or the Markov event bound; an all-height assertion; uniformity in `L`, as `k → 0`, as `B0 → ∞`, or for `d ≥ 3`; any numerical constant; nonauthor review or acceptance (the landing disposition REVIEWED_SCOPED is transcribed by the register, not here). -/
def FixedAnnulusWindowCandidateA2 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi A0 B0 : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi → 1 < A0 → A0 < B0 →
    ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          ∀ E0 : Set (E 2), MeasurableSet E0 → E0 ⊆ {p : E 2 | A0 ≤ ‖p‖ ∧ ‖p‖ ≤ B0} →
            meanWindowCount 2 L Ω I r b k R j (scaledAt 2 R r 0 E0) ≤
              C * k * r ^ 3 * (volume E0).toReal

/-- Shape of the all-height fixed-annulus bridge candidate statement, display (2): `d = 2`,
fixed `L > 0`, compact marks, fixed scaled annulus `K_AB = {A ≤ |(u,v)| ≤ B}` with
`1 < A < B`; there are `C, r_* > 0`, uniform in the marks, frames, Borel `E ⊆ K_AB` and
`j ∈ {0,1,2}`, such that `E_(Q_r^W) N_j(rE) ≤ C r^3 area(E)` for `0 < r ≤ r_*`, where
`N_j(rE)` counts index-`j` critical points in `rE` at ALL heights. The review interfaces
R1-R7 of the source are the steps of its proof, not parts of this statement.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/rn_annulus_bridge_20260925/PROOF.md sha256 d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05
Anchor: "E_(Q_r^W) N_j(rE) <= C r^3 area(E), 0<r<=r_*."
Interface: `I` is a given object; `rE` is embedded through the frame by `scaledAt`; the intensity formula (16) and the normalizer (5) are not transcribed.
Does not claim: the result or its proof (R1 subtraction and remainders, R2 full-rank compactification (12), R3 conditional moments (14), R4 three-Hessian bound (6)-(8), R5 original `Z`, R6 weighted Kac-Rice, R7 crossover and cover); the rewriting with a `k` factor through `k_-`; uniformity as `k → 0`; pin neighbourhoods, `d ≥ 3`, intermediate distances or collisions; any numerical constant; nonauthor review or acceptance (the R1-R7 ACCEPT record is transcribed by the register, not here). -/
def AnnulusBridgeAllHeight2 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi A B : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi → 1 < A → A < B →
    ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          ∀ E0 : Set (E 2), MeasurableSet E0 → E0 ⊆ {p : E 2 | A ≤ ‖p‖ ∧ ‖p‖ ≤ B} →
            meanAllHeightCount 2 L Ω I r b k R j (scaledAt 2 R r 0 E0) ≤
              C * r ^ 3 * (volume E0).toReal

/-! ### The D5 reconciliation rows D5-a ... D5-f (d = 2, fixed torus, compact marks with
`k ≥ k_- > 0`, all frames and indices, existential constants) -/

/-- Shape of row D5-a (P2, punctured pin disks): all heights; for Borel
`E ⊂ {0 < p² + q² ≤ 1/16}` in the endpoint chart, `E_{Q_r^W} N_j(M + rE) ≤ C r³|E|`, and
the same at `S`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "E_{Q_r^W} N_j(M + rE) ≤ C r³|E|"
Interface: `I` is a given object; the chart `M + rE` is `scaledAt` based at the pin.
Does not claim: the P2 source (PUNCTURED_PIN_PROOF.md) or its proof; the P2 continuum review or the proposed register transitions of the reconciliation; any numerical constant; nonauthor review or acceptance. -/
def D5aPuncturedPinDisks (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          ∀ E0 : Set (E 2), MeasurableSet E0 →
            E0 ⊆ {p : E 2 | 0 < ‖p‖ ^ 2 ∧ ‖p‖ ^ 2 ≤ (1 : ℝ) / 16} →
              meanAllHeightCount 2 L Ω I r b k R j (scaledAt 2 R r (pinM 2 r (R 0)) E0) ≤
                C * r ^ 3 * (volume E0).toReal ∧
              meanAllHeightCount 2 L Ω I r b k R j (scaledAt 2 R r (pinS 2 r (R 0)) E0) ≤
                C * r ^ 3 * (volume E0).toReal

/-- Shape of row D5-b (microdisk): all heights; for fixed `κ > 0`, the expected index-`j`
count in the punctured disk of physical radius `κ r²` about each pin is `O(r⁵)`: there are
`C, r_* > 0` with `E_{Q_r^W} N_j(disk) ≤ C r^5` for `0 < r ≤ r_*`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "the nested disks of radius `κr²` about each pin, for fixed `κ`: `O(r⁵)`"
Interface: `I` is a given object; the disk is punctured at the pin (the pin itself is a critical point under the law), following the dimension-lift reading "nested ball" inside the punctured ball `D`.
Does not claim: the P2 source or its nested-disk corollary; any numerical constant; uniformity in `κ`; nonauthor review or acceptance. -/
def D5bMicrodisk (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∀ κ : ℝ, 0 < κ → ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          meanAllHeightCount 2 L Ω I r b k R j
            {x : E 2 | 0 < ‖x - pinM 2 r (R 0)‖ ∧ ‖x - pinM 2 r (R 0)‖ ≤ κ * r ^ 2} ≤ C * r ^ 5 ∧
          meanAllHeightCount 2 L Ω I r b k R j
            {x : E 2 | 0 < ‖x - pinS 2 r (R 0)‖ ∧ ‖x - pinS 2 r (R 0)‖ ≤ κ * r ^ 2} ≤ C * r ^ 5

/-- Shape of row D5-c (C1, compact collar): all heights; for fixed `R ≥ 1` and
`0 < η ≤ 1/4`, for Borel `E ⊂ C_{η,R}` in midpoint scaled coordinates,
`E_{Q_r^W} N_j(rE) ≤ C r³|E|`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "for Borel `E ⊂ C_{η,R}`, `η ≤ 1/4`, fixed `R ≥ 1`"
Interface: `I` is a given object; `C_{η,R}` is `collar 2 η R` (definition taken from the dimension-lift source, since RECONCILIATION.md does not define it).
Does not claim: the C1 source (COLLAR_PROOF.md) or its proof; the normalizer (P14) it imports; any numerical constant; nonauthor review or acceptance. -/
def D5cCollarC1 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∀ η R0 : ℝ, 0 < η → η ≤ (1 : ℝ) / 4 → 1 ≤ R0 → ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          ∀ E0 : Set (E 2), MeasurableSet E0 → E0 ⊆ collar 2 η R0 →
            meanAllHeightCount 2 L Ω I r b k R j (scaledAt 2 R r 0 E0) ≤
              C * r ^ 3 * (volume E0).toReal

/-- Shape of row D5-d (C2, the summed pin-neighbourhood bound): all heights; for fixed
`R ≥ 1`, `E_{Q_r^W} N_j(r B_R ∖ {M,S}) ≤ C_R r³`, the scaled ball of radius `R` with the
two pins removed.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "E_{Q_r^W} N_j(r B_R ∖ {M,S}) ≤ C_R r³"
Interface: `I` is a given object.
Does not claim: the composition `D5-a + D5-c at η = 1/4` as a proof; the "collar to the reviewed annulus" remark; any numerical constant; nonauthor review or acceptance. -/
def D5dSummedPinNeighbourhoodC2 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∀ R0 : ℝ, 1 ≤ R0 → ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          meanAllHeightCount 2 L Ω I r b k R j (scaledAt 2 R r 0
            {p : E 2 | ‖p‖ ≤ R0 ∧ p ≠ EuclideanSpace.single (0 : Fin 2) (-(1 : ℝ) / 2) ∧
              p ≠ EuclideanSpace.single (0 : Fin 2) ((1 : ℝ) / 2)}) ≤ C * r ^ 3

/-- Shape of row D5-e (I3/I4, intermediate shells): window `I_r`; the dyadic shells and
`{A_0 r ≤ |X| ≤ ρ}` satisfy `E N ≤ C r³(A_0^{−2} + ρ²)`: there are `C, s_0 > 0` such that
for `A_0 ≥ 4` and `A_0 r < ρ ≤ s_0` the expected index-`j` window count in
`{A_0 r ≤ |X| ≤ ρ}` (distance from the midpoint) is at most `C r^3 (A_0^(-2) + ρ^2)`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "E N ≤ C r³(A_0^{−2} + ρ²)"
Interface: `I` is a given object; the side conditions `A_0 ≥ 4`, `A_0 r < ρ ≤ s_0` are taken from the dimension-lift display (1.6), since RECONCILIATION.md states the row without them.
Does not claim: the single-shell display (1.5) (stated in `DimensionLiftIntermediateShellsI`); the I3/I4 source (intermediate_window_20260928) or its proof; any numerical constant; nonauthor review or acceptance. -/
def D5eIntermediateShellsI3I4 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C s0 : ℝ, 0 < C ∧ 0 < s0 ∧
      ∀ A0 ρ r : ℝ, 4 ≤ A0 → 0 < r → A0 * r < ρ → ρ ≤ s0 →
        ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
          ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
            meanWindowCount 2 L Ω I r b k R j {x : E 2 | A0 * r ≤ ‖x‖ ∧ ‖x‖ ≤ ρ} ≤
              C * r ^ 3 * ((A0 ^ 2)⁻¹ + ρ ^ 2)

/-- Shape of row D5-f (I5, global window first moment and planar event order): window
`I_r`; `E_{Q_r^W} N_{r,j}(T² ∖ {M,S}) ≤ C r³`, and for the event `A_r` that some critical
point other than `M, S` has height in `I_r`, `c r³ ≤ P(A_r) ≤ C r³` with `c > 0`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "E_{Q_r^W} N_{r,j}(T² ∖ {M,S}) ≤ C r³"
Interface: `I` is a given object; `A_r` is the event `{N ≥ 1}` for the all-index torus-wide window count.
Does not claim: the tiling composition `D5-d (R = 4) + I4 + remote Theorem A` as a proof; the lower bound's source (remote_collision Corollary F-) or the I5 source; any numerical constant; nonauthor review or acceptance; any register transition proposed by the reconciliation; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def D5fGlobalWindowFirstMomentI5 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    (∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2), ∀ j : ℕ, j ≤ 2 →
          meanWindowCount 2 L Ω I r b k R j (torusMinusPins 2 L r (R 0)) ≤ C * r ^ 3) ∧
    (∃ c C rStar : ℝ, 0 < c ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2),
          c * r ^ 3 ≤ ((I.QW r b k R) {ω | 1 ≤ torusWindowCount 2 L Ω I r b k R ω}).toReal ∧
          ((I.QW r b k R) {ω | 1 ≤ torusWindowCount 2 L Ω I r b k R ω}).toReal ≤ C * r ^ 3)

/-! ### The dimension lift: Theorems P_d, C_d, I_d, G_d (every fixed d ≥ 2) -/

/-- Shape of Theorem P_d (all-height punctured pin ball), display (1.2): fixed `d ≥ 2`,
`L > 0`, compact marks; there are `C, C', r_* > 0` such that for `0 < r ≤ r_*`, all `b, k, R, j`
and every Borel `E ⊆ D = {0 < p² + |q|² ≤ 1/16}` in the endpoint chart,
`E_(Q_r^W) N_j(M + r E) ≤ C r^3 ∫_E omega_d ≤ C' r^3`, and the same about `S`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "E_(Q_r^W) N_j(M + r E) <= C r^3 integral_E omega_d(p, q) dp dq <= C' r^3."
Interface: `I` is a given object; the chart is `scaledAt` based at the pin; the weight is `omegaD`.
Does not claim: the result or its proof (devices D1-D5, stable gradient frame, density and conditional moments, weighted Kac-Rice (4.10), crossover); the nested-ball `O(r^5)` corollary (stated for `d = 2` as `D5bMicrodisk`); the planar P2 identification; any numerical constant; uniformity in `d`; nonauthor review or acceptance. -/
def DimensionLiftPuncturedPinBallP (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C C' rStar : ℝ, 0 < C ∧ 0 < C' ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
          ∀ E0 : Set (E d), MeasurableSet E0 →
            E0 ⊆ {p : E d | 0 < ‖p‖ ^ 2 ∧ ‖p‖ ^ 2 ≤ (1 : ℝ) / 16} →
              (meanAllHeightCount d L Ω I r b k R j (scaledAt d R r (pinM d r (R 0)) E0) ≤
                  C * r ^ 3 * ∫ p in E0, omegaD d r p ∧
                C * r ^ 3 * ∫ p in E0, omegaD d r p ≤ C' * r ^ 3) ∧
              (meanAllHeightCount d L Ω I r b k R j (scaledAt d R r (pinS d r (R 0)) E0) ≤
                  C * r ^ 3 * ∫ p in E0, omegaD d r p ∧
                C * r ^ 3 * ∫ p in E0, omegaD d r p ≤ C' * r ^ 3)

/-- Shape of Theorem C_d (compact collar), display (1.3): fixed `d ≥ 2`, `L > 0`, compact
marks, fixed `R ≥ 1` and `0 < η ≤ 1/4`; there are `C, r_* > 0` such that for every Borel
`E ⊆ C(η, R)`, `E_(Q_r^W) N_j(r E) ≤ C r^3 |E|` for `0 < r ≤ r_*`, all `b, k, R, j`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "E_(Q_r^W) N_j(r E) <= C r^3 |E|."
Interface: `I` is a given object; `C(η, R)` is `collar d η R`.
Does not claim: the consequence (1.4) for the scaled ball with pins removed (stated for `d = 2` as `D5dSummedPinNeighbourhoodC2`); the result or its proof; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftCompactCollarC (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∀ η R0 : ℝ, 0 < η → η ≤ (1 : ℝ) / 4 → 1 ≤ R0 → ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
          ∀ E0 : Set (E d), MeasurableSet E0 → E0 ⊆ collar d η R0 →
            meanAllHeightCount d L Ω I r b k R j (scaledAt d R r 0 E0) ≤
              C * r ^ 3 * (volume E0).toReal

/-- Shape of Theorem I_d (intermediate height-window shells), displays (1.5) and (1.6):
there are `C, s_0 > 0` such that for `0 < r ≤ s/4` and `0 < s ≤ s_0`,
`E_(Q_r^W) N_(r,j)({ s ≤ |X| ≤ 2 s }) ≤ C r^3 [ (r/s)^2 + s^2 ]`, and hence, uniformly for
`A_0 ≥ 4` and `A_0 r < ρ ≤ s_0`, `E_(Q_r^W) N_(r,j)({ A_0 r ≤ |X| ≤ ρ }) ≤ C r^3 (A_0^(-2) + ρ^2)`.
`|X|` is the distance from the midpoint.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "E_(Q_r^W) N_(r,j)({ s <= |X| <= 2 s }) <= C r^3 [ (r/s)^2 + s^2 ],"
Interface: `I` is a given object.
Does not claim: the result or its proof (polynomial row rank, Euler suppression, shell ledger, dyadic sum); the planar I3/I4 identification; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftIntermediateShellsI (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C s0 : ℝ, 0 < C ∧ 0 < s0 ∧
      (∀ r s : ℝ, 0 < r → r ≤ s / 4 → 0 < s → s ≤ s0 →
        ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
          ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
            meanWindowCount d L Ω I r b k R j {x : E d | s ≤ ‖x‖ ∧ ‖x‖ ≤ 2 * s} ≤
              C * r ^ 3 * ((r / s) ^ 2 + s ^ 2)) ∧
      (∀ A0 ρ r : ℝ, 4 ≤ A0 → 0 < r → A0 * r < ρ → ρ ≤ s0 →
        ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
          ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
            meanWindowCount d L Ω I r b k R j {x : E d | A0 * r ≤ ‖x‖ ∧ ‖x‖ ≤ ρ} ≤
              C * r ^ 3 * ((A0 ^ 2)⁻¹ + ρ ^ 2))

/-- Shape of Theorem G_d (global height-window first moment), display (1.7): fixed `d ≥ 2`,
`L > 0`, compact marks; there are `C, r_* > 0` such that
`E_(Q_r^W) N_(r,j)( X minus {M, S} ) ≤ C r^3` for `0 < r ≤ r_*`, all `b, k, R, j`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/d5_dimension_lift_20260929/PROOF.md sha256 6fb94b247e8e45bca9dc02f46670ac6029fb54e0ef906001efe48fe908549b80
Anchor: "E_(Q_r^W) N_(r,j)( X minus {M, S} ) <= C r^3."
Interface: `I` is a given object.
Does not claim: the tiling proof (P_d, C_d, I_d and the fixed-remote Theorem A); Corollaries F_d and M_d; the planar I5 identification; the `d = 3` SIDE24 application; any regional supersession of historical carriers; any numerical constant; nonauthor review or acceptance. -/
def DimensionLiftGlobalWindowG (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C rStar : ℝ, 0 < C ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), ∀ j : ℕ, j ≤ d →
          meanWindowCount d L Ω I r b k R j (torusMinusPins d L r (R 0)) ≤ C * r ^ 3

/-! ### C6: Fourier cutoff (F4), Palm route (Theorem Q, Corollary Theta) -/

/-- Shape of the planar Fourier-cutoff consequence (F4): in `d = 2`, GIVEN the
dimension-matched first-moment hypothesis `G_2` (`E_(Q_r^W) N ≤ C r^3` for small `r`,
uniformly in the marks and frames), there are `C` and `r_* > 0` such that
`E_(Q_r^W) N(N-1) ≤ C r^3 log(1/r)` for `0 < r ≤ r_*`, all `b, k, R`. The source derives
this from Theorem W with `p = 2`; the hypothesis `G_2` is stated as a hypothesis here, not
discharged.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_fourier_cutoff_20260929/PROOF.md sha256 c1692379a3a066589bd2522aaa5b4d480e736b737c8c39a474d1735185793733
Anchor: "E_Qr^W N(N-1) <= C r^3 log(1/r),"
Interface: `I` is a given object; `N` is `torusWindowCount`; `(N)_2 = N(N-1)` is `Nat.descFactorial N 2`.
Does not claim: Theorem F (the cap `Psi_F` and its tail (F1)-(F2)); Theorem W for general `d` and `p` (F3); that `G_2` holds (the source takes it from the reviewed planar I5 composition); Proposition B; that the logarithm is sharp or removable; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def FourierCutoffPlanarSecondFactorialF4 (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface 2 L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    (∃ C0 r0 : ℝ, 0 < r0 ∧
      ∀ r : ℝ, 0 < r → r ≤ r0 → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2),
          ∫ ω, (torusWindowCount 2 L Ω I r b k R ω : ℝ) ∂(I.QW r b k R) ≤ C0 * r ^ 3) →
    ∃ C rStar : ℝ, 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2),
          ∫ ω, (Nat.descFactorial (torusWindowCount 2 L Ω I r b k R ω) 2 : ℝ) ∂(I.QW r b k R) ≤
            C * r ^ 3 * Real.log (1 / r)

/-- Shape of Theorem Q (Palm route, factorial moments in every fixed dimension): fixed
`d ≥ 2`, `L > 0`, compact marks; for every fixed integer `q ≥ 2` there are `C_q, r_* > 0`
such that `E_(Q_r^W)[ (N)_q ] ≤ C_q r^3` for `0 < r ≤ r_*`, all `b, k, R`. The radius
threshold is allowed to depend on `q` (the source's quantifier order is "There are
`C_q, r_* > 0` such that for `0 < r ≤ r_*`, all `b, k, R` and every fixed integer `q ≥ 2`";
the weaker reading with `r_*` after `q` is stated).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "E_(Q_r^W)[ (N)_q ] <= C_q r^3."
Interface: `I` is a given object; `N` is `torusWindowCount`; `(N)_q` is `Nat.descFactorial N q`.
Does not claim: the result or its proof (the cap `Psi` in `d` dimensions, the parametric Rouché tail, the marked Kac-Rice formula with Hölder insertion, the regimes R1-R4); the dependence on [C6L] and [DL] at their bytes; a lower bound for `q ≥ 3`; Corollary P; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def PalmRouteFactorialMomentsQ (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∀ q : ℕ, 2 ≤ q → ∃ Cq rStar : ℝ, 0 < Cq ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          ∫ ω, (Nat.descFactorial (torusWindowCount d L Ω I r b k R ω) q : ℝ) ∂(I.QW r b k R) ≤
            Cq * r ^ 3

/-- Shape of Corollary Theta (the C6 order is `r^3`): in every fixed `d ≥ 2`, there are
`c > 0`, `C` and `r_* > 0` such that `c r^3 ≤ E_(Q_r^W)[ N(N-1) ] ≤ C r^3` and
`c r^3 ≤ Q_r^W{ N ≥ 2 } ≤ C r^3` for `0 < r ≤ r_*`, all `b, k, R`.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_palm_route_20260929/PROOF.md sha256 aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b
Anchor: "c r^3 <= E_(Q_r^W)[ N(N-1) ] <= C r^3,"
Interface: `I` is a given object; `N` is `torusWindowCount`.
Does not claim: the lower bounds' source ([EDL] (A4)) or the upper bounds' sources (Theorem Q, [DL] Corollary M_d); that the logarithm of the Fourier cutoff is removed as a reviewed fact; any statement about the positions of the pairs; any numerical constant; nonauthor review or acceptance; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus. -/
def PalmRouteSecondFactorialTheta (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ c C rStar : ℝ, 0 < c ∧ 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          c * r ^ 3 ≤
            ∫ ω, (Nat.descFactorial (torusWindowCount d L Ω I r b k R ω) 2 : ℝ) ∂(I.QW r b k R) ∧
          ∫ ω, (Nat.descFactorial (torusWindowCount d L Ω I r b k R ω) 2 : ℝ) ∂(I.QW r b k R) ≤
            C * r ^ 3 ∧
          c * r ^ 3 ≤ ((I.QW r b k R) {ω | 2 ≤ torusWindowCount d L Ω I r b k R ω}).toReal ∧
          ((I.QW r b k R) {ω | 2 ≤ torusWindowCount d L Ω I r b k R ω}).toReal ≤ C * r ^ 3

/-! ### C6 rare-cluster consequences: Theorem O (abstract, on laws on ℕ) -/

/-- Total-variation distance between two probability mass functions on `ℕ`: half the `l^1`
distance (the source's convention; equal to the supremum over events for probability laws).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_rare_cluster_laws_20260929/PROOF.md sha256 c52a3cf197b5cf26071a8cc951e15ef3ac564b3d7453d41647f91af540a1b8eb
Anchor: "For probability laws, dTV is the supremum over measurable events, equivalently half the l1 distance on this countable state space."
Interface: none; if the `l^1` series is not summable the `tsum` is `0`.
Does not claim: equality with the event supremum (true for probability laws, not stated). -/
noncomputable def pmfTV (μ ν : ℕ → ℝ) : ℝ :=
  (1 / 2) * ∑' n : ℕ, |μ n - ν n|

/-- The Poisson probability mass function `Pois(lam)(n) = exp(-lam) lam^n / n!`
(for `lam ≥ 0`; `Pois(0) = δ_0`).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_rare_cluster_laws_20260929/PROOF.md sha256 c52a3cf197b5cf26071a8cc951e15ef3ac564b3d7453d41647f91af540a1b8eb
Anchor: "dTV(Law N, Pois(lambda))"
Interface: none; an explicit formula (the source's `Pois(lambda)`).
Does not claim: that it sums to one (true for `lam ≥ 0`, not stated). -/
noncomputable def poissonPMF (lam : ℝ) (n : ℕ) : ℝ :=
  Real.exp (-lam) * lam ^ n / (Nat.factorial n : ℝ)

/-- Shape of Theorem O (sharp ordinary-Poisson obstruction): for a family of laws
`μ_r` on `ℕ` (the law of the window count `N_r`) which, for all sufficiently small `r`
(that is, for `0 < r ≤ r_0` for some `r_0 > 0`), are probability mass functions with
summable mean satisfying only (H1) `E N_r ≤ A r^3` and (H2) `P(N_r ≥ 2) ≥ a r^3` with
`a > 0`: then, for all sufficiently small `r`,
`(a/2) r^3 ≤ inf_{lambda ≥ 0} dTV(Law N_r, Pois(lambda)) ≤ A r^3`: every nonnegative
intensity is at distance at least `(a/2) r^3`, and some intensity is within `A r^3`. The
hypotheses are imposed only on `0 < r ≤ r_0`, as in the source ("for all sufficiently small
r"); imposing (H2) for every `r > 0` would be unsatisfiable (`a r^3 ≤ 1` fails for large
`r`) and would make the statement hold for every family. The source's remark that the
premises imply `2a ≤ A` is not taken as a hypothesis.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/c6_rare_cluster_laws_20260929/PROOF.md sha256 c52a3cf197b5cf26071a8cc951e15ef3ac564b3d7453d41647f91af540a1b8eb
Anchor: "(a/2) r^3 <= inf_{lambda>=0} dTV(Law N, Pois(lambda)) <= A r^3."
Interface: the laws `μ_r` are given probability mass functions on `ℕ` (nonnegative, summing to one, with summable mean) for `0 < r ≤ r_0`; the hypotheses (H1), (H2) are hypotheses on that range only; nothing ties `μ_r` to the Gaussian model here (the source says the only analytic inputs are H1, H2, Hq).
Does not claim: the result (Theorem O) or its proof; that `2a ≤ A` (derived in the source, not a hypothesis here); that H1 or H2 hold for the Gaussian window count (the source takes them from [D5] Theorem G_d and [C6] Corollary Theta); the mean-matched statement `lambda = mu`; Theorems C and S, the factorial-moment matching or the independent-replica limit; the conditional-law liminf remark; nonauthor review or acceptance. -/
def RareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) : Prop :=
  0 < a →
  (∃ r0 : ℝ, 0 < r0 ∧ ∀ r : ℝ, 0 < r → r ≤ r0 →
    (∀ n : ℕ, 0 ≤ μ r n) ∧ Summable (μ r) ∧ ∑' n : ℕ, μ r n = 1 ∧
    Summable (fun n : ℕ => (n : ℝ) * μ r n) ∧ ∑' n : ℕ, (n : ℝ) * μ r n ≤ A * r ^ 3 ∧
    a * r ^ 3 ≤ 1 - μ r 0 - μ r 1) →
  ∃ rStar : ℝ, 0 < rStar ∧ ∀ r : ℝ, 0 < r → r ≤ rStar →
    (∀ lam : ℝ, 0 ≤ lam → (a / 2) * r ^ 3 ≤ pmfTV (μ r) (poissonPMF lam)) ∧
    (∃ lam : ℝ, 0 ≤ lam ∧ pmfTV (μ r) (poissonPMF lam) ≤ A * r ^ 3)

/-! ### Remote collision (Corollary D) and the OPEN witness-collision target -/

/-- Shape of Corollary D (second factorial moment in the fixed remote region): fixed `d ≥ 2`,
`L > 0`, `0 < ρ < L/4`, compact marks; there are `C` and `r_* > 0` such that for
`0 < r ≤ r_*`, all marks and frames and every Borel `E ⊆ D_ρ`,
`E_(Q_r^W) N(E)(N(E)-1) ≤ C r^5 |E|` and `Q_r^W{ N(E) ≥ 2 } ≤ C r^5 |E|`, where
`N(E) = sum_j N_j(E)` is the sum of the index-`j` window counts in `E`
(`windowCountIndexSum`, the source's definition). Pairs at every mutual separation inside
`D_ρ`, down to coincidence, are included.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/remote_collision_20260928/PROOF.md sha256 b9b8b58fd8266db7ffe6537078003888ef138b445d3289ec5588f116af9050c2
Anchor: "E_(Q_r^W) N(E)(N(E)-1) <= C r^5 |E|,"
Interface: `I` is a given object; the pair Kac-Rice formula (3.1) and the divided-difference coordinates (4.1) are not transcribed.
Does not claim: Theorem C (the ordered near-pair count) or its proof (Lemmas 1-5); Corollaries E and F; a torus-wide second factorial moment or `math.rn-region.witness-collision` in its full sense; pairs with a witness near the pins; equality of `N(E)` with the all-critical-point count `windowCountAll` (they agree only off the degenerate locus); any numerical constant; nonauthor review or acceptance. -/
def RemoteCollisionSecondFactorialD (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (ρ bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → 0 < ρ → ρ < L / 4 → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C rStar : ℝ, 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          ∀ A : Set (E d), MeasurableSet A → A ⊆ remoteRegion d L ρ →
            ∫ ω, (Nat.descFactorial (windowCountIndexSum d (I.F ω) b k r A) 2 : ℝ) ∂(I.QW r b k R) ≤
              C * r ^ 5 * (volume A).toReal ∧
            ((I.QW r b k R) {ω | 2 ≤ windowCountIndexSum d (I.F ω) b k r A}).toReal ≤
              C * r ^ 5 * (volume A).toReal

/-- OPEN OBLIGATION, stated as a TARGET and not as a result. GRAPH node
`math.rn-region.witness-collision` (classification OPEN_ACTIVE, catalog C6): the regional
shrinking pin/witness-collision mechanism. The reconciliation describes what is missing as
a torus-wide second factorial moment of the window count "with pairs where at least one
witness is within fixed `η` of a pin", at every mutual separation (the fixed-remote `P_η`
of Corollary D covers only pairs with both witnesses at distance `≥ η` from the pins). The
Prop below is that target: fixed `d ≥ 2`, `L > 0`, `η > 0`, compact marks; there are `C` and
`r_* > 0` with `E_(Q_r^W) #{ordered window pairs with the first witness within η of a pin} ≤ C r^3`
for `0 < r ≤ r_*`, all `b, k, R`. No source in this packet establishes it; its register
status is OPEN and this declaration does not change that.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path reviews/d5_reconciliation_20260929/RECONCILIATION.md sha256 14342b7eb0ad47c1c99b06be58fb6c901f5c7c72fc70f73e81de3dde50c23432
Anchor: "A torus-wide second factorial moment `E_{Q_r^W} N(N−1)` for the window count, with pairs where at least"
Interface: `I` is a given object; the pair count is `nearPinPairCount`.
Does not claim: that this target holds, is attainable, or is implied by any reviewed source (the Palm route's Theorem Q is an author-side candidate whose `q = 2` case would imply it; that candidate is stated separately as `PalmRouteFactorialMomentsQ` and is not consumed here); that the lower obstruction `E N(N−1) ≥ 2c r³` is stated here; the optimal order; any status transition of the node. -/
def WitnessCollisionNearPinPairsTarget (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    (I : PinnedLawInterface d L Ω) (η bLo bHi kLo kHi : ℝ) : Prop :=
  2 ≤ d → 0 < L → 0 < η → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
    ∃ C rStar : ℝ, 0 < rStar ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d),
          ∫ ω, (nearPinPairCount d L (I.F ω) b k r η (R 0) : ℝ) ∂(I.QW r b k R) ≤ C * r ^ 3

end UniversalLaw.Spec.RN
