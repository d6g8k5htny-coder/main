# Nonauthor review of the restricted collinear coalescence calculation

16 October 2026 UTC. Reviewer: OpenAI/Codex `/root/coalescence_review_20261016`.
Author reviewed: OpenAI/Codex `/root/core_theorem_audit_20261015`.
This is a separate nonauthor review within the same provider and parent thread;
organizational independence is **0**. I read the complete proposed calculation,
its complete diagnostic program and JSON, and the pinned source portions actually
used below. I did not author or edit those three inputs. No remote write, formal
execution, applicability change, scientific status transition or process claim
was performed by this review.

**Verdict: ACCEPT, restricted conventional mathematics only.** I find no blocking
mathematical error in the fixed-law, planar, sufficiently short collinear bounds
(8) and (11), the coordinate-aligned ranks and K2 obstruction in (16), or the
K>=4 exact-contact upper bound (20). Two nonblocking wording corrections are
listed below. Acceptance here does not extend the estimates off the collinear
stratum, make constants effective, identify a global selected-bar measure, or
establish a point-process limit. None of those conclusions follows from this
review.

## Exact inputs and source exposure

| Input | SHA-256 |
| --- | --- |
| `COALESCING_CONTACT_CALCULATION.md` | `39a63e1b00393b49e7b6d049c0c9ea7be82436352eed64014155a63aa8c7fb36` |
| `coalescing_diagnostics.py` | `6fe14ca7f518d3c42236421ddd5b8f7470accfec855f8889d17e8aaa62b89e2b` |
| `COALESCING_DIAGNOSTICS.json` | `946a962bf47589a76affae03977f3d03376a463e26b7943ea675cceb4ad2799a` |

I independently compared all four report-named local source copies with their
Git objects at Main `85b928b326ff0586715a56ae91f51d3fbd89557a`. Their bytes, Git
blob identities, sizes and SHA-256 values match the report's table. The substantive
source inputs consumed were the literal finite Fourier covariance multiplier,
six-row pin/target and physical Hessian weight in
`finite_gaussian_contact/PROOF.md` §§2–5, and the actual infinite-law/separated
center boundary in `iid_short_bar_process/PROOF.md` §6. I did not rerun the
underlying global H0 selection/counting proofs or import their conclusions into
this local estimate. The candidate/H0 source copies were checked for source
identity and their distinction is preserved.

The current [owner mandate](../gaussian-jet-coefficient-20261016-root01a11cbf/checkout/governance/OWNER_MANDATE_20261015.md)
and [standing delegation](../gaussian-jet-coefficient-20261016-root01a11cbf/checkout/governance/OP-AUTONOMY-20260923-v2.1.md)
were read. They authorize this work and do not supply mathematical acceptance.
For the elementary interpolation remainder I also checked the primary
[NIST DLMF interpolation reference, equation 3.3.5](https://dlmf.nist.gov/3.3.E5).
The proof below is reconstructed directly from Rolle/Taylor arguments rather
than treating that reference as an acceptance authority.

## Reconstruction of the eight-gradient density and determinant cancellation

For a fixed positive full-square Gaussian spectrum, write the field as a linear
map of a fixed finite standard Gaussian coefficient vector. A vanishing variance
of a linear combination of the eight limiting gradient jets forces its complex
frequency polynomial to vanish at every point of the full square. Its degree
in each original frequency coordinate is at most four. A grid with at least
five points in each coordinate therefore forces polynomial identity. After the
invertible orthogonal change to frequency coordinates `(a,c)`, the monomials
`a,a²,a³,a⁴,c,ac,a²c,a³c` are distinct, so every coefficient vanishes. The factors
of `i`, omega and factorials are nonzero row scalars and do not alter this
argument. It works for arbitrary positive symmetric weights; isotropy or D4
symmetry is unnecessary.

For each fixed law the limiting covariance depends continuously on the frame.
Compactness of O(2) gives a positive minimum eigenvalue. Finite Fourier rows have
uniform derivative bounds, and the divided-difference integral representation
gives convergence in mean square with an error controlled by the segment
diameter, independently of unequal node spacings. This last qualification is
essential: a Taylor expansion with denominator-dependent errors would not
suffice. The proposed integral argument supplies the needed uniformity.

The Newton map for four values has determinant `1/V`. Applying it separately to
the two gradient components gives determinant `V^-2`, hence
`det Cov(G)=V^4 det Cov(U)` and `p_G(0)=Theta(V^-2)`. The density normalization
`(2pi)^-4` is correct for eight real observations. Stationarity handles the
basepoint, and all constants remain fixed-law constants.

For a C4 scalar function vanishing at all four nodes, subtracting its value at
a fifth point times the monic nodal polynomial and applying Rolle four times
gives `|a(s)| <= ||a^(4)|| infinity product_j |s-x_j| / 24`. Dividing by
`|s-x_i|` and taking the endpoint limit proves
`|a'(x_i)| <= D_i ||a^(4)|| infinity / 24`. The argument holds also for the
two interior nodes; no common intermediate Rolle point for several functions
is asserted or needed.

With `a=F_u` and then `a=F_v`, the required derivatives are exactly `F_(u^5)`
and `F_(u^4 v)`. Bounding `F_vv` by a fixed coordinate C5 norm gives
`|det H_i| <= C M² D_i` after shrinking the physical segment so `D_i<=1`.
The squared mixed-Hessian term is included. Since `product_i D_i=V²`, the
product of all four physical determinant weights is at most `C M^8 V²`.

Conditioning on the zero gradient target has canonical coefficient law `PZ`,
where P is an orthogonal projection. It has zero mean even near covariance
degeneracy. Coupling it to the same standard Z gives `||PZ||<=||Z||`, so all
needed conditional norm moments are bounded independently of the node ratios.
This proves the bounded Rice integrand (8). It is not a four-dimensional
position integration or a new Kac-Rice counting identity.

## Positive cubic marks and the action identity

Under `r_i<=rho eta`, `0<rho<1`, each cross-pair distance lies between
`(1-rho)eta` and `(1+rho)eta`. The four sites are ordered and disjoint. The
first interval therefore satisfies

```text
|ell_1| <= ||F_(u^5)|| infinity /24
           * (1+rho)² eta² * integral_0^r1 s(r1-s) ds
         = (1+rho)² eta² ||F_(u^5)|| infinity r1³ /144.
```

The second interval has the same absolute bound. This verifies both the eta²
power and the constant 1/144 in (9). Requiring either positive lower cubic mark
forces a C5 norm of order `k_- eta^-2`. Combining the preceding cancellation
with the finite-dimensional Gaussian norm tail proves (11); polynomial moment
factors can be absorbed into a smaller positive Gaussian exponent. Its useful
suppression requires `k_-/eta²` large. At a moving lower mark of order eta² it
only gives a bounded estimate. The report states this limitation correctly.

I independently expanded Taylor formulas in the stated linear functional L1.
The coefficients of `a'(0)`, `a''(0)` and `a'''(0)` cancel, leaving the kernel
`s(eta-s)²/eta²` multiplying `a^(4)(s)`. Its integral is `eta²/12`. Thus
`Var(L1)<=m10(u) eta^4/144`, while its exact contact target is `12k1`.
Covariance Cauchy-Schwarz gives the action lower bound
`20736 k1²/(m10(u) eta^4)` and the density exponent with constant 10368.
The reflected identity and the summed kernel `s(eta-s)/eta` are also correct;
the latter has integral `eta²/6`, giving the necessary derivative bound
`72(k1+k2)/eta²` for the stated same-oriented targets. The individual endpoint
identity is the correct one to retain when the second contact direction is
reversed. The inverse covariance statement is explicitly restricted to
nonsingular joint contact covariance; it must not be used as a twelve-dimensional
density statement on the K2/K3 coordinate strata.

## Finite-carrier obstruction and the K>=4 bound

On a coordinate line the longitudinal restriction has dimension `2K+1`; its
value/first-derivative observations provide eight Hermite rows. The transverse
restriction has four rows and can be prescribed independently by using a sine
factor in the other coordinate. The exact ranks are therefore 9,11,12 for
K=2,3,>=4 on sufficiently short distinct configurations. Injectivity when the
number of constraints exceeds dimension follows by trigonometric zero counting;
surjectivity of the smaller Hermite lists follows equivalently by the ordinary
polynomial Hermite matrix after multiplying the Laurent polynomial by its
lowest-frequency monomial. For the two exact contact centers the multiplicity
lists are four/four longitudinal and two/two transverse, giving the same ranks.

At K2, `F_x` on that circle is a zero-mean degree-two trigonometric polynomial.
Two double zeros with nonzero second derivative exhaust its possible four zeros
counting multiplicity and prevent a sign change. Zero mean then forces the
polynomial to vanish identically, contradicting a nonzero contact cubic target.
The separate four-pin statement uses the local nonsingularity of four value
evaluations on the four-dimensional nonzero-frequency derivative space. Hence
the longitudinal derivative vanishes on the whole coordinate line, and
`F_xx=0` at the pins. A nondegenerate maximum is impossible there, although
saddles and nonsingular eight-gradient observation covariance remain possible.
This proves the claimed typed-kernel zero and avoids conflating observation
rank with maximum admissibility.

These obstructions are coordinate-stratum statements. My exact rank checks give
12 for the twelve coincident normalized jet polynomials at the rational frames
`u=(3/5,4/5)` and `u=(5/13,12/13)` even at K2. They do not supply a uniform
floor as those frames approach a singular orientation. This supports the
report's call for geometry stratification rather than a blanket K2 rank claim.

For K>=4 the twelve limiting monomials have degree at most seven in each
original coordinate and are independent on the nine-point square. Hermite
normalization is uniform also for arbitrary small node ratios. Four sites
with multiplicity two in the longitudinal channel give determinant `V^4`,
and four transverse values give `V`, proving the combined density factor
`V^-5`. At two contact centers the respective determinant powers are eta^16
and eta^4. With literal, unfactored derivative rows, the longitudinal constant
is 144; squaring the combined determinant gives `Theta(eta^40)` for the
covariance determinant. This confirms (18)-(19), including the one-scale
epsilon^-30 and the two-pair `r1^-5 r2^-5 eta^-20` prefactors.

The normalized inverse covariance is bounded; transformation entries are
`O(eta^-7)`. Consequently the conditional finite coefficient mean and each
transverse curvature mean are `O(eta^-7)` on compact contact targets. The
centered covariance stays bounded by identity, giving the stated crude
`E[A1² A2² | U=v] <= C eta^-28`. Multiplying by eta^-20 and the contact
action bound gives (20). The single-contact physical weight has exactly the
source factor `36 k_i² A_i² 1{A_i<0}`. No extra Hessian determinant or independent
contact assumption was inserted. Uniform interchange of finite-radius and
coalescing-center limits is deliberately not supplied.

## Replay and independent finite algebra

`NONAUTHOR_COALESCENCE_CHECKS.py` and its JSON retain my local checks. The author
program replay produced JSON equal to the supplied JSON, including all number
strings, using bundled Python 3.12.14. Independent Fraction checks verify the
ordinary and multiplicity-two Hermite determinants, the two-center constants
144 and 1, 39 polynomial instances of L1, and nine rational-frame rank cases.
Those finite cases support, rather than replace, the all-frame argument above.

Independent exact covariance algebra at the coincident normalized jets gives
these limit constants for the diagnostic law and target:

| Normalized diagnostic | Exact limit |
| --- | --- |
| `det Cov(G_K2)/V^4` | `129024/78125 = 1.6515072` |
| `det Cov(O_K4)/V^10` | `1550348800000/10460353203` |
| `det Cov(U_K4)/eta^40` | `396889292800000/129140163` |
| `eta^4 * contact action` | `28782/25 = 1151.28` |

For the last identity, the degree-five Hermite interpolant is
`F(s)=2s³-3eta^-1 s^4+(6/5)eta^-2 s^5`. In the repeated-node Newton coordinates,
eta² times the target tends to `(6/5)e5`; evaluating its quadratic form against
the exact limiting covariance gives the displayed rational action. The
reported decimals approach all four constants consistently.

Repeating the decimal computations at 130 rather than 95 significant digits
changes the supplied normalized values by relative differences between roughly
10^-90 and 10^-67. This is a useful precision crosscheck, not certified interval
arithmetic. The original Taylor sin/cos routine remains uncertified. No seed,
field sample, numerical H0 test, asymptotic window or error constant was created.

## Nonblocking wording corrections and remaining boundary

1. In §3 explicitly rebase `z=z1, x1=0`, or put the limiting jets at `z+x1 u`.
   Small diameter alone does not force the four numerical x_i to approach zero.
   Stationarity and uniformity already make this harmless to every bound.
2. State the divided-difference simplex convention once: either integrate the
   derivative against simplex volume of total mass `1/j!`, or average under a
   probability measure and multiply by `1/j!`. Do not apply both normalizations.
   The displayed limiting jets already have the correct factorials.

The assumptions essential to this review are a fixed finite planar Gaussian
law with strictly positive full-square weights, an embedded short torus chart,
four distinct ordered sites, the declared disjoint-pair ratio when eta is used,
canonical zero-target regression, and fixed positive lower cubic marks for
exponential suppression. Constants are not uniform over vanishing spectral
weights, increasing K, varying L or singular orientation families for the
twelve-row K2/K3 route. H0 selectors occur solely as Borel marks bounded by one.
The transverse collision tube, the small-k boundary, endpoint incidence
partition, increasing-domain estimates, Lean interfaces and global selected-bar
or process laws remain work to perform. A collinear set has zero full position
measure, so these accepted local estimates alone cannot resolve any of them.
