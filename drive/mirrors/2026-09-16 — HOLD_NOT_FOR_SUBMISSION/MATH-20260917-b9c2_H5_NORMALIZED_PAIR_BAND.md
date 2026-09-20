# Uniform normalized pair-Gram and pin-inverse extension

Task DQ-MATH-20260917-b9c2; 2026-09-17 UTC. Author-side continuation of
H5_ANALYTIC_ADVANCE.md. The same 3,307-byte reconnaissance memo is checked
by SHA256 29a117b8702e590e0a34005497bda19409c167c31d7d8ec291cb1b4452badb9a.
Shared scouting and source exposure are disclosed; no blind-review claim.

The finite-scope conclusion is stronger than the scalar c2 certificate:
the complete transformed pair-Gram matrix W(r) G12(r) W(r)^T is enclosed
uniformly for 0<=r<=0.05, including its continuous endpoint. The top-left
6x6 Hermite pin covariance A(r) is uniformly invertible. Its smallest
eigenvalue is greater than 0.1107638 throughout this interval. This does
not discharge the full 24-jet, spatial conditioning, or envelope obligations.

## Exact source binding

The canonical pin transform T is bound to H2 pin_transform.py SHA256
c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41.
When supplied that source, the executable symbolically compares all 36
entries of its exact rational transform against the source matrix_T(r).
No intake source is changed.

The raw vector consists of

    P6 = (f(M),fx(M),fy(M),f(S),fx(S),fy(S)),
    H_M = (fxx(M),fxy(M),fyy(M)),
    H_S = (fxx(S),fxy(S),fyy(S)),

where M=(-r/2,0), S=(r/2,0). Set W=diag(T,I6); the transformed vector
is (U,H_M,H_S), U=T P6. The exact T rows are:

    (1/2, r/8, 0, 1/2, -r/8, 0),
    (-3/(2r), -1/4, 0, 3/(2r), -1/4, 0),
    (0, 0, 1/2, 0, 0, 1/2),
    (0, -1/(2r), 0, 0, 1/(2r), 0),
    (0, 0, -1/r, 0, 0, 1/r),
    (2/r^3, 1/r^2, 0, -2/r^3, 1/r^2, 0).

The point formula has removable singularities in its covariance. The
method below removes them algebraically before interval evaluation.

## Exact cancellation and remainder proof

Each nonzero W entry is a rational monomial a*r^e with -3<=e<=1 or
an identity entry. Each raw covariance entry has the form

    s K1^(n1)(c*r) K1^(n2)(0),

where c is 0, -1 or 1; s=(-1)^|beta|; and n1,n2 are the corresponding
coordinatewise sums of derivative orders. Odd n2 entries vanish exactly.

For a product of two W entries with exponent sum e, expand K1^(n1)
through degree N=9-e. Multiplication by r^e then yields a polynomial
through total degree 9 plus a remainder of order r^10. The coefficient
of each term is an exact rational multiple of a product of exact-torus
even moments. Terms are collected by (r power, sorted moment-order pair).
The identity a0=1 is handled exactly. Every coefficient with a negative
r power must vanish as a rational number in every moment group; failure
raises CertificateFailure. There are 27 negative-power groups exercised
and cancelled in the present matrix.

No approximation a_(2m)=(2m-1)!! is used for these cancellations. Sorting
moment factors only uses commutativity; odd derivatives vanish by the
exact symmetry of the spectral law.

For one nonzero-c contribution a*r^e K1^(n1)(c*r) K1^(n2)(0), Taylor's
integral remainder and the already proved absolute spectral majorant give

    |error| <= |a| |c|^(N+1) L_(n1+N+1) a_n2
                         r^(e+N+1)/(N+1)!
             = C_term r^10.

The executable sums these positive bounds. For c=0 the covariance
derivative is already a constant and has zero Taylor remainder.
All needed moment bounds include both numerator and normalization tails.
The resulting coefficients and remainder constants for all 78 upper
triangular matrix entries are written to NORMALIZED_PAIR_BAND_RECEIPT.json.
Interval polynomial evaluation on [0,0.05] is therefore a uniform enclosure.

The endpoint U0 is

    (f(0), fx(0), fy(0), fxx(0)/2, fxy(0), fxxx(0)/6).

The top-left matrix at r=0 is also computed from these derivative jets
and compared to the cancelled polynomial limit as a separate check.

## Uniform inverse and eigenvalue floor

Let B be the rational symmetric matrix with nonzero entries

    B00=3/2, B03=B30=1, B33=2,
    B11=5/2, B15=B51=3, B55=6,
    B22=B44=1.

B is used as an approximate inverse. It happens to be the inverse of
the planar limiting covariance; this is only a choice of preconditioner,
not a substitution in the exact-torus covariance being certified.
Its infinity norm is exactly 9.

Uniform interval multiplication proves

    rho = sup_(0<=r<=0.05) ||I-B A(r)||_infinity
         < 0.003125196 < 1.

The Neumann series then proves invertibility of A(r) for every r in the
whole interval, and

    A(r)^(-1) = (I-E(r))^(-1) B,   E=I-B A,
    ||A(r)^(-1)-B||_infinity
       <= 9*rho/(1-rho) < 0.028214935.

This bound gives certified entrywise inverse enclosures, recorded in the
receipt. A(r) is a covariance matrix, hence positive semidefinite; its
proved invertibility makes it positive definite. Since its inverse is
symmetric, its spectral norm is at most its infinity norm, giving

    lambda_min(A(r)) >= 1/(9 + 9*rho/(1-rho)) > 0.1107638.

This is a uniform band result including r=0, not a collection of
point-rung inverse checks. It resolves this specific pair-pin conditioning
interface on the stated interval, subject to the supplied proof and
executable arithmetic. No broader program gate is automatically promoted.

## Validation and remaining transfer

The executable normalized_pair_band.py verifies all 144 matrix entries
against independent raw spectral covariance sums followed by explicit
T multiplication at r=0.0125, 0.025 and 0.05; exact zero entries use
overlap at zero because the raw cancellation may produce tiny interval
noise. Nonzero entries require direct-interval containment. All local
enclosures must lie within the whole-band enclosure. These finite checks
corroborate the formulas; the uniform proof is the Taylor remainder chain.

A deliberately perturbed cubic-row coefficient is rejected by the exact
Laurent cancellation guard. A zero approximate inverse is rejected by
the Neumann contraction guard. The source transform is compared
symbolically when --source is supplied. The 8 named checks and 432
individual positive-r entry comparisons passed.

Run, with the adjacent torus_jet_certificate.py and its dependencies:

    python normalized_pair_band.py --recon ../MATH-20260917-b9c2_EXTERNAL_RECON.md

Add --source /absolute/path/to/extracted/K3_SIDE24_LB/UPPER2D for the
canonical-transform comparison. The receipt binds both executable hashes.

Remaining work includes the missing explicit 24-jet enumeration and
scaling exponents, spatial y-dependent conditioning and scaled conditional
covariance estimates, propagation through the complete H5 operator and
geometry, the ZBAND upper bracket and remote threshold, and replay of
archived Taylor-model consumers affected by the odd-moment defect.
The all-r pair-pin result here supplies a concrete prerequisite; it is
not a claim of those closures. OBL-H5-JETMOD remains OPEN.
