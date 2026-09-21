# H5 analytic advance and source counterexample

Task: DQ-MATH-20260917-b9c2. Date: 2026-09-17 UTC.

Author-side result: a cancellation-free, exact-torus certificate for the scalar
jet c2 on the whole interval 0 <= r <= 0.05, six reusable demonstration stencils,
and a reproduced defect in the archived Taylor remainder majorant.
OBL-H5-JETMOD remains OPEN. No theorem status, premise, or frozen evidence is changed.

## Reconnaissance and exposure

The controlling fresh reconnaissance memo is
MATH-20260917-b9c2_EXTERNAL_RECON.md, 3,307 bytes, SHA256
29a117b8702e590e0a34005497bda19409c167c31d7d8ec291cb1b4452badb9a,
Drive ID 1HMwqF8Z0k8JOAjg7MPpvuflOm4jCr-cz. The executable checks its hash,
size, and detached custody receipt before computation. Shared scouting and
source intake were used; this is not a blind review or an independent-provider
assessment. The cancellation-removal approach is a standard divided-difference
technique, not a novelty claim. The primary source inspected by the shared
scouting was de Boor, *Divided Differences*, Eq. (52):
https://arxiv.org/pdf/math/0502036.

## 1. Exact law and certified moments

Let h=pi/12, k_j=jh, w_j=exp(-k_j^2/2),

    Z = sum_{j in Z} w_j,
    K1(s) = (1/Z) sum_{j in Z} w_j cos(k_j s),
    a_(2m) = (1/Z) sum_j w_j k_j^(2m),
    L_n = (1/Z) sum_j w_j |k_j|^n.

The 2D side-24 covariance is K1(x1-y1) K1(x2-y2). Thus K1(0)=1
exactly and c2(r)=(1-K1(r))/r^2, continuously extended by c2(0)=a2/2.
The derivative majorant is |K1^(n)(s)| <= L_n for every real s.
For even n, L_n=a_n; for odd n, the signed spectral moment is zero and
is not a derivative majorant.

The code retains j=-116,...,116. Put j0=117. For any integer n>=0,

    rho_n = exp(-(2*j0+1)*h^2/2) (1+1/j0)^n,
    T_n = 2 exp(-(j0*h)^2/2) (j0*h)^n / (1-rho_n).

When rho_n<1, the omitted positive-and-negative absolute numerator is
at most T_n: the ratio between successive positive summands is decreasing
for j>=j0 and bounded by rho_n. Every use checks rho_n<1 with interval
arithmetic. If S_n is the retained nonnegative absolute numerator,

    Z in [S_0, S_0+T_0],
    L_n in [S_n, S_n+T_n] / [S_0, S_0+T_0]  (n>0).

All sums, pi, exponentials, and endpoint operations use mpmath.iv at 160
decimal digits. The n=0 identity L_0=1 is used exactly. Both numerator
and normalization-denominator tails are included. The kernel evaluation
used for direct comparisons encloses its signed omitted numerator by
[-T_n,T_n] and divides by the same full denominator enclosure.

The certificate resolves the nonzero torus correction:

    1-a2 approximately 9.6525417989599130550e-123 > 0.

It also proves a4>3. Consequently planar substitutions a2=1 or a4=3 are
rejected by explicit controls, even though they coincide at ordinary display
precision. The JSON's exact dyadic endpoints are authoritative; printed
decimal values here are rounded summaries.

## 2. Uniform c2 theorem

For every real t, the cosine Taylor polynomial of even degree has an error
with alternating sign and magnitude at most the first omitted term.
One proof of the sign starts with cos(t)-1<=0 and repeatedly integrates
the relation R_m''=-R_(m-1), using R_m(0)=R_m'(0)=0. The magnitude also
follows from Taylor's theorem and |cos^(n)(t)|<=1.

Averaging against the positive spectral measure and dividing by r^2 gives,
for every r>=0 including the continuous endpoint,

    0 <= c2(r) - [a2/2 - a4*r^2/24] <= a6*r^4/720.

Therefore on 0<=r<=0.05 the quadratic residual is less than 1.303e-7.
On the prototype's first band [0.025/sqrt(2),0.025], it is less than
8.139e-9. These are analytic uniform inequalities with certified constants,
not inferences from sampled rungs.

More strongly, write

    P8(r) = a2/2 - a4*r^2/24 + a6*r^4/720
            - a8*r^6/40320 + a10*r^8/3628800.

Then for every r>=0,

    -a12*r^10/12! <= c2(r)-P8(r) <= 0.

On 0<=r<=0.05 the magnitude is less than 2.120e-18. Every coefficient
is an interval enclosing the exact finite-torus moment. This is the main
finite-scope advance over direct subtraction of two nearly equal kernel
intervals.

## 3. Band variation and the prototype's width comparison

The identity

    c2(r) = E[k^2 integral_0^1 (1-t) cos(k*r*t) dt]

and |sin(x)-x|<=|x|^3/6 imply

    c2'(r)/r <= -a4/12 + a6*r^2/180.

The certificate proves the right side is negative on 0<r<=0.05.
Thus the first-band range lies between the certified values at its two
endpoints. Its enclosing width is approximately 3.90563970406e-5;
its endpoint values are approximately 0.49992188313738509 and
0.49996093953442574.

This exposes a units-of-comparison issue in the prototype: a whole-band
range width cannot generally be required to fall below the structural
halfwidth about a band center. Here the true variation already exceeds
the cited approximately 1.95e-5 halfwidth. The new results distinguish
the actual variation across r from the much smaller residual of a
polynomial tube. The previous approximately 0.025722 slab hull is
approximately 659 times wider than the new whole-band hull.

## 4. Generic cancellation-free jet stencils

For rational weights w_i, rational offsets c_i, derivative order d>=0,
and integer p>=0, consider

    F(r) = r^(-p) sum_i w_i K1^(d)(c_i*r).

Let mu_j=sum_i w_i*c_i^j. Before evaluating, the code requires
mu_j*K1^(d+j)(0)=0 for every j<p. It proves these cancellations using
exact rational arithmetic and the exact vanishing of odd derivatives
at zero. In particular, a small numerical residual is never treated
as an exact cancellation.

For N>=p, Taylor's integral remainder gives

    F(r) = sum_{j=p}^N mu_j K1^(d+j)(0) r^(j-p)/j! + E(r),
    |E(r)| <= L_(d+N+1) [sum_i |w_i| |c_i|^(N+1)]
                         r^(N+1-p)/(N+1)!.

Here K1^(2m)(0)=(-1)^m a_(2m). The formula extends continuously to r=0
and is evaluated as a polynomial without negative powers or cancellation.
The executable certifies six demonstration stencils on [0,0.05]: c2,
negative centered second difference, centered fourth and sixth differences,
K1'(r)/r, and [K1''(r)-K1''(0)]/r^2. These six examples are not asserted
to be the program's still-unenumerated full 24-jet set.

## 5. Reproduced archived-source counterexample

Source: archive 09152026OKComputer_Project_Gap_Closure.zip,
Drive ID 1vSI-evINWskhXVyiZ0slt-rLT74sPpXH,
SHA256 a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b.

Hash-bound inputs:

    h5_kernel.py
    b2e6014c2fd6eb9e72b2104cc828a764e280f76b8396a02edd37654b6ad83ad7
    cov_exact.py
    f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783
    pin_transform.py
    c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41

Lattice.lambda_q(q) in the archived H5 kernel sums w_j*k_j^q over
positive and negative j. For odd q these retained terms cancel. The
function therefore returns an interval near zero instead of L_q.
At the archived default 100-digit precision, two certified witnesses are:

    q=1: old upper approximately 5.76e-101;
         |K1'(0.05)| approximately 0.0499375390462.
    q=7: old upper approximately 1.21e-98;
         |K1^(7)(0.05)| approximately 5.23033954938.

The defect reaches an actual archived consumer, rather than remaining an
unused helper defect. KernelSeries.tm_factor(n=1,arg='dxM',axis=0),
with r=0.05, yc=(0.025,0), eta1=eta2=0.001 and TD=5, requests q=7.
At the positive edge delta=0.001, direct full-law interval evaluation proves

    K1'(0.051) - P5(0.001) approximately 7.28495881102e-21,

while the claimed archived remainder upper endpoint is approximately
1.68559496223e-119. The exact endpoint interval and claimed Taylor
enclosure are disjoint. This is a falsifying witness of the archived
remainder contract.

Using the positive absolute moment L_7 yields a valid remainder about
5.31923041621e-20 and encloses the endpoint. The reusable repair is
L_n, including full normalization and truncation tails, as implemented
by Torus.absolute_moment. The intake sources and their receipts remain
unchanged. This finding requires a dependency review and new replay of
affected consumers before old Taylor-model certificates can be relied
upon. It does not, by itself, prove the underlying mathematical theorem
false or establish the exact set of affected downstream packages.

## 6. Verification and limits

The executable passes 33 recorded checks: exact-torus substitution
controls; uniform numerical constants; direct spectral comparisons at
seven rational rungs; r=0 extension; a deliberately false residual cap;
an uncancelled-stencil mutation; and three direct comparisons for each
of the six generic stencils. The optional hash-bound source audit additionally
checks two derivative-majorant witnesses and the actual Taylor-model
counterexample plus repaired containment. Samples corroborate the
implementation; the uniform conclusion comes from the proved remainder
inequalities and interval moment bounds, not from those samples.

Run with Python 3, mpmath==1.3.0 and sympy==1.14.0:

    python torus_jet_certificate.py --recon ../MATH-20260917-b9c2_EXTERNAL_RECON.md

To replay the archived-source counterexample, add
--source /absolute/path/to/extracted/K3_SIDE24_LB/UPPER2D.
The code reads the source but does not change it or write bytecode there.
The exact receipt embeds the verifier's SHA256 and exact dyadic interval
endpoints. Rounded decimal strings are display-only.

Unclosed interfaces: the full 24-jet enumeration and powers p_J; interval-r
PinFrame inversion and conditional covariance transfer; complete band
operator/geometry evaluation; ZBAND upper bracket; REMOTE-THRESHOLD and
RN-UNIF; the infinite-band/all-small-r assembly beyond this scalar
analytic endpoint theorem; and propagation/replay of the source defect.
No claim is made that the present certificate discharges any of them.
