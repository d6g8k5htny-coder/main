# A direct planar elder-failure lower bound and the sharp compact-window density loss

Object: OA-ELDER-LOWER-DENSITY-GAP-20260928-v1.
Author: OpenAI / ChatGPT, foreground research session, 28 September 2026.
Disposition: AUTHOR-SIDE PROOF CANDIDATE; NONAUTHOR ANALYTIC REVIEW REQUIRED.
Scientific effect: NONE. This does not change any historical certificate or governing status.

## 1. Exact elder probability and conclusions

Use the exact planar periodized Gaussian field and the original typed pair law
Q_r^W defined in LOCAL_MULTIPLICITY.md. The torus L is fixed, birth b is in a fixed
compact interval, k is in [k_-,k_+] with k_->0, and frames range over O(2). The six
pins are M=-(r/2)u, S=(r/2)u, f(M)=b, f(S)=b-k r^3, and both gradients zero.
W_r=F_2(H_M)F_1(H_S) and the FULL normalizer is Z_r=E_Qr W_r. This is not a law
conditioned on adjacency.

Let p_r(b,k,R) be the probability that the global ordinary superlevel elder death
partner of M is S. This is **exactly** the p_r definition in [LP, Section 1], not
a new selector. On the generic locus its death height is the maximin

    d_f(M)=sup_{gamma(0)=M, f(gamma(1))>f(M)} min_t f(gamma(t)), (E1)

with the empty supremum convention of [LP, Section 8]. A path to a point strictly
above the birth height, entirely above f(S), therefore prevents S from being M's
elder death partner. No gradient-trajectory or saddle-saddle transversality theorem
is required for this sufficient obstruction.

**Theorem E1 (direct lower bound).** Uniformly on the stated compact planar model,
there are c,r_*>0 such that

    1-p_r(b,k,R) >= c r^3, 0<r<=r_*.                      (E2)

The proof supplies a more specific subevent of this probability on which

    d_f(M) >= b-k r^3/4 > f(S),
    d_f(M)-f(S) >= 3k r^3/4.                              (E3)

It does not use a subtraction of competing witness-intensity bounds, numerical
rungs, an asymptotically fitted exponent, or an assumed independence of witnesses.
In particular a window-count lower bound alone would not prove (E2); the explicit
path below is essential.

**Corollary E2 (two-sided selection rate, with a separate upper input).** If the
precise Theorem A upper bound in [LP] is consumed, then in d=2

    c r^3 <= 1-p_r(b,k,R) <= C r^3.                       (E4)

The lower proof below is independent of that upper theorem. A defect in the upper
input would affect (E4)'s upper half, not the direct lower mechanism (E2).

## 2. A rational polygonal path with strict quantitative margins

Use the local scaled cubic

    G_k(X,Z)=k(2X^3-3X/2-1/2-3Z^2/4-XZ^2).

Consider the three-segment path

    (-1/2,0) -> (-3/4,0) -> (-3/4,9/4) -> (-1,9/4).       (E5)

Every segment lies in B(0,3): the largest endpoint squared norm is 97/16<9,
and the ball is convex. On the three segments, parameterized consecutively by
v in [0,1], the exact restricted polynomials divided by k are

    g_1(v)=-3v^2/16-v^3/32,
    g_2(v)=-7/32,
    g_3(v)=-7/32+51v/64-9v^2/32-v^3/32.                 (E6)

The first is nonincreasing, the second constant, and

    g_3'(v)=(51-36v-6v^2)/64 >= 9/64>0.

It follows that the minimum on the entire path is -7k/32, while its final value
is +17k/64. The path passes through a level region, but its proof does not rely on
that level remaining exactly constant under perturbation.

If a field F on B(0,3) obeys

    ||F-G_k||_infinity < k/64, F(-1/2,0)=0,

then the entire path has F-values above -15k/64>-k/4 and the endpoint has value
above 16k/64=k/4>0. Consequently, for the physical field with
F(X,Z)=(f(rX,rZ)-b)/r^3, the same path scaled by r connects M to a point with
value above b, entirely above b-k r^3/4. Definition (E1) proves (E3).

This is stable C^0 connectivity, not a statement that a prescribed path is a
separatrix. It remains a sufficient elder obstruction even without identifying
the exact saddle where the merger occurs. The existence of the two extra saddles
proved in the local multiplicity module is useful additional information, but is
not needed for the path argument.

## 3. Positive probability of the path under the actual tilted Gaussian law

For completeness, the ingredients from the local module can be isolated without
using its global upper bound or its extra-saddle implicit-function argument.
With s=f_zz(0)/r, a=f_xxz(0), beta=f_xzz(0), c=f_zzz(0), the exact six pins give

    (f(rX,rZ)-b)/r^3
      =2kX^3-3kX/2-k/2+(s/2)Z^2
          +(a/2)(X^2-1/4)Z+(beta/2)XZ^2+(c/6)Z^3
          +O_{C^2(B3)}(r||f||C4).                         (E7)

The term -aZ/8 retains both transverse gradient pins. The target values
s=-3k/2, a=0, beta=-2k, c=0 recover G_k. Choose a fixed small epsilon and the
physical jet box

    |f_zz+(3k/2)r|<epsilon k r,
    |f_xxz|<epsilon k, |f_xzz+2k|<epsilon k, |f_zzz|<epsilon k.

The endpoint contact observations plus these four physical jets are exactly the
ten independent planar jet coordinates of degree at most three. Positive Fourier
weights, continuity of the observation frame, and compactness give a uniformly
positive conditional density on these bounded targets. The box volume is
16epsilon^4 k^4 r, so its Q_r probability is at least c_1 r.

Condition on those **physical** jet values as well as the six pins before bounding
the C^4 norm. The joint covariance remains uniformly invertible and targets are
bounded, so full-field Gaussian regression yields a uniform conditional C^4 moment.
Choose a fixed finite K so the conditional probability of ||f||C4<=K is at least
1/2 at every jet target in the box. Integrating over the box leaves probability
at least (c_1/2)r. One may not replace this step by subtracting a fixed unconditional
tail probability from a rare event of size r.

Choose epsilon and then r_* small enough that (E7) is within k/64 of G_k in C^0
on B3 and sufficiently close in C^2 at the two pinned endpoints. Their scaled
Hessians at the cubic are diag(-6k,-k/2) and diag(6k,-5k/2), respectively. Their
indices persist and their determinant magnitudes retain positive lower bounds.
Since each physical Hessian is r times its scaled Hessian,

    W_r >= c_2 r^4

on this event. The full normalizer obeys 0<Z_r<=C_Z r^2 by the gradient-pin
averages and uniform Q_r C^3 moments (or [RM, (11)]). Thus its tilted probability
is at least

    (c_1 c_2/(2C_Z)) r*r^4/r^2 = c r^3.                  (E8)

Every field in this event satisfies the path obstruction (E3). This proves (E2).
The sufficient event also has the two local window saddles when the stronger
C^2 stability neighborhood from LOCAL_MULTIPLICITY.md is used. All constants are
existential but independent of r,b,k,frame on the stipulated compact ranges.

## 4. Sharp loss between candidate and elder lifetime densities

This section additionally consumes the source-bound weighted Kac-Rice and radial
ledger of [LP, Sections 9-11]. Let B and K have positive lengths, and let
nu_cand(ell), nu_eld(ell) be exactly its **compact birth/gap-window** candidate and
ordinary elder densities, per unit midpoint volume. The candidate intensity is

    r A_r(b,k,u) dr db dk d sigma(u),
    A_r=12 pi_r(v_r)(Z_r/r^2) -> A_0>0                   (E9)

uniformly on the compact parameters, where sigma is the ordinary surface measure
and pi_r is the density of the endpoint observation frame. Selected intensity has
the same factors times p_r. There is no factor 1/2 for the differently typed,
ordered maximum/saddle pair.

Changing variables ell=k r^3 at fixed k gives the exact density-gap version

    nu_cand(ell)-nu_eld(ell)
      =(ell^(-1/3)/3) integral_B integral_K integral_S
          k^(-2/3) A_{(ell/k)^(1/3)}(b,k,u)
          [1-p_{(ell/k)^(1/3)}(b,k,u)] d sigma dk db.      (E10)

For sufficiently small ell all these radii are below the common cutoff. The
uniform positive lower and finite upper bounds for A_r, and (E2)/(E4), imply

    c_{B,K} ell^(2/3)
       <= nu_cand(ell)-nu_eld(ell)
       <= C_{B,K} ell^(2/3).                             (E11)

The lower follows directly from (E2); the upper consumes the already identified
parent upper theorem. The k factor in its bounding integral is k^(-5/3), since
r^3=ell/k. It is finite and has positive integral on the fixed gap window.

Thus the parent's compact-window O(ell^(2/3)) **selection-density loss** is optimal
in order in the planar model; it cannot be replaced by o(ell^(2/3)) under these
assumptions. This does not prove sharpness of the different O(1) remainder in a
leading density expansion, and does not remove the compact-mark restriction.
Combining with nu_cand(ell)~c ell^(-1/3) gives a rejected-pair fraction of order ell.
Integrating (E11) from zero to epsilon gives a cumulative rejected-pair intensity
of order epsilon^(5/3), versus total compact-window intensity of order
 epsilon^(2/3). No leading numerical constant is determined by these inequalities.

## 5. Relation to the lower-rate campaign and review boundaries

The new direct event proves a uniform, all-small-r positive lower constant for
the exact typed pair law and ordinary elder selector defined in [LP]. Fixing any
birth and any positive gap mark in these ranges gives the corresponding one-mark
specialization. This is an existential analytic lower theorem, not a collection
of finite-r numerical checks. It does not rely on any disputed window-saddle
Cauchy-Schwarz estimate or on its numerical budget subtraction.

A historical artifact using the symbol q(r,b) is not automatically identified
with this p_r: its covariance, scaling convention, pin meanings, tilt and selector
must match. No historical numerical c,r_0, finite-band certificate, lower-campaign
registry bit, or prize is advanced here. In particular an adjacency-conditioned
law is a different law and is excluded from the statement.

The ordinary elder obstruction itself uses only the maximin definition and the
strict polygonal path, not the SARD-G no-saddle-connection theorem. The local
Gaussian jet-density and extra-conditioned C^4 control are the core new analytic
obligations. The parent upper theorem and weighted density identity are separately
named dependencies for (E4) and (E11). This separation permits a reviewer to accept
or reject the direct lower mechanism without inheriting a broader parent verdict.
