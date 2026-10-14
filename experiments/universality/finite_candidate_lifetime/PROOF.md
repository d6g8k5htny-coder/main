# Finite Gaussian candidate-pair lifetime theorem — research preparation

Prepared on 8 October 2026 by OpenAI/Codex `/root/universality_review`,
in task `01a11cbf-068d-7102-861e-75814e715c98`, at the coordinator's request;
incorporated by `/root` in a separate source proposal after merged PR319.
This is a conventional mathematical argument about candidate pairs. It is not
a scientific-status register, a formal Lean proof, a numerical confirmation,
or a promotion of any catalogue model. No compiler state, sample or seed was
changed or consumed in preparing or incorporating this argument.

The preparation author was the source-exposed nonauthor reviewer and a
mathematical-input contributor to the analytic source below. Its author was
`/root`. `/root/benchmark_formal_audit` independently attacked the candidate
counting route, primary Kac–Rice hypotheses, genericity, weighted-type
continuity and near/far ledger before this file was written. That contributor
also authored the catalogue implementation. These roles are not blind or
organizationally independent; organizational-independence credit is zero.
Neither that route discussion nor the preparation author's own checks are
an exact-hash nonauthor review of this file. The [review record](REVIEW.md)
binds the separate saved-preparation and incorporated-source reviews.

## 1. Frozen sources and precisely named external imports

The principal consumed analytic source is
[finite Gaussian contact proof](../finite_gaussian_contact/PROOF.md): **25,367 UTF-8 bytes**, SHA-256
**`7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`**.
Its full content was read and its byte identity independently reproduced.
Call this source **FG**. References to its sections below are to these exact
bytes; this preparation does not amend FG.

The implementation identities read with FG are models.py, 12,385 bytes,
SHA-256 `7ba088158b5e97ba6e7d715e66e6f9d14268ac3a82a6d187a776821936cc6686`,
and models.json, 12,914 bytes,
SHA-256 `10d334b27cac985d61506b4f26118699cd1689a82a0f6e454cef8d00df21e595`.
They identify the intended spectral formulas and distinguish the ideal law
from the retained floating profile and finite-word sampler. The mathematical
law in Section 2, rather than a PRNG implementation, is the input here.

FG consumes parent **P**, [UNIFORM_MATRIX_CAP_AND_LIFETIME.md at Math-
13af1089fd7991105a3d8828539bdd9a41ca87c3](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md):
40,261 bytes, blob `dfed3b8d318a3ab1950957f393307733a4bef3f2`, SHA-256
`9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`.
The complete P source was read and its local byte identity checked.
Read its normalizer argument with [E1](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md):
1,782 bytes, blob `213594d6ca6a86fb938110f4d166d9ce275a02d0`, SHA-256
`bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028`.
E1 was completely read and its bytes checked. The [D1 reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md)
supplies the complete parent reading rule, including E2 and W1/embedding.

The independent conventional Kac–Rice route was checked against:

* **AAL:** Diego Armentano, Jean-Marc Azaïs and José Rafael León,
  [*On a general Kac-Rice formula for the measure of a level set*,
  arXiv:2304.07424v3, 5 December 2023](https://arxiv.org/html/2304.07424v3).
  Proposition 2.1, Theorems 2.1, 2.2 and 7.1, Remarks 7–8 and the equal-dimension
  area-formula proof in Section 5 were read in the primary paper.
* **E2:** [REPAIR.md at the same Math- commit](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/reviews/d1_section9_borel_repair_20260925/REPAIR.md),
  blob `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a`, source-record identity
  9,062 bytes/SHA-256
  `845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f`.
  Its complete fetched text was read. Sections 9.1–9.3 give the two-finite-measure
  extension from continuous cylinders to Borel marks. Its elder-mark and
  cap-selection conclusions are not imported here.

Two classical external theorem imports are stated explicitly, not silently
replaced by a numerical test or an all-jet Gaussian assumption:

**Sard, equal dimensions.** For a C-infinity map from an open subset of R^N
to R^N, the set of images of points where its derivative has rank less than N
has N-dimensional Lebesgue measure zero. Use countable chart/localization
covers when the domain is not one bounded Euclidean region. The primary
reference is Arthur Sard, [*The measure of the critical values of differentiable
maps*, Bulletin AMS 48 (1942), 883–890](https://doi.org/10.1090/S0002-9904-1942-07811-6).
The original paper's statement was read through the [University of Edinburgh
hosted primary-paper copy](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/sard.pdf).

**Weighted area formula, equal dimensions.** If Phi is locally Lipschitz from
an open subset U of R^N to R^N, and h is nonnegative Borel, then

    integral_U h(z) |det D Phi(z)| dz
      = integral_R^N sum_{z in U : Phi(z)=xi} h(z) dxi.       (AF)

The equality is an extended nonnegative integral; no prior finiteness of the
root count is assumed. Localize to countably many bounded subsets if needed.
This is the equal-dimension weighted area formula of Herbert Federer,
*Geometric Measure Theory*, Section 3.2.3, originally 1969; [publisher edition
and identity](https://link.springer.com/book/10.1007/978-3-642-62010-2).
AAL Section 5 independently exhibits the area-formula counting mechanism.
The theorem (AF) is a named classical import; the incidence-map application
and its determinant calculation are proved below.

Read-status precision: the publisher's Federer metadata was inspected, not
the complete licensed book or a new proof of (AF). A single attempt to open
the AMS-hosted Sard PDF returned `Internal Error`; it was not retried. The
primary-paper copy above supplied the displayed classical statement. There
is no claim of a new verification of the entire classical literature.

## 2. Exact field, observable and theorem

Fix an integer K>=2, L>0 and omega=2 pi/L. Let

    S_K={-K,...,K}^2,
    q_n>0, q_n=q_-n, sum_{n in S_K} q_n=1.

Choose one member of each nonzero pair {n,-n}. With independent standard real
Gaussians C_0,C_n,S_n, put

    F(x)=sqrt(q_0) C_0
       + sum_pairs sqrt(2q_n)
           [C_n cos(omega n.x)+S_n sin(omega n.x)].          (1)

The torus is X=R^2/(L Z^2), with area V=L^2. The real coefficient vector xi
has dimension N=(2K+1)^2 and standard Gaussian Lebesgue density rho_N.
The field is centered, variance one, stationary and C-infinity. It need not
be rotationally isotropic. Every fixed C^j norm is bounded by C_j ||xi||.

For I a Borel subset of (0,infinity), define

    N_cand(I)=sum_{p!=q, gradF(p)=gradF(q)=0}
        1{H_p<0} 1{det H_q<0} 1{F(p)-F(q) in I}.           (2)

Here H is the physical two-dimensional Hessian. The first role is a local
maximum; the second is a nondegenerate index-one saddle. Both roles are
ordered. This population includes every such positive-gap pair, without
requiring a persistence pairing or excluding a maximum because it supports
the essential class. Set mu_cand(I)=E N_cand(I)/V.

**Candidate theorem.** For every fixed law (1), mu_cand has a finite continuous
density nu_cand(ell) on ell>0, and

    nu_cand(ell) ~ C_loc ell^(-1/3), ell decreasing to zero,
    0<C_loc<infinity.                                     (3)

The coefficient is

    C_loc=Gamma(7/6)/(24^(1/3) sqrt(pi))
      * integral_S^1 p_G(0) p_Vu(0) tau_u^(4/3)
                              [s_u^2/2] d sigma(u),       (4)

where G=(F_u,F_v), V_u=(F_uu,F_uv), T_u=F_uuu, A_u=F_vv,

    tau_u^2=Var(T_u | G=0), s_u^2=Var(A_u | V_u=0).

All derivatives and covariances are from (1); v is either perpendicular unit
direction. The functional is unchanged by reversing v. d sigma is ordinary
unnormalized arclength on S^1, with mass 2 pi. The 24 in (4) is a numerical
Jacobian/gamma factor, not the torus side. Expected whole-torus counts have
coefficient V C_loc; at side24, V=576.

The cumulative consequence is separate:

    E N_cand((0,t])/V ~ (3/2) C_loc t^(2/3).               (5)

The theorem does not claim the stronger remainder
nu_cand=C_loc ell^(-1/3)+O(1). It does not count actual persistence bars.

## 3. The finite list of analytic premises actually proved in FG

The following facts hold for every fixed K>=2,L,q above. They are the named
inputs consumed here, with short mechanisms recorded to make their scope
auditable:

**R1 — seven contact rows (FG Sections 2–3).**
(F,F_u,F_uu,F_uuu,F_v,F_uv,F_vv) has positive unconditional covariance in every
orthonormal frame. Its Fourier multiplier is a polynomial of degree at most
three in each original frequency coordinate. If its variance is zero, positivity
of every q_n makes that polynomial zero on the full square grid. At least five
points in each coordinate force polynomial identity, and its distinct
homogeneous terms force every coefficient to vanish. Frame compactness gives
a contact covariance floor. Literal divided-difference coefficient matrices
extend continuously to contact, so that floor persists for every sufficiently
small r, uniformly over basepoints and frames.

**R2 — original two-site first jets (FG Section 7).**
For every distinct torus pair,

    O_t=(F(p),gradF(p),F(q),gradF(q))

has positive covariance. Choose a coordinate in which the torus sites differ,
and let its relative phase be t!=1. Holding the other frequency fixed, a
zero-variance combination has form (a+bj)+(c+dj)t^j. Four consecutive j values
give the invertible confluent Vandermonde determinant t(t-1)^4. Varying the
other frequency then kills the remaining value and transverse-gradient
coefficients. K>=2 suffices. On any fixed compact off-diagonal domain this
six-row covariance and its four-gradient submatrix have uniform floors.
The two endpoint heights conditional on the four gradients have positive
two-dimensional covariance by the Schur complement.

**R3 — actual conditional law and moments (FG Section 4).**
For any full-row-rank observation matrix R, conditioning R xi=z gives mean
R^T(RR^T)^(-1)z and centered covariance
P=I-R^T(RR^T)^(-1)R. P is an orthogonal projection. Uniform observation
covariance floors give bounded compact-target means and polynomial growth
in arbitrary targets. Finite coefficient expansion gives every fixed-order
field moment. Further scalar regression supplies the independent transverse
residual used in FG; no arbitrary residual jet rank is asserted.

**R4 — literal contact normalizer (FG Section 5 with E1).**
For M=x-r u/2,S=x+r u/2, exact pins

    F(M)=b, F(S)=b-k r^3, gradF(M)=gradF(S)=0

give U_r target v_r=(b-k r^3/2,-k r^2,0,12k,0,0). The original six-pin
observation transform has absolute determinant 12 r^-5. With

    W_r=|det H_M det H_S| 1{H_M<0} 1{det H_S<0},
    Z_r=E_Q W_r,

one has, uniformly on compact b windows and 0<k_-<=k<=k_+,

    Z_r/r^2 -> z_0=36 k^2 E[A_0^2 1{A_0<0}]>0,            (6)

where A_0 is F_vv conditioned on U_0=(b,0,0,12k,0,0).
Endpoint gradient averages give |F_uu(i)|/r,|F_uv(i)|/r<=M_3/2, while the
height pins give F_uu(M)/r -> -6k and F_uu(S)/r -> 6k. The correct congruence
diag(r^-1/2,1) has off-diagonal sqrt(r) beta. The transverse Schur variance,
Gaussian density bound and uniform moments give type convergence and uniform
integrability. This is compact-mark convergence, not a uniform inverse-Z
bound over all marks.

**R5 — all-mark unnormalized envelope (FG Section 6).**
On one small band 0<r<=r_0, independent of b,k, define

    A_r(b,k,u)=12 pi_r(v_r) Z_r/r^2.

Then

    A_r<=H(b,k)=C(1+|b|+k)^4 exp[-c(b^2+k^2)],
    integral_Rx(0,infinity) k^(-2/3) H(b,k) db dk<infinity. (7)

This follows from target coercivity, the conditional coefficient mean
O(|b|+k), centered covariance <=I, and the gradient-average determinant bound
|det H_i|/r<=C(1+||F||_C3)^2. No division by Z_r or k is involved.

**R6 — off-diagonal bound (FG Section 7).**
For torus distance at least fixed r_0>0, original six-pin density at
(b,0,b-ell,0), 0<ell<=1, has Gaussian decay in b; conditional determinant
moments have at most degree-four polynomial target growth. The integral
over b and the finite spatial domain is bounded uniformly in ell.

**R7 — parity functional (FG Section 8).** Odd and even derivative blocks
are jointly independent, while correlations within each block are retained.
R1 gives tau_u^2>0 and s_u^2>0. The centered scalar law A_u|V_u=0 has
E[A_u^2 1{A_u<0}|V_u=0]=s_u^2/2. These factors are continuous in the frame.

No part of this proof consumes positive covariance of both full two-site
second jets. That stronger fact holds at K>=3 and fails in an axial K2
example. Neither it nor an arbitrary finite-jet premise is needed here.

## 4. Sard genericity before any Rice argument

This step avoids a Morse-to-Rice-to-Morse cycle. At a physical point p, write

    gradF(p)=B(p) xi.

B(p) has row rank two: positive coordinate-mode weights give positive
gradient variance in every nonzero direction. Cover the torus by coordinate
patches on which some two coefficient columns B_C(p) are invertible. A
finite or countable such cover exists by continuity and the finite set of
possible minors. In each patch split xi=(zeta,eta), with zeta in R^2.
The critical-point equation is equivalent to

    zeta=a(p,eta)=-B_C(p)^(-1) B_D(p) eta.

The smooth map

    Psi(p,eta)=(a(p,eta),eta)

has N-dimensional domain and target. Differentiating the constraint at
xi=Psi(p,eta), while taking the physical derivative of gradF at fixed xi,
gives

    D_p a=-B_C(p)^(-1) H_F(p),
    |det D Psi|=|det H_F(p)|/|det B_C(p)|.                 (8)

If a field has a degenerate critical point in this patch, its coefficient
vector is a critical value of Psi. Sard makes those vectors Lebesgue-null.
A countable patch union is still null, and the coefficient law has Lebesgue
density rho_N. Thus F is Morse almost surely, before any counting formula.

The critical set is closed in the compact torus. Nondegenerate critical
points are isolated; an infinite critical set would accumulate at a critical
point and contradict local uniqueness. Hence the set is finite.

The Morse coefficient set is open. Near a Morse coefficient vector, finitely
many implicit-function critical branches persist; outside their chosen
neighborhoods the gradient stays bounded away from zero. Countable local
coefficient neighborhoods therefore make the resulting finite root counts,
with arbitrary Borel location/coefficient marks, Borel. On the null non-Morse
set define a count as zero. No claim about every fixed conditional pin law
has entered this argument.

## 5. Direct coefficient-space area formula for all Borel pair marks

Let T=(XxX) minus the torus diagonal and t=(p,q). Define

    G_t=(gradF(p),gradF(q))=R(t) xi.

R2 implies row rank four on T. On a coordinate patch U in T choose an
invertible four-column minor R_C(t). Split xi=(zeta,eta), zeta in R^4.
Then G_t=0 is equivalent to

    zeta=a(t,eta)=-R_C(t)^(-1) R_D(t) eta,
    Phi(t,eta)=(a(t,eta),eta).

Phi is smooth from UxR^(N-4) to R^N. Its preimages of a fixed coefficient
vector correspond exactly to that field's pair-gradient zeros in U.
At constraints,

    D_t a=-R_C(t)^(-1) D_tG,
    D_tG=diag(H_p,H_q),
    |det D Phi|=Delta_t/|det R_C(t)|,
    Delta_t=|det H_p det H_q|.                             (9)

Apply (AF) with rho_N(Phi(t,eta)) times any nonnegative Borel mark w(t,xi)
and a Borel restriction to a chosen subset E of U. Initially allow both sides
to be infinite. It gives

    E sum_{t in E:G_t=0} w(t,xi)
      = integral_E integral_R^(N-4)
          rho_N(a(t,eta),eta) w(t,(a(t,eta),eta))
                Delta_t / |det R_C(t)| d eta dt.           (10)

Localize in t and eta to apply the locally Lipschitz theorem, then use
nonnegative monotone convergence. A countable Borel partition of T subordinate
to the minor patches avoids counting a pair twice.

For fixed t, the linear transformation (zeta,eta) -> (G_t,eta) has absolute
Jacobian |det R_C(t)|. Thus

    integral rho_N(a(t,eta),eta) f(a(t,eta),eta)
                                      / |det R_C(t)| d eta
       =p_Gt(0) E[f(xi)|G_t=0],                            (11)

using the canonical Gaussian regression law. Substituting f=Delta_t w and
combining the patches yields the all-Borel pair Rice identity

    E sum_{t in D:G_t=0} w(t,xi)
      =integral_D p_Gt(0) E[Delta_t w(t,xi)|G_t=0] dt.     (12)

The root-count formula was not assumed to establish genericity or finiteness.
On a compact D separated from the diagonal, the four-gradient covariance
floor bounds p_Gt(0), and conditional centered coefficients have covariance
<=I and mean zero. Delta_t is bounded by C||xi||^4. Consequently (12) with
w=1 is finite on D. This derives the local finiteness needed by a separate
E2 two-finite-measure proof; it is not a hidden premise of (10).

Sard applied also to Phi shows regular pair roots almost surely directly.
The single-point construction in Section 4 already establishes that result
via a Morse field, so neither direction relies on a Rice/genericity cycle.

The typed mark is

    tau_t=1{H_p<0} 1{det H_q<0},
    W_t=Delta_t tau_t.

It is Borel, and its type sets are open. In fact W_t, unlike the bare
indicator, is continuous in both Hessians: changing type at a boundary
requires a zero determinant, which kills the jump. It has degree-four
polynomial growth. Equation (12) contains Delta exactly once, not an
additional determinant inserted into the mark.

## 6. Distinct critical values and the independently checked AAL/E2 route

R2 makes the two endpoint heights conditional on G_t=0 nondegenerate Gaussian.
Thus the conditional probability of F(p)=F(q) is zero for each distinct t.
Since Delta_t is finite and integrable under that conditional law, (12) with
w=1{F(p)=F(q)} gives expected tie count zero on every compact off-diagonal
domain. Countably exhaust T, for example by torus separation at least 1/j
and finitely many coordinate domains. No two distinct critical points have
equal heights almost surely. This proof uses two-site first jets only.

For comparison, actual AAL hypotheses hold: G is C-infinity and Gaussian;
Cov(G_t) is positive definite; its density has a compact-domain bound; and

    law(xi|G_t=z)=law(R(t)^T(R(t)R(t)^T)^(-1)z+P_t eta),
    P_t=I-R(t)^T(R(t)R(t)^T)^(-1)R(t)

is continuous in (t,z) through the whole C^j field. These verify the Gaussian
base and conditional requirements. The continuous weighted base can use xi
as a constant extra Gaussian field. Its bounded continuous cylinders generate
Borel(DxR^N); equality of the two finite counting/kernel measures then extends
to the typed and height marks by E2's monotone-class mechanism.
This is an independently checked alternative to Section 5, not an assertion
that AAL's lower-semicontinuity condition covers every arbitrary Borel mark.
E2's global elder event is not used.

## 7. Height disintegration and an actual density version

For t=(p,q), let Q^t_(b,s) be continuous Gaussian regression on the six
observations O_t=(F(p),gradF(p),F(q),gradF(q)) at (b,0,s,0). Define

    K_t(b,s)=p_Ot(b,0,s,0) E_Q^t_(b,s) W_t.               (13)

The order of observations may be permuted, and physical gradients rotated
orthogonally, with absolute Jacobian one. The six-row rank makes the Gaussian
height disintegration legitimate. Equation (12), (11) and nonnegative
Tonelli give the expected typed-pair height measure with kernel K_t(b,s).
Push it forward under ell=b-s; ds has absolute Jacobian one. Therefore

    nu_cand(ell)=(1/V) integral_T integral_R
                        K_(p,q)(b,b-ell) db dp dq, ell>0, (14)

and mu_cand(I)=integral_I nu_cand(ell) d ell for every positive-lifetime
Borel I. Initially this is a nonnegative density version, possibly infinite;
the next sections prove finiteness, continuity and its asymptotic directly.
No differentiation of a cumulative asymptotic is used.

For distinct sites, canonical conditional coefficient means and covariance
square roots depend continuously on locations and heights. The continuous
weighted type W and degree-four growth, with finite Gaussian moments, imply
continuity of its conditional expectation. This argument allows singular
conditional Hessian covariance; it does not need full two-site second-jet rank.

## 8. Near-diagonal ledger and dominated convergence

Choose the R1/R5 band 0<r<=r_0 below one and the injectivity scale. In that
band write p=x-r u/2, q=x+r u/2. The midpoint/separation transformation has
absolute Jacobian one, and dh=r dr d sigma(u). Stationarity removes the
midpoint integral and cancels V; it does not remove directional dependence.

Set s=b-k r^3, with k>0. The height change has absolute Jacobian r^3.
The literal P contact transform and R4 give original-pin density
12 r^-5 pi_r(v_r), v_r=(b-k r^3/2,-k r^2,0,12k,0,0).
The determinant expectation is Z_r. The factors are thus

    (r dr d sigma) (r^3 db dk) (12 r^-5 pi_r) Z_r
      =r A_r dr db dk d sigma,
    A_r=12 pi_r Z_r/r^2.                                  (15)

Endpoint roles are ordered. u ranges once over the full circle; no factor
1/2 or normalized-angle convention is introduced.

At fixed k the pushforward ell=k r^3 has
dr/dell=1/(3k r^2). Formula (15), or equivalently the fixed-ell change
r=(ell/k)^(1/3), gives

    ell^(1/3) nu_near(ell)
      =integral_Rx(0,infinity)xS^1
         1{k>=ell/r_0^3} A_(ell/k)^(1/3)(b,k,u)
                                  /(3k^(2/3)) db dk d sigma. (16)

For each fixed b,k>0,u, R4 gives A_r -> 12 pi_0(v_0) z_0, and the indicator
in (16) tends to one. R5 supplies the integrable majorant H/(3k^(2/3)).
Dominated convergence gives

    ell^(1/3) nu_near(ell) ->
      4 integral k^(-2/3) pi_0(v_0) z_0 db dk d sigma
      =C_loc.                                             (17)

This conclusion handles all birth heights and all positive k; its proof uses
the unnormalized product pi_r Z_r. There is no global inverse-Z estimate,
no division by k in a conditional probability, and no selection probability.

Equation (16) bounds nu_near(ell) by C ell^-1/3 for every ell>0. It also proves
continuity for each positive ell: for ell_j -> ell>0, the integrand converges
at every fixed b,k,u except k=ell/r_0^3, a measure-zero boundary. Conditional
kernel continuity from Section 7 and the same H majorant permit dominated
convergence. This yields a particular continuous Radon–Nikodym density.

## 9. Far contribution, finiteness and the full theorem

On dist(p,q)>=r_0 the six-pin covariance has a compact-domain floor.
For 0<ell<=1, at target (b,0,b-ell,0), R3 bounds the determinant expectation
by C(1+|b|)^4, and the Gaussian pin density by C exp(-c b^2).
Integrating b and the finite spatial domain in (14) proves

    0<=nu_far(ell)<=C, 0<ell<=1.                          (18)

For any fixed compact lifetime interval in (0,infinity), the same argument
gives a uniform Gaussian-in-b bound and continuity by dominated convergence.
Together with the near result, nu_cand is finite and continuous for ell>0.

There is also no hidden infinite-total-count premise. For the near population,
integrating (15) over all positive gaps and all b,k is bounded by
integral_0^r0 r dr times integral H db dk d sigma, which is finite.
For the far population, the target (b,s) has Gaussian decay in both heights,
and its conditional determinant moment has degree-four polynomial growth;
the db ds integral is finite. Hence mu_cand((0,infinity))<infinity.

Equations (17) and (18) prove (3) for the actual expected candidate density.
The finite positive coefficient is evaluated in Section 10. Elementary
integration of the density asymptotic gives (5); its exponent 2/3 is not
the density exponent -1/3. Dominated convergence establishes no quantitative
near-term remainder, so an O(1) remainder is not asserted.

## 10. Exact coefficient, conditional pins and units

At contact the odd block (G,T_u) is independent of the even block (F,V_u,A_u),
by inversion parity of the stationary real covariance. Within those blocks,
the actual correlations remain. Reorder the six contact coordinates to obtain

    pi_0(v_0)=p_(F,V_u)(b,0) p_G(0) phi_tau_u(12k),
    z_0=36 k^2 E[A_u^2 1{A_u<0}|F=b,V_u=0].              (19)

Here phi_tau is the centered scalar Gaussian density with standard deviation
tau. Birth disintegration gives

    integral_R p_(F,V_u)(b,0)
          E[A_u^2 1{A_u<0}|F=b,V_u=0] db
      =p_Vu(0) E[A_u^2 1{A_u<0}|V_u=0]
      =p_Vu(0) s_u^2/2.                                  (20)

The last equality is scalar Gaussian symmetry. The law conditioned on F=b
need not be centered; replacing it prematurely would be incorrect.

Substitute (19)–(20) into (17). The remaining scalar integral is

    144 integral_0^infinity k^(4/3) phi_tau(12k) dk
      =12^(-1/3) integral_0^infinity t^(4/3) phi_tau(t) dt
      =Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt(pi)).            (21)

The last step is the Gaussian half-moment/gamma integral, with t=12k.
This proves (4). R1 implies all relevant densities and conditional variances
are strictly positive, and they are continuous over the compact circle.
Thus 0<C_loc<infinity. Physical omega factors remain in every derivative
covariance. Changing a field's spectrum or amplitude changes its coefficient;
no model-independent numerical coefficient or SIDE24 reference digits follow.

## 11. K2, the twenty fixed Gaussian profiles, and nonuniform boundaries

Every step above consumes only the one-point gradient rank, the seven contact
rows, and the six two-site first jets, followed by finite Gaussian regression.
Those facts hold for K=2 as well as K=3. At K2 the coefficient dimension is
25; the incidence maps in Sections 4–5 have dimension25 in domain and target,
and the same Sard/area argument works without change.

In particular, two-site second-jet covariance may be singular at K2, but this
does not invalidate (10)–(14): gradient rank is four, endpoint heights have
the conditional two-dimensional density, and the determinant-weighted type
expectation remains continuous even if its conditional Hessian law is
degenerate. Local positivity and the coefficient limit use R1/R4 instead.

At L=24 the ten positive symmetric catalogue spectra, each with Gaussian
coefficients, at K=2 and3 therefore give **twenty fixed ideal Gaussian cases**
to which (3)–(5) apply. These are not twenty independent experiments or fifty
admitted continuum models. Since the family is finite, the small-radius
bands and all analytic constants can be chosen uniformly across it, and

    max_j |ell^(1/3) nu_j(ell)-C_loc,j| -> 0.

Each C_loc,j is positive; taking a finite minimum also gives uniform relative
leading asymptotics across these twenty cases. There is no asserted uniform
window or constant over arbitrary q approaching the boundary of the positive
weight simplex, over unbounded K, or over variable L. For each fixed positive
q, integer K>=2 and L>0, the theorem holds with its own constants.

The K1 contact dependence and the K2 second-jet singularity in FG are retained
negative boundaries. The seventh-order axial annihilator at K3 blocks reuse
of P's arbitrary finite-jet premise but does not obstruct this finite list
of ranks. No counterexample to (3) under its stated assumptions was found.

## 12. Exact remaining scope

This preparation supplies a complete conventional candidate-count derivation
from the named FG inputs and the stated classical Sard/area imports. There
is no additional unproved model-specific analytic premise in that route.
Independent exact-source review is recorded separately in [REVIEW.md](REVIEW.md);
the earlier review of FG and pre-file route discussions do not substitute
for an incorporated-source review. No formal Lean implementation or kernel
execution is claimed.

Actual persistence bars require further source-bound elder selection,
global once-counting and the appropriate complement proof. Conditional
all-point genericity for enlarged cap/collision/factorial-moment lists has
not been proved here and is not silently imported. Higher homology and a
process limit remain separate. This theorem also does not certify numerical
sampling/coupling, a usable asymptotic window, blind-field validation,
publication acceptance or external uptake.

In particular the actual finite-word sampler is not the Gaussian coefficient
law used by Sard/null-set and Gaussian density arguments. The non-Gaussian
coefficient laws are not covered. Exact mathematical weights, actual
conditioning, density versus cumulative normalization, field-dependent
coefficient and per-area versus whole-torus counts must be preserved in any
consumer. Existing catalogue admissibility labels, applicability arrays,
graphs and scientific-status records remain unchanged.
