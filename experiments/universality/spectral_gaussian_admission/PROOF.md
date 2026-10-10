# Spectral Gaussian admission for regular-fold H0 lifetimes and iid-copy processes

Incorporated mathematical source,8October2026. Original author OpenAI/Codex
universality_review; repository wrapper and binding author root,
task01a11cbf-068d-7102-861e-75814e715c98, exactclaim6070584604 amended6071035594.
This records a fixed-law spectral/carrier admission theorem with named
classical imports. Its actual C4 regularity is weaker than smooth samples;
no existing smooth premise is silently declared met.

Original42038B/825-line source SHA43e8180947b559e53f7afcf95ba61136f3fffcbe05a19e7a5953018fda27e294
received two separate fresh full nonauthor mathematical PASS verdicts.
The preserved mathematical Sections1-11 are unchanged. This repository
source has its own identity and fresh review in REVIEW.md; original-file
verdicts do not certify later wrapper or dependency bytes.
All performers are AI agents with substantial same-provider source/route
exposure, organizational independence0. No numerical draw,blind seed,
compiler,formal closure or scientific promotion is inferred.

## 1. Admission theorem and a concrete spectral verifier

Fix an integer d>=2 and L>0. Let X=R^d/(L Z^d), omega=2pi/L, and
let a declared real stationary Gaussian law have spectral weights

    q_n=q_(-n)>=0, sum_(n in Z^d) q_n=1.                  (1)

The field is ensemble centered. If q_0>0, it has its independent random
constant mode; if q_0=0, that mode is absent. In either case its point
variance is one. Its physical covariance is

    K(z)=sum_n q_n exp(i omega n.z),                      (2)

not a covariance inferred from a name, approximate grid or truncated
experiment. The coefficient law is Gaussian; specifying this covariance
alone for a non-Gaussian field would not specify the law used below.

Here is a simple sufficient verifier:

    (P5) q_n>0 for every |n|infinity<=5;
    (A4) sum_n sqrt(q_n)(1+|omega n|)^4<infinity.          (3)

This is full positivity on the fixed box K5, including n=0. An optional
extension replaces it by (P5^0): positivity only for 0<|n|infinity<=5,
with q_0 unrestricted; Section 3 proves the additional zero-mean witness
step. The main verifier (P5) does not need that extension. Modes outside the box
may have arbitrary holes or may all vanish. No positive lower bound on
all frequencies, isotropy, analytic covariance or Gaussian-shaped
spectral decay is required. For each fixed law there is a positive
minimum among its finitely many required weights. A common minimum
over a family is a further condition, not part of (3).

**Spectral admission theorem.** Under (1) and (3), the actual field
has a C4 version and supplies the finite-regularity versions of COUNT,
CONTACT, SELECT, ENVELOPE and FAR proved below. The expected ordinary
finite superlevel H0 lifetime measure per volume has a specified finite
nonnegative Borel density representative satisfying

    nu_bar(ell) ~ c_q ell^(-1/3),
    E N_t/L^d ~ (3/2)c_q t^(2/3), ell,t down to zero,       (4)

where 0<c_q<infinity is the actual law-specific jet functional in (27).
The candidate-pair coefficient is the same. Bars are selected by the
global ordinary elder rule; the essential class is excluded.

For independent whole copies of this one fixed field law, retain the
marks lifetime a=ell/t, birth b, physical gap k=ell/dist(M,S)^3,
birth location M and ordered direction from M to its killing saddle S.
If n t_n^(2/3)->lambda in (0,infinity), their superposition converges
to the Poisson random measure with intensity lambda eta_q, where

    d eta_q=4 a^(-1/3) k^(-2/3) pi_0(v_0) z_0(b,k,u)
                         da db dk dx d sigma(u).          (5)

After integrating birth/gap/direction, its whole-torus lifetime
intensity is lambda L^d c_q a^(-1/3) da. A single fixed field's process
instead tends to empty on bounded rescaled-lifetime windows.

This is a C4 theorem, not a statement that (A4) makes samples C-infinity.
The existing structural and physical-law sources literally assume
smooth samples and prove all-Cq norm bounds. Sections 2-8 independently
check the finite regularity actually needed and prove the C4 variant;
they do not declare that an A4 field meets those stronger literal words.
No selected-density continuity, O(1) full near remainder, quantitative
cutoff, Hk theorem, spatial expanding-domain limit, coalescing-contact
estimate or global second-factorial theorem follows.

The theorem remains true under the weaker alternative (R4): the actual
Gaussian law has a C4 version with E||F||C4^(2d)<infinity. This version
must be supplied rather than asserted from finite pointwise moments.
Section 10 gives a spectral polynomial-tail family for which (R4) is
proved directly while (A4) fails. Neither (P5) nor (A4) is claimed
necessary or a minimal characterization.

## 2. Gaussian representation, C4 moments and the exact regularity ledger

Choose one representative from each nonzero pair {n,-n}, omitting
zero-weight terms. Independent standard real Gaussian coordinates give

    F(z)=sqrt(q_0) C_0+sum_pairs sqrt(2q_n)
              [C_n cos(omega n.z)+S_n sin(omega n.z)].     (6)

The constant term is omitted when q_0=0. Every real Fourier polynomial
with support in a positive-weight finite set is a coefficient direction
of this actual law; its coordinate scaling by sqrt(q_n) is invertible.
The block retains the original weights. It is never renormalized to
variance one after removal of the tail.

Under (A4), a positive weighted sum of |C_n|+|S_n| bounds ||F||C4.
Its expectation is finite, hence it is finite almost surely. Minkowski
gives its Lp norm at most a constant times the sum in (A4), for every
finite p>=1. Thus the series and derivatives through order four
converge absolutely uniformly, with

    F in C4 almost surely,
    E||F||C4^p<infinity for every finite p>0.              (7)

All derivative jets used below are the actual derivatives of this
version. In particular A4 also implies sum_n q_n(1+|omega n|)^8<infinity:
write a_n=sqrt(q_n)(1+|omega n|)^4 and use sum a_n^2<=(sum a_n)^2.
This explains why mixed derivative covariances needed in C4 regression
exist, without asserting pathwise eighth derivatives. Gaussian
uniqueness on a countable dense evaluation set identifies the Borel
law on C4. A theoretical representation is not an empirical draw.

Under (R4), the same argument below starts with its actual C4 version.
Its finite positive Fourier block, obtained by the continuous Fourier
coefficient maps, is independent of its C4 Gaussian tail. All uses of
field-norm moments need at most order 2d; all uses of Bochner covariance
need order two, which is included for d>=2. No Fernique theorem is
needed to upgrade the stated moment premise.
Here ||F||C4 is any fixed norm formed from supremum norms of its
coordinate derivatives through order four on this flat torus.

The finite-regularity imports and their uses are explicit:

| Step | Regularity actually used |
| --- | --- |
| Gradient coefficient incidence | A C4 tail gives a C3 map between equal-dimensional finite spaces. |
| Critical-value nullity of that projection | The equal-dimensional C1 case, also directly obtained from the area formula by zero Jacobian on its critical set. No scalar high-dimensional Sard theorem is invoked. |
| All-Borel critical counts | Weighted Euclidean area formula on locally Lipschitz incidence maps, localized to compact charts. C3 is ample. |
| Morse local form and H0 handles | Classical C2 Morse local form; the quadratic local model and critical-free level flow suffice for component attachment. |
| Critical-free bands | grad F/|grad F|^2 is C3 away from roots, so its ordinary flow is defined on each compact band. |
| Local cubic ridge | F_t is C3, its implicit ridge is C3, and phi'=F_s on the ridge makes phi C4. |
| Conditional convergence and endpoint estimates | Global C4 convergence, physical C3 bounds and finite 2d-th norm moments. |

The C2 Morse local-form theorem is used only on nondegenerate critical
points. Together with the displayed critical-free flow, its quadratic
coordinate form proves the same H0 handle statement used by the smooth
source: maxima create components, and an index-(d-1) critical point
merges two components or attaches a loop without changing H0. Higher
handle attaching sets are connected and do not merge H0 components.
This depends on the local homeomorphism/flow, not on infinite
differentiability. The conventional C2 Morse lemma and the Lipschitz
Euclidean area formula are named classical imports, not new theorems
about arbitrary low-regularity fields. No weakened rule below uses C1
samples or a nonexistent pathwise higher derivative.

## 3. Specific finite rank witnesses, including the absent-zero-mode case

Put

    theta_a(z)=sum_j[1-cos(omega(z_j-a_j))].

It is positive away from a, has a zero of order two at a, and has
coordinate frequency degree one. Theta_a^2 vanishes through derivative
order three and has coordinate degree two.

At p the coordinates sin(omega(z_j-p_j))/omega have derivative I.
Composing their degree-three Taylor polynomials realizes any prescribed
third-order jet at p. Their support lies in K3={|n|infinity<=3}.
This proves independence of the contact rows, including the independent
transverse Hessian entries, at every point and orthonormal frame.

For two distinct first-jet sites p,q, take B_p=theta_q^2 and the
functions B_p and B_p sin(omega(z_j-p_j))/omega. They kill the other
first jet; the value/gradient map at p is triangular with positive
diagonal B_p(p). The symmetric construction gives full two-site
value/gradient rank. Support is in K3.

For a contact x and an offsite p use B_p=theta_x^2; this kills the
entire third jet at x while freely varying the value and gradient at
p. For distinct x,p,q use

    B_p=theta_x^2 theta_q^2, B_q=theta_x^2 theta_p^2.

Their products with the local sine coordinates lie in K5, kill the
other offsite first jet and the full third contact jet, and give both
offsite value/gradient jets freely. Combining them with the single-site
contact directions gives the joint ranks. This is a proof for the
named rows, not an inference of arbitrary multijet rank.

If q_0=0, a witness's constant Fourier coefficient must be removed.
For each relevant jet configuration there is a nonnegative polynomial
g with positive integral, zero at all prescribed jets:

| Configuration | g, with the indicated jet vanishings |
| --- | --- |
| One contact through order three | theta_x^2 |
| First jets at p,q | theta_p theta_q |
| Contact at x and first jet at p | theta_x^2 theta_p |
| Contact at x and first jets at p,q | theta_x^2 theta_p theta_q |

The degrees are respectively 2,2,3,4. At finitely many distinct sites
the factors are positive on a nonempty open set, so integral g>0.
Replace each already constructed witness h by

    h_tilde=h-(integral_X h/integral_X g) g.              (8)

It has exactly the same specified jets and zero mean; its support
stays in K5. Thus every required rank is witnessed using only nonzero
positive K5 modes, proving the optional (P5^0) extension of the theorem.
Removing q_0 still changes the law and birth-mark
distribution; it is not declared to preserve the covariance or
normalization of a field which previously had that mode.

In particular the actual finite block and the full Gaussian law have
positive covariance for precisely these vectors:

    O_(p,q)=(F(p),grad F(p),F(q),grad F(q)), p!=q;
    U_0=(F,F_u,F_uu,F_uuu,(F_vi,F_uvi)_i);
    U_0 plus independent symmetric entries of A=Hess F|u-perp;
    U_0 plus (F(p),grad F(p)), p!=x;
    U_0 plus both first jets at p,q, x,p,q distinct.        (9)

Indeed zero variance of a row combination would make it annihilate
every positive Fourier direction, contradicting the witnesses. The
enlarged Hessian list uses independent symmetric entries; it does not
pretend all d^2 entries are independent. After U_0 is conditioned its
coordinates are deterministic; the residual ranks are the positive
Schur complements of these joint covariances.

Continuity gives positive floors only on the declared compact separated
domains. Contact floors are uniform in frames because O(d) is compact.
No floor is inferred as additional sites approach contact or the pair
diagonal. Nor is a K7 four-site rank result inferred from K5 support.
The iid proof below needs the bounded one-contact global mark, not K7.

## 4. COUNT by finite incidence and actual Gaussian fibers

Split F=F_B+T into its actual K5 coefficient block and independent C4
tail; omit the zero coordinate if absent. The finite block alone has
all ranks (9), with its original positive weights. Given almost every
tail realization, it is a deterministic C4 function added to a finite
smooth Gaussian basis.

At a point choose d coefficient columns whose gradient minor is
invertible locally. Write the block coordinates as (a,eta). The zero
gradient equation solves

    a(p,eta)=-B(p)^(-1)[grad T(p)+C(p)eta].               (10)

The coefficient projection (p,eta)->(a(p,eta),eta) is a C3 map between
N-dimensional spaces. Its derivative is singular exactly when the
physical Hessian at p is singular, after invertible minor factors.
On compact chart pieces the map is Lipschitz. Area formula applied
to its critical set, whose absolute Jacobian is zero, proves that
its critical image has Lebesgue measure zero: every image point has
at least one critical preimage. This is the needed equal-dimensional
Sard conclusion, requiring only C1 here. The Gaussian block has a
Lebesgue density. Countable chart/minor exhaustion and integration of
the tail prove unconditional Morse genericity before any Rice count.
Compactness then gives a finite critical set.

For a pair p!=q use 2d invertible gradient coefficient columns. The
spatial derivative is block diagonal (H_p,H_q). The coefficient
projection Jacobian is |det H_p det H_q|/|det B(p,q)|, with the usual
coordinate-volume factors. Weighted area formula, coefficient
change of variables and nonnegative Tonelli give

    E sum_(p!=q,grad F(p)=grad F(q)=0) w(p,q,F)
      =integral_(p!=q) p_(grad F(p),grad F(q))(0)
          E[|det H_p det H_q| w | both gradients=0] dp dq  (11)

for every nonnegative Borel mark of the full C4 field and pair.
Both sides initially may be infinite. Partition the countably many
minor/chart pieces disjointly and exhaust relatively compact parameter
domains. Local Lipschitz regularity suffices; no global coefficient
Lipschitz assertion is made. Conditional determinant moments and
covariance floors make unmarked integrals finite on compact annuli.

The conditional version in (11) is fixed at every target. The
finite coefficient fiber integral is an ordinary full-rank Gaussian
disintegration. Integrating its Bayes-weighted tail gives, for each
finite evaluation vector, the usual full Gaussian regression at that
target. Equality on a countable dense evaluation set identifies the
whole C4 kernel. Thus one does not extend a continuous-mark formula
to Borel marks by assertion or select a favorable unspecified null
fiber. All-Borel marks are allowed directly by area formula.

The conditional two-height covariance at distinct sites is positive
by O rank in (9). Apply (11) with F(p)=F(q) on each compact annulus:
the conditional height density makes its contribution zero. Countable
annulus exhaustion gives distinct critical values almost surely.
Point incidence similarly gives finite expected critical count,
using a uniform gradient covariance floor and Hessian moments.

The intrinsic rational-superlevel connectivity selector of ST is
Borel on C4 fields and zero outside the open Morse/distinct locus.
The local-form/level-flow argument of Section 2 counts every finite
H0 bar once as an ordered maximum/merging-saddle pair. Its unique
essential global-maximum class is excluded. Apply (11) with the
Hessian type indicator and selector; only one determinant product
enters. Two-height disintegration yields the specific Borel kernel

    K_sel(p,q,b,s)=p_O(b,0,s,0) E_Q[W_(p,q) sigma],
    nu_bar(ell)=L^(-d) integral_(p!=q) integral_R
                         K_sel(p,q,b,b-ell) db dp dq.      (12)

This proves COUNT in the C4 setting, with the chosen actual versions.
It does not use two-site full Hessian rank or a scalar Sard theorem
on F:X^d->R.

## 5. Actual contact regression and ENVELOPE

At M=x-r u/2,S=x+r u/2 choose an orthonormal transverse frame v_i.
Use the literal row order

    U_r=((F(M)+F(S))/2,(F(S)-F(M))/r,
         (F_u(S)-F_u(M))/r,
         (6/r^2)[F_u(M)+F_u(S)-2(F(S)-F(M))/r],
         ((F_vi(M)+F_vi(S))/2,(F_vi(S)-F_vi(M))/r)_i),
    v_r=(b-k r^3/2,-k r^2,0,12k,(0,0)_i), k>0.           (13)

The contact rows are exactly U_0 in (9), target
v_0=(b,0,0,12k,(0,0)_i). In an axial C4 path psi(s), with a=r/2,
the fourth row is the positive average

    (3/(4a^3)) integral_(-a)^a (a^2-s^2) psi'''(s) ds.

Integration by parts verifies this identity. Its kernel has mass one.
All other rows are averages/divided differences of derivatives of
order at most two. Thus, uniformly in x and frames,

    |U_r F|<=C||F||C3,
    |(U_r-U_0)F|<=Cr||F||C4.                             (14)

Let Sigma_r=Cov(U_r), and let c_r=E[F U_r F] be a finite vector of
C4-valued Bochner expectations. E||F||C4^2<infinity makes them exist.
The estimates in (14) show

    Sigma_r->Sigma_0 uniformly,
    ||c_r-c_0||C4<=Cr E||F||C4^2.                         (15)

This supplies global C4 cross-covariance convergence without pathwise
C8 derivatives or an all-Cq claim. Stationarity removes x from the
contact floor. Rank (9) and compactness of O(d) give Sigma_0>=lambda I,
and then Sigma_r>=lambda I/2 for a common small r_0>0. Upper covariance
bounds also hold. Density pi_r(v_r) therefore converges at every
fixed b,k,u to pi_0(v_0).

On one actual full-field probability space set

    F_r=F+c_r Sigma_r^(-1)(v_r-U_r F).                    (16)

This satisfies its pins exactly. Its centered residual is independent
of U_r F by Gaussian orthogonality on a dense evaluation set. Thus it
is the canonical full-field conditional law Q_r, including at null
pin events. Equations (14)-(15) give F_r->F_0 globally in C4 and

    ||F_r||C4<=C||F||C4+C(|b|+k), 0<r<=r_0.              (17)

The constants depend on the declared fixed law, d,L, not on b,k,r.
The same coupling is continuous along convergent compact parameter
sequences. It is not a finite-truncation convergence argument.

Under the zero endpoint gradients, the physical Hessians satisfy

    H_i=[r alpha_i, r beta_i^T; r beta_i, A_i],
    det H_i/r=alpha_i det A_i-r beta_i^T adj(A_i) beta_i.

This polynomial identity holds also for singular A_i. Equation (14)
and the endpoint zero average bound alpha_i,beta_i by C||F_r||C3.
Global C4 convergence gives alpha_M->-6k, alpha_S->6k and A_i->A_0.
The correct congruence diag(r^(-1/2),I) gives the maximum/index-(d-1)
types when A_0<0. If A_0 is nonsingular but not negative definite,
the maximum condition fails; if singular, the determinant kills the
boundary. Consequently, with W_r the actual typed determinant product,

    V_r=W_r/r^2 -> V_0=36k^2(det A_0)^2 1{A_0<0},
    V_r<=C(1+||F_r||C3)^(2d).                            (18)

The moment in (R4) suffices for domination. Enlarged contact rank
makes the conditional Gaussian law of A_0 full-support on symmetric
transverse matrices. Its b-conditioned mean may be nonzero, but
every open negative-definite cone has positive probability. Thus
z_0=E V_0 is positive and finite at every fixed b in R,k>0 and
E_Qr V_r->z_0. No pin covariance itself remains positive after pinning.

The target's first and fourth coordinates are b-k r^3/2 and 12k;
for bounded r they give |v_r|^2>=c(b^2+k^2). Covariance floors and
(17)-(18) therefore yield the all-mark unnormalized bound

    12 pi_r(v_r) E_Qr V_r
      <=H(b,k)=C(1+|b|+k)^(2d) exp[-c(b^2+k^2)],
    integral_Rx(0,infinity) k^(-2/3) H db dk<infinity.      (19)

This proves CONTACT and ENVELOPE. Continuous positivity of z_0 gives
a floor on compact birth/gap/frame windows, k bounded away from zero.
There is no all-mark inverse normalizer, inverse random curvature or
unbounded-law-family uniformity.

## 6. Contact posterior genericity and actual SELECT

Fix x,frame,b in R,k>0 and condition on U_0=v_0. This conditional
field has a forced degenerate contact at x; it is not globally Morse.
Unconditional genericity cannot be substituted for its needed
off-contact assertion.

In the finite-block/tail split the block contact covariance Sigma_B
is positive. The conditional tail has the exact posterior density

    dmu_tail,v(t)/dmu_tail(t)
      =p_(U_0 F_B)(v_0-U_0 t)/p_(U_0 F)(v_0).             (20)

Its denominator is positive and the density integrates to one. The
posterior is absolutely continuous with respect to the original C4
tail, so it remains C4 almost surely. Given the tail, the block is a
Gaussian on its affine contact fiber. Its free coefficient directions
are the kernel of U_0; the centered free coordinates have a positive
Lebesgue density independent of the affine mean.

The residual ranks in (9) hold on that kernel. With the fixed posterior
tail, apply the same C3 equal-dimensional point incidence on X minus
{x}: countable compact exhaustion proves all other critical points
Morse. The residual single value/gradient density makes the mark
F(p)=b contribute zero; residual paired first-jet density makes
F(p)=F(q) contribute zero when x,p,q are distinct. Use the all-Borel
one- and two-point area formulas and separated compact exhaustions,
then integrate (20). Thus, for each fixed contact target, all other
critical heights are distinct and different from b almost surely.
No event simultaneously covering uncountably many targets is asserted.

The full-density conditional A_0 is nonsingular almost surely. With
invertible A_0 the local C3 transverse ridge exists; its reduced
function has phi_0'=phi_0''=0 and phi_0'''=12k at contact. Hence contact
is isolated. The remaining compact critical set consists of isolated
Morse points and is finite; their heights have positive finite gaps
from each other and b.

On V_0>0, A_0<0. The deterministic elder argument in ST Section 4
needs only the C4 regularity checked in Section 2, so it applies to
the actual coupling (16). To make its topological content explicit,
take a fixed small cylinder in axial s and transverse t. Strict
transverse concavity gives a unique ridge h_r(s), C3-converging to
h_0. Its reduced phi_r has phi_r''' positive on the cylinder. The
exact pins give its only critical roots at s=+-r/2, a maximum then
an index-(d-1) saddle. A compact gradient floor and implicit old
branches exclude any other critical point globally.

The left cylinder cut at S has its left and transverse boundary
below the death height; its right face is bounded above by the saddle
height. For intermediate levels its maximum component is enclosed,
connected along the concave fibers and ridge, and cannot connect
outside. The positive ridge past S reaches a fixed point higher than
the limiting birth b. At S the newborn component therefore merges
with an older component; the elder rule gives precisely this actual
finite pair. All old critical levels are separated from b, so the
coupled field is globally Morse/distinct eventually as well.

Thus sigma_r=1 eventually on V_0>0. On V_0=0, V_r(1-sigma_r)->0
without any genericity claim at the corresponding finite-r fibers.
The same norm domination gives

    E_Qr[V_r(1-sigma_r)]->0,
    E_Qr[V_r sigma_r]->z_0.                              (21)

This is SELECT. A varying-parameter compactness argument gives
qualitative compact-window convergence if desired, but the theorem
uses the pointwise result and (19). Neither a common physical chart
radius nor a probability rate is claimed.

## 7. FAR, density composition and the coefficient

For dist(p,q)>=r_0, the original O_(p,q) covariance has a positive
floor by (9), continuity and compactness. Conditional regression at
target (b,0,b-ell,0), 0<ell<=1, gives a C2-norm moment polynomial
in |b| and a Gaussian observation density bounded by C exp(-c b^2).
Integrating the determinant product over b and the compact pair domain
gives

    0<=nu_bar,far(ell)<=nu_cand,far(ell)<=C.               (22)

Any fixed compact positive-lifetime band similarly has a finite
local bound. This is a direct density bound for the representative
(12), not differentiation of a cumulative estimate. It proves FAR.

The literal linear observation transform in (13) has exact absolute
determinant 12r^(-(d+3)). Its four height/axial rows contribute 12r^-4,
and the d-1 transverse pairs contribute r^(-(d-1)). With spatial
polar volume r^(d-1)dr, height Jacobian r^3 db dk and typed Hessian
weight r^2 E[V_r sigma], the product is

    r B_r dr db dk dx d sigma(u),
    B_r=12 pi_r(v_r) E[V_r sigma].                       (23)

The flat torus midpoint Jacobian is exactly one in the unique short
band. The dimensional powers cancel to r; a cubic gap alone would
not determine this cancellation. Substituting ell=k r^3 in the
specific density (12) gives

    ell^(1/3) nu_bar,near(ell)
      =integral 1{k>=ell/r_0^3}
           B_((ell/k)^(1/3))(b,k,u)/(3k^(2/3)) db dk d sigma.

Equations (19),(21) give its all-mark dominated limit

    c_q=4 integral_Rx(0,infinity)xS^(d-1)
                  k^(-2/3) pi_0(v_0) z_0 db dk d sigma.   (24)

It is finite and positive; the far term vanishes after multiplication
by ell^(1/3). This proves (4) as a pointwise statement for the chosen
canonical finite Borel representative. A different density modified
on a null set need not retain that pointwise statement. Global
selection's discontinuity supplies no density-continuity theorem.
The cumulative factor 3/2 follows by integration.

For explicit physical evaluation, write xi=omega n. The exact jet
covariances include

    M_G=sum q_n xi xi^T,
    M_Vu=sum q_n (xi.u)^2 xi xi^T,
    h_u=Cov(F_uuu,G)=-sum q_n (xi.u)^3 xi,
    tau_u^2=sum q_n (xi.u)^6-h_u^T M_G^(-1)h_u.           (25)

Hessian covariances use sum q_n xi_i xi_j xi_k xi_l, and
Cov(F,H_ij)=-sum q_n xi_i xi_j. All these sums are actual and
absolutely convergent. Stationarity and q_n=q_-n make the covariance
even; the odd Gaussian block (G,F_uuu) is independent of the even
block (F,V_u,A_u). No rotational or coordinate-product independence
is inferred. Put

    D_u=E[(det A_u)^2 1{A_u<0}|V_u=0].                   (26)

The full V_u has d coordinates, not merely F_uu. Birth disintegration
removes F=b while retaining V_u/A_u correlations. The b-conditioned
A need not be centered; the A conditioned only on V_u=0 is centered.
The half-Gaussian integral is
2^(-1/3)Gamma(7/6)tau_u^(4/3)/sqrt(pi). Combining the exact 12 factor
in (24) yields

    c_q=Gamma(7/6)/(24^(1/3)sqrt(pi))
         *integral_S p_G(0) p_Vu(0) tau_u^(4/3) D_u d sigma(u),
    p_G(0)=(2pi)^(-d/2)(det M_G)^(-1/2),
    p_Vu(0)=(2pi)^(-d/2)(det M_Vu)^(-1/2).               (27)

Full enlarged contact rank makes D_u>0; frame compactness controls
the integrals. For d=2 it is Var(A_u|V_u=0)/2. For higher d one
keeps the negative-definite cone integral. Ordinary full-sphere area
and ordered maximum-to-saddle direction introduce no factor 1/2.
The numeral 24 is the Jacobian/gamma factor, not L. The exponent is
shared in this admitted class, while c_q varies with the physical law.

## 8. Marked iid-copy process: physical cluster avoidance, not factorial transfer

On the open C4 Morse/distinct locus define N_t as the sum of its
finite Borel elder pairs with gap <=t, and assign zero on its complement.
Implicit critical branches on a countable parameter-chart cover make
N_t jointly Borel in field and threshold. Use a Borel shortest torus
displacement convention away from the near band; its far mark
contribution vanishes under (22). The near ordered displacement is
M=x-r u/2,S=x+r u/2. Local/Borel transverse frames suffice.
Explicitly Xi_t is the sum of delta masses at (ell/t,b,k,M,u) over
finite bars, on E=(0,infinity) x R x (0,infinity) x X x S^(d-1),
and is zero on bad fields. Its restriction to a<=T has count N_(Tt).

At ell=ta with 0<a<=T, substitution in (23) gives the rescaled mean
integrand

    a^(-1/3) 1{k>=ta/r_0^3}
       B_((ta/k)^(1/3))(b,k,u)/(3k^(2/3))

with birth location x-r u/2. Its majorant is a^(-1/3)H/(3k^(2/3)),
integrable over the full window and all marks. Pointwise location
convergence and (21) give the mean limit (5), including tightness of
small-a and unbounded birth/gap tails. Far expected counts are O(t).
In particular eta_q{a<=T}=(3/2)L^d c_q T^(2/3).

For a fixed contact-law realization with V_0>0, Section 6 gives a
complete list: only the pinned M,S collapse, and all old critical
branches persist with distinct limiting heights. The pinned actual
elder bar consumes their unique birth/merger roles. Every other bar
has two old endpoints, whose finitely many possible limiting gaps
have a positive minimum. Thus at r=(ta/k)^(1/3), N_(Tt)=1 eventually.
The pairing among old endpoints need not stay fixed. On V_0=0 the
weight V_r tends to zero regardless of the count.

Insert the bounded global Borel mark 1{N_(Tt)>=2} in (11)-(12).
At every fixed a,b,k,u,x its weighted contact expectation tends to
zero. The preceding all-mark majorant applies without multiplying
by an additional count. Dominated convergence and the O(t) far
anchor bound prove the genuine physical estimate

    E[N_(Tt)1{N_(Tt)>=2}]=o(t^(2/3)).                    (28)

For a compact-supported nonnegative process test f, let h=1-exp(-f).
The difference between E Xi_t h and E[1-exp(-Xi_t f)] is between
zero and the left side of (28). Hence, for n independent whole
copies with n t_n^(2/3)->lambda,

    E exp[-sum_(j=1)^n Xi_(t_n)^(j) f]
       ->exp[-lambda integral(1-exp(-f))d eta_q].        (29)

Equivalently censor multi-point copies; their union probability is
o(n t_n^(2/3))->0. The remaining independent Bernoulli singletons
have convergent tight mark laws and binomial counts converging to
Poisson. This proves the process theorem, including finite windows
a<=T. It is independent whole copies, not within-one-field spatial
independence. For one fixed field the finite positive lifetime list
makes the local rescaled process eventually empty almost surely.

Nothing in (28) bounds E[N_(Tt)(N_(Tt)-1)]: that would insert an
unbounded multiplicity into the contact kernel. K5 does not license
the auxiliary K7 four-site claims of PP. Coalescing-contact factorial
and regional additional-witness problems therefore remain separate.
PP's two-point-cluster comparison also still applies: first moment
alone can give a compound Poisson limit, and (28) is the substantive
extra ingredient used here.

## 9. More general finite carriers and why K5 is not necessary

The proof only needs a finite symmetric positive-weight carrier B
whose real trigonometric span supplies the five exact joint ranks
in (9), for every declared point/frame/distinct-site tuple. Under
(R4), these **named carrier tests** replace (P5) throughout Sections
4-8: select finite minors there, use that block's contact posterior,
and retain the same full Gaussian regression and moments. A covariance
floor still follows only on separated compact domains. This is a
more general sufficient admission theorem, not an assertion of a
minimal spectral criterion or a license to ignore the residual tests.

There is a concrete example failing literal K5 positivity but meeting
these tests in every d>=2. Start with any positive full K5 finite law
H, and take an integer unimodular shear A with z_1 mapped to
z_1+m z_2, m>5. Let F(z)=H(Az). It is again stationary Gaussian on
the same torus and variance one, with positive support on A^T K5.
The mode e_1 is absent: A^(-T)e_1=(1,-m,0,...,0) is outside K5.
The finite C4 sum is automatic.

All needed carrier tests nevertheless hold. Pull back the earlier
finite witnesses through the torus diffeomorphism A. Full third jets
at a contact transform by an invertible jet map; first jets at
distinct sites transform invertibly too; full contact-jet vanishings
are preserved. Thus the same joint/residual interpolation maps are
surjective. The transformed support, rather than an imaginary full
coordinate box, supplies the finite minors. The geometric constants
depend on A, and A is not treated as an isometry. This proves that
full coordinate K5 positivity is sufficient, not necessary.

Even within the simple verifier, arbitrary outside holes are harmless.
For example remove any chosen symmetric pair of modes outside K5
from an admitted infinite A4 spectrum and renormalize the remaining
weights. K5 remains positive and A4 remains finite, so the theorem
applies to this changed physical law. The change may alter the jet
coefficient and birth marks; it is not claimed to preserve them.

## 10. A4 is not necessary: polynomial tails with a proved C4 moment version

Let

    q_n=Z^(-1)(1+|n|infinity)^(-beta),
    d+8<beta<=2d+8.                                     (30)

The normalizer is finite; symmetry, positivity and (P5) hold. On a
dyadic frequency shell 2^j<=|n|infinity<2^(j+1), the A4 shell mass
is comparable to 2^(j(d+4-beta/2)). Thus A4 diverges throughout the
stated upper-bound range, including its endpoint. We now prove
that the actual Gaussian Fourier law still meets (R4), without a
claim from pointwise eighth moments alone.

Let F_j be its finite Gaussian trigonometric shell. For each derivative
alpha of order <=4, its value at any fixed point has standard
deviation at most

    M_j=C 2^(j(d+8-beta)/2).

The elementary coefficient-L1 bound for the gradient of D^alpha F_j
has Lp norm at most C_p 2^(j(d+5-beta/2)), for every finite p>=1.
Use a torus grid of mesh delta_j=2^(-j(d/2+2)), with fixed-L
constants. Its logarithmic cardinality is O(j+1). A union bound on
Gaussian marginal tails, followed by tail integration, gives

    ||max_grid |D^alpha F_j|||Lp
       <=C_p sqrt(j+1) M_j.

No independence of grid values is needed. Interpolation from the
nearest grid point costs at most C delta_j||grad D^alpha F_j||infinity,
whose Lp norm is at most C_p 2^(-j)M_j. There are finitely many
derivatives through order four. Consequently,

    ||F_j||Lp(C4)<=C_p sqrt(j+1)2^(j(d+8-beta)/2),
    sum_j ||F_j||Lp(C4)<infinity.                        (31)

The base finite block is handled separately. Minkowski proves all
finite C4-norm moments for the block sum. In particular the sum of
its C4 norms has finite expectation and is finite almost surely,
so the shell series converges absolutely in the C4 Banach space.
It has covariance (2) by summable q and is the actual Gaussian law.
This proves (R4), hence the full density and iid-copy conclusions,
despite the divergence of A4. It does not claim samples C-infinity.
All constants in this calculation are for fixed d,L,beta.

This alternative also shows why a finite positive carrier plus an
actual C4/moment version is a more intrinsic admission condition than
one particular Fourier summability test. No assertion is made that
(30)'s range or C4 itself is necessary for a regular-fold theorem.

## 11. Refusals, lack of law-family uniformity and structural meaning

Smoothness or summability alone does not ensure admission. If q_0=1
and all other weights vanish, the field is a random constant: all
moments exist, but the rank, Morse and positive fold-weight conclusions
fail. Spectra supported on a proper sublattice produce exact spatial
copies of critical values, defeating distinct-value COUNT. Additive
coordinate-only spectra can force mixed Hessian rows to vanish and
collapse the contact density. These are refusal diagnostics for the
present interfaces, not proved alternative exponent laws.

Nor does variance normalization make the coefficient universal.
Take any admitted normalized Gaussian H and an independent standard
constant C. For a fixed 0<delta<1 put

    F_delta=sqrt(1-delta^2) C+delta H.

This is another normalized stationary Gaussian law with positive
nonzero K5 modes and finite A4 when H has them. Its derivative
covariance floors shrink as delta^2. Addition of its constant shifts
all critical heights equally, while multiplication by delta scales
every lifetime, so the exact density transformation gives

    c_(F_delta)=delta^(-2/3)c_H.                         (32)

It diverges as delta->0, although every fixed delta is admitted and
even A4 can stay bounded. Thus uniform tail summability without a
common finite-carrier floor does not supply law-family uniformity.
No joint delta/lifetime assertion is imported from the fixed-law theorem.

The admitted exponent is structural in a precise sense: the cubic
height gap, regular contact density, spatial polar power and typed
Hessian product have the orders in (23), giving r dr and then -1/3.
The spectral tests verify those probabilistic and topological
interfaces for this Gaussian family. They do not say the cubic gap
alone fixes the exponent, or that covariance fixes a non-Gaussian
contact density. A forced transverse null block, contact rank loss,
different singularity order, nonintegrable marks or a comparable far
contribution needs a different proof. No higher-order singularity
taxonomy is asserted as an established field law here.

For a finite collection of already admitted laws one can take finite
minima/maxima of analytic constants, while retaining each c_q. For
an arbitrary family approaching a support boundary, varying dimension,
side length or geometry, the present individual theorem makes no
uniform claim. Hk, spatial mixing/expanding volume, regional rejected
candidate/additional-witness populations, all factorial moments,
numerical approximation windows, blind fields and formal owner targets
remain distinct consumers.

## 12. Exact repository bindings, proof reuse and review scope

Current operational sources are the unchanged reviewed files of the
structural/physical/process increment. The initial incorporation base is
native1966a74693f5cebfc63ace00b85d9efa0d07a3d7, reconciled to mainf08dfdf9.
These mathematical dependencies have their own actual source identities:

| Input | Current source | UTF-8 bytes / lines | SHA256 |
| --- | --- | --- | --- |
| ST | [Structural fold theorem](../fold_structural_universality/PROOF.md) | 43927 / 804 | 3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531 |
| IG | [Exact periodized law](../periodized_gaussian_h0/PROOF.md) | 34629 / 637 | c8bd489ca3fbb3760f7b2d53e7385131ac5336c51ba983e00e4d540fe2f00aaa |
| PP | [Iid marked process](../iid_short_bar_process/PROOF.md) | 30935 / 618 | 320bc988c468bea4938bb777a2d05952c6ab8e0a384f33725b50de847aeec1eb |

The original author freshly read all three actual current sources in full,
including separately filling its initially truncated structural middle.
Those reads covered these exact unchanged dependencies. Its prior original
preparation reads remain distinct; neither is a claimed full read of new
wrapper bytes. Root independently fully read all three current sources,
all825 original spectral lines and the complete actual incorporation diff.

Reuse is narrow. STSections2-4 supply the Borel selector, literal row
transform, determinant identity and elder ridge route; Sections5-7 supply
the geometric ledger and corrected parity coefficient. Sections2-8 here
independently establish their C4 variants. IGSections3-6 supply theta
interpolation, finite-block posterior incidence and full-field regression;
the image-sum law, all-mode positivity and physical moment numerals are
not general-family premises. PPSections2-5 supply Borel counts, complete
endpoint isolation and iid Laplace; Section8 supplies the first-moment
insufficiency comparison. Its auxiliary K7/factorial result is not admitted
from K5. The original task began while PP review was pending; root later
reported the unchanged incorporated sources' full reviews completed.
That historical report was not the spectral author's own review of those
records or a verdict on later spectral wrapper bytes.

The original spectral saved source received final_contract_review fresh
FULL825-line mathematical PASS, and benchmark_formal_audit separately
fresh FULL825-line mathematical PASS. Both are nonauthor AI reviews with
substantial prior route exposure and credited prefile contributions.
The former freshly rehashed ST/IG/PP with its immediately preceding full
incorporated-source reads; the latter freshly rehashed them and indirect
sources, reading ST750-804,IG585-637,PP545-618 custody sections with prior
full original-source reads and earlier regularity excerpts. Those hashes
and bounded dependency reads are not additional whole dependency audits.
Earlier prefile checks remain bounded route attacks rather than full
saved-file reviews. Original preparations remain frozen in ordinary
coordinator custody, not unverified repository links.

The mathematical Sections1-11 above retain their exact original bytes.
The local review companion separately binds this current whole file,
construction snapshots, completed verdicts and any subsequent amendments.
No metadata pending snapshot is rewritten as a completed earlier review.

Named classical imports remain weighted Lipschitz Euclidean area formula,
finite-differentiability Morse local form and critical-free flow/H0 handle
consequences, with Gaussian regression,Bochner expectation,Tonelli/Fubini,
Minkowski and dominated convergence under the displayed domains/moments.
Both saved-file reviewers retained a wording qualification: actual use is
on C4 functions, for which finite-differentiability Morse coordinates
suffice; neither freshly verified the stronger C2 boundary statement.
Root's separate bounded primary read of [Conrad's Morse handout,Remark2.2](https://math.stanford.edu/~conrad/diffgeomPage/handouts/morselemma.pdf)
confirms that Cp,p>=3 has Cp-2 coordinates, sufficient for this C4 theorem.
It is a parsed statement read, not a full-book/raw-PDF custody claim.
Root's separate Milnor h-cobordism open returned initial passages only;
no complete-book review is attributed to it or to the other agents.

The admitted law/carrier tests must actually be checked for each proposed
Gaussian support. No covariance-only non-Gaussian admission, spectral
family uniformity, Hk result, spatial expanding-domain limit, coalescing
or global factorial closure, numerical window, sampler/count coupling,
blind50 confirmation or literal Lean discharge is supplied. Scientific
promotion:NONE. Full landmark goal ACTIVE.
