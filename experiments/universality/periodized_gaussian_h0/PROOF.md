# Actual H0 lifetime intensity for the exact periodized Gaussian law

Source proof by OpenAI/Codex universality_review, 8 October 2026; repository
composition and current structural-source binding by OpenAI/Codex root, task
01a11cbf-068d-7102-861e-75814e715c98, under standing authorization and exact
claim6070584604. Initial source base is main2c84170712c59d9de580c172815bd30bac5d93cd.
Mathematical Sections1-8 are byte-identical to the frozen complete preparation.
The source identifies the actual physical law before deriving all five analytic
interfaces; it does not import a finite-cutoff limit or historical remainder.

The [review](REVIEW.md) separates the original full-source verdict from this
incorporated source's exact-file review. Source and proof-route exposure is
substantial, including the coordinator's theta-polynomial rank contribution.
Organizational independence is zero. No sampler, coefficient enclosure, model
admissibility, scientific register, formal target or old parent is promoted.

## 1. Actual source, law and exact theorem scope

The actual source read is root project commit
`6541afbdc93114b8cd35c40e77605e8491214330`,
`formal/sources/side24_v1/PROOF.md`, 10,272 UTF-8 bytes / 216 lines,
SHA256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`.
Its complete actual text was read; its local Git blob
`44b66f04f89fcd87383b3603fa69f1feb64cdddd` matched that commit's path.
Its explicitly defined covariance is

    K_24(z)=sum_(j in Z^d) exp(-|z+24j|^2/2)
             /sum_(j in Z^d) exp(-|24j|^2/2), d=2 or 3.     (1)

It evaluates its parent's coefficient expression; it expressly does not
independently accept the parent's global elder-selection or Kac-Rice proof.
The imported formal source does not define a finite Fourier-cutoff field.
The actual parent P also allows each fixed d>=2 and L>0, with covariance

    K_L(z)=sum_(j in Z^d) exp(-|z+Lj|^2/2)/S_(d,L),
    S_(d,L)=sum_(j in Z^d) exp(-|Lj|^2/2).                 (2)

We derive the adapters below for that fixed law; (1) is its L24 instance.
The torus is X=R^d/(L Z^d). The field is centered in its ensemble, stationary,
variance one, and has its random constant mode. Removing a sample mean,
changing width, empirical variance normalization, a minimum-image kernel,
or a finite spectral cutoff defines a different law.

For omega=2pi/L the exact spectral weights are

    q_n=exp(-omega^2 |n|^2/2)/Z_(d,L),
    Z_(d,L)=sum_(j in Z^d) exp(-omega^2 |j|^2/2), n in Z^d. (3)

The catalogue's finite Gaussian profile uses `exp(-|n|^2)` on a finite
square support. It has neither this exponent nor this support. Shared
side length, Gaussian coefficient law or moment numerals do not identify
the two random fields. There is no limiting-cutoff admission in this proof.

**Adapter theorem, subject to the named classical imports.** For every
fixed d>=2,L>0, the actual infinite Gaussian law (2) supplies COUNT, CONTACT,
SELECT, ENVELOPE and FAR of the corrected structural theorem ST below.
The expected ordinary finite superlevel H0 lifetime measure per volume has
a particular finite nonnegative Borel density representative satisfying

    nu_bar(ell) ~ c_(d,L) ell^(-1/3),
    E N_bar((0,t])/L^d ~ (3/2)c_(d,L) t^(2/3),             (4)

where c_(d,L) is finite and positive and is the exact covariance-derived
functional (19). The same leading coefficient holds for candidate pairs.
The proof gives no selected-density continuity, quantitative pairing rate,
O(1) full near remainder, usable lifetime window or numerical digits.
In particular it does not re-admit the historical SIDE24 intervals or the
parent's quantitative compact-window claims. These are separate consumers.

The five-interface ledger is:

| Interface | Actual infinite-law discharge here |
| --- | --- |
| COUNT | A fixed finite coefficient block conditioned on its smooth tail; finite incidence Sard before Rice; all-nonnegative-Borel area formula; Gaussian height disintegration; unconditional Morse/distinct values and the actual Borel elder selector. |
| CONTACT | Specific trigonometric interpolation proves full contact/enlarged-transverse rank; compact covariance floor; the same infinite field's finite-observation regression converges globally in C4; polynomial moments give the full normalizer limit. |
| SELECT | The contact-conditioned finite block gives Morse other critical points and excludes all height collisions; the contact is isolated; the actual deterministic ridge/enclosure elder argument applies on positive limiting weight. |
| ENVELOPE | Summable Fourier derivative weights and the exact regression formula give a common all-mark bound C(1+|b|+k)^(2d) exp[-c(b^2+k^2)]. |
| FAR | Actual original value/gradient rank at distinct pairs, a compact covariance floor away from the diagonal, and conditional determinant moments give a direct O(1) far lifetime density. |

All ranks, couplings and expectations in this table refer to (2)-(3).
They are proved below, rather than inherited from a finiteN label.
The constants are existential for a fixed d,L. No joint unbounded-d,
variable-L, cutoff or model-family uniformity is asserted.

## 2. Fourier identity and smoothness with actual supremum moments

One can derive (3) directly, without assuming a truncated Poisson formula.
Integrate the periodic image sum over one fundamental cube. At Fourier
frequency omega n, shifting its images tiles R^d and gives coefficient

    a_n=(1/(L^d S_(d,L))) integral_Rd
                   exp(-|z|^2/2) exp(-i omega n.z) dz
        =(2pi)^(d/2)/(L^d S_(d,L)) exp(-omega^2 |n|^2/2).

The one-dimensional Gaussian Fourier integral follows by integration by
parts: its derivative in frequency t is -t times the integral, and its
value at zero is sqrt(2pi). The product gives the displayed d-dimensional
integral. The resulting Fourier coefficients are absolutely summable;
their smooth series equals the image sum by Fourier uniqueness for a
continuous periodic function. This last elementary Fourier uniqueness
fact is the only Fourier-series import used for identification. Evaluating
at zero gives sum a_n=1 and therefore a_n=q_n in (3).

Use one representative of each nonzero pair {n,-n}, and independent standard
real Gaussian coordinates C_0,C_n,S_n. The actual smooth version is

    F(z)=sqrt(q_0) C_0+sum_pairs sqrt(2q_n)
                   [C_n cos(omega n.z)+S_n sin(omega n.z)]. (5)

For every nonnegative integer q,

    A_q:=sqrt(q_0)+sum_pairs sqrt(2q_n)(1+omega|n|)^q<infinity.

A fixed coordinate Cq norm is bounded by the corresponding positive
weighted sum of |C_0|, |C_n|+|S_n|, up to a dimension/q constant.
That sum has finite expectation, hence is finite almost surely. Minkowski
gives its Lp norm at most a constant times A_q ||N(0,1)||Lp for every
finite p>=1; lower positive moments follow too. Intersecting these
probability-one events over integer q gives absolute uniform convergence
of every derivative series. Thus F is smooth almost surely and

    E ||F||Cq^p<infinity for each integer q>=0 and finite p>0. (6)

The expansion is a theoretical representation of the exact law. No actual
random coordinates are drawn. No coefficient vector is asserted to lie
in unweighted l2 almost surely; the summable weighted norms are what the
smoothness proof uses. Its covariance is (2), so centered Gaussian
uniqueness identifies the law on the Borel space of smooth functions.

## 3. Named ranks by explicit finite trigonometric interpolation

Here is a concrete proof for precisely the needed rows, rather than an
unproved arbitrary-jet premise. At a torus site a put

    theta_a(z)=sum_(j=1,...,d) [1-cos(omega(z_j-a_j))].

It is positive at every other torus site, vanishes to order two at a,
and has coordinate frequency degree one. Thus theta_a^2 kills every
derivative through order three at a and has coordinate degree two.
At a fixed site p, the coordinates sin(omega(z_j-p_j))/omega have
derivative I. A polynomial of degree at most three in these local
coordinates realizes any specified third-order Taylor jet at p, by
Taylor expansion through that local diffeomorphism. It lies in K3.
This proves independence of the contact rows, including all independent
transverse Hessian entries, at a single point in any rotated frame.

For the two-site first jets at p!=q, take B_p=theta_q^2 and the d+1
polynomials

    h_p,0=B_p,
    h_p,j=B_p sin(omega(z_j-p_j))/omega, j=1,...,d.

At p the value/gradient matrix is triangular with diagonal B_p(p)>0:
h_p,0 has value B_p(p), the remaining values vanish, and the remaining
gradient columns are B_p(p)e_j. These polynomials kill first jets at q
and lie in K3. The symmetric construction at q proves full two-site
value/gradient rank.

For the contact at x and one offsite p, use B_p=theta_x^2 in the same
construction. It kills the full third-order contact jet, a stronger
condition than U_0 h=0. It gives arbitrary value/gradient variations
at p in K3. For x,p,q distinct, use

    B_p=theta_x^2 theta_q^2, B_q=theta_x^2 theta_p^2.

Both have coordinate degree four. Multiplication by the displayed sine
raises it to at most five. They give independent value/gradient maps
at p and q, kill the contact jet, and kill the other offsite first
jet. Thus the fixed block {-5,...,5}^d suffices for every needed
three-site residual row. Combining these residual variations with the
K3 contact directions gives the full contact-plus-offsite joint ranks.
This is an exact support calculation, not a numerical minor check;
no minimality of cutoff five is claimed.

Any nonzero linear combination of the specific independent jet rows
can therefore be detected by a finite trigonometric polynomial. Under
(5), every mode of that polynomial has strictly positive variance.
If the variance of the combination vanished, it would annihilate every
Fourier mode, hence that polynomial, a contradiction. This proves the
following actual named positive covariance statements:

* The original two-site values and gradients O_(p,q), 2d+2 rows, at p!=q.
* The contact U_0=(F,F_u,F_uu,F_uuu,(F_vi,F_uvi)_i), 2d+2 rows.
* U_0 enlarged by the independent symmetric entries of the transverse
  block A. None duplicates the axial Hessian column.
* U_0 together with (F(p),grad F(p)) when p!=x.
* U_0 together with values and gradients at p,q when x,p,q are distinct.

Their contact covariance is unconditional. After conditioning U_0, its
coordinates are deterministic; only the remaining Schur complements
are asserted positive. Independent symmetric Hessian entries are used,
not a list of d^2 allegedly independent entries.

For the finite block as well as the full field, these positive ranks
hold at every specified tuple. Continuity yields positive floors only
on compact domains with their declared separations. No floor is assumed
as an additional site approaches contact or two sites approach each other.

## 4. Infinite-field regression and the contact moments/envelope

Use the literal U_r, original observations, target v_r and row order of
ST Sections 3-4. Put Sigma_r=Cov(U_r),
c_r(z)=Cov(F(z),U_r), and on one actual full-field Gaussian probability
space define

    F_r(z)=F(z)+c_r(z) Sigma_r^(-1)(v_r-U_r F).             (7)

Its observation is exactly U_r F_r=v_r. The residual
F-c_r Sigma_r^-1 U_r F is independent of U_r F, since it is Gaussian
and uncorrelated with it. Independence on a countable dense evaluation
set and continuity give independence of the whole smooth field. Hence
(7) is the genuine canonical conditional field law Q_r for every target,
not a conditioning on a positive-probability pin event.

The integral divided-difference formulas express the fourth row as
an average of F_uuu and the other divided rows as averages of derivatives
of order at most two. Thus

    U_r F -> U_0 F in every finite Lp and almost surely,
    sup_(x,R,0<r<=1) |U_r F|<=C ||F||C3,

with an error bounded by Cr||F||C4 for the row convergence. Covariances
therefore converge uniformly in frames. Cross-covariances c_r converge
in global Cq for any fixed q, using (6) and differentiation under the
summable series. The named contact rank and compactness of O(d) give
Sigma_0>=lambda I for some lambda>0 and Sigma_r>=lambda I/2 for a
common sufficiently small r_0. Upper bounds hold as well. The stationarity
removes x from the eigenvalue question; no rotation invariance is invoked.

Consequently, at each fixed (x,R,b,k), (7) converges globally in C4 to

    F_0=F+c_0 Sigma_0^-1(v_0-U_0 F),
    v_0=(b,0,0,12k,0,...,0).                               (8)

It has exactly that contact row target. Its conditional transverse
covariance is strictly positive on all independent symmetric entries,
by the enlarged rank in Section 3. Thus A_0 has a full-support Gaussian
law under Q_0, even when its b-conditioned mean is nonzero. In particular
z_0=36k^2 E[(det A_0)^2 1(A_0<0)] is positive for every fixed k>0.

The convergence is also valid along varying parameter sequences on a
compact birth/gap/frame domain, by the same formula and continuous
matrices. It is a coupling of the same actual infinite field, not a
finiteN convergence claim. For every q the deterministic covariance
functions and inverse matrices are bounded, giving the pathwise bound

    ||F_r||Cq<=||F||Cq+C||F||C3+C(|b|+k).                 (9)

The random part on the right has all finite moments by (6). ST's exact
physical determinant identity gives W_r/r^2<=C(1+||F_r||C3)^(2d).
On compact marks this is dominated by one polynomial of the actual
unconditional field norm. Its determinant/type limit is
36k^2(det A_0)^2 1(A_0<0), with the saddle inertia mark d-1. Dominated
convergence supplies the full normalizer limit E[W_r/r^2]->z_0.
The correct congruence is diag(r^(-1/2),I), not diag(sqrt(r),I).

For all b in R,k>0, (9) supplies the polynomial conditional moment bound
C(1+|b|+k)^(2d). The Gaussian contact density pi_r at its exact target
is bounded by C exp[-c(b^2+k^2)]: Sigma_r has upper and lower floors,
and the target's first/fourth entries are b-k r^3/2 and 12k, uniformly
coercive in (b,k). Therefore

    12 pi_r(v_r) E_Qr[W_r/r^2]
      <=C(1+|b|+k)^(2d) exp[-c(b^2+k^2)],
    integral_Rx(0,infinity) k^(-2/3) RHS db dk<infinity.    (10)

This discharges CONTACT and ENVELOPE for the actual infinite law.
It needs neither inverse Hessian moments nor a global inverse normalizer.
Continuity and positivity of z_0 give a compact-mark minimum; they do
not give an all-mark minimum. Density convergence and conditional kernels
are the explicit canonical Gaussian versions from (7)-(8), compatible
with the original observation versions used next.

## 5. COUNT by finite incidence conditioned on the smooth tail

We spell out the infinite-to-finite step; finiteN proofs are not applied
to the full coefficient sequence. Split (5) into the fixed frequency
block K5, with independent standard coefficients xi_B in R^N, and its
independent smooth Gaussian tail T. For almost every tail realization,
T is a smooth deterministic function by Section 2.
The block retains the original weights q_n in (3); it is not renormalized
to variance one or identified with a catalogue finite field.

The finite block alone has every named rank in Section 3. In particular,
its point-gradient and pair-gradient maps have ranks d and 2d. On
local spatial charts choose an invertible d- or 2d-column coefficient
minor. By continuity the same minor works on a neighborhood. There
are finitely many possible minors and a countable spatial chart cover;
partition overlapping neighborhoods into disjoint Borel pieces.

For the point-gradient equation and a fixed smooth tail, write xi_B=(a,eta)
where a consists of the d selected coefficient coordinates. The equation
solves exactly

    a(p,eta)=-B(p)^(-1)[grad T(p)+C(p)eta].

The incidence parametrization (p,eta)->(a(p,eta),eta) is a smooth map
between N-dimensional manifolds. At a critical point its derivative is
singular exactly when the physical Hessian is singular (local metric
conversion factors are invertible). Equal-dimensional smooth Sard makes
the critical values in coefficient space null. The Gaussian coefficient
law has a Lebesgue density; conditioning on the smooth tail and then
integrating it proves F is Morse almost surely, before any Rice argument.
Compactness gives a finite critical set.

For a pair t=(p,q), choose 2d coefficient coordinates instead. The
derivative of the pair-gradient with respect to t at a zero is block
diagonal (H_p,H_q). The coefficient projection Jacobian is
|det H_p det H_q|/|det B(t)|, with the spatial coordinate-volume factors.
The equal-dimensional weighted area formula applied to this smooth
incidence map gives the critical-root sum for every nonnegative Borel
mark of the complete field T+F_B. Integrate the remaining finite
coordinates and the tail with nonnegative Tonelli; the chosen minor's
coefficient Jacobian gives the gradient density and conditional law.
Sum the disjoint minor/chart pieces. Thus the all-Borel formula is

    E sum_(p!=q,grad F(p)=grad F(q)=0) w(p,q,F)
      =integral_(p!=q) p_(grad F(p),grad F(q))(0)
          E[|det H_p det H_q| w | both gradients=0] dp dq. (11)

Both sides start as extended nonnegative integrals. Smooth area formula
is applied on countably many relatively compact parameter domains;
their union gives the full domain by monotone convergence. No global
Lipschitz claim for a chart on unbounded coefficient space is needed.

Here and below the conditional versions are fixed by exact Gaussian
regression. In the finite-tail proof, the coefficient change-of-variables
fiber integral is defined at every gradient target, using its nonsingular
linear map. The conditional tail has its explicit Bayes density, and
the finite block has its exact Gaussian fiber law. For any finite field
evaluation vector, integrating that fiber is ordinary finite joint
Gaussian disintegration, giving the actual Gaussian regression distribution
at every target. Gaussian uniqueness on a countable dense set of
field evaluations identifies the whole smooth-field kernel. Therefore
(11) uses that canonical version at zero, without extending a continuous
mark formula to Borel marks or inferring a value on a null fiber by
continuity. Arbitrary global Borel marks are already allowed by area
formula and Tonelli.

On each compact annulus away from the diagonal, covariance floors and
conditional determinant moments make the unmarked integral finite.
The original value/gradient observations have full covariance rank, so
the two heights given zero gradients have an actual nondegenerate
Gaussian density. The mark 1{F(p)=F(q)} therefore has zero contribution
in (11). Countable annulus exhaustion proves distinct critical values
almost surely. The single-point area formula similarly gives a finite
expected number of critical points: its gradient covariance has a
uniform positive floor and conditional Hessian moments are bounded.

ST Section 2 supplies the intrinsic everywhere Borel elder selector
sigma and actual H0 once-counting on this unconditional Morse/distinct
locus. Apply (11) with the type and selector mark. Delta appears once.
Gaussian two-height disintegration gives the actual selected density
kernel and hence COUNT. No full two-site Hessian rank is needed, even
though stronger named finite-jet ranks are available for this law.

## 6. Contact-conditioned global genericity and actual SELECT

Fix x and R, b in R, and k>0; condition the actual law on U_0=v_0. This law has a
forced degenerate critical contact at x. It is not globally Morse.
We must prove that other critical points are Morse with distinct values
different from b; unconditional genericity alone does not prove this.

Use the same finite block/tail split. The block contact covariance
Sigma_B=Cov(U_0 F_B) is positive definite. Under Q_0, the conditional
tail distribution has density

    dmu_tail,v(t)/dmu_tail(t)
       =p_(U_0 F_B)(v_0-U_0 t)/p_(U_0 F)(v_0).            (12)

The denominator is strictly positive. This is Bayes' formula for a
finite Gaussian observation of independent block and tail. It integrates
to one and is absolutely continuous with respect to the original smooth
tail law; hence the conditional tail remains smooth almost surely.

Given that tail, the block coefficients are Gaussian on the exact affine
fiber U_0 F_B=v_0-U_0 t. Choose an orthonormal basis for its finite
coefficient kernel. The free coordinates are independent standard
Gaussians; their covariance projection is independent of the target
and tail. Their field directions are exactly the finite trigonometric
polynomials in the block with U_0 h=0.

Section 3's interpolation proves the required residual ranks directly:
at p!=x the residual map to (F(p),grad F(p)) is surjective; at distinct
x,p,q the residual map to both values and gradients at p,q is surjective.
Indeed the displayed theta-polynomial directions give those first jets
while prescribing zero full third-order jet at x, a stronger condition
than U_0 h=0. The polynomials stay in K5. Thus all these conditional
covariances are positive, independent
of the smooth tail's deterministic mean. No K2 exceptional mean argument
or hypothetical arbitrary-jet premise is transferred.

Apply the finite free-coordinate incidence/Sard construction on X minus
{x}, with that fixed tail. Countable compact chart exhaustion proves
all other critical points Morse for almost every free coefficient vector.
Integrate the conditional tail law (12). This gives the contact-law
Morse conclusion at every fixed target v_0.

The same finite free-coordinate area formula is valid for all Borel
marks on single and paired critical points away from x. Residual value
rank given residual gradients makes the single-point mark F(p)=b have
zero expected count on each compact domain away from x. Residual two-height
rank given the two gradients makes F(p)=F(q) have zero expected count
when x,p,q are distinct. Exhaust by countably many domains separated
from x and from the pair diagonal. Thus no other critical value equals
b, and other critical values are pairwise distinct almost surely.
There is no simultaneous assertion over all targets; these conclusions
hold for each specified fixed contact law, which is what the limit uses.

The enlarged contact rank also gives a full-density Gaussian A_0, so
det A_0!=0 almost surely. With an invertible transverse block, the local
implicit ridge has phi_0'=phi_0''=0 and phi_0'''=12k>0 at contact;
it has no other nearby critical point. This isolation argument uses
invertibility only; strict concavity is needed later on A_0<0. Once
contact is isolated, the remaining compact critical set consists of
isolated Morse points and is finite. Their heights have finite nonzero
gaps from one another and from b. This completes the precise
contact-conditioned global genericity premise.

On positive limit weight A_0<0, the actual coupled fields (7)-(8) meet
ST Section 4's deterministic vector-ridge elder theorem: exact pins,
global C4 convergence, finite other Morse/distinct critical branches,
and the older positive ridge. It follows that the actual pair is
eventually the finite ordinary H0 elder pair and F_r is globally Morse
with distinct values along that coupled limit.

Put V_r=W_r/r^2. On V_0>0, sigma_r is eventually one. On V_0=0 the
weighted error V_r(1-sigma_r) tends to zero regardless of the selector.
The actual polynomial field-norm envelope from (9) gives

    E_Qr[V_r(1-sigma_r)] -> 0,
    E_Qr[V_r sigma_r] -> z_0>0,
    Q_r^W(sigma_r=1) -> 1.                               (13)

This discharges SELECT for every fixed mark/frame. No inverse random
curvature or all-mark inverse normalizer is used. Continuous varying
parameters, common compact-mark moment domination, and contact genericity
at one fixed limiting parameter give qualitative compact uniformity by
the usual contradiction subsequence argument. Since z_0 is continuous
and positive on that compact domain it has a positive minimum, allowing
normalized uniformity there. Pointwise genericity is not turned into a
single samplewise event for uncountably many parameters. There is no
rate such as Cr^3, all-mark normalized uniformity or common physical
chart radius in this preparation.

## 7. FAR and direct all-mark lifetime composition

On the compact set dist(p,q)>=r_0 the original full value/gradient
covariance has a positive floor by Section 3 and continuity. Gaussian
regression there, by the same full-field formula as (7), gives conditional
Cq moments growing polynomially in |b| for target (b,0,b-ell,0),
0<ell<=1. Its density is at most C exp(-c b^2). Therefore

    integral_R p_O(b,0,b-ell,0)
              E_Q[|det H_p det H_q|] db <=C,
    0<=nu_bar,far(ell)<=nu_cand,far(ell)<=C.                (14)

For each other compact positive-lifetime band the same reasoning gives
a finite local density bound. This is direct height disintegration of
the compact spatial complement. It does not bound a remote population
conditional on a rare cap failure. It discharges FAR for the canonical
selected kernel representative from COUNT.

In the near band the actual original observation transform has absolute
determinant 12r^(-(d+3)), exactly as in P/ST. Write

    B_r=12 pi_r(v_r) E_Qr[V_r sigma_r].

The spatial, height, density and determinant factors are

    r^(d-1) dr, r^3 db dk, 12r^(-(d+3)) pi_r, r^2 E[V_r sigma_r],

and give r B_r dr db dk d sigma. The torus midpoint Jacobian is exactly
one in the embedded short-pair band; stationarity cancels L^d. Hence

    ell^(1/3) nu_bar,near(ell)
       =integral 1{k>=ell/r_0^3}
          B_((ell/k)^(1/3))(b,k,u)/(3k^(2/3)) db dk d sigma(u). (15)

Canonical versions and a direct fixed-ell radial substitution make (15)
a pointwise formula for the same specified density. The unnormalized
majorant (10) controls every birth and positive gap mark, including
k->0 and unbounded k. Equations (8),(13) give B_r->12pi_0(v_0)z_0.
Dominated convergence therefore gives

    c_(d,L)=4 integral_(R,(0,infinity),S^(d-1))
                    k^(-2/3) pi_0(v_0) z_0 db dk d sigma. (16)

It is positive and finite by the full-support transverse Gaussian and
the envelope. Equation (14) vanishes after multiplication by ell^(1/3),
proving (4). The Borel selector counts all bars, not only local candidates;
the essential class is excluded. The full sphere counts each ordered
displacement once, with no factor 1/2. Cumulative integration gives
the factor 3/2. Only a finite Borel density representative, not selected
continuity or a near O(1) remainder, is obtained. The missed near
candidate density is o(ell^-1/3), with no sharper rate here.

## 8. Exact covariance-derived coefficient and reference-moment refusal

For any actual derivatives L_a,L_b at the same point, the exact covariance
is obtained from (3) by the absolutely convergent spectral sum

    Cov(L_a F,L_b F)=sum_n q_n P_a(omega n) conjugate(P_b(omega n)),

where P is the actual derivative multiplier. Equivalently differentiate
the normalized image sum (2). These are identical exact laws.
For instance

    Cov(G_i,G_j)=omega^2 sum q_n n_i n_j,
    Cov(H_ij,H_kl)=omega^4 sum q_n n_i n_j n_k n_l,
    Cov(F_uuu,G_j)=-omega^4 sum q_n (n.u)^3 n_j,
    Var(F_uuu)=omega^6 sum q_n (n.u)^6.                    (17)

No value is replaced by its unperiodized reference moment. In particular
the independent block is odd (G,F_uuu), while the even block is
(F,V_u=Hess F u,A_u). Gaussian independence follows from the actual
even stationary covariance, without rotational isotropy. Put

    tau_u^2=Var(F_uuu|G=0),
    D_u=E[(det A_u)^2 1(A_u<0)|V_u=0].                    (18)

The true full axial column V_u has d coordinates. Birth disintegration
removes F=b but retains V_u/A_u correlations. The correct half-Gaussian
moment is 2^(-1/3) Gamma(7/6) tau^(4/3)/sqrt(pi). Combining its
12^(-1/3) multiplier gives the exact expression

    c_(d,L)=Gamma(7/6)/(24^(1/3) sqrt(pi))
       *integral_(S^(d-1)) p_G(0) p_Vu(0) tau_u^(4/3)
                                             D_u d sigma(u). (19)

Every covariance/regression/cone expectation is from (2)-(3). Enlarged
contact rank gives full conditional transverse support, so D_u>0;
compact frame continuity makes the angular integral finite. Equation
(19) is algebraically P (15.2), the expression S24 evaluates. This
identification is not a new verification of S24 arithmetic, endpoint
intervals, review acceptance or compiled formal semantics. The numeral
24 in the prefactor is the scalar Jacobian/gamma factor, not L.

One can prove a strict diagnostic at L24 without numerical calculation.
Because the image sum factorizes, write its one-dimensional normalized
factor as k_L(t). Let m_2=-k_L''(0), m_4=k_L^(4)(0),
m_6=-k_L^(6)(0), and S_1=sum_j exp(-L^2 j^2/2). For L=24,

    m_2=1-[sum_(j!=0) L^2 j^2 exp(-L^2 j^2/2)]/S_1 <1,
    m_4=3+[sum_(j!=0) L^2 j^2(L^2 j^2-6)
                                      exp(-L^2 j^2/2)]/S_1 >3,
    m_6=15-[sum_(j!=0) L^2 j^2((L^2 j^2)^2-15L^2 j^2+45)
                                      exp(-L^2 j^2/2)]/S_1 <15.

All summands giving the strict signs are positive at L24; positive
spectral rank also gives m_2>0. On a coordinate axis u=e1,
Cov(G)=m_2 I and Cov(F_uuu,G)=(-m_4,0,...,0). Consequently

    0<tau_e1^2=m_6-m_4^2/m_2<6.                         (20)

In d=2, for transverse A=F_22, the actual full column V=(F_11,F_12)
has covariance diag(m_4,m_2^2) and Cov(A,V)=(m_2^2,0). Thus

    Var(A|V=0)=m_4-m_2^4/m_4>8/3.                       (21)

So neither tau^2=6 nor Var(A|V=0)=8/3 is an exact periodic equality.
Those are S24 Section 1's unperiodized reference values. The reference
Rayleigh-cone simplification also cannot simply be substituted in d=3:
at u=e1 for A=[[s+x,y],[y,s-x]], its centered conditional law has
Var(x)=(m_4-m_2^2)/2 while Var(y)=m_2^2, which are unequal at L24.
This is an exact law diagnostic, not a numerical enclosure or new
claim about the size of the already bounded periodization error.

## 9. Current consumed sources, historical identities and residual scope

The current structural dependency is
[the regular-fold theorem](../fold_structural_universality/PROOF.md),
43927 UTF-8 bytes / 804 lines,
SHA2563d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531. Its mathematical Sections1-10 are byte-identical to corrected
ST43761B/813lines SHAb077e2465174cadf78f2df120a80992534f99b838e18b5ba15fed250f4c28cca,
which the original adapter author fully read. Root has read the complete original
adapter and actual incorporated dependencies; new wrapper/source bindings have
separately recorded review. The original ST79e8 error and full AMEND, corrected
delta PASS and fresh full corrected-source PASS remain distinct historical records.

The actual law source is [SIDE24](../../../formal/sources/side24_v1/PROOF.md),
10272B/216lines SHAc06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769,
Git blob44b66f04f89fcd87383b3603fa69f1feb64cdddd. Its complete actual text was
freshly read by the original author and root. This unchanged source was first
bound at rootproject6541afbdc93114b8cd35c40e77605e8491214330; its identity also
matches the initial incorporated base2c8417. SIDE24's expression/enclosure scope
is separate from the new actual-bar proof. Its old arithmetic and parent
quantitative assertions are not reaccepted here.

Other bound original inputs are P, finite-contact-parent-source.md,
40261B/494lines SHA9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7;
[the candidate source C](../finite_candidate_lifetime/PROOF.md),29190B/577lines
SHAed31f562f0ffaaf6d4bd2ceb1ae3d234798e983a55d55c6983b128fce27133f5;
and original ET21138B/418lines SHA1cac9b525831b7e1f37f0d0561d3231fa73c8713c4c92283dca1d5b51b8a67ce.
P had a prior complete actual read and its model/coefficient/scope sections were
reread. C supplied the route rederived with the actual smooth tail here. ST's
higher-dimensional vector-ridge argument supplies the consumed adaptation;
ET's original finite two-dimensional law is not automatically transferred.
The current [ET successor](../finite_elder_transfer/PROOF.md)c9e4 preserves its
mathematical Sections2-13, with the separately bound whole-file review recorded
there. External preparation names in this paragraph are custody identities,
not broken repository links or new public uploads.

The complete original adapter34281B/639lines
SHAcd0abd19cfb0bdd398142c136729d3bf96724939b3e774a3773a683310078a7f received
final_contract_review fresh full639line PASS after root's full read. The reviewer
rehashed ST,S24,P,C,ET and independently matched the actual S24 blob. Its ST
consumption was the earlier full original plus complete corrected delta, rather
than a second full corrected-file read. It additionally reread P's model/final
coefficient/scope and fully read both measurement/manuscript companions. No fresh
primary classical fetch, compiler, test or sampler run is attributed to that
mathematical review. The present bytes receive their own verdict in REVIEW.md.

The measurement companions are
[SIMULATION_PROTOCOL](../../../docs/SIMULATION_PROTOCOL.md),2529B/35lines
SHA7d69e479612805a9fe024f575b8feb8752eac457075d356997937db8690c281c,
and [PAPER1_NOTE](../../../docs/PAPER1_NOTE.md),6322B/69lines
SHA161ba87433879ee77ade047f1516d8c919fa3097812577cac54d7fe978a7ad63.
They were completely read by author, root and the original adapter reviewer.
They identify the exact field and preserve manuscript/remainder/experiment
boundaries; they are not new premises supplying the five analytic interfaces.

Named classical imports are equal-dimensional smooth Sard, weighted area formula
on smooth incidence maps for nonnegative Borel integrands, smooth Morse local
form, critical-free level flow and isolated H0 handle attachment. Gaussian finite
linear regression, Tonelli/Fubini,Minkowski,DCT and continuous periodic Fourier
uniqueness are used explicitly. C/ET record earlier bounded primary statements;
no stronger full-field Borel theorem or full-book custody is attributed to them.
The direct finite-tail argument supplies the all-Borel formulas11-12.

The analytic theorem concerns each fixed d>=2,L>0 and the exact infinite
periodized law, including conditional nonlocal genericity. It supplies a
particular Borel density and positive finite coefficient, with no quantitative
selection rate, unrestricted near O(1) remainder, continuity, coefficient digits,
usable numerical window or finite-word/grid admission. Higher homology, spatial
expanding-domain/regional-witness limits, blind-field validation, formal closure
and publication uptake remain separate. No scientific status changes here.
