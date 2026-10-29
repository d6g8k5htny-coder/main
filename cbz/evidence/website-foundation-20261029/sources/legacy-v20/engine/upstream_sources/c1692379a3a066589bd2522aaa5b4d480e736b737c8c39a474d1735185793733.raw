# C6 Fourier cutoff: an exponential planar count cap and a single logarithm

Object: OA-C6-FOURIER-20260929-v1.
Author: OpenAI / ChatGPT, foreground continuation, 29 September 2026.
Disposition: AUTHOR-SIDE MATHEMATICAL CANDIDATE; nonauthor analytic review required.
Scientific effect NONE. No existing proof, review disposition, register, graph,
lemma flag, premise, prize or numerical certificate is changed.

## 1. Model, results and exact logical scope

Use the ORIGINAL normalized periodized Gaussian field on X=R^d/(T Z^d), for a
FIXED d>=2 and T>0. Its covariance is the normalized sum of
exp(-|x-y+Tn|^2/2). Put omega=2pi/T. The Fourier variances are

    a_n = c_T exp(-omega^2 |n|^2/2), n in Z^d, a_n>0.

Real-valuedness imposes conjugacy of the complex Fourier coefficients. No
independence of their conditional laws will be used. All orthonormal original
pin frames are allowed; the lattice used for Fourier truncation is FIXED in the
torus coordinates and is NOT rotated with that pin frame.

Retain the exact original value/gradient observations at M=-(r/2)u and S=(r/2)u,
with f(M)=b, f(S)=b-kr^3 and both gradients zero. Let Q_r be their continuous
Gaussian regression, equivalently the bounded-target nonsingular frame U_r=v_r
of LP. Birth and positive-gap marks range over the original compact sets, with
k>=k_->0. Use the full typed endpoint weight W_r and full Z_r=E_Qr W_r:

    dQ_r^W=(W_r/Z_r)dQ_r.

N counts critical points other than M,S with values in I_r=(b-kr^3,b), all
indices. C(f) counts ALL real critical points on the torus, including the pins
and all heights. The two are not interchangeable. Constants depend on fixed
d,T and compact parameter ranges; no growing-dimension, k->0 or numerical
uniformity is claimed. Reduce r_* as needed within the parent validity regime.

**Theorem F (count cap, no G_d input).** There is a measurable field functional
Psi_F>=C(f) such that uniformly in this original pin family

    Q_r(Psi_F>x) <= C exp(-c x^(2/d)),
    Q_r^W(Psi_F>x) <= C exp(-c x^(2/d)),                         (F1)

for x>=1, after changing constants. Thus all its fixed moments are uniformly
finite under either law. In particular, for some a>0,

    sup E_Qr^W exp(a Psi_F^(2/d)) < infinity.                   (F2)

The same statements hold for the unconditioned field, without the pin/tilt
steps. In d=2 this is a uniform EXPONENTIAL tail for the total critical count,
not just a finite collection of moments.

**Theorem W (window consequence).** Let G_d denote the dimension-matched
hypothesis E_Qr^W N<=C r^3. Given G_d, for every fixed integer p>=2,

    E_Qr^W (N)_p <= C_p r^3 [log(1/r)]^(d(p-1)/2).             (F3)

For d=2, G_2 is the reviewed planar I5 composition under the complete D5
reading rule, including its actual full-depth P2 continuum review. Hence

    E_Qr^W N(N-1) <= C r^3 log(1/r),                           (F4)
    E_Qr^W (N)_p <= C_p r^3 [log(1/r)]^(p-1).

This is a NEW candidate improvement over the integrated Taylor-radius bound
r^3[log(1/r)/log log(1/r)]^2. For d>=3 only (F1)-(F2) are asserted without G_d;
the factorial composition remains explicitly conditional on that still-separate
first-moment input. No assertion that #141 or #145 is accepted is made here.

**Proposition B (an inference boundary, not a Gaussian counterexample).** The
factor [log(1/r)]^(d(p-1)/2) cannot in general be removed using only G_d, a
nonnegative count cap, and the tail/moment statements (F1)-(F2). Section 8 gives
exact rare-count variables attaining that factor. This does not prove that the
factor is optimal for this Gaussian field and does not refute the Palm route.

## 2. Inputs, including their conditioning boundaries

LP Sections 3-5 provide the uniform covariance sandwich for U_r, bounded v_r,

    sup v_r^T Cov(U_r)^(-1)v_r < infinity,
    ||W_r||_(L2(Q_r)) <= C r^2,       Z_r>=c r^2.              (F5)

The matrix/moment facts are consumed in the original law; no witness is added.
C6L supplies the compact-analytic-set/holomorphic-degree argument and the
protected covering of the torus. C6S Section 4 explicitly extends its
polynomial-symbol independence and complex small-ball density argument to
fixed d. Section 4 below restates why the needed density estimate holds; no
uniform lower covariance bound is asserted at a real pinned critical point.
I5 and D5R supply only the planar small-window mean for (F4). Their original
source and reviewer identities are in SOURCE_MAP.json. Historical narrower
algebra-only HOLDs are preserved, not counted as full analytic acceptance.

The new ingredients are a cutoff of the FULL conditioned field in its original
Fourier coordinates, uniform Gaussian-frequency error bounds, and conversion
of each truncated gradient to a Laurent system. The Taylor truncation tail of
C6S and the new witness-conditioning claims of #145 are not inputs.

## 3. Uniform Fourier error AFTER conditioning

Let c_n(f) be the complex Fourier coefficient of a realization. Define

    f_[m](z)=sum_(|n|_infinity<=m) c_n(f) exp(i omega n.z),
    P_m(z)=grad f_[m](z),       F(z)=grad f(z).                (F6)

This is the Fourier truncation of f drawn under Q_r, not the prior field
conditioned on a truncated set of observations. f_[m] generally does NOT satisfy
the original pin equations; no argument below requires it to do so. F, N, W
and Z always refer to the original, exactly pinned field.

For a real standardized prior Fourier coordinate xi, Gaussian regression gives

    |E[xi | U_r=v_r]|^2 <= v_r^T Cov(U_r)^(-1)v_r,
    Var(xi | U_r=v_r) <= 1.                                  (F7)

The first is covariance Cauchy-Schwarz after whitening U_r, and the second is
variance reduction by orthogonal projection. Thus, for every fixed s>=1,

    ||c_n||_(Ls(Q_r)) <= C_s sqrt(a_n),                       (F8)

uniformly in m,r and marks/frames, with an immaterial factor for real and
imaginary components. The coefficients need NOT be conditionally independent.
This bound includes the conditional mean, not just the centered residual.

Fix a complex tube |Im z|<=Y, with real parts taken modulo the torus. For every
fixed derivative order j, Minkowski and absolute convergence give

    ||sup |D^j(f-f_[m])|||_(Ls(Q_r))
      <= C_s sum_(|n|_infinity>m) (1+|n|)^j
                         exp(-omega^2 |n|^2/4 + omega Y |n|).

To bound this tail put A=omega^2/4 and B=omega Y. Since
Bv <= A v^2/2+B^2/(2A), and |n|^2>m^2 in the omitted set,

    exp(-A|n|^2+B|n|)
       <= exp(B^2/(2A)) exp(-A m^2/4) exp(-A|n|^2/4).

The remaining polynomial-weighted lattice sum is finite. In particular, there
is a fixed a>0, chosen smaller if necessary, such that for the fixed tube used
by the cover,

    R_m := sup |F-P_m|,
    E_Qr R_m^2 <= C exp(-4a m^2),
    E_Qr [sup ||DF||]^2 <= C.                                (F9)

The constants absorb the finitely many small cutoffs. This also proves uniform
absolute convergence of the relevant complex derivative series. Differentiating
a finite Fourier sum and taking its complex analytic extension commute. No
Cauchy radius tending to infinity is needed.

## 4. Protected annuli and small-gradient volume

Choose 0<eta<T/20 and reduce r_*<=eta/2. Use one ball centered at the midpoint
with radius eta, and a finite eta/16-net of the complement |x|>=7eta/8 with
radii eta_j=eta/8; set eta_0=eta. Their real balls cover the torus. For every
centre c_j take fixed complex test radii rho in [eta_j,2eta_j], and write

    A'_j={eta_j/2<=|z-c_j|<=3eta_j}.

Its real slice stays at least eta/4 from the original pins. In each chart choose
real lifts of the centre. Distances near the torus seam are understood using
lifts; the local balls have diameter less than T and no aliasing occurs there.

For z=x+itu, t=|Im z|>0, define the REAL 2d-dimensional observation

    G(z)=(Re F(z), Im F(z)/t).

It extends continuously at t=0, with direction u retained, to (grad f(x),H_f(x)u).
For t>0 the evaluation symbols at z, conjugate z, and the real pin sites are
independent exponential-polynomial characters of Z^d. At t=0 the extra symbols
at remote x are i b.n and -(u.n)(c.n), independent because their linear and
quadratic parts cannot cancel and u is nonzero. At r=0 the pin frame consists
of its nonduplicated contact jets at the midpoint. Independence of characters
with polynomial coefficients follows by separation of the joint generalized
eigenspaces of the commuting lattice shifts. This is the C6L Lemma E/C6S
Section 4 argument, not an assumed isotropic GOE law.

The joint covariance of the pin frame and G is therefore positive at each
point of the compact annulus/mark/direction/frame parameter set, including the
contact limit. Continuity and the Schur complement give a UNIFORM covariance
floor for G under Q_r. Mean shifts cannot increase the Gaussian density supremum.
Consequently, uniformly on these annuli for 0<h<1,

    Q_r(|F(z)|<=3h) <= C min(1,h^(2d)/t^d).                  (F10)

At t=0 use the trivial bound one. Integrate first over imaginary parts in R^d.
Splitting at t=h^2 gives

    E_Qr Leb_(2d){z in A'_j: |F(z)|<=3h}
       <= C h^(2d)[1+log(1/h)].                              (F11)

The power of t is (d-1)-d=-1, explaining the logarithm in this intermediate
estimate. It does not force a final log-logarithm in the cutoff theorem.
This estimate is NOT asserted on arbitrary complex sets touching a real pin.

## 5. Laurent-polynomial zero cap, with no genericity shortcut

For cutoff m>=1, introduce w_l=exp(i omega z_l), l=1,...,d. Each component of
P_m is a Laurent polynomial with exponent support in [-m,m]^d. Multiplication
of each component by the nonvanishing monomial (w_1...w_d)^m clears the negative
exponents and gives an ordinary polynomial Q_(m,l) of total degree at most 2dm.
Extra zeros on coordinate hyperplanes w_l=0 are irrelevant to the original
complex torus but may be counted in an upper bound.

The exponential map is injective on every closed test ball of radius at most
2eta: equality of exponentials implies an original coordinate difference in
T Z^d; the distance between two points in the ball is at most 4eta<T. Its complex
Jacobian is invertible everywhere, so it is biholomorphic on a neighborhood of
each ball (enlarge slightly within the same injectivity radius).

Suppose on a test sphere

    |F-P_m| < |P_m|.                                         (F12)

P_m has no zero on the sphere. Its interior zero set is a compact analytic
subset of the open complex ball and hence finite; a positive-dimensional
compact analytic subset of C^d is impossible. Under the exponential map these
zeros become isolated common zeros of the d cleared polynomials. The isolated
zero form of Bezout bounds their multiplicity sum by

    product_l deg Q_(m,l) <= (2dm)^d.                         (F13)

This form permits other positive-dimensional components elsewhere. A system
with an identically zero component causes no exception: if there are interior
zeros, they cannot form an isolated d-equation complete intersection with fewer
than d nonzero equations; the same local dimension/compactness argument rules
out such zeros. No assumption of generic random Fourier coefficients is needed.

The homotopy P_m+t(F-P_m) has no boundary zero by (F12). Its real 2d-dimensional
Brouwer degree is constant. F also has no boundary zero and has a finite interior
analytic zero set; local holomorphic degrees equal positive multiplicities.
Thus F has at most (2dm)^d complex zeros in that ball, and in particular at most
that many real critical points. Laurent exponent-clearing is a bound on isolated
zeros, not a claim that a trigonometric system has no positive-dimensional set.

Define k_j as the smallest m>=1 for which (F12) holds on at least one test
sphere, or infinity if none works. Let J be the finite number of cover balls,

    Psi_F=(2d)^d sum_j k_j^d.                                (F14)

On the event all k_j are finite, C(f)<=Psi_F. The successful radius can vary
with j and f; all of them still cover the fixed real balls. Boundaries of one
cover ball need not carry any zero: a strict Rouché success excludes them.

Measurability is explicit. For fixed m the sphere minimum of
|P_m|-|F-P_m| is continuous in rho and in the compact analytic field norm.
Strict success at some rho in the closed interval can be detected at a rational
radius in its interior, by continuity (also at endpoint success). The event
{k_j<=m} is the UNION of success events for cutoffs 1,...,m, not just success at
cutoff m; monotonicity of individual Fourier tests is not assumed.

## 6. Gaussian cutoff tail and stretched-exponential count tail

Fix j and define h_m=exp(-a m^2), lambda_m=exp(a m^2/(4d)) and

    delta_m=h_m/lambda_m.

Let L_j be the supremum of ||DF|| on the fixed ball of radius 3eta_j. For large
m, delta_m<=eta_j/2. If k_j>m, the cutoff-m Rouché test fails at EVERY radius,
so each sphere contains z_rho with |P_m(z_rho)|<=|F-P_m|(z_rho). On
{R_m<=h_m,L_j<=lambda_m}, this gives |F(z_rho)|<=2h_m and |F|<=3h_m on the ball
of radius delta_m around it.

Take radii eta_j+2i delta_m within [eta_j,2eta_j]. There are at least
eta_j/(2delta_m) such radii. Their centres are separated by at least 2delta_m,
so the OPEN balls have disjoint interiors; touching boundaries have zero volume.
All lie in A'_j. Thus failure forces sublevel volume at least

    c_d eta_j delta_m^(2d-1).

Combining this lower volume with (F11) and Markov gives

    Q_r(k_j>m,R_m<=h_m,L_j<=lambda_m)
      <= C h_m lambda_m^(2d-1)(1+a m^2)
      = C(1+a m^2) exp[-a(2d+1)m^2/(4d)].                    (F15)

The other two events are bounded using (F9):

    Q_r(R_m>h_m) <= C exp(-2a m^2),
    Q_r(L_j>lambda_m) <= C exp[-a m^2/(2d)].                  (F16)

Absorb the fixed polynomial and finite small-cutoff range into constants:

    Q_r(k_j>m) <= C exp(-c m^2).                              (F17)

This proves k_j<infinity almost surely and so establishes the cap. If
Psi_F>x, some k_j exceeds (x/[(2d)^d J])^(1/d); integer floors change only
constants for x large. A finite union gives the Q_r part of (F1).

Under the ORIGINAL tilt, covariance Cauchy-Schwarz and (F5) give

    Q_r^W(A) <= ||W_r/Z_r||_2 Q_r(A)^(1/2) <= C Q_r(A)^(1/2).

The square root changes c, not the power x^(2/d). This proves the other part
of (F1). For Y=Psi_F^(2/d), integrate its exponential tail to obtain (F2) for
sufficiently small a. All constants are uniform only in the specified original
pin family. No further witness-conditioned cap bound has been proved.

## 7. Factorial moments: exactly one mean and a tail

For an integer p>=2, N<=Psi_F gives the pathwise inequality

    (N)_p <= N Psi_F^(p-1)
          <= Lambda^(p-1) N + Psi_F^p 1_{Psi_F>Lambda}.        (F18)

By Cauchy-Schwarz under Q_r^W, its uniformly bounded 2p-th cap moment and (F1),

    E_Qr^W[Psi_F^p 1_{Psi_F>Lambda}]
       <= C_p exp(-c Lambda^(2/d)).                          (F19)

Assume G_d. With L_r=log(1/r)>=1 choose Lambda=(A L_r)^(d/2), where A is a fixed
large constant making cA>=3. Then (F18)-(F19) give (F3). Each p has its own
finite constant; no uniform assertion for growing p is made. In d=2 insert
the dimension-matched planar I5, yielding (F4).

The planar conclusion is asymptotically stronger than the Taylor bound because
L_r/[L_r/log L_r]^2=(log L_r)^2/L_r ->0. In fixed general d the analogous ratio
is (log L_r)^d/L_r^(d/2)->0. Neither comparison makes a numerical radius promise.
G_d for d>=3 remains a separate analytic dependency and is NOT proved by (F1).

## 8. Why a marked estimate is still needed for O(r^3)

The following is an EXACT abstract construction, not a Gaussian field. Fix d>=2
and an integer m>=2. Set r_m=2^(-m^2), n_m=m^d, and let

    N_m=Psi_m=n_m with probability 2^(-3m^2)/n_m,
    N_m=Psi_m=0 otherwise.

Then E N_m=r_m^3 and for 0<=x<n_m,

    P(Psi_m>x)=2^(-3m^2)/n_m <= exp[-2(log 2) x^(2/d)].

Above n_m the tail is zero. Also E exp[(log 2)Psi_m^(2/d)]<=2 uniformly. For
fixed integer p>=2, eventually n_m>=p and

    E (N_m)_p = r_m^3 product_(j=1)^(p-1)(n_m-j)
               ~ r_m^3 m^(d(p-1))
               = constant * r_m^3[log(1/r_m)]^(d(p-1)/2).      (F20)

Thus even an exponential planar cap tail does not turn an unconditional small
mean into an O(r^3) factorial bound. The exact size-biased identity is

    E(N)_p = (E N) E_sizebiased[(N-1)_(p-1)],
    dP_sizebiased = N dQ_r^W / E N.

A uniform bound on that SIZE-BIASED moment, or an appropriate marked Kac-Rice
estimate, would be an additional input. This identifies the role of #145's
proposed marked argument, not a proof of it. The previous #146 obstruction for
the Taylor-cap tail is credited as motivation; (F20) is a new simpler exact
construction matched to the present Fourier-tail class. The true Gaussian C6
order may still be Theta(r^3).

## 9. Review and execution boundaries

New analytic targets: (i) the FULL conditioned Fourier coefficient bound,
including its mean; (ii) keeping truncation distinct from exact pin constraints;
(iii) exponential-coordinate injectivity and isolated-zero Bezout, even for
nongeneric coefficient systems; (iv) the two cutoffs and all three probabilities
in (F15)-(F16); (v) the correct dimension power and separate G_d dependency.

The finite checker verifies Laurent clearing and degree counts, rational Gaussian
regression inequalities, completing-square algebra, the probability exponents,
measurability's finite-union convention, and the exact inference-boundary
example. It does not establish Gaussian uniformity, complex analytic zero-set
facts, Kac-Rice or nonauthor acceptance. No hidden import or old missing runner
is used. The exact dependency identities are in SOURCE_MAP.json; local packet
replay does not claim a full upstream checkout, and the hosted workflow separately
checks the pinned upstream bytes.

The statement is a source-consumer extension, not a novelty/priority claim.
There is no new persistent-pair law, Poisson approximation, same-index refinement,
quantitative SARD rate, numerical C103 certificate, status promotion, or resumed
stopped process. References and the actual limited reconnaissance scope are in
RECONNAISSANCE.md.
