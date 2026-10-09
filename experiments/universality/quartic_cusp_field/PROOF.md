# Actual stationary quartic contact: sampling-dependent selected H0 law

Incorporated on 8 October 2026 by OpenAI/Codex root, task
01a11cbf-068d-7102-861e-75814e715c98, under exact GitHub307 amendment
6071510105 and the existing README claim. Mathematical author:
OpenAI/Codex universality_review. Initial detached base is published
amplitude draft334 head `6bd133d591e43e75a9292463bb4bfde7a34984dc`.
No main landing or model-catalogue admission is implied.

Sections1-11 preserve the original32,334-byte/695-line preparation,
SHA256 `24e6f4149c0a2df2f8df1dbfa8d66da3e01a7d12bee4b803bac093cfea36bf3f`,
exactly: 26,625 bytes/602 lines, SHA256
`9f1426a21b40b524ad4b25c81368489b0aa7a0c29f79659eb556bb4d1579f6d0`.
Both final_contract_review and benchmark_formal_audit have separate fresh
FULL original695-line mathematical PASS verdicts. Root fully read the
actual original1-350/351-695. These new wrapper/context/README bytes await
their own incorporated-source and metadata reviews; original verdicts are
not rebound. The [review record](REVIEW.md) retains source identities.

The self-contained construction gives an actual globally selected finite
H0 bar in a stationary smooth field, with quartic contact and exact
lifetime C*lambda^2. Sampling weight near zero determines its density
power. The path is restricted and only C1 in its parameter at zero;
no generic full two-parameter cusp or Gaussian contact law is admitted.
All performers are prior source/route-exposed OpenAI/Codex AI agents,
organizational independence0. No numerical, formal, blind or scientific
promotion follows from these conventional analytic results.

## 1. Exact claim and conventions

Fix an integer d>=2, a side length L>0, and the flat torus

    X=(R/LZ)^d,   V=L^d,   omega=2 pi/L.

Homology is ordinary superlevel persistent homology over an arbitrary
fixed field K. A finite H0 interval has a maximum birth value B and a
merging-saddle death value D, with lifetime ell=B-D>0. The component with
the larger maximum value survives a merger. One H0 class is essential
because X is connected. Essential intervals are excluded from all counts.

For any

    0<eta<2/(3 sqrt(3)),    gamma>0,                       (1)

we construct a deterministic family F_lambda, 0<=lambda<=lambda_0,
of C-infinity functions on X and a stationary random field Z. Its random
parameters are a uniform translation and an independent lambda with
density gamma lambda^(gamma-1)/lambda_0^gamma on (0,lambda_0).
The positive constant lambda_0 is chosen by explicit deterministic
inequalities in Section 5. Optional deterministic centering and variance
normalization are included in Z, and never change the pairing.

There are constants C>0 and ell_0=C lambda_0^2 such that each sample of Z
has exactly one finite H0 bar, and

    ell=C lambda^2,
    P(ell<=t)=(t/ell_0)^(gamma/2),       0<t<ell_0,
    nu_Z(t)=(gamma/2) ell_0^(-gamma/2)
                    t^(gamma/2-1) 1{0<t<ell_0}.          (2)

Here nu_Z is the whole-torus expected finite-H0 lifetime density. The
density per unit spatial volume is nu_Z/V. These are exact identities
on the stated interval, not only leading asymptotics.

Write beta=gamma/2. For independent whole-field copies, if

    t_n -> 0,       n t_n^beta -> Lambda in (0,infinity),  (3)

their finite-H0 points with lifetime mark ell/t_n and birth-location
mark x converge to a Poisson random measure with intensity

    Lambda beta ell_0^(-beta) a^(beta-1) da dx/V.          (4)

The exact direction from the birth maximum to its death saddle is the
positive first coordinate direction, so that mark can be added as a
Dirac mass at e_1. Section 10 proves the Laplace functional directly.
This is an independent-copies process, not a spatial expanding-domain
limit or a Poisson statement inside one field.

## 2. An explicit Morse seed and exact maximum coordinates

Let w_i=3^(-i), i=1,...,d, and define

    h(y)=sum_i w_i cos(omega y_i),       b=sum_i w_i.      (5)

The critical points have y_i in {0,L/2}. At each one the Hessian is
diagonal with nonzero entries -omega^2 w_i cos(omega y_i).
There are exactly 2^d critical points, all nondegenerate, and a unique
maximum at y=0. Their values are distinct. Indeed, if two sign patterns
differ first at index i, their value difference has leading magnitude
2w_i, while the absolute sum of all later differences is at most
2 sum_(j>i) w_j < w_i. In particular every other critical value is at
most b-Delta, where

    Delta=2 w_d>0.

Use representatives |y_i|<L/4 near the maximum and set

    z_i=sqrt(2w_i) sin(omega y_i/2),
    s=z_1,       t=(z_2,...,z_d).

This is a smooth product chart with positive diagonal differential.
It gives the exact identity

    h=b-s^2-|t|^2.                                      (6)

Choose R>0 with R^2<min(w_d,1). The chart contains the closed z-ball
of radius R, with a collar around it. More precisely, if b-h<R^2,
then each 2w_i sin^2(omega y_i/2)<R^2<w_i. In representatives
|y_i|<=L/2 this forces |y_i|<L/4. Hence the entire superlevel region
{h>b-R^2} is exactly the chart ball D_R={|z|<R}.
Outside D_R we have h<=b-R^2. No old critical point other than the
maximum belongs to its closure.

This seed removes the need to import a Morse lemma or to assert the
existence of a suitable global completion of a local polynomial.

## 3. A globally matched quartic surgery with no extra critical points

Set r_1=R/4. Choose a smooth nondecreasing function c:[0,infinity)->[0,1]
such that

    c(v)=v/4 for 0<=v<=r_1^2,
    c(v)=1   for v>=R^2/2,
    c(v)>0  for v>0.                                    (7)

Such a function exists with c'>=0. To see this without a matching
assumption, take a smooth nonnegative derivative d_0 equal to 1/4 on
[0,r_1^2], tapering to zero strictly before R^2/2. Its integral I_0 is
less than 1, since R^2<1. Add a nonnegative smooth bump, supported in
(r_1^2,R^2/2), with integral 1-I_0, and integrate their sum from 0.
The resulting c has exactly the properties (7).

Replace h on D_R by

    F_0=b-s^2 c(v)-|t|^2,       v=s^2+|t|^2,             (8)

and retain h elsewhere. On the collar v>=R^2/2 the replacement equals
h exactly. Thus it defines a global C-infinity function, including all
derivatives at the chart boundary. In the chart,

    (F_0)_s=-2s[c(v)+s^2 c'(v)],
    (F_0)_(t_i)=-2t_i[1+s^2 c'(v)].                     (9)

The second equations force t=0 at any critical point. For s!=0 the
bracket in the first equation is strictly positive. Consequently the
origin is the only critical point in D_R; all other critical points
and their values remain those of h.

Near the origin, (8) is exactly

    F_0=b-s^4/4-(1+s^2/4)|t|^2.

The smooth coordinate change u=t sqrt(1+s^2/4) gives

    F_0=b-s^4/4-|u|^2.                                  (10)

Its Hessian has corank one, with a negative definite transverse block;
its third axial derivative vanishes and its fourth axial derivative is
-6. In this source an A3 quartic contact means precisely this real
function-germ normal form, up to the displayed nonsingular coordinate
changes and a constant. We use no classification theorem to infer it.
The usual A_k function-germ notation is independently identifiable in
Arnold--Gusein-Zade--Varchenko's primary chapter preview, which gives
x^(k+1)+y^2 as its A_k series. Our signs specify the real maximum-type
variant. [Actual primary preview, pp. 242--243](https://page-one.springer.com/pdf/preview/10.1007/978-1-4612-5154-5_15).
This is function-germ terminology, not an identification of
wave-front or caustic names in a different convention.

## 4. The restricted unfolding and its exact three critical values

Take a smooth radial cutoff chi(z)=psi(|z|^2), with

    0<=chi<=1,
    chi=1 on |z|<=r_1/2,
    chi=0 on |z|>=r_1.

Extend it by zero outside D_R and define, for lambda>=0,

    F_lambda=F_0+chi(z)[lambda s^2/2+eta lambda^(3/2)s].   (11)

Each field is globally C-infinity in the spatial variable. On the core
|z|<=r_1/2 the formula is exact:

    F_lambda=b+Psi_lambda(s)-(1+s^2/4)|t|^2,
    Psi_lambda(s)=-s^4/4+lambda s^2/2+eta lambda^(3/2)s.

For lambda>0 put s=sqrt(lambda) z and

    Phi_eta(z)=-z^4/4+z^2/2+eta z.

At a core critical point t=0, and the axial equation is

    -z^3+z+eta=0.                                       (12)

For (1) it has exactly three simple real roots z_-<z_0<z_+, with

    -1<z_-<-1/sqrt(3),
    -1/sqrt(3)<z_0<0,
    1<z_+.

These bounds follow from the signs of -z^3+z+eta at -1,
-1/sqrt(3), 0 and 1, and its two turning points. Write
Z_* = max(|z_-|,|z_0|,|z_+|). When sqrt(lambda) Z_*<r_1/4,
all these points are strictly inside the core. Their z-coordinates,
critical values and Hessians are

    M_-=(sqrt(lambda) z_-,0),
    S  =(sqrt(lambda) z_0,0),
    M_+=(sqrt(lambda) z_+,0),

    F_lambda(M_i)=b+lambda^2 Phi_eta(z_i),
    Hess_z F_lambda(M_i)=diag(
        lambda(1-3z_i^2), -2(1+lambda z_i^2/4) I_(d-1)).  (13)

Thus M_- and M_+ are maxima and S has one positive and d-1 negative
Hessian eigenvalues. A chart change preserves these inertias because
at a critical point its Hessian transforms by congruence.

The values satisfy

    Phi_eta(z_+)>Phi_eta(z_-)>Phi_eta(z_0).               (14)

For the second inequality, Phi' is negative on (z_-,z_0), so its
integral there is strictly negative. For the comparison of the two
maxima, extend their simple root branches to eta=0. The maximum-value
difference is zero at eta=0 and has derivative z_+(eta)-z_-(eta)>0,
since the derivative of Phi at a root vanishes. This proves the first
inequality for the whole interval (1). Define

    kappa_eta=Phi_eta(z_-)-Phi_eta(z_0)>0.                (15)

The larger maximum has Phi_eta(z_+)>0; this also follows by comparing
its value with Phi_eta(1)=1/4+eta. We do not require the smaller maximum
to lie above b. For eta sufficiently close to the endpoint of (1) its
value can lie below b, which does not affect (14) or the argument.

## 5. Deterministic parameter bounds and the complete critical list

All constants in this section depend only on fixed d,L,eta,R,c,chi.
They are chosen before lambda is randomized. Let

    A={r_1/2<=|z|<=r_1},
    g_* = min_A |grad_z F_0|>0.

Positivity follows from (9) and compactness; this annulus avoids the
only critical point of F_0. Define finite constants

    C_0=sup |chi s^2/2|+eta sup |chi s|,
    C_1=sup |grad_z(chi s^2/2)|
                         +eta sup |grad_z(chi s)|,
    C_P=1+max_i |Phi_eta(z_i)|,
    delta=r_1^4/64.

For 0<=lambda<=1, the perturbation in (11) has sup norm at most C_0
lambda and z-gradient norm at most C_1 lambda on its support. Choose
lambda_0>0 so small that

    lambda_0<1,
    C_1 lambda_0<g_*/2,
    sqrt(lambda_0) Z_*<r_1/4,
    C_0 lambda_0<delta/4,
    C_P lambda_0^2<min(delta/4,Delta/4).                 (16)

If a bound has a zero coefficient it imposes no restriction. There is
always a positive lambda_0 satisfying this finite set of strict bounds.

For 0<lambda<=lambda_0 there are no roots on A, by its gradient floor.
Inside |z|<r_1/2 the exact equations give precisely the three roots
(13). Outside |z|>r_1, the perturbation vanishes and (9) excludes all
new roots. Its support and boundary are covered by these regions.
All old critical points lie outside D_R and remain unchanged.
The complete critical list consequently contains

    (2^d-1) old critical points plus M_-,S,M_+,
    hence 2^d+2 points in total.

Every one is nondegenerate at positive lambda. The three new critical
values are distinct by (14), and lie in (b-Delta/4,b+Delta/4) by (16).
All old values are at most b-Delta and are mutually distinct. Thus
F_lambda is Morse with distinct critical values. In particular it has
exactly two maxima: the seed's only maximum was replaced by M_- and
M_+. The right maximum M_+ is the unique global maximum.

There is also a uniform old-height gap. The set of unchanged old
critical values is finite and distinct, so its distinct pairwise
differences have a positive minimum when that set has at least two
elements. Its distance from each of the new values is at least
3Delta/4. There is no vanishing old-critical-value collision in this
family; the only closing critical-height gaps are among the three
explicit local points.

## 6. A fixed high-level barrier and an actual H0 elder bar

For any chart point with |z|>=r_1/2, monotonicity gives

    c(|z|^2)>=c(r_1^2/4)=r_1^2/16.

Since c<=1, its loss in (8) is at least

    s^2 c(|z|^2)+|t|^2
       >= (r_1^2/16)|z|^2 >= delta.

Outside D_R the loss is at least R^2>delta. Combining this with (16)
gives the global barrier

    F_lambda <= b-3delta/4 outside {|z|<r_1/2},
    F_lambda(M_-),F_lambda(S),F_lambda(M_+)>b-delta/4.   (17)

Every superlevel set at a height between F_lambda(S) and F_lambda(M_-)
is therefore contained strictly inside the exact core. A path outside
that core cannot connect its two components at any such height.

The topology inside is explicit. At height q in this interval, its
axial section is determined by

    Psi_lambda(s)>=q-b,

and the transverse fiber at an allowed s is the closed ball

    |t|^2 <= [b+Psi_lambda(s)-q]/(1+s^2/4).              (18)

There are exactly two axial intervals, one containing M_- and one
containing M_+, because Psi_lambda is a negative-leading quartic with
the three simple critical points already listed and q is strictly
between its middle minimum and its lower maximum. The intervals and
their fibers lie within the core: its boundary is below q by (17).
One can also see this by following the axial quartic from its local
maxima to either boundary; it has no further turning point there.

The homotopy (s,t)->(s,(1-u)t), 0<=u<=1, remains in the core and
increases F_lambda. It retracts the whole superlevel set to these two
axial intervals. Each interval is connected, its fibers are connected,
and no path between intervals exists because every point projects to
an allowed s. Thus there are exactly two superlevel components.

At q=F_lambda(S), those intervals first meet at s=sqrt(lambda)z_0;
the fiber there is the single point t=0. Just below this level the
axial section is one interval and its superlevel set is connected.
This proves an actual merger in the complete torus superlevel set.
It is not only a local saddle candidate or a presumed desired pair.

At its birth M_- forms a new component, while M_+ has the strictly
larger birth value by (14). The elder rule therefore pairs M_- with
S and preserves the component born at M_+. Its exact finite lifetime
before fixed normalization is

    F_lambda(M_-)-F_lambda(S)=kappa_eta lambda^2.        (19)

There are no other finite H0 intervals. A component of a nonempty
superlevel set contains a maximum: maximize F_lambda on that
component, or use the elementary critical-free gradient flow between
critical levels. Each maximum gives one H0 birth and there are only
two maxima. The explicit merger accounts for the only finite death;
the unique global maximum gives the one essential interval. Later
lower levels cannot create new H0 births without another maximum.

If one additionally tracks ordinary higher homology, the standard
one-handle/one-endpoint rule for a smooth Morse function over K gives
a useful bounded consequence. The three local critical points have
already spent their endpoint roles on the essential H0 birth and
this finite H0 birth/death. Any other finite interval has two old
critical values and hence a lifetime uniformly bounded below by the
old-height gap in Section 5. This uses that named Morse-handle fact,
not an asserted Hk elder generalization. The H0 theorem (2)--(4) does
not need this additional consequence.

## 7. A genuine stationary random-field law

Let Y be Haar-uniform on X and independent of a variable lambda with
density

    p_lambda(v)=gamma v^(gamma-1)/lambda_0^gamma,
                  0<v<lambda_0.                        (20)

Define G(x)=F_lambda(x-Y). For any translation a in X,
G(x+a)=F_lambda(x-(Y-a)), and Y-a has the same uniform distribution
as Y, still independently of lambda. This proves stationarity of
the full field law, including every finite-dimensional distribution.
It is not merely a spatial average of one covariance function.
No rotation invariance is claimed.

Each sample is a global C-infinity Morse function with distinct
critical values, because lambda=0 has probability zero. For every
fixed integer q>=0,

    sup_(0<=lambda<=lambda_0) ||F_lambda||_(C^q(X))<infinity.

The cutoff is fixed and the coefficients lambda,lambda^(3/2) remain
bounded. Translations preserve these bounds in the flat coordinates.
In particular every positive moment of every spatial C^q norm is
finite. There is no blowup of sample regularity or derivative moments
behind the power in (2).

If a centered variance-one field is desired, let

    mu=E[G(0)],
    sigma^2=Var(G(0)),
    Z(x)=(G(x)-mu)/sigma.                               (21)

These are deterministic constants for the specified law. They are
finite. Moreover sigma>0: conditional on any positive lambda,
the nonconstant continuous function F_lambda evaluated at a uniform
point has strictly positive variance. The law of total variance
then gives sigma^2>0. Centering and multiplying by this positive
constant preserve all critical locations, inertias and elder pairs.
Stationarity, smoothness and norm-moment bounds persist.

Set C=kappa_eta/sigma for (21), or set sigma=1, mu=0 when using G
without normalization. The parameters gamma and the deterministic
construction are fixed before sigma is computed. Changing gamma
may change sigma and the covariance. We make no claim that the
different gamma laws are covariance matched.

The parameter family is smooth for lambda>0 and converges globally
in every spatial C^q norm to F_0 as lambda decreases to zero. As a
family at lambda=0 it is C^1 but not C^2, since the nonzero coefficient
eta lambda^(3/2) has that regularity. In the standard two coefficient
controls (u,v) of the quartic potential, our path is

    (u,v)=(lambda,eta lambda^(3/2)).                    (22)

It stays inside the three-root cusp region and approaches its tip
along a prescribed path. It is neither a generic full two-parameter
random unfolding nor a smooth transverse one-parameter crossing of
that tip. The restriction is part of the actual law, not hidden in
a universal-class claim.

## 8. Exact lifetime density, location and physical distance

The unique finite-H0 lifetime of Z is C lambda^2 by (19),(21).
The change of variables lambda=sqrt(ell/C) in (20) proves (2):

    p_ell(ell)=gamma/(2 lambda_0^gamma C^(gamma/2))
                         ell^(gamma/2-1)
             =beta ell_0^(-beta) ell^(beta-1),
                   0<ell<ell_0=C lambda_0^2.            (23)

It integrates to one. Because exactly one finite H0 bar occurs in
each whole field, this probability density is also its expected
whole-field counting density. Translation gives the spatially
marked intensity p_ell(ell) d ell dx/V. It is important to retain
the factor 1/V when switching to a per-volume coefficient.

Its birth and death locations, before the random translation, are
the images of (sqrt(lambda)z_-,0) and (sqrt(lambda)z_0,0) under
the inverse sine chart. All transverse coordinates are zero and

    y_1(s)=(2/omega) arcsin(s/sqrt(2w_1)).

The birth position x=Y+y_1(sqrt(lambda)z_-)e_1 is Haar-uniform
and independent of lambda: this holds conditionally for each
lambda, so also jointly. Both points are in a fixed small chart
with a unique shortest displacement. Its direction from birth
to death is exactly e_1, since z_0>z_-. Its length is

    r_lambda=(2/omega)[
       arcsin(sqrt(lambda)z_0/sqrt(2w_1))
      -arcsin(sqrt(lambda)z_-/sqrt(2w_1))]
       =D sqrt(lambda)+O(lambda^(3/2)),
    D=2(z_0-z_-)/(omega sqrt(2w_1))>0.                  (24)

All parameters in this expansion are fixed. Thus the actual physical
height gap has quartic order in endpoint distance,

    ell=C D^(-4) r_lambda^4 [1+O(lambda)].             (25)

The exact law is quadratic in lambda; (25) is a distance asymptotic
and is not promoted to an exact Euclidean-radius identity.
The regular-fold cubic mark ell/r_lambda^3 tends to zero like
(C/D^3)sqrt(lambda). These samples approach a different contact
boundary from fixed positive-cubic-mark windows in the Gaussian
regular-fold theorem.

The global mean-zero birth height is

    B_Z=b_0+(Phi_eta(z_-)/kappa_eta) ell,
    b_0=(b-mu)/sigma.                                 (26)

In particular unscaled physical birth heights collapse to b_0 as
lifetimes decrease to zero. If desired the rescaled mark
(B_Z-b_0)/t is exactly (Phi_eta(z_-)/kappa_eta)(ell/t).
The centered lifetime still has no dependence on mu.

## 9. What this proves about the order taxonomy

For the identical limiting germ (10) and the identical fixed path
shape eta, changing only gamma in (20) gives

| Sampling near lambda=0 | Exact small-lifetime density power |
| --- | --- |
| gamma=1, flat lambda density | ell^(-1/2) |
| gamma=4/3 | ell^(-1/3) |
| gamma=2 | ell^0 |
| gamma>2 | ell^(gamma/2-1), vanishing at zero |
| any gamma>0 | ell^(gamma/2-1) |

The constants in (23) may change with the fixed variance normalization,
but these powers do not. The associated cumulative power is gamma/2.
This is an actual stationary-field construction with an actual selected
bar, rather than a formal replacement of the cubic order by four in
a Gaussian Kac--Rice formula.

The underlying probability order can also be stated in physical radius.
By (24), the induced small-r parameter mass is proportional to
r^(2gamma-1) dr; together with ell proportional to r^4 this gives
the same density exponent gamma/2-1. Uniform translation supplies
uniform contact location, not the ordinary off-diagonal polar contact
measure of a nondegenerate two-site jet law. There is no reason to
substitute the Gaussian observation/determinant orders into this law.

Consequently a quartic singularity fixes the local relation between
the parameter, endpoints and height gap, but does not by itself fix
an observed lifetime density exponent. Contact sampling weight is a
necessary part of a taxonomy. The flat-parameter example realizes a
different exponent from the admitted regular Gaussian fold law; the
gamma=4/3 example also shows that the exponent -1/3 alone cannot
identify the singularity class.

No claim is made about a generic two-parameter cusp sampled with a
specified two-dimensional density, about all A3 laws, or about an
Arnold class assigning one covariance-independent persistence power.
Those would require their own actual contact law and global selection
proof, just as the regular-fold interface theorem requires them.

## 10. An actual iid-copy marked process, with exact cluster avoidance

For a single sample let Xi_t be its point measure with lifetime
ell/t and birth location x. For every finite T, on {ell<=Tt} it
has exactly one point, and otherwise it has none on (0,T]xX.
As t decreases to zero the measure on this window is eventually
empty for each sample, since its one lifetime is strictly positive.
Its expected count is exactly (Tt/ell_0)^beta when Tt<ell_0.

Take n independent copies and superpose their Xi_(t_n). Let f>=0
be a bounded continuous compactly supported test function on
(0,infinity)xX, with lifetime support in (0,T]. For t<ell_0/T,
the exact one-field Laplace deficit is

    D_t(f)=E[1-exp(- integral f dXi_t)]
      =t^beta beta ell_0^(-beta)
          integral_(0,T] integral_X
              [1-exp(-f(a,x))] a^(beta-1) dx/V da.      (27)

For independent copies the Laplace functional is (1-D_t(f))^n.
Using (3), it tends to the exponential of minus the integral of
1-exp(-f) against (4), which is the Poisson random-measure Laplace
functional. The limiting intensity is finite on (0,T]xX because
beta>0, including at the lifetime boundary zero. The same calculation
also applies to bounded continuous tests that extend to zero.

There is no inference of a Poisson limit from first moments alone.
Single-field multiple-witness avoidance is exact here:

    N_(H0,Tt) is in {0,1},
    E[N_(H0,Tt); N_(H0,Tt)>=2]=0.                       (28)

The process is a rare-event limit of independent entire field copies.
It does not assert spatial independence, a factorial limit within a
copy, a growing-domain regime, or convergence of unscaled barcodes.
Adding degree and direction marks puts a Dirac mass at degree zero
and e_1 in (4). Adding the unscaled physical birth height puts a
Dirac mass at b_0, by (26); it does not retain a nontrivial birth law.

One may retain the explicit rescaled birth height in (26), or the
separation mark r_lambda/t^(1/4). The latter converges, at lifetime
mark a, to D(a/C)^(1/4) by (24). These give deterministic mark
pushforwards of (4), not extra probabilistic independence assertions.
If the higher-degree old-gap consequence in Section 6 is used, all
other finite bars are absent from (0,Tt] for sufficiently small t,
so the identical limit holds for the entire degree-marked finite
barcode. The main theorem is the H0 assertion and needs no such
higher-degree extension.

## 11. Admission boundaries, measurable selectors and law deficits

The chosen finite-H0 pair is a Borel function of (lambda,Y): its
locations, values and lifetime are explicit continuous expressions
for positive lambda. Equivalently it is the actual elder pair on
the open Morse/distinct field locus, proved in Section 6. The count
in (28) is its Borel lifetime indicator. Thus there is no unresolved
root-count or global-selector measurability premise in this family.

Every sample has good topology and uniformly bounded spatial
derivatives, but the law fails the full joint-density contract of
the Gaussian or regular contact-counting adapters. To make this
deficit precise, fix two distinct sites p,q and consider their
values and gradients, a vector in R^(2d+2). For positive lambda this
vector is a smooth function of (lambda,Y), which has only d+1
real parameters. Cover the parameter space by countably many
compact coordinate charts with lambda bounded away from zero.
The map on each is Lipschitz. A Lipschitz image of a bounded subset
of R^(d+1) has zero (2d+2)-dimensional Lebesgue measure: subdividing
into cubes of side epsilon covers the image by O(epsilon^(-(d+1)))
balls of radius O(epsilon), with total (2d+2)-volume tending to zero.
Taking the countable union proves that this observation law is
singular, and hence has no full Lebesgue joint density.

This is a law-density deficit, not a claim that its covariance matrix
must be singular. A nonlinear distribution with a small-dimensional
support can have a nonsingular covariance; covariance rank alone
does not decide this issue. Similarly we do not assert any particular
positive or zero Fourier block for the constructed covariance.

The translated smooth templates give a real stationary, non-Gaussian
law with rapidly controlled spatial derivatives. Non-Gaussianity also
has a direct scalar witness: G(0) has bounded support and positive
variance, whereas a nonconstant Gaussian variable has unbounded
support. Its Fourier phases
are correlated through Y and its deformation through lambda. No
independent Gaussian coefficient carrier, arbitrary jet-rank property,
or generic full two-parameter contact density is supplied. It is not
admitted by covariance positivity plus sample smoothness alone.

The apparent abundance of nearby critical points is also explicit:
three local points, not the two branches of a regular cubic fold.
The two maxima and one saddle have different roles. Choosing the
wrong maximum--saddle candidate would give the essential older
component or a different height difference; the actual finite bar
uses the lower maximum in (14). Its enclosure and elder comparison
are why this source can make an actual selected statement.

The construction is not covariance matched to the spectral Gaussian
family, and it is not a random scalar-amplitude mixture of that law.
It supplies a different local contact order in an actual spatial
field while keeping its restricted path and sampling dependence
visible. The previously prepared amplitude example isolates a
different failure mechanism, namely a global scale changing the
effective contact probability even at fixed covariance.

## 12. Operational context, source review and reading boundary

These current repository references were freshly hashed by root. They
are context, not Gaussian assumptions imported into this non-Gaussian
law. Sections1-11 directly prove the principal H0 field/count/process
result. The optional higher-degree old-gap consequence consumes only the
explicitly named ordinary Morse one-handle/one-endpoint fact.

| Context | Current repository source | Bytes / lines | SHA256 |
| --- | --- | --- | --- |
| ST | [regular-fold structural theorem](../fold_structural_universality/PROOF.md) | 43927 / 804 | 3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531 |
| SG | [spectral Gaussian admission](../spectral_gaussian_admission/PROOF.md) | 42478 / 826 | 6c0f34cb318d337b3b8c2c306f3b965f5e5a9d2fe38d5e001b8044de74b7f225 |
| HK | [higher-homology fold](../higher_homology_fold/PROOF.md) | 46174 / 911 | 87f43e6ed94c5c63c385ecf177f68212afa5db54b24062fa008f081f47a78999 |
| AMP | [exact amplitude crossover](../spectral_amplitude_crossover/PROOF.md) | 37461 / 778 | aeea6b55a1876c357d9beeeb5a96ee54a5b17b53c6138e2993c454f1117f65d5 |

The author originally read the externalSG825/43e818 andHK884/de1cad
preparations and both amplitude720/5b185 and corrected730/9e8f sources;
the following historical ledger retains those exact identities and
actual reading extents. Root checked original/current SG Sections1-11
at36656B/4cad43d3 and HK Sections1-12 at37220B/c770d4f7. No new whole
currentSG/HK author reading is inferred. The amplitude result is context;
no amplitude theorem is a premise of this quartic construction.

Fresh exact original-source audits are separately bound. Final read all
695 actual lines in1-240/241-480/481-695, and benchmark read all695 in
1-232/233-464/465-695; both returned mathematicalPASS with no must-fix.
Both freshly reproduced the five original context hashes and additionally
reread ST691-737 and HK388-442. Their prior whole-source exposure is
separate; no five new whole dependency audits are claimed. Final supplied
the earlier fold-chart route and amplitude wrapper, benchmark CG/ET and
catalogue code, root the prefile flattening/unfolding idea, and the author
all related spectral/Hk/amplitude routes. Organizational independence0.

The author's primaryAGV exposure is bounded to its two-page nomenclature
preview, not classification. Root and final each attempted that preview
once and received inaccessible/InternalError; neither obtained fresh
primary-content verification or retried. Benchmark made no new primary
fetch. The actual real normal form is explicitly derived in Section3;
no unseen classification theorem is used. No rawPDF/fullbook custody,
compiler, numerical field, sampler, seed or experiment is credited.

The author's original source ledger below is preserved as historical
provenance, including its original request for exact-file review. Its
filesystem paths identify original artifacts, while current operational
links are above. The original's completed two reviews are not a new full
review of this incorporated wrapper. New incorporated-source review and
README metadata remain PENDING at this construction snapshot.

## 13. Preserved original author context and primary ledger

All paths below are under
`/Users/dylanroy/Documents/Codex/2026-10-08/turn-the-below-into-a-reality/`.
Their hashes were freshly reproduced before saving this source.
The author had previously read the full actual sources listed here;
the current task does not rebadge those prior reads as new independent
reviews, nor infer any current publication or scientific status.

* `fold-universality-worktree/experiments/universality/fold_structural_universality/PROOF.md`,
  SHA256 `3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531`.
  Prior full source read; Section 10, actual lines 691--737, was freshly
  reread for its conditional order taxonomy. It explicitly leaves a
  higher-order field theorem unproved. The present construction supplies
  one such actual restricted law, not a general admission of its class.
* `spectral-gaussian-universality-preparation.md`,
  42,038 bytes / 825 lines, SHA256
  `43e8180947b559e53f7afcf95ba61136f3fffcbe05a19e7a5953018fda27e294`.
  Author's prior full actual-byte read and prior same-team technical
  reviews are historical context. None of its Gaussian COUNT,
  CONTACT, ENVELOPE, SELECT, FAR or coefficient conclusions are
  imported into this non-Gaussian law.
* `higher-homology-fold-obstruction-preparation.md`,
  44,258 bytes / 884 lines, SHA256
  `de1cad971e113f097a8e29ba82fb295d93b23f57aeda5b265e6f3c88cef47f2e`.
  Prior full author read; Section 7, actual lines 388--442, was freshly
  reread for the ordinary Morse-handle endpoint rule used only in the
  optional old-higher-bar-gap consequence. Its independent review state
  is not a premise for the self-contained H0 enclosure proof here.
* `spectral-amplitude-crossover-preparation.md`,
  33,421 bytes / 720 lines, SHA256
  `5b1859080a137aed9eeeba20e1b6f0c5a61387e5eb6fa20bb1b8704860fb73d2`,
  and its separate corrected preparation, 34,110 bytes / 730 lines,
  `spectral-amplitude-crossover-corrected-preparation.md`, SHA256
  `9e8f10162e9fa21f308454df9463459cdd67ef39a2d8103a78a397b4ab82dd0d`.
  Both had full saved-byte author reads. The original's two full reviews
  found a missing t_n->0 hypothesis in its critical process normalization;
  its AMEND history remains intact. The corrected source is a successor,
  not a rewrite of that frozen source or its review record. No amplitude
  theorem is consumed in this proof. In particular (3) includes the
  small-threshold hypothesis explicitly.

For the A_k nomenclature, the author freshly read the official publisher
book and chapter pages and the actual two-page primary chapter preview
of Arnold, Gusein-Zade and Varchenko, *Singularities of Differentiable
Maps*, Volume I, Section 15, printed pages 242--243. The preview states
the A_k series of function germs. It does not expose the whole chapter,
and no full-book or full real-classification read is claimed. The exact
real quartic normal form (10) is proved directly, so no unseen
classification result is needed. Primary links:

* [AGV, Lists of singularities](https://link.springer.com/chapter/10.1007/978-1-4612-5154-5_15).
* [Actual chapter preview, pp. 242--243](https://page-one.springer.com/pdf/preview/10.1007/978-1-4612-5154-5_15).

The optional all-degree old-gap statement consumes only the classical
fact that crossing a nondegenerate Morse level attaches one handle,
whose field-coefficient relative homology is one-dimensional in its
handle degree. In the prior higher-homology task the author actually
read the relevant bounded primary portions of Milnor's *Lectures on
the h-Cobordism Theorem*: Section 5's handle setup and Theorem 5.4,
and Section 7's relative-homology/intersection material, including
Corollary 7.3. The current proof uses no cancellation theorem, generic
gradient-trajectory assumption, or stronger Hk selected claim from
that book. [Primary Milnor scan](https://webhomes.maths.ed.ac.uk/~v1ranick/surgery/hcobord.pdf).

The actual dependencies of (2)--(4) are elementary smooth cutoffs,
the explicit coordinate and derivative calculations, compact gradient
and height bounds, the connected-component computation (18), Haar
translation invariance, a scalar change of variables, and the iid
Laplace product. No Gaussian regression, numerical draw, compiler,
external message, repository source edit or status change was used.
The proposed stationary A3 family and its exact selected H0 law are
new analytic preparation, awaiting independent exact-file review.
