# Smooth two-control cusp population: actual H0 bars and fold-edge dominance

Repository incorporation by OpenAI/Codex `final_contract_review`, the bounded
wrapper constructor, under integration owner `root`, task
`01a11cbf-068d-7102-861e-75814e715c98`. Mathematical author:
`universality_review`. Native claim `6071732615` authorizes these three paths
within its eleven-path scope through 02:20 UTC on 9 October 2026. The initial
detached base is published quartic PR336 head
`3ab12c6542c88554ad0d42e0e19aff2461dc3c0d`; no main integration is implied.

Sections 1-10 preserve the frozen external preparation exactly: 32,809 UTF-8
bytes / 696 lines, SHA256
`001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6`.
Its 28,073-byte / 620-line mathematical core has SHA256
`d4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213`.
Both `final_contract_review` and `benchmark_formal_audit` completed separate
fresh FULL original-source mathematical PASS reviews, with no must-fix.
Root fully read the actual original 696 lines. Their exact reading ranges
and the current context bindings are recorded in Section 11.

The constructor first reviewed that original source and now authors this
repository wrapper, review custody and README addition. This is an integration
contribution, not an independent review of the constructor's own wrapper.
A fresh full mathematical review of the incorporated source, root's current
full read and new metadata review are **PENDING** at this construction snapshot.
Original verdicts do not certify the new wrapper bytes. The
[review record](REVIEW.md) preserves these separate source-bound states.

The result concerns one specified stationary non-Gaussian field law. Its
smooth two-control rectangle yields an explicit globally selected H0 bar,
with fold-edge leading density and a lower-order restricted-tip contribution.
All performers are source/route-exposed members of the same OpenAI/Codex team;
organizational independence is zero. No general A3, Gaussian, Hk, numerical,
spatial, formal, blind or scientific-status admission is implied.

## 1. The fixed law and the exact assertions

Fix an integer d>=2, L>0, X=(R/LZ)^d, spatial volume V=L^d, and
omega=2 pi/L. Persistent homology is ordinary superlevel H0 over a fixed
field K, with the elder rule and the essential global-maximum class excluded.
Use the explicit smooth seed and quartic surgery of Section 2. The family is

    F_(u,v)=F_0+chi(s,z_perp)[u s^2/2+v s],              (1)

on the full control rectangle

    -u_0<u<u_0,      -v_0<v<v_0,
    v_0>=eta_c u_0^(3/2),      eta_c=2/(3 sqrt(3)).       (2)

Section 2 supplies compatible explicit smallness conditions, including
root containment, annular exclusion and a global high-level barrier.
All constants and rectangle sides are fixed before taking any lifetime limit.
Choose (u,v) uniformly in this full rectangle, with density

    c_0=1/(4u_0 v_0),                                   (3)

and independently choose Haar-uniform Y in X. The random field is

    G(x)=F_(u,v)(x-Y).

We leave its scale unnormalized in the main theorem. This law is stationary,
has smooth sample fields and uniform bounds on every spatial C^q norm, and
is almost surely Morse with distinct critical values. The density (3) is
positive and smooth in the interior, including at the control origin;
its extension by zero across the rectangle boundary is not claimed smooth.

The only controls producing a finite H0 bar are the open cusp wedge

    W={0<u<u_0, |v|<eta_c u^(3/2)}.                      (4)

At good controls in W there is exactly one finite H0 interval, with

    eta=|v|/u^(3/2),
    ell=u^2 kappa(eta),        0<eta<eta_c,              (5)

where kappa decreases strictly from 1/4 to zero. Controls with v=0 in W
have tied maxima and form a null exception. Outside the wedge, away from
the discriminant, the field has one maximum and no finite H0 intervals.
The discriminant and tied-value locus have two-dimensional control measure
zero, so their arbitrary zero-count convention does not change the law.

Let q(h)=|d eta/d kappa| at eta=kappa^(-1)(h), 0<h<1/4.
The whole-torus expected finite-H0 lifetime measure has the explicit density

    nu(t)=2c_0 integral_(2 sqrt(t))^u_0
                         u^(-1/2) q(t/u^2) du,
                         0<t<u_0^2/4,                  (6)

and is zero for t>=u_0^2/4. This is derived by a direct pushforward of
control measure, not by differentiating a cumulative asymptotic. Write

    K_c=4/3^(5/4),
    Q_c=(2/3) K_c^(-2/3),
    C_edge=(8c_0/7) K_c^(-2/3) u_0^(7/6).              (7)

Then

    nu(t) ~ C_edge t^(-1/3),
    E N_t ~ (3/2) C_edge t^(2/3),        t down to zero. (8)

These are whole-field coefficients. Per unit spatial volume divide by V.
For any fixed 0<epsilon<eta_c, the population restricted to
eta<=eta_c-epsilon has, for all sufficiently small t, the exact density

    nu_tip,epsilon(t)=C_tip,epsilon t^(1/4),
    C_tip,epsilon=c_0 integral_0^(eta_c-epsilon)
                                      kappa(eta)^(-5/4) d eta. (9)

Its cumulative contribution is exactly (4/5)C_tip,epsilon t^(5/4)
there. The uniform small-u bound in Section 5 makes fold-edge dominance
and the absence of an atom at the A3 tip precise.

For n independent whole-field copies, if t_n->0 and n t_n^(2/3)->lambda
in (0,infinity), their actual short-H0 points with lifetime mark ell/t_n
and birth position x converge to a Poisson random measure with intensity

    lambda C_edge a^(-1/3) da dx/V.                     (10)

The full latent-control, sign, physical birth and cubic marks are specified
in Sections 7--8. This uses the exact single-field bound N_t in {0,1}.
It is not a same-field spatial or expanding-domain process claim.

## 2. Actual global construction and uniform rectangle bounds

Here we reproduce the needed global surgery, rather than presuming a local
normal form has a harmless completion. Put w_i=3^(-i) and

    h(y)=sum_i w_i cos(omega y_i),       b=sum_i w_i.

Its 2^d critical points have y_i in {0,L/2}, diagonal nonzero Hessians,
and distinct values. In comparing two sign patterns, their first differing
weight dominates the sum of all later weights. The origin is the unique
maximum, and all other critical values are <=b-Delta, Delta=2w_d.

In |y_i|<L/4 use z_i=sqrt(2w_i)sin(omega y_i/2), s=z_1, and
z_perp=(z_2,...,z_d). These smooth coordinates satisfy exactly

    h=b-s^2-|z_perp|^2.

Choose R^2<min(w_d,1). The superlevel {h>b-R^2} is precisely the
chart ball D_R. Indeed each coordinate's nonnegative sine-square loss
is then <R^2<w_i, forcing |y_i|<L/4 in torus representatives.
Outside this ball h<=b-R^2, and no other old critical point lies in it.
Let r_1=R/4. Take a smooth nondecreasing c:[0,infinity)->[0,1] with

    c(r)=r/4 for r<=r_1^2,
    c(r)=1 for r>=R^2/2,
    c(r)>0 for r>0,       c'(r)>=0.

For example integrate a nonnegative smooth derivative equal to 1/4
on [0,r_1^2], tapering to zero before R^2/2, and add an interior bump
to make its total integral exactly one. This is possible because R^2<1.
Define on D_R, retaining h outside,

    F_0=b-s^2 c(s^2+|z_perp|^2)-|z_perp|^2.             (11)

It equals h on a full boundary collar, so is globally C-infinity.
Its derivatives are

    (F_0)_s=-2s[c(r)+s^2 c'(r)],
    (F_0)_(z_i)=-2z_i[1+s^2 c'(r)],     r=s^2+|z_perp|^2.

They vanish in D_R only at the origin. All old critical points outside
are unchanged. In |z|<=r_1 the exact formula is

    F_0=b-s^4/4-(1+s^2/4)|z_perp|^2.                   (12)

Let chi be a fixed radial smooth cutoff, equal to one for |z|<=r_1/2,
zero for |z|>=r_1, and between zero and one elsewhere. Its zero extension
is smooth on X. Formula (1) therefore defines a global field jointly
C-infinity in x,u,v, including at u=v=0. In its exact core,

    F_(u,v)=b+P_(u,v)(s)-(1+s^2/4)|z_perp|^2,
    P_(u,v)(s)=-s^4/4+u s^2/2+v s.                    (13)

For explicit uniform guards, set

    a_*=r_1/4,        delta=r_1^4/64,
    g_*=min_(r_1/2<=|z|<=r_1)|grad_z F_0|>0,
    B_j=||chi s^2/2||_(C^j),    D_j=||chi s||_(C^j), j=0,1,

where for j=1 the gradient norm alone suffices. Fix a constant
c_v>=eta_c, set v_0=c_v u_0^(3/2), and choose u_0>0 small enough that

    u_0<a_*^2/4,          v_0<a_*^3/4,
    B_1 u_0+D_1 v_0<g_*/2,
    B_0 u_0+D_0 v_0<delta/4.                           (14)

These strict inequalities are compatible: their left sides tend to zero
as u_0 tends to zero. They also permit any larger v_0 satisfying (2)
and the same upper bounds. Coverage of the wedge is not being confused
with the smallness guards.

Every real root of -s^3+u s+v=0 in the rectangle has |s|<a_*.
For if |s|>=a_*, its root equation and (14) would give

    |s|^3<=u_0|s|+v_0<|s|^3/4+|s|^3/4,

a contradiction. Thus all axial roots are strictly inside the region
where chi=1. On the transition annulus the perturbation gradient is
<g_*/2, so no new critical points occur. Elsewhere in D_R the unperturbed
gradient excludes roots. Old critical points outside are unchanged.

The core transverse equations force z_perp=0. Hence the complete new
critical list is exactly the real roots of that axial cubic, together
with the old 2^d-1 critical points. Its core Hessian is diagonal,

    diag(u-3s^2, -2(1+s^2/4) I_(d-1)).                 (15)

Coordinate changes preserve its inertia at critical points.
Every new critical value is within delta/4 of b: using |s|<a_* and
a_*^4=delta/4 gives

    |P_(u,v)(s)|<=a_*^4/4+u_0 a_*^2/2+v_0 a_*
                   <5delta/32<delta/4.                (16)

Also delta=R^4/16384<Delta. Thus these new values remain uniformly
separated from every old one by more than 3Delta/4. The old values are
fixed and mutually distinct, so all their distinct differences have a
positive minimum. This excludes hidden old-level collisions in the
entire rectangle.

Finally, outside the core |z|<r_1/2, monotonicity and c<=1 imply

    b-F_0>= (r_1^2/16)|z|^2>=delta

in the chart, while the exterior chart loss is at least R^2>delta.
The last inequality of (14) gives the global barrier

    F_(u,v)<=b-3delta/4 outside the core,
    all new critical heights >b-delta/4.               (17)

These are uniform bounds over the full rectangle, not only over the wedge.

## 3. Complete root classification and actual global H0 selection

The cubic -s^3+u s+v has three simple real roots exactly when

    u>0,       27v^2<4u^3.

This follows from evaluating the cubic at its two turning points
s=+-sqrt(u/3). For u<0 it is strictly decreasing and has one real root.
For u=0,v!=0 it also has one simple real root. For u>0 outside the
closed three-root wedge it has one root beyond the turning points, at
which u-3s^2<0. Thus outside W and off the discriminant there is one
new maximum; with the unchanged old list, it is the only maximum on X.
The field is Morse and has distinct critical values by (16). It has
no finite H0 intervals: a connected closed torus with just one maximum
has only the essential H0 birth, and no second component birth.

Inside W scale s=sqrt(u)z. For epsilon=sign(v) and eta=|v|/u^(3/2),
the axial profile is u^2 Phi_(epsilon eta), where

    Phi_eta(z)=-z^4/4+z^2/2+eta z.

For positive eta denote its three roots of Phi'=0 by z_-<z_0<z_+.
They lie respectively in (-1,-1/sqrt(3)), (-1/sqrt(3),0), and (1,infinity).
The outer roots give two maxima, the middle root gives the saddle with
one positive and d-1 negative physical Hessian eigenvalues. The left
maximum is lower. To verify the ordering rather than prescribe it,
Phi(z_-)>Phi(z_0) follows by integrating its negative derivative between
these two roots. The right-minus-left peak value is zero at eta=0 and
has derivative z_+-z_->0, proving strict positivity at eta>0.
For negative v, reflect s to -s: the right maximum is now younger.

All critical heights near this pair satisfy (17). Therefore every
superlevel set between its saddle and its lower maximum is entirely
inside the exact core. At a height q there its axial section is
{P_(u,v)(s)>=q-b} and its transverse fibers are the balls

    |z_perp|^2 <= [b+P_(u,v)(s)-q]/(1+s^2/4).          (18)

For levels strictly between the saddle and the lower maximum there
are two axial intervals. Contracting each transverse fiber to its
center increases F and stays in the core, so the full global superlevel
has exactly two components, one containing each maximum. At the saddle
level the intervals first touch at its axial coordinate, with a
single-point transverse fiber; just below, the superlevel is connected.
The global barrier rules out an earlier external bypass. This proves
an actual component merger in the whole torus field.

The higher maximum is older, and its component survives. The lower
maximum is paired with the middle saddle, with lifetime (5). There are
exactly two global maxima and therefore exactly one finite H0 interval;
the higher maximum gives the essential class. This completes the
actual selection proof in both regions of the control rectangle.

On the discriminant v=+-eta_c u^(3/2), u>0, the cubic has a double
root and another simple root. At u=v=0 it has a triple root. These
fields are not Morse at the contact. At v=0,u>0 the two maxima are
nondegenerate but have equal values. These graphs and the line v=0
are Lebesgue-null in the control plane by Fubini. We set counts to
zero there for the probabilistic formulas, without asserting a
distinct-values elder convention on that exceptional set.

## 4. The gap function and a derived fold-edge inverse density

For 0<=eta<eta_c define by endpoint continuation

    kappa(eta)=Phi_eta(z_-(eta))-Phi_eta(z_0(eta)).

It is smooth before eta_c and

    kappa(0)=1/4,
    kappa'(eta)=z_-(eta)-z_0(eta)<0,
    kappa(eta) -> 0 as eta increases to eta_c.           (19)

The derivative formula uses that Phi' vanishes at each critical root;
there is no endpoint differentiation error. At eta_c the two roots
coalesce at z_c=-1/sqrt(3). Thus kappa maps (0,eta_c) bijectively to
(0,1/4), and its exact inverse derivative is

    q(h)=1/[z_0(kappa^(-1)(h))-z_-(kappa^(-1)(h))].     (20)

It is positive, smooth for 0<h<1/4, and tends to one as h increases
to 1/4. We use this extension at the upper endpoint if needed.

We now derive the singular endpoint constant. Set delta_eta=eta_c-eta
and write z=z_c+H. The cubic is exactly

    Phi_eta'(z_c+H)=sqrt(3) H^2-H^3-delta_eta.           (21)

Let rho=sqrt(delta_eta), H=rho w. The two nearby roots solve
sqrt(3)w^2-rho w^3-1=0. At rho=0 they are +-a_c, a_c=3^(-1/4),
and are simple in w. The implicit function theorem gives

    w_-(rho)=-a_c+O(rho),       w_0(rho)=a_c+O(rho).

Their order identifies them with the lower-maximum and saddle roots.
The exact difference of potential values is the integral of minus
(21) between these roots. It gives

    kappa(eta)=delta_eta^(3/2)
       integral_(w_-(rho))^(w_0(rho))
                [1-sqrt(3)w^2+rho w^3] dw
       =K_c delta_eta^(3/2)[1+O(sqrt(delta_eta))],
    K_c=integral_(-a_c)^(a_c)(1-sqrt(3)w^2)dw
          =(4/3)a_c=4/3^(5/4).                         (22)

Also z_0-z_-=2a_c sqrt(delta_eta)[1+O(sqrt(delta_eta))].
Substituting in the exact formula (20), rather than differentiating
the asymptotic (22), proves

    lim_(h down to zero) h^(1/3)q(h)
       =K_c^(1/3)/(2a_c)=(2/3)K_c^(-2/3)=Q_c.          (23)

In particular h^(1/3)q(h) extends continuously to [0,1/4] and is
bounded there. For some finite A_q depending only on this fixed
profile,

    0<q(h)<=A_q h^(-1/3),       0<h<=1/4.              (24)

This global normalized-gap bound controls the entire population;
an endpoint expansion alone would not suffice for the later domination.

## 5. Direct density, global domination and the leading coefficient

Inside W use u>0, epsilon=+-1 and v=epsilon u^(3/2)eta. Its control
measure is exactly c_0 u^(3/2)du d eta for each sign. For any nonnegative
Borel lifetime test f, actual selected counting and Tonelli give

    E sum_finiteH0 f(ell)
      =2c_0 integral_0^u_0 integral_0^eta_c
                        u^(3/2) f(u^2 kappa(eta)) d eta du. (25)

The null tie and discriminant controls contribute nothing. For fixed u,
the ordinary one-variable substitution h=kappa(eta), then ell=u^2 h,
turns (25) into integral f(ell)nu(ell)d ell, with exactly (6).
Thus nu is an actual density representative of the expected selected
bar measure. For each t>0 it is finite. It is continuous on positive
t: the q argument lies in a compact subinterval of (0,1/4] locally
in t, the moving lower endpoint is regular, and the integration interval
shrinks to zero as t approaches u_0^2/4. No continuity of an arbitrary
global selected Kac--Rice kernel is being imported.

The total mass checks the rectangle normalization:

    P(one finite H0 bar)=E total finiteH0
      =2c_0 eta_c integral_0^u_0 u^(3/2)du
      =(4c_0 eta_c/5)u_0^(5/2)
      =eta_c u_0^(3/2)/(5v_0)<=1/5.                    (26)

With the simplest choice v_0=eta_c u_0^(3/2), it is exactly 1/5.
The remaining controls have no finite H0 bar. Nu is therefore not a
probability density of total mass one; treating it as one would lose
the full symmetric rectangle's population factor.

Extend the integrand in (6) by zero for u<=2sqrt(t). Then

    t^(1/3)nu(t)=2c_0 integral_0^u_0
       1{t/u^2<1/4} u^(1/6)
                    [(t/u^2)^(1/3)q(t/u^2)] du.        (27)

For each fixed u>0 its bracket tends to Q_c. Bound (24) supplies the
majorant A_q u^(1/6), which is integrable both at u=0 and at u=u_0.
Dominated convergence therefore gives

    lim t^(1/3)nu(t)=2c_0 Q_c (6/7)u_0^(7/6)=C_edge.

This proves the first part of (8), with the constant (7). There is a
uniform positive-lifetime bound nu(t)<=C_bound t^(-1/3), from (27).
Integrating the already established density, or applying dominated
convergence after ell=t a, gives its cumulative asymptotic in (8).
We have not differentiated an asymptotic count to infer a density.

The small-u control is quantitative. The contribution from u<=r to
t^(1/3)nu(t) is at most

    (12c_0 A_q/7) r^(7/6),       0<r<=u_0,             (28)

uniformly in t. Its limiting coefficient is exactly
(12c_0 Q_c/7)r^(7/6). Hence taking r down to zero removes the A3-tip
tail uniformly. For every fixed positive u the asymptotic samples
approach one of its two ordinary fold edges. No mass concentrating at
u=0 is hidden in the all-u limit.

## 6. The genuinely quartic-tip contribution away from the fold edges

Fix 0<epsilon<eta_c and restrict (25) to 0<=eta<=eta_c-epsilon.
On that interval kappa is bounded below by
kappa_min=kappa(eta_c-epsilon)>0. Change variables u=sqrt(ell/kappa(eta))
for fixed eta. The factor two for the signs cancels the derivative's
one-half, and the selected measure has density

    c_0 ell^(1/4) integral_0^(eta_c-epsilon)
          kappa(eta)^(-5/4)
                       1{ell<u_0^2 kappa(eta)} d eta. (29)

For 0<ell<u_0^2 kappa_min the indicator is identically one, proving
the exact formula (9). Its coefficient is finite and positive, and
its cumulative is exactly (4/5)C_tip,epsilon t^(5/4) below that cutoff.
It is lower order than t^(2/3), and its density is lower order than
t^(-1/3). The controls producing these small lifetimes necessarily
have u=O(sqrt(t)) and v=O(t^(3/4)), so they approach the actual quartic
tip while staying away from the two fold-edge directions.

The area element u^(3/2)du d eta is the reason this tip population
differs from a flat one-parameter restricted path. A fixed-eta path
sampled with flat u density gave a lifetime density ell^(-1/2) in the
earlier construction. Integrating a full smooth two-dimensional control
density through its shrinking cusp width instead gives the tip power
ell^(1/4). Neither calculation assigns one sampling-independent power
to the A3 label.

## 7. The actual limiting marks and their physical coordinate conversion

Retain epsilon=sign(v), the positive control u, and actual birth position
x. On the cusp these are Borel. The birth location is Haar-uniform and
independent of every control: conditional on controls it is Y plus a
deterministic critical-point location. The direction from the younger
maximum to its saddle is exactly epsilon e_1. All roots lie in a fixed
small coordinate chart, so their physical displacement is the unique
shortest one. For epsilon=+1, their chart positions are sqrt(u)z_- and
sqrt(u)z_0; for epsilon=-1 they are their reflected positions.

At the positive edge, the coalescing chart point is
s_c=-sqrt(u)/sqrt(3). The reflected edge has +sqrt(u)/sqrt(3).
The inverse physical chart is

    y_1(s)=(2/omega)arcsin(s/sqrt(2w_1)),
    g(u)=y_1'(s_c)
        =2/[omega sqrt(2w_1)] [1-u/(6w_1)]^(-1/2)>0.    (30)

The smallness conditions ensure the displayed denominator is positive.
Since g is even in the coalescing sign, it applies to both edges.
For fixed u>0, as eta increases to eta_c, the physical pair distance r is

    r=2g(u)sqrt(u)a_c sqrt(delta_eta)
                                      [1+O(sqrt(delta_eta))].

Combining this with ell=u^2 kappa(eta) and (22) gives the actual
physical cubic mark limit

    k_phys=ell/r^3 -> k(u)
          =sqrt(u)/(2sqrt(3)g(u)^3)>0.                 (31)

Equivalently the oriented axial third derivative at the coalescing point
is 2sqrt(3)sqrt(u)/g(u)^3=12k(u): the axial first and second derivatives
vanish there, so a coordinate change contributes no lower-jet terms.
This checks the physical normalization, not just a normalized chart ratio.

The physical birth height is b+u^2 Phi_eta(z_-), with reflection at the
negative edge. At eta=eta_c, Phi_eta(z_c)=-1/12, giving

    B -> B_c(u)=b-u^2/12.                               (32)

The older maximum at the edge has value b+2u^2/3, so it remains strictly
older for each fixed u>0. This gap need not be uniform as u tends to
zero; uniformity is instead supplied by the integrable tail (28).

For each sign the joint lifetime/control/spatial density before scaling is

    c_0 1{0<ell<u^2/4} u^(-1/2)q(ell/u^2)
                                  d ell du dx/V.      (33)

Therefore t^(-2/3) times the expected point measure with ell/t=a has
the weak limit

    eta_edge(da,du,d epsilon,dx)
      =c_0 Q_c a^(-1/3) da u^(1/6)du dx/V
                  [delta_(+1)+delta_(-1)](d epsilon),
                  a>0, 0<u<u_0.                       (34)

This follows directly by substituting ell=t a in (33), using (23),
and dominating by c_0 A_q a^(-1/3)u^(1/6). The majorant is integrable
on 0<a<=T, 0<u<u_0, including both small-lifetime and small-u boundaries.
Bounded continuous tests of the additional actual marks converge to the
pushforward of (34) under B=B_c(u), k_phys=k(u), direction=epsilon e_1.
Continuity for fixed positive u,a and this same majorant prove the
statement; no unjustified uniform positive-u floor is required.

After normalizing the latent-u part of (34), its density is

    (7/6)u_0^(-7/6)u^(1/6)du,

and the two signs are equiprobable. Lifetime, latent u, sign and spatial
position factor in this limiting intensity. Birth and cubic marks are
deterministic functions of u and retain that correlation. There is no
atom at u=0 or k_phys=0. In fact k(u) is asymptotic to a positive constant
times sqrt(u) at zero, so the mass below a small cubic mark has order
k_phys^(7/3). Summing the two signs and integrating u in (34) gives
C_edge a^(-1/3)da dx/V, confirming (10)'s coefficient.

## 8. Independent-copy PRM from exact single-field multiplicity

For almost every control the field has zero or one finite H0 interval.
Define its marked point measure on good controls by the explicit selected
pair above, and zero on the null bad controls. This is Borel: the root
branches are smooth on each sign/wedge component, their values and
positions are continuous, and the control-region indicators are Borel.
Outside the wedge the measure is empty. In particular for every t,

    N_t in {0,1},
    E[N_t;N_t>=2]=0.                                   (35)

For a single field, its point process on lifetime window (0,Tt] is
eventually empty almost surely as t decreases to zero. If a sample has
a finite bar, its lifetime is fixed and positive; otherwise its process
is always empty. This also follows in probability from (8).

Let f>=0 be a bounded continuous test of the scaled lifetime and the
retained marks, supported in 0<a<=T. Because there is at most one point,
the exact one-copy Laplace deficit equals its one-point integral:

    D_t(f)=E[1-exp(-sum f)]
           =E sum [1-exp(-f)].                          (36)

The measure convergence in (34), with its integrable domination, gives

    t^(-2/3)D_t(f) -> integral [1-exp(-f)]d eta_edge,    (37)

or the corresponding physical-mark pushforward if those marks are used.
For n independent copies the Laplace functional is (1-D_(t_n)(f))^n.
Under t_n->0 and n t_n^(2/3)->lambda>0 it tends to

    exp(-lambda integral [1-exp(-f)]d eta_edge),        (38)

the PRM Laplace functional. For lifetime and birth position alone this
is exactly (10). Its mean on (0,T]xX is
lambda(3/2)C_edge T^(2/3). The location measure is normalized Haar
dx/V; C_edge was already a whole-field coefficient, so multiplying it
by a further V would be incorrect.

This is a genuine rare-event process theorem for the actual constructed
law, and uses more than a first-moment exponent: the complete field
construction proves the multiplicity statement (35). No same-field
Poisson, factorial, spatial-growth, coalescing-contact or expanding-domain
result is asserted. There is no numerical field draw or independent seed
in this argument.

## 9. Smooth local controls, density deficits and optional normalization

The core coordinate change w=z_perp sqrt(1+s^2/4), independent of the
controls, makes the whole local family exactly

    b-s^4/4+u s^2/2+v s-|w|^2.                         (39)

Thus its two controls are the actual canonical quadratic and linear
controls of the quartic potential, smooth on an open rectangle through
the origin. At the origin the germ's derivative ideal is generated by
s^3,w_1,...,w_(d-1). Taylor's formula identifies its local Jacobian
quotient with the span of 1,s,s^2; after omitting the additive height
constant, the derivatives with respect to v and u supply its two
independent generators s and s^2/2. This verifies the essential two
local directions directly. We invoke no unseen classification or
versal-deformation theorem to assert equivalence of arbitrary smooth
families to (39). The exact displayed canonical unfolding suffices.

The full random law still does not have the Gaussian two-site observation
contract. At fixed distinct sites p,q, their two values and two gradients
form a vector in R^(2d+2), a smooth function of the d translation
parameters and the two controls. Since d+2<2d+2 for d>=2, its image is
Lebesgue-null. Explicitly, cover the parameter space by countably many
compact coordinate charts, use their bounded derivatives to make the
maps Lipschitz, then cover each d+2 dimensional chart by epsilon cubes.
Their images have total (2d+2)-volume O(epsilon^d), tending to zero.
The observation law is therefore singular. This does not imply that
its covariance matrix is singular; nonlinear distributions can have
full covariance on a lower-dimensional support.

Stationarity follows exactly from Haar translation Y, independently of
controls. Uniform spatial C^q bounds follow from the fixed smooth
surgery/cutoff and bounded u,v, for each q, so all positive norm moments
are finite. The scalar G(0) is bounded and has positive variance:
conditional on each control it is a nonconstant smooth function at a
uniform point. It is therefore not a nonconstant Gaussian variable.
We neither supply independent Gaussian coefficients nor claim any
covariance match, P5 spectral carrier, arbitrary-jet rank, full contact
density, or Gaussian theorem admission.

If deterministic centering and variance normalization are desired, put
mu=E G(0), sigma^2=Var G(0)>0 and Z=(G-mu)/sigma. Its pairing and
positions are unchanged, and every lifetime is divided by sigma.
Consequently

    nu_Z(t)=sigma nu_G(sigma t),
    C_edge,Z=sigma^(2/3)C_edge,G,
    C_tip,epsilon,Z=sigma^(5/4)C_tip,epsilon,G.           (40)

The limiting physical birth height becomes (B_c(u)-mu)/sigma and the
cubic mark becomes k(u)/sigma. These are fixed-law scale transformations,
not comparisons of covariances between different populations.

## 10. What the actual population establishes about taxonomy

The field at u=v=0 has the explicit A3 function-germ normal form
b-s^4/4-|w|^2. The family now samples both of its essential local
coefficients with a positive interior density, rather than following
one prescribed C1-not-C2 path. Almost every realization is Morse,
and the actual bar has been selected globally, not assumed.

Nevertheless the dominant short bars in the full population approach
ordinary cubic folds at fixed positive u. Their expected density has
power -1/3 and a coefficient determined by the control population,
while the away-edge quartic-tip contribution has density power +1/4
and cumulative power 5/4. The small-u domination (28) is what permits
this conclusion without silently replacing the whole cusp population
by its fixed-u edges.

Together with the restricted-path example, this falsifies two shortcuts:
an A3 label alone does not assign a sampling-independent density power,
and a population containing an A3 contact need not have its leading
short-bar statistics governed by the tip. One must identify the actual
sampling measure, the singular strata it weights, the physical height-gap
law and the global selector. The present result supplies these inputs
for this particular family; it is not a theorem about every A3 ensemble.

The -1/3 exponent in (8) is compatible with the ordinary cubic edge
relation (31), despite the law's singular full observation distribution.
This does not extend the Gaussian counting theorem by covariance alone.
We counted by explicit control pushforward, and the coefficient (7) is
not inferred from the Gaussian jet functional. No Hk, scientific uptake,
formal implementation, sampler coupling or full singularity taxonomy is
closed by the present preparation.

## 11. Operational context, source review and constructor boundary

The following current repository sources were freshly hashed in this detached
construction checkout. The quartic seed and matched surgery are reused narrowly;
the full rectangle guards, root classification, selected counting and population
limits are explicitly rederived in Sections 1-10. The earlier one-parameter
probability law is not transferred. ST is conditional-taxonomy context only.

| Context | Current repository source | UTF-8 bytes / lines | SHA256 |
| --- | --- | --- | --- |
| Quartic construction | [stationary quartic field](../quartic_cusp_field/PROOF.md) | 35931 / 751 | `33ba8362069346058ba33bbd8c1bd1feed62b749a96b6cbbc8836ddcbc6e4f2e` |
| ST taxonomy | [regular-fold structural theorem](../fold_structural_universality/PROOF.md) | 43927 / 804 | `3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531` |

The mathematical author consumed the original quartic preparation, 32,334 bytes /
695 lines, SHA256
`24e6f4149c0a2df2f8df1dbfa8d66da3e01a7d12bee4b803bac093cfea36bf3f`.
The author had a prior whole saved-byte read of that source and additionally
reread its actual lines 72-353 for this two-control task. The original/current
quartic Sections 1-11 match exactly: 26,625 bytes / 602 lines, SHA256
`9f1426a21b40b524ad4b25c81368489b0aa7a0c29f79659eb556bb4d1579f6d0`.
The constructor checked the actual core mapping here. It does not create a new
whole-751-line author read. The author's original ST whole read and earlier
bounded lines 691-737 remain separately exposed in the historical ledger.
No Gaussian, higher-homology or amplitude theorem premise is consumed here.

The original two-control preparation is
`two-parameter-cusp-population-preparation.md`, 32,809 bytes / 696 lines,
SHA256 `001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6`.
`final_contract_review` freshly read all actual lines in untruncated ranges
1-240, 241-480 and 481-696: mathematical PASS, no must-fix.
`benchmark_formal_audit` separately freshly read all actual lines in untruncated
ranges 1-232, 233-464 and 465-696: mathematical PASS, no must-fix.
Root fully read the actual original in untruncated ranges 1-230, 231-464 and
465-696. These original-source performances do not review the new incorporated
wrapper or become provider-independent acceptance.

Both nonauthor original-source reviewers freshly reproduced the quartic695
and ST804 hashes, with bounded rereads of quartic lines 72-353 and ST lines
691-737. Their preceding separate whole-source exposure is retained; two
additional whole dependency audits are not claimed. The constructor now checks
the preserved two-control mathematical core, the historical-ledger body,
current context identities, baseline README preservation and local links.
These are bounded integration checks, not a new full incorporated-source
mathematical verdict. Current source and metadata review remain PENDING until
separately completed against these new actual bytes.

Source/route exposure is substantial. The mathematical author wrote the earlier
quartic, spectral, selected-bar, process, higher-homology and amplitude sources.
Root supplied the prefile full-rectangle, edge/tip and physical-mark questions.
Benchmark supplied the separate prefile warning that wedge coverage does not
supply the full global guards. Final_contract_review contributed a separate
fold-chart route, prior carrier/amplitude attacks and the amplitude wrapper;
for this source it is the original696 reviewer and subsequent wrapper constructor.
All performers are OpenAI/Codex AI agents. Organizational independence remains zero.

No new primary-source fetch is credited in this incorporation. The author's
prior bounded AGV nomenclature preview is recorded below; no full classification,
versal-deformation theorem or new full primary-book read is asserted. The
local canonical family and quotient statement are explicitly derived in Section 9.
Earlier failed preview opens by root and final remain without primary-content
credit and were not retried as part of this construction.

Section 12 preserves the original author's ledger body verbatim, including its
historical request for an exact-file review and original artifact paths. Those
statements describe preparation chronology. The completed original696 reviews
above and pending incorporated-source reviews are separate. Publication, PR
attachment, hosted checks, merge and ordinary receipt bindings will be recorded
only when obtained; this source records none of those new integration facts.
No compiler, sampler, seed, installation, scientific-status change or external
post belongs to this construction.

## 12. Preserved original author bindings, reading extents and exposure

The only reused actual-field input is
`/Users/dylanroy/Documents/Codex/2026-10-08/turn-the-below-into-a-reality/quartic-cusp-field-preparation.md`,
32,334 UTF-8 bytes / 695 lines, SHA256
`24e6f4149c0a2df2f8df1dbfa8d66da3e01a7d12bee4b803bac093cfea36bf3f`.
The author previously reread all actual saved bytes before freezing it.
For this task its hash was freshly reproduced and actual lines 72--353
were freshly read, covering the seed, matching surgery, restricted
roots, complete critical list and enclosing barrier. We reuse that
construction narrowly and explicitly rederive the full-rectangle
guards, root classification and selection; its one-parameter probability
law is not transferred to the new two-dimensional population. No
completed independent exact-source verdict on this new file is inferred
from a review of the preceding source.

The structural source's conditional-order taxonomy is context, not a
counting theorem imported into this singular law:
`fold-universality-worktree/experiments/universality/fold_structural_universality/PROOF.md`,
SHA256 `3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531`.
It had a prior full actual-source read; its Section 10, lines 691--737,
was freshly read during the preceding quartic task. It separates height
order, observation density, radial measure and selected weight rather
than assigning an unproved higher-order class law.

Prior SG, higher-homology, selected-bar, iid-process and amplitude source
authorship supplied substantial route exposure. Their mathematical
conclusions are not needed for (6)--(10): the actual dependencies are
the explicit smooth cutoff construction, root analysis, elementary
connected-component topology, Haar translation, ordinary substitutions,
Tonelli, the implicit function theorem, dominated convergence and the
iid Laplace product. In particular no arbitrary conditioned-Gaussian
genericity or independence premise is transferred.

Root's prefile contributions included the proposed full rectangle,
coverage condition, fold-edge expansion and constants, tip pushforward,
latent-u weighting, birth-height and physical-cubic-mark targets. The
author checked them by the calculations (19)--(40). The communicated
benchmark prefile attack highlighted that wedge coverage alone does not
supply the global smallness guards; (14)--(17) supply those independently.
Any later exact-file verdict must be separately bound to the final saved
bytes, not relabel this communicated route discussion as a whole-file read.

For A_k nomenclature only, the author had actually read the official AGV
publisher pages and the two-page primary *Lists of singularities* preview,
printed pages 242--243, during the preceding task. It identifies the A_k
function-germ series. No whole book, real-classification theorem or
versal criterion was read or consumed here. [Actual primary preview](https://page-one.springer.com/pdf/preview/10.1007/978-1-4612-5154-5_15).
Formula (39) and its elementary quotient calculation provide the exact
local statement required by this proof, without an unseen primary import.

No repository, frozen predecessor, GitHub artifact, model registry,
formal source or status was changed. No compiler, sampler, seed,
installation, empirical draw or external post was used. This is a
source-exposed same-team analytic preparation, organizational credit zero,
awaiting a fresh nonauthor full actual-byte review.
