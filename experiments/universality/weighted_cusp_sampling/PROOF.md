# Weighted cusp sampling: selected-bar phase laws and physical marked processes

Mathematical source author: OpenAI/Codex
/root/universality_next_decisive_gap_20261025, 25 October 2026.
Root01a11cbf selected and coordinates this finite source-only task under
claim2860/f6d6f589, actual pickup
[main340/6399519074](https://github.com/d6g8k5htny-coder/main/pull/340#issuecomment-6399519074).
Root retains native publication, custody and integration; this author writes
the three new sibling documents only. All work is source/route exposed,
same provider, organization/blind/human independence0, scientific effect NONE.
No mathematical program, field draw, numerical integration, sampler, compiler,
Lean or test was run. The [review record](REVIEW.md) preserves the dated V1
preparation and subsequent actual source-bound review dispositions; no prior
verdict is silently rebound to this metadata successor. The
[reader guide](README.md) states the scope.

The sole theorem input is the actual global family in
[TWO at f02758555b4bce4554e34e8913947f69578d0816](https://github.com/d6g8k5htny-coder/main/blob/f02758555b4bce4554e34e8913947f69578d0816/experiments/universality/two_parameter_cusp_population/PROOF.md):
38350 UTF-8 bytes / 782 lines, SHA256
9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326,
Git blob d86cc513b620190f601d483c59ad4687cbe9ff62.
Its mathematical Sections1-10 are 28073 bytes / 620 lines, SHA256
d4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213.
The whole/core identities are distinct. This immutable input is an unmerged
source at this preparation cut, not an assertion that its path has landed.
Its original author is universality_review, wrapper constructor
final_contract_review, retained coordinator Root01a11cbf.
No old verdict on TWO or the separate #595 documentary repair reviews this
new source. Its global geometry is retained; the probability law below is new.

## 1. The complete statement and conventions

All lengths, heights and derivatives use TWO's fixed physical torus and
unnormalized field units. Fix its coefficient field K, and ordinary superlevel
H0 with the elder rule. Exclude the essential global-maximum class.
All construction choices in Section2 and all sampling choices below are fixed
before any lifetime limit. The whole-torus expected selected lifetime density
is denoted nu(t); divide by V=L^d only for a per-volume density.

Let the fixed family be F_(u,v), with rectangle |u|<u0, |v|<v0 and independent
Haar translation Y. Write eta_c=2/(3sqrt(3)), h_max=1/4. Inside
W={0<u<u0, |v|<eta_c*u^(3/2)}, put epsilon=sign(v), eta=|v|/u^(3/2).
The actual selected lifetime is ell=u^2*kappa(eta). The inverse of kappa is
eta(h), and q(h)=|deta/dh|. The construction, not an abstract point-process
assumption, will prove that there is one finite bar on good controls in W
and none off W.

Choose fixed a>-5/2 and theta>-1. For each epsilon in{+,-}, let w_epsilon be
nonnegative and continuous on every compact subset of [0,eta_c), with
w_epsilon(eta)/(eta_c-eta)^theta -> c_epsilon>0 at eta_c.
No global positive smooth control-density premise is asserted.
Define
 gamma=a/2+5/4>0, beta=2(theta+1)/3>0,
 Wsum=sum_epsilon integral_0^eta_c w_epsilon(eta)deta,
 Z=u0^(2gamma)*Wsum/(2gamma), C=p/Z, 0<p<=1.             (1)
With respect to du dv, put
 rho(u,epsilon*u^(3/2)*eta)=C*u^a*w_epsilon(eta) on W.   (2)
For p<1, choose a declared absolutely continuous probability density rho_out
on the rectangle outside W, and use (1-p)*rho_out there. For p=1 this
component is absent. This fully specifies a probability law together with the
independent Haar Y. Boundary, discriminant and tie controls carry no mass.

Set
 Kc=4/3^(5/4), r_epsilon=(2/3)c_epsilon*Kc^(-beta),
 Rsum=r_++r_-,
 R_epsilon(h)=w_epsilon(eta(h))*q(h).                   (3)
The following are the conclusions proved below.

For 0<t<u0^2/4 there is the exact selected density
 nu(t)=C sum_epsilon integral_(2sqrt(t))^u0
             u^(a-1/2) R_epsilon(t/u^2)du
      =(C/2)t^(gamma-1) sum_epsilon integral
             h^(-gamma)R_epsilon(h)dh,
             with lower limit h=t/u0^2 and upper limit h=1/4. (4)
Here h IS kappa, and the UPPER limit is one quarter. There is no fourth-root
substitution. The last integral equivalently integrates
w_epsilon(eta)kappa(eta)^(-gamma)deta over kappa(eta)>=t/u0^2.
The density is zero above u0^2/4. The exact cumulative count is
 E N_t=(C/(2gamma)) sum_epsilon integral_0^eta_c
  w_epsilon(eta) min{u0^(2gamma),(t/kappa(eta))^gamma}deta. (5)
There is no atom at zero or at the maximum lifetime. Almost surely N_t is0
or1 for every t, so P(N_t=1)=E N_t exactly.

The density asymptotics have the following three coefficients D:
 beta<gamma: nu(t)~D*t^(beta-1),
  D=C*Rsum*u0^[2(gamma-beta)]/[2(gamma-beta)];            (6)
 beta=gamma: nu(t)~D*t^(beta-1)*log(u0^2/t),
  D=C*Rsum/2;                                          (7)
 gamma<beta: nu(t)~D*t^(gamma-1),
  D=(C/2)sum_epsilon integral_0^eta_c
                  w_epsilon(eta)kappa(eta)^(-gamma)deta. (8)
Every D is positive and finite in its declared regime.
Writing delta=min(beta,gamma), the cumulative is asymptotic to
(D/delta)t^delta off equality and (D/beta)t^beta*log(u0^2/t) at equality.
The fixed units inside the logarithm change no leading coefficient.

The complete joint physical marked independent-copy processes converge to
the PRM pushforwards in Sections7-10. They use
 s(t)=t^beta for beta<gamma,
 s(t)=t^gamma for gamma<beta,
 s(t)=t^beta*log(u0^2/t) for beta=gamma,                 (9)
and t_n->0, n*s(t_n)->lambda in(0,infinity).
Their lifetime/location marginal intensity is
 lambda*D*x^(delta-1) dx dX/V,                          (10)
where x=ell/t and X is actual birth position.
For any fixed T>0, conditional on an actual bar with ell<=Tt, the observable
 J_t=log(r/r_*)/log(t/ell_*)                            (11)
has limit point mass1/3 in (6), Uniform(1/4,1/3) in (7), and point mass1/4
in (8), for any fixed positive physical reference units r_*,ell_*.
Here r is the actual shortest physical birth-to-killing-saddle distance.
The theorem concerns independent whole copies of this declared law on one
fixed torus. A single field's shrinking-window process becomes empty.

## 2. Unchanged global family and actual selected multiplicity

We restate every guard from TWO Sections1-3. Fix integer d>=2, L>0,
X=(R/LZ)^d, V=L^d, omega=2pi/L. Let W_i=3^(-i),
b=sum_i W_i, Delta=2W_d, and H(y)=sum_i W_i cos(omega*y_i).
Its 2^d critical points have coordinates0 or L/2 and diagonal nonzero
Hessians. The first differing W_i dominates all later weights, so their
values are distinct; the origin is the unique maximum and every other old
height is at most b-Delta.

Choose R with R^2<min(W_d,1), put r1=R/4, and use the exact chart
 z_i=sqrt(2W_i)sin(omega*y_i/2), s=z_1.
There H=b-s^2-|z_perp|^2. Its superlevel above b-R^2 is exactly the chart
ball D_R: the nonnegative coordinate losses force |y_i|<L/4.
Fix the smooth nondecreasing function c:[0,infinity)->[0,1] with
 c(r)=r/4 for r<=r1^2; c(r)=1 for r>=R^2/2;
 c(r)>0 for r>0; c'(r)>=0.
These compatible cutoff choices are the actual fixed ones of TWO.
In D_R set
 F0=b-s^2 c(s^2+|z_perp|^2)-|z_perp|^2,
and retain H outside. The equality on a full collar makes F0 smooth.
Its chart derivatives vanish only at the origin, since
 (F0)_s=-2s[c(r)+s^2 c'(r)],
 (F0)_(z_i)=-2z_i[1+s^2 c'(r)].
In |z|<=r1 the formula is exactly
 F0=b-s^4/4-(1+s^2/4)|z_perp|^2.

Fix the same radial smooth chi,1 on |z|<=r1/2 and0 on |z|>=r1, and set
 F_(u,v)=F0+chi*(u*s^2/2+v*s).
Set
 a_*=r1/4, d_*=r1^4/64,
 g_*=min_(r1/2<=|z|<=r1)|grad_z F0|>0,
 B_j=||chi*s^2/2||_(C^j), D_j=||chi*s||_(C^j), j=0,1.
For j=1 the gradient norm suffices. Fix the same u0,v0 satisfying
 v0>=eta_c*u0^(3/2),
 u0<a_*^2/4, v0<a_*^3/4,
 B_1*u0+D_1*v0<g_*/2,
 B_0*u0+D_0*v0<d_*/4.                                 (12)
These strict conditions, not merely wedge coverage, remain in force.

The perturbation gradient excludes roots on the transition annulus.
In the exact core transverse criticality forces z_perp=0 and axial roots
solve -s^3+u*s+v=0. All such roots have |s|<a_*: otherwise
|s|^3<=u0|s|+v0<|s|^3/4+|s|^3/4, a contradiction.
Thus there are no unlisted chart critical points. Old critical points outside
the chart are unchanged. At an axial root the chart Hessian is
 diag(u-3s^2,-2(1+s^2/4)I_(d-1)).
Its inertia is the physical inertia under the chart diffeomorphism.
Every new critical value is within d_*/4 of b: the bound
|P_(u,v)(s)|<5d_*/32 follows from
P=-s^4/4+u*s^2/2+v*s and a_*^4=d_*/4.
Also d_*<Delta, so old levels remain separated from all new levels.

Outside |z|<r1/2, monotonicity gives
 b-F0 >=(r1^2/16)|z|^2>=d_*
within the chart; here r1^2/16<1 follows from R^2<1.
Outside the chart the loss is at least R^2>d_*.
The height perturbation bound in (12) therefore gives
 F_(u,v)<=b-3d_*/4 outside the exact core,
 while all new critical heights exceed b-d_*/4.         (13)
This is a global high-level barrier.

There are three simple axial roots exactly in W. For v>0 write
s=sqrt(u)z and Phi_eta=-z^4/4+z^2/2+eta*z. Its roots satisfy
z_-<z_0<z_+, with the outer roots maxima and middle root an index-(d-1)
saddle. The left maximum is lower: the peak difference vanishes at eta=0
and has derivative z_+-z_->0 for eta>0. For v<0 reflect s; the right
maximum is lower. The gap to the middle saddle is u^2*kappa(eta)>0.

At levels between that saddle and the younger maximum, (13) confines the
whole global superlevel to the core. The transverse fibers are exactly
 |z_perp|^2 <=[b+P_(u,v)(s)-level]/(1+s^2/4).
Contracting them to their centers increases F and stays in the superlevel.
The axial section has two intervals, one containing each maximum; they first
touch at the middle saddle. There is no exterior bypass by (13).
Thus this is the actual elder merger, killing the younger maximum.
The full field has exactly two maxima, so it has exactly one finite ordinary
H0 bar, with the older/global maximum essential. This conclusion holds over
the fixed coefficient field K without a candidate-to-bar substitution.

Off W and off the discriminant, the cubic has one root with negative axial
second derivative. The whole field has exactly one maximum and hence only
the essential H0 birth on the connected torus; there is no finite H0 bar.
The discriminant graphs v=+-eta_c*u^(3/2), u>0, the origin and the Maxwell
line v=0 inside W are plane-Lebesgue-null. We retain TWO's zero-count
convention there, without claiming that those fields are Morse/distinct.

On each good sign/wedge region the roots, critical values and physical
positions are continuous, with smooth local branches. The explicitly selected
bar measure is therefore Borel. Off W it is empty, and the bad locus is Borel.
The new law in (2) is absolutely continuous after the normalization proved
next; all these exceptions still have probability0. Consequently its actual
whole-field count has N_t in{0,1} simultaneously for all t.
This derives multiplicity from the unchanged complete global construction.
It is not a P5 Gaussian admission or a singleton assumption.

Independent Haar translation makes G(x)=F_(u,v)(x-Y) stationary.
Conditional on every good bar control, its actual birth position is Y plus a fixed
critical location and is Haar-uniform, independent of the controls and all
translation-invariant marks. The physical direction is exactly epsilon*e1.
All roots stay in the same small chart, so the displacement is the unique
shortest torus displacement. The smooth bounded control family has uniform
spatial C^q bounds for every fixed q, regardless of its new density.
This spatial smoothness does not assert a smooth control density.

## 3. Gap inverse, sampling envelopes and normalization

For eta in(0,eta_c), define
 kappa(eta)=Phi_eta(z_-)-Phi_eta(z_0).
Endpoint differentiation uses Phi'_eta(z_i)=0, giving
 kappa'(eta)=z_--z_0<0, kappa(0)=1/4, kappa(eta_c)=0.
Hence kappa is invertible and q(h)=1/[z_0(eta(h))-z_-(eta(h))].
At eta=0 the gap in roots is1, so q(h)->1 at h=1/4.

For completeness the endpoint constants are derived from the actual cubic.
Set z_c=-1/sqrt(3), d_eta=eta_c-eta, H=z-z_c. Then
 Phi'_eta(z_c+H)=sqrt(3)H^2-H^3-d_eta.
Putting H=sqrt(d_eta)w, the two nearby roots are
w=+-a_c+O(sqrt(d_eta)), a_c=3^(-1/4), by the implicit function theorem.
Integrating the negative derivative between the roots gives
 kappa(eta)=Kc*d_eta^(3/2)[1+O(sqrt(d_eta))],
 Kc=integral_(-a_c)^a_c(1-sqrt(3)w^2)dw=4a_c/3,
 z_0-z_-=2a_c*sqrt(d_eta)[1+O(sqrt(d_eta))].             (14)
Using the exact inverse derivative rather than differentiating an asymptotic,
 h^(1/3)q(h)->Kc^(1/3)/(2a_c)=(2/3)Kc^(-2/3).         (15)
Continuity on the rest of the interval gives q(h)<=A_q h^(-1/3).

The positive function d_eta(h)/h^(2/3) extends continuously to both endpoints,
with positive finite endpoint values, including Kc^(-2/3) at0.
Thus there are fixed m_delta,M_delta>0 such that
 m_delta*h^(2/3)<=eta_c-eta(h)<=M_delta*h^(2/3)
 for every 0<h<=1/4.                                   (16)

The hypotheses on w_epsilon imply one global bound
 w_epsilon(eta)<=(M_w)(eta_c-eta)^theta.
Indeed the ratio has finite endpoint limit and is bounded on every remaining
compact interval. This is derived from the actual hypotheses, not an added
integrability assumption. Because theta>-1, Wsum is finite; c_epsilon>0
makes it positive. Equations (15)-(16) then give, for a fixed A_R,
 R_epsilon(h)<=A_R*h^(beta-1),
 R_epsilon(h)/h^(beta-1)->r_epsilon.                    (17)
When theta<0, the lower bound in (16) supplies the upper bound after taking
the theta power; for theta>=0 use its upper bound.

The change v=epsilon*u^(3/2)*eta has Jacobian u^(3/2) for each sign.
Since a+3/2=2gamma-1, the W mass is
 C sum_epsilon integral_0^u0 u^(2gamma-1)du
                  integral_0^eta_c w_epsilon(eta)deta = C Z=p.
It is integrable at the origin because gamma>0 and at the fold boundary
because theta>-1. Adding the stated exterior mass gives total probability1.
An integrable density may be unbounded at these boundaries while assigning
them mass0. No random field is drawn in this argument.

## 4. Exact selected measure, density and cumulative identities

For any nonnegative Borel lifetime test f, Section2 and Tonelli give
 E sum_finiteH0 f(ell)
 =C sum_epsilon integral_0^u0 integral_0^eta_c
      u^(2gamma-1)w_epsilon(eta)f(u^2*kappa(eta))deta du. (18)
The exterior component and null bad controls contribute no bar.

For fixed u substitute h=kappa(eta) and ell=u^2*h. This yields the first
density form in (4), with actual selected counting and no extra spatial
volume factor. Substituting u=sqrt(t/h) in that form gives its second:
u^(a-1/2)|du|=(1/2)t^(gamma-1)h^(-gamma)dh.
The lower limit is h=t/u0^2 and the upper limit h=1/4.
For fixed eta, integrating u up to min(u0,sqrt(t/kappa)) gives (5).
These direct substitutions prove a density and a cumulative measure
independently; no asymptotic cumulative formula is differentiated.

On a compact positive-t interval, h is bounded away from0, R is bounded,
and the moving endpoints are regular. The integral is finite and continuous
there; it shrinks to0 at the maximum lifetime. This chosen representative
is a density of total mass p. It is not a conditional lifetime probability
density of mass1 unless p=1. All lifetimes on good bar controls are positive,
and the displayed pushforward is absolutely continuous, so there is no zero
or endpoint atom. The source's chosen bad-control convention changes no measure.

## 5. Three asymptotic regimes and all-lifetime bounds

First suppose beta<gamma. Divide the first expression in (4) by t^(beta-1).
After zero extension for u<=2sqrt(t), its integrand is
 C*u^[2(gamma-beta)-1]
     [R_epsilon(t/u^2)/(t/u^2)^(beta-1)].
It converges pointwise at every fixed u>0 to C*r_epsilon times that power,
and (17) bounds it by the integrable C*A_R*u^[2(gamma-beta)-1].
Dominated convergence proves (6). In particular the normalized contribution
from u<=r is bounded by
 C*2*A_R*r^[2(gamma-beta)]/[2(gamma-beta)].               (19)
This is the requisite uniform shrinking-tip tail bound.

If gamma<beta, change u for fixed eta in (18). The density is exactly
 (C/2)t^(gamma-1) sum_epsilon integral_0^eta_c
 w_epsilon(eta)kappa(eta)^(-gamma)
                    1{t<u0^2*kappa(eta)}deta.           (20)
Near eta_c its integrand is comparable to
(eta_c-eta)^[theta-3gamma/2], which is integrable exactly when gamma<beta.
On the rest it is bounded. Dominated convergence proves (8), with a positive
finite D. The same integrable function controls every discarded eta tail.

If beta=gamma, let Rtot=sum_epsilon R_epsilon.
The second expression of (4) is
 (C/2)t^(beta-1) integral [Rtot(h)/h^(beta-1)] dh/h,
 with LOWER limit h=t/u0^2 and UPPER limit h=1/4.        (21)
The bracket is bounded and tends to Rsum>0. Split at a fixed small h0.
The upper part is finite; on the lower part bound the bracket between
Rsum+-epsilon, integrate dh/h, divide by log(u0^2/t), then let epsilon go
to0. This proves (7). The factor one-half is the actual Jacobian in (4);
in u coordinates it is the length of the log band from sqrt(t) to fixed u.
There is no second logarithm or missing one-half.

The same bounds give constants, depending on this fixed law, with
 nu(t)<=A*t^(delta-1) in either noncritical regime,
 nu(t)<=A*t^(beta-1)*(1+log(u0^2/t)) at criticality,      (22)
for all sufficiently small t>0. The positive-t rest is harmless.
Substitute t*x into the already established density and integrate
0<x<=T. In noncritical regimes use x^(delta-1); at criticality, for
L(t)=log(u0^2/t)>=1, use x^(beta-1)(1+|log x|) with fixed-T constants.
These are integrable at0 because delta,beta>0. Dominated convergence gives
the asserted cumulative coefficients, including D/beta at criticality.

It also proves uniform tightness of the rescaled mean near lifetime0:
after division by s(t), the expected mass for ell<=epsilon*t is bounded
by a constant times epsilon^delta off equality, or a constant times
integral_0^epsilon x^(beta-1)(1+|log x|)dx at equality.
Both vanish with epsilon. None of these bounds differentiates a count
asymptotic or uses a hidden numerical lifetime window.

## 6. Actual physical marks and mark spaces

For a good bar let B be its younger-maximum height, P its actual birth
position, r its shortest physical endpoint distance, k3=ell/r^3, and
direction epsilon*e1. Write
 D_eta=z_0(eta)-z_-(eta)>0,
 y(s)=(2/omega)arcsin(s/sqrt(2W_1)),
 g0=y'(0)=2/[omega*sqrt(2W_1)].
Reflection gives the same heights and distances at negative sign, so exactly
 B=b+u^2*Phi_eta(z_-),
 r=y(sqrt(u)z_0)-y(sqrt(u)z_-),
 ell=u^2*kappa(eta).                                   (23)
All positions are in the actual physical y coordinates. This is not a chart
distance substituted into a physical observable.

For fixed u>0, (14) and the mean value theorem give as eta approaches eta_c
 r=2*g(u)*sqrt(u)*a_c*sqrt(eta_c-eta)[1+o(1)],
 g(u)=y'(-sqrt(u)/sqrt(3))
     =g0*[1-u/(6W_1)]^(-1/2),
 B->B_c(u)=b-u^2/12,
 k3->k(u)=sqrt(u)/[2sqrt(3)*g(u)^3]>0.                  (24)
The denominator is positive by the unchanged smallness guards.
The constant in k follows from Kc/(8a_c^3)=1/(2sqrt(3)).
For u in a compact subinterval bounded away from0, these expansions are
uniform. We use that only after a u truncation.

For small u, all z roots used in (23) stay in a fixed bounded interval.
The mean value theorem and smooth y' near0 give, uniformly there,
 r=g0*sqrt(u)*D_eta*[1+O(u)].                            (25)
In particular it holds uniformly for eta on compact subsets of[0,eta_c).
For limits involving an approaching fold edge we first perform the log-band
truncation in Section9 and use its uniform h->0 and u->0.
This avoids deriving moving-endpoint uniformity from a fixed-eta assertion.

Every marked process below retains (ell/t,P,epsilon*e1) and the specified
additional latent and physical marks. Unscaled physical birth/cubic marks
live in R_B x [0,infinity)_k3. Including k3=0 is essential for the tip and
critical limits. This is different from asserting a limit in a space that
requires k3 to remain strictly positive. Natural separation/height rescalings
and logarithmic marks live in their stated ordinary real/positive spaces.
All these finite products are separable metric mark spaces. H0 degree is the
constant0 if a degree mark is appended. Positions use normalized Haar dP/V.

## 7. Edge-dominated full joint mean measure

Let beta<gamma and x=ell/t. In variables (x,u,epsilon,P), the rescaled
one-bar mean density, divided by t^beta, is exactly
 C*x^(beta-1)*u^[2(gamma-beta)-1]
 [R_epsilon(tx/u^2)/(tx/u^2)^(beta-1)]
 1{tx<u^2/4} dx du dP/V.                               (26)
Its integrable majorant on0<x<=T is C*A_R*x^(beta-1)
u^[2(gamma-beta)-1]. Thus it converges as a finite measure, including total
mass and all small-lifetime/u tails, to the density with the bracket replaced
by r_epsilon and the indicator removed.

Equivalently the limiting mean measure is
 D*x^(beta-1) dx dP/V times the independent latent law
 P(epsilon)=c_epsilon/(c_++c_-),
 f_U(u)=2(gamma-beta)*u^[2(gamma-beta)-1]/u0^[2(gamma-beta)],
 0<u<u0.                                               (27)
For fixed x>0,u>0, h=tx/u^2->0. The joint actual physical maps converge to
 B=B_c(u), k3=k(u), direction=epsilon*e1,
 r/t^(1/3)=(x/k(u))^(1/3), J=1/3.                       (28)
For J, use the positive finite limiting separation factor in (28) and the
fixed reference units in (11). The term log of that factor divided by
log(t/ell_*) tends to0.

To justify the full pushforward, first restrict x>=epsilon_life and u>=r0
(and, if desired, away from the upper boundary, which has no atom).
On those compact parameters (24) is uniform and all displayed physical
maps converge uniformly. Then remove the truncations using (19), the
integrable x power and the u density in (27). Noncompact separation factors
as u->0 cause no extra moment premise: their probability tail is bounded by
the discarded u mass. This proves finite weak convergence of the complete
joint marked mean measure, not independent convergence of its marginals.
In particular B and k3 are correlated functions of the same U; they are not
replaced by independent marks. The source's original mark law is recovered
only in its specific uniform specialization.

## 8. Tip-dominated full joint mean measure

Let gamma<beta. Change u for fixed eta in the selected measure with ell=tx.
After division by t^gamma the exact latent mean measure is
 (C/2)x^(gamma-1) w_epsilon(eta)kappa(eta)^(-gamma)
 1{tx<u0^2*kappa(eta)} dx deta dP/V.                    (29)
The integrable function in (20), times x^(gamma-1), dominates it on0<x<=T.
Its limit is D*x^(gamma-1) dx dP/V times
 P(deta,epsilon)=w_epsilon(eta)kappa(eta)^(-gamma)deta
     /[sum_signs integral w_sign*kappa^(-gamma)].        (30)
This probability law has no endpoint atom. It need not make eta and epsilon
independent. It factors from x and P before the physical pushforward.

For each fixed eta in(0,eta_c), u=sqrt(tx/kappa(eta))->0. Set
 A4(x,eta)=g0*x^(1/4)*D_eta/kappa(eta)^(1/4),
 S(x,eta)=x*Phi_eta(z_-)/kappa(eta),
 Q4(eta)=kappa(eta)/(g0^4*D_eta^4).
Equations (23)-(25) give the complete joint limits
 r/t^(1/4) -> A4,
 (B-b)/t = S exactly,
 ell/r^4 -> Q4 = x/A4^4,
 B->b, k3->0, k3/t^(1/4)->x/A4^3,
 direction=epsilon*e1, J->1/4.                         (31)
The shared x,eta dependence in these marks is part of the assertion.

For rigorous passage to physical marks, first bound x away from0 and
restrict eta to a compact subinterval of(0,eta_c). Then u->0 uniformly,
all root/chart functions are uniformly controlled, and (31) is uniform.
The omitted eta=0 neighborhood has vanishing mass by local continuity and
positive kappa there; the eta_c tail has vanishing mass by (30)'s proven
integrability. The x tail vanishes by Section5. These bounds remove the
truncations for every bounded continuous joint test, and give tightness of
even the unbounded rescaled birth or separation marks. No claim of bounded
moments of those marks is needed. Retaining the original unscaled birth
and cubic marks appends the joint boundary value(b,0).

## 9. Critical log band and full joint physical mean measure

Let beta=gamma, tau=t/u0^2, L=log(1/tau), and
 Theta_t=log(u/u0)/log(tau).
For t small L>0, u=u0*exp(-L*Theta_t); since0<u<u0, Theta_t>0.
At ell=tx,
 h=tx/u^2=x*exp[-L(1-2Theta_t)].
The exact range h<1/4 is
 0<Theta_t<q_t(x)=1/2-log(4x)/(2L).                     (32)
For any fixed0<epsilon_life<=x<=T this upper endpoint tends uniformly to1/2.

In (26), now with beta=gamma, change u to Theta and divide its already
t^(-beta)-normalized density by L. The exact critical latent mean density is
 C*x^(beta-1)*[R_epsilon(h)/h^(beta-1)]
 1{0<Theta<q_t(x)} dx dTheta dP/V.                      (33)
The factor L from |du/u| cancels the normalizing L exactly.
For any fixed interior band d<=Theta<=1/2-d and epsilon_life<=x<=T,
both h->0 and u->0 uniformly. Equation (17) makes its limit
 C*r_epsilon*x^(beta-1) dx dTheta dP/V.                 (34)

Here are the boundary estimates needed before using physical expansions.
Equation (17) bounds (33) by C*A_R*x^(beta-1) on its exact Theta range.
For x in[epsilon_life,T], each omitted endpoint strip has Theta length at
most d+O(1/L); its normalized mass is bounded by a constant times that
length. For0<x<=T and L>=1, the whole range has length bounded by
A_T*(1+|log x|), so the omitted small-lifetime mass is bounded by
A*integral_0^epsilon_life x^(beta-1)(1+|log x|)dx.
It tends to0 uniformly with epsilon_life. In particular the possible
prelimit region Theta>1/2 has no surviving mass. These estimates establish
the full finite weak mean limit on the log mark, not only convergence on
interior rectangles.

Integrating (34) over0<Theta<1/2 gives coefficient C*Rsum/2=D.
Thus the normalized latent critical mark law is
 Theta Uniform(0,1/2), P(epsilon)=c_epsilon/(c_++c_-),
 with Theta, epsilon, x and Haar P independent.           (35)
The uniform density of Theta is2, not1; its interval length accounts for
the one-half in (7).

On a fixed interior band above, use (14) with h->0 and (25) with u->0:
 D_eta=2*a_c*Kc^(-1/3)*h^(1/3)[1+o(1)],
 r=A0*(tx)^(1/3)*u^(-1/6)[1+o(1)],
 A0=2*g0*a_c*Kc^(-1/3).                                (36)
All errors are uniform on that truncated band. Equation (23) gives B->b.
Equation (36) gives k3~A0^(-3)*sqrt(u)->0, where
A0^(-3)=1/(2sqrt(3)*g0^3).
For arbitrary fixed positive reference units, uniformly on the same band,
 J_t -> 1/3-Theta/6,
 H_t=log(k3/k_*)/log(t/ell_*) -> Theta/2.                (37)
Indeed log u/log t differs from Theta by fixed-unit terms tending to0,
and log x is bounded on the truncated lifetime interval.
The relation in the limit is H=1-3J, a deterministic correlation.

Combining (33)-(37) on the interior, then removing its endpoint/lifetime
strips with the explicit preceding bounds, proves convergence of the
complete joint critical mean measure. Its physical pushforward appends
 B=b, k3=0, direction=epsilon*e1,
 J=1/3-Theta/6, H=Theta/2.                              (38)
If H is omitted the remaining result is unchanged.
Theta and J cannot be declared independent; neither can H and J.
J has Uniform(1/4,1/3), with density12 on that interval.
No single power normalization of physical r is asserted to resolve all
critical scales. The logarithmic mark resolves those scales.
Using a mark space excluding k3=0 would discard the leading physical mass,
so (38) expressly uses the boundary-containing space in Section6.

## 10. From actual one-bar means to joint marked PRM laws

For the regime's joint marked object let Xi_t be its actual bar measure,
zero on nonbar controls and on the null bad locus.
Restrict to scaled lifetime0<x<=T. Section2 gives at most one point under
the actual normalized whole-field law, for every t.
Sections7-9 prove, as finite measures on this complete window and mark space,
 s(t)^(-1) E Xi_t -> mu_T,                              (39)
including total masses, tight lifetime0 tails and all physical mark tails.
Its lifetime/location marginal is D*x^(delta-1)dx dP/V and
mu_T(total)=(D/delta)T^delta>0.
The measures for different T are consistent restrictions.

Take n independent complete draws (u_j,v_j,Y_j) of this same fixed law.
There is no independence of marks within a bar. For nonnegative bounded
continuous f on the T window, the at-most-one property gives exactly
 q_t(f)=E[1-exp(-Xi_t f)]=E Xi_t(1-exp(-f)).
By (39), q_t/s(t)->integral(1-exp(-f))dmu_T.
Independence of the whole draws, t_n->0 and n*s(t_n)->lambda give
 E exp[-sum_(j=1)^n Xi_(t_n)^(j) f]
 =(1-q_(t_n)(f))^n
 ->exp[-lambda integral(1-exp(-f))dmu_T].                (40)
Here n*q_t^2=O(n*s(t)^2)->0. The right side is the PRM Laplace functional
of the entire joint physical/latent measure, so it preserves every
correlation of the pushforwards in Sections7-9.

For an elementary tightness/convergence formulation, each copy on the T
window is a Bernoulli point with success probability
p_t=E N_(Tt)~s(t)mu_T(total). Conditional on success its complete mark law
is E Xi_t/p_t, which converges weakly by (39). The number of successful
copies is binomial and converges to Poisson(lambda*mu_T(total)); their marks
are independent across copies and converge to normalized mu_T. This proves
finite-window marked point-process convergence. Counts are tight by their
bounded means; (39)'s explicit tails make all marks tight. There is no
limiting atom at lifetimeT, so continuous window approximation is valid.
Increasing T gives the corresponding local point-process convergence.
This argument requires no global factorial theorem or unproved mixing.

Equation (39) divided by its total mass also proves the actual conditional
short-bar laws for one field. Applying its bounded-test convergence to J
gives (11)'s trichotomy for any fixed T>0. This conditioning uses actual
P(N_(Tt)=1)=E N_(Tt), not a substitution of candidate/Palm counts.
A single fixed realization has at most one fixed positive finite lifetime;
therefore its process on0<ell<=Tt is eventually empty almost surely.
Only the independent-copy regime in (40) is nontrivial.

## 11. Exact recovery of the uniform source

Put a=theta=0 and w_+=w_-=1. Choose C=c0=1/(4u0v0) and
 p=(4*c0*eta_c/5)u0^(5/2)<=1/5.
The off-wedge density matching c0 on the rest of the rectangle recovers
TWO's full uniform law, not merely its conditional bar law.
Equation (1) gives exactly this W mass. Now
 gamma=5/4, beta=2/3, r_epsilon=(2/3)Kc^(-2/3),
 2(gamma-beta)=7/6.
Equation (6) becomes
 D=(8*c0/7)Kc^(-2/3)u0^(7/6)=TWO's C_edge.
Equation (5) and the proven cumulative law give (3/2)D*t^(2/3).
The edge U density in (27) is
 (7/6)u0^(-7/6)u^(1/6),
the signs are equal, and the complete physical maps are precisely TWO's
B_c(u),k(u),epsilon*e1 and normalized Haar birth position.
Its marked intensity and whole-field volume factors therefore agree.

For a fixed away-edge restriction eta<=eta_c-epsilon, (20) is valid as a
restricted exact transform even though the full uniform law is edge dominated.
Below t<u0^2*kappa(eta_c-epsilon), it gives exactly
 c0*t^(1/4) integral_0^(eta_c-epsilon) kappa(eta)^(-5/4)deta.
This recovers TWO's restricted-tip coefficient and cumulative factor4/5.
It is not an assertion that the full uniform law has a tip-dominated limit.

## 12. Four predeclared new-law predictions

Fix a=0, a common p>0, the same u0,v0 and entire guarded spatial family.
For each separate law set w_+=w_-=(eta_c-eta)^theta. The actual normalizer is
 Wsum=2*eta_c^(theta+1)/(theta+1),
 C_theta=5*p*(theta+1)/[4*u0^(5/2)*eta_c^(theta+1)].       (41)
Use a fixed declared exterior component of mass1-p.
Each law has full open-wedge support. Its field samples remain spatially
smooth; the control density need not extend smoothly at the cusp origin.
If theta=0 is intended to be the original uniform rectangle, use precisely
the p and exterior choice in Section11.

For theta<7/8 the edge coefficient is
 D_theta=[8*C_theta/(7-8theta)]*
          Kc^[-2(theta+1)/3]*u0^[(7-8theta)/6].
At theta=7/8 it is D_theta=(2*C_theta/3)*Kc^(-5/4).
At theta=1 the tip coefficient is
 D_1=C_1 integral_0^eta_c (eta_c-eta)kappa(eta)^(-5/4)deta,
which is finite by the already proved tip criterion.

| theta | Selected cumulative asymptotic | Density asymptotic | Conditional J limit |
|---|---|---|---|
| 0 | (3D_0/2)t^(2/3) | D_0*t^(-1/3) | point1/3 |
| 1/2 | D_(1/2)*t | D_(1/2) | point1/3 |
| 7/8 | (4D_(7/8)/5)t^(5/4)log(u0^2/t) | D_(7/8)*t^(1/4)log(u0^2/t) | Uniform(1/4,1/3) |
| 1 | (4D_1/5)t^(5/4) | D_1*t^(1/4) | point1/4 |

These are analytic predictions for four identified changed laws, made before
any new computation. The last two share the power but differ in their
logarithm and physical geometry. A fitted mean slope alone misses that
distinction. The exact density/CDF identities and physical correlations
provide independent observable targets for a future admitted falsifier.

No experiment is executed or admitted here. A later experiment needs its own
frozen sampling law, normalization, actual field/sampler/global ordinary-H0
selector/coupling, physical metric, uncertainty, resolution/lifetime window,
resource caps and failure predicates. Direct control-pushforward integration
tests this analytic population formula under its source semantics; it does
not independently verify the full field or global selector.
The completed Main343 uniform-law campaign remains its exact old law/outcome.
No new numerical/global-selector/complete-field/PRM confirmation is inferred.

## 13. Boundaries, authorship and source-review custody

The theorem changes sampling on one explicitly constructed actual global
family. It classifies its competing strata and full physical iid-copy law.
It is not a generic A3/versal-coordinate theorem, perturbation-stability
theorem, arbitrary stationary non-Gaussian admission, Gaussian full-jet
theorem, Hk theorem, same-field spatial increasing-volume process, regional
collision or all-factorial result. Scalar covariance and spatial norm bounds
do not remove these boundaries. The parameterized observation remains a
smooth image of d+2 controls/translation variables in dimension2d+2 for d>=2,
as in TWO; no Gaussian observation-density transfer is being made.

TWO's full actual782-line proof was read by this author during the immediately
preceding scout and its exact bytes were freshly rehashed for this task.
Sections2-3 here retain its exact field, guards, critical classification and
physical normalization, while explicitly reconstructing the actual one-bar
consequence under the new absolutely continuous law. The new probability
normalization, weighted transforms/envelopes, critical log band, all joint
marked limits and physical separation trichotomy are this author's new
composition. Named conventional imports are the elementary implicit function
theorem, smooth coordinate change/one-variable substitution, ordinary H0 elder
semantics for the stated Morse fields, Tonelli, dominated convergence and
binomial-to-Poisson/finite marked probability laws, with domains and
integrable bounds supplied above. No new primary-book review is claimed.

The author previously wrote the private read-only weighted-cusp scout and
therefore has substantial source/route exposure. Root supplied the selected
whole-task specification, independent checks of the Jacobian/exponents and
physical separation law, and the warning to distinguish h=kappa from a
fourth-root variable. The earlier ASCII integral meant upper endpoint h=1/4;
the clarification resolved notation rather than a mathematical defect.
This source states the lower/upper limits explicitly and preserves that
history. The private scout is input/route evidence, not an earlier verdict
on this final proof. Historical SG/PP/AMP sources explain the project
direction but are not additional theorem premises.

The exact V1 three-file composition received a distinct full conventional
mathematical/specification/metadata PASS, C0/I0/M0, sealed15:47:55UTC on
25October2026 by OpenAI/Codex /root/weighted_cusp_whole_math_review_20261025.
The companion REVIEW binds that actual V1 verdict and its whole-source reading
to the V1 identities. This successor changes metadata only and preserves the
entire mathematical body; its own final metadata/composition review is PENDING
at this author cut. Final source identities are recorded externally and in
REVIEW.md without a self-hash cycle. No prospective successor PASS, compiled
theorem, numerical acceptance, organization/provider/human/blind independence
credit or scientific promotion is claimed.

Source-only work does not clear Root403/R1/R2/R3, C257 unstarted/capability
hold, stronger NUM/item6, R83/R85, ComponentKernel/C299/H03 or foreign
selector/runtime/formal/integration/scientific scopes. Main336's new inherited
composition was UNOBSERVED at the governing cut and was not polled/retried.
Old340/343/595 sources, refs and shared README are untouched by this author.
Root's separate Morse/compiler work has its own scoped contract; this source
author neither executes it nor interprets its technical holds globally.
The full landmark objective remains ACTIVE_UNMET.
