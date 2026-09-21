# H3 scalar optimization: a whole-band lower coefficient of 1747/1000

**Author-side proved candidate; independent-method arithmetic, not independently
authored review.** This is a numbered additive extension of the fixed-axis H3
argument in `h3_floor/PROOF.md`, frozen inside source archive SHA-256
`73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21`.
The source proof is 10,423 bytes, SHA-256
`b48d8baa39b80be9a15af9833234d9339ea0afa4cdff2e1c60ae9ef79f5cb4fa`.
No source body or previous scientific disposition is replaced.

## Statement and exact scope

Let

\[
k(s)=\frac{\sum_{j\in\mathbb Z}e^{-(s+24j)^2/2}}
{\sum_{j\in\mathbb Z}e^{-(24j)^2/2}},\qquad K(x,y)=k(x)k(y).
\]

Use the centered stationary Gaussian field with covariance K, the fixed x-axis
points M=(-r/2,0), S=(r/2,0), and canonical Gaussian conditioning on
f(M)=6/5, f(S)=6/5-r^3/6 and both gradients zero. Define Z(r) as the conditional
expectation of |det H_M det H_S| times the indicator that M is a maximum and S
is a saddle. Then the argument below and the accompanying rational certificate give

\[
\boxed{Z(r)\ge \frac{1747}{1000}r^2,\qquad 0<r\le\frac1{20}.}
\]

The coefficient is 47/1700, approximately 2.7647%, larger than the prior stated
17/10 coefficient. This does not mean the true Z increased; the proved lower
estimate improved. At r=1/20 this gives 1747/400000=0.0043675. The already-recorded
sharper fixed-radius floor 0.0077592917375327855 remains preferable for the existing
RN consumer. Nothing here substitutes the new band estimate into that RN budget.

No all-angle, 3D, full-annulus, actual elder-selection, q0, prize or independent
acceptance claim is made. The original upper bound is not re-proved by this work.

## 1. Generalize the source's two free event parameters

Write R=1/20, choose epsilon in (0,1), let a=1-epsilon, and choose d>0. The new
selection is epsilon=103/500 and d=9/100. These are event cutoffs, not changes to
the field, pins, conditioning measure or radius domain.

Set m_(2j)=(-1)^j k^(2j)(0), sigma_e^2=m4-m2^2 and beta=m4*m2/4. Retain the source's
cancellation-free pin vector

\[
V=(f_M,f_{x,M},f_{y,M},(f_{x,S}-f_{x,M})/r,
(f_{y,S}-f_{y,M})/r,
(f_S-f_M)/r^3-(f_{x,S}+f_{x,M})/(2r^2)).
\]

Its conditioned value is v=(6/5,0,0,0,0,-1/6). The limiting vector is
(f,fx,fy,fxx,fxy,-fxxx/12). Its only nonsingleton covariance blocks are
[[1,-m2],[-m2,m4]] and [[m2,m4/12],[m4/12,m6/144]]. The other variances are m2,m2^2.
Exact positive shifted pivots establish G(0)>(3/80)I. Mean-square integral
identities give

\[
\|c\cdot(V(r)-V(0))\|_2^2\le r^2\|c\|_2^2
\left(m_2/4+m_4/4+m_2^2/4+m_6/16+m_4m_2/16+m_8/4096\right).
\]

For example the first three coordinate differences use endpoint displacement
r/2; the next two use average derivatives with mean |t|=1/4; the last uses
kernel (1/2)(1/4-t^2), whose absolute first moment is 1/64. Squared L2 bounds
sum by Cauchy-Schwarz. The displayed coefficient is below (31/20)^2.
Since (193/1000)^2<3/80, reverse triangle and Loewner inversion imply

\[
Q(r)=v^TG(r)^{-1}v\le Q_*=\frac{17/6}{[1-(31/20)R/(193/1000)]^2}
=\frac{1266466}{160083}<8.
\]

Here Q(0)=36m4/[25(m4-m2^2)]+4m2/(m2*m6-m4^2)<17/6, also recomputed. Thus every
positive radius in the band has a nonsingular regression law. No finite-radius
matrix grid is being used as a substitute for this transport argument.

## 2. Sector decomposition and axial event

As in the source, the complete line processes X=f(x,0), Y=fy(x,0), and
E=fyy(x,0)+m2*f(x,0) are mutually independent: separability and parity make all
cross-covariances zero. The pins condition X and Y separately and leave E
unchanged, with covariance sigma_e^2*k(x-z).

Put A_M=fxx(M)/r and A_S=fxx(S)/r. Integration by parts after subtracting the pins
gives residuals R_M=A_M+1 and R_S=A_S-1 with Peano kernels
(1/2-t)^2(1/2+t) and (1/2+t)^2(1/2-t), respectively, integrated over [-1/2,1/2]
against r*X''''(rt). Both are nonnegative with mass 1/12. Hence each unconditional
residual variance is at most s(r)^2=r^2*m8/144. Regression bounds the conditional
mean magnitude by sqrt(Q_*)s(r), and the conditional variance by s(r)^2.

For the event A={A_M<=-a,A_S>=a}, a union bound therefore gives

\[
P(A)\ge p_A=1-2\overline\Phi\left(\frac{\epsilon}{R\sqrt{m_8}/12}-\sqrt{Q_*}\right).
\]

The standardized argument is positive. A smaller r improves the bound; a
zero conditional variance is covered by the deterministic mean bound. The two
residuals are not assumed independent. Unlike the old fixed 14/15 floor, this
uses the full certified probability bound and the sharper Q_* rather than 8.

The centered mixed derivative B_M=fxy(M)/r belongs to the independent Y sector.
The pin-subtracted second-derivative kernel has mass 1/2, and conditioning on
zero Y values cannot increase variance, so E B_M^2<=beta. No B_S bound is needed.

## 3. Transverse event and moments

Let q_M=fyy(M), q_S=fyy(S), Qbar=(q_M+q_S)/2 and Delta=q_S-q_M. Then

\[
E Qbar=-m_2(6/5-r^3/12),\quad Var(Qbar)=\sigma_e^2(1+k(r))/2,
\]
\[
E Delta=m_2r^3/6,\quad Var(Delta)=2\sigma_e^2(1-k(r)).
\]

Stationarity gives zero covariance between midpoint and difference, hence their
Gaussian independence. Both are independent of X and Y. The mean-square
fundamental theorem gives 2(1-k(r))<=m2*r^2. Consequently

\[
Var(Qbar)\in[\sigma_e^2(1-m_2R^2/4),\sigma_e^2],
\quad sd(Delta)\le R\sqrt{\sigma_e^2m_2}.
\]

For D={|Delta|<=2d}, retain both tails and the positive nonzero mean:

\[
P(D)\ge p_D=1-2\overline\Phi\left(
\frac{2d-m_2R^3/6}{R\sqrt{\sigma_e^2m_2}}\right).
\]

Let T=(-Qbar-d)_+. Its underlying normal mean mu=m2*(6/5-r^3/12)-d and variance
sigma^2=Var(Qbar) lie in explicit whole-band intervals. With t=mu/sigma,

\[
M_1=E T=\sigma\phi(t)+\mu\Phi(t),\qquad
M_2=E T^2=(\sigma^2+\mu^2)\Phi(t)+\mu\sigma\phi(t).
\]

The program interval-evaluates both quantities over their entire parameter
ranges, including overestimation of correlated intervals. The approximation
used in parameter search is not used to certify these moments.

## 4. Typed determinant bound

Write H_i=[[rA_i,rB_i],[rB_i,q_i]], det(H_i)=rD_i with
D_i=A_i*q_i-rB_i^2. On A and D, whenever T>0, both transverse entries are at most
-T. Thus D_M>=aT-rB_M^2 and D_S<=-aT. A positive first expression ensures M is a
maximum and S a saddle. Pointwise, the rescaled typed integrand is at least
1_A*1_D*aT*(aT-rB_M^2)_+. The zero positive-part cases cause no difficulty.
Using x_+>=x and the established independence gives

\[
Z(r)/r^2\ge P(A)P(D)[a^2M_2-ar\beta M_1].
\]

First verify the bracket is positive; only then replace probabilities by lower
bounds and r by R in the negative term. The sufficient lower-bound expression
is enclosed by exact rational arithmetic, with outward decimal display

\[
[1.747106129274,\ 1.747994749902].
\]

**This interval encloses the sufficient-bound formula, not the true value of
Z(r)/r^2.** Its lower endpoint exceeds 1747/1000. All r dependence was bounded
uniformly before this scalar evaluation, proving the displayed candidate for
every 0<r<=R, not only tested radii.

## 5. Separate rational implementation and explicit tails

The scalar arithmetic is deliberately separate from the original repository
implementation. It uses Fraction endpoints, outward 2^-256 rounding, integer
square-root brackets, Machin's pi identity and alternating series with a proved
first-omitted-term remainder. CDF arguments outside [-8,8] and exponential
arguments outside [0,50] are refused, not extrapolated.

A positive Taylor partial sum proves e^12>100000. Hence e^-288<10^-120,
e^-1152<10^-480 and e^-1440<10^-600. For n<=8, writing C_n as the sum of absolute
Hermite coefficients, the |j|>=2 image tail is bounded by

\[
2C_n48^n10^{-480}/[1-(3/2)^n10^{-600}].
\]

The j=+/-1 contribution is bounded by 2|H_n(24)|10^-120. Keeping normalization,
the resulting moments differ from (1,3,15,105) by less than 10^-100. These are
rationally checked bounds, not deletion of small image terms. Rounding error is
then enclosed independently. Source hashes select the declared source, not an
extraction transform chosen to fit a desired hash.

The tests include both-tail omissions, understated moments, scope expansion,
wrong-quantity refutations, cache corruption and actual candidate recomputation.
The optional mpmath comparison is diagnostic only. Arb was not executed.
Nonauthor mathematical review, a second rigorous implementation, and any wider
research application remain separate tasks.
