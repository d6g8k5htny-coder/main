# SIDE24 blind note: axial suppression on \(\rho\asymp d\)

**Date:** 2026-08-02  
**Disposition:** **PROVED as a local density lemma** under the fixed-side
analytic/full-spectrum hypotheses already used by the V3.4 chart  
**Method:** exact Hermite factorization plus normalized Schur positivity;
no Kimi artifact, Drive source, or web source was read

## 1. Variable map

This note uses

\[
M=0,\qquad S=dt,\qquad y=M+\rho t,\qquad
\lambda=\rho/d.                                           \tag{1.1}
\]

Thus \(d\) is the pair separation and \(\rho\) is a **signed axial witness
offset**. In the V3.4 facewise note these variables are called \(r,q\).
The \(\rho\) here is not its angular endpoint coordinate.

The cone is

\[
\epsilon d<\rho\le2d,\qquad\text{equivalently}\qquad
\epsilon<\lambda\le2,                                    \tag{1.2}
\]

with the pinned point \(\lambda=1\) deleted from the witness set. The
normalized extension through that divisor is used only to compare limits.

## 2. Exact Hermite factors

After centering by the conditional mean, the pair endpoint values and
gradients vanish. Cubic Hermite division gives, exactly,

\[
a=\rho(\rho-d),\qquad \chi=2\rho-d,\qquad
d_0^2=4\chi^2+a^2,                                       \tag{2.1}
\]

and residuals \(A,B,C^v,C^w\) such that

\[
\begin{aligned}
\widetilde f(y)&=a^2A,\\
\partial_t\widetilde f(y)&=a(2\chi A+aB),\\
\partial_v\widetilde f(y)&=aC^v,\qquad
\partial_w\widetilde f(y)=aC^w.
\end{aligned}                                             \tag{2.2}
\]

With \(U=(2\chi A+aB)/d_0\),

\[
\det\operatorname{Cov}(\nabla f(y)\mid V_d)
=a^6d_0^2\det\operatorname{Cov}(U,C^v,C^w\mid V_d).       \tag{2.3}
\]

The limiting symbols are a nonzero projective combination of
\(T^4,T^5\), together with \(T^2V,T^2W\). They are independent modulo the
contact pins. Strictly positive side-24 Fourier weights therefore give a
positive conditional Gram matrix. The projective midpoint blow-up keeps
that floor uniform when \(\chi=0\); compactness is used only after this
normalization.

Consequently, for sufficiently small \(d\),

\[
0<c\le
\det\operatorname{Cov}(U,C^v,C^w\mid V_d)
\le C.                                                     \tag{2.4}
\]

## 3. Mean mismatch and uniform cone exponent

For pair marks in a compact set with \(\kappa\ge\kappa_->0\), analytic
regression and exact Hermite divisibility give

\[
\mathbb E[\partial_tf(M+\rho t)\mid V_d]
=a\,m_{d,\lambda},\qquad
m_{d,\lambda}\longrightarrow\kappa                       \tag{3.1}
\]

uniformly for \(\lambda\in[\epsilon,2]\), including the normalized
midpoint and endpoint faces. After reducing \(d_*\),

\[
|m_{d,\lambda}|\ge\kappa_-/2.                             \tag{3.2}
\]

A scalar covariance ceiling from (2.4) gives

\[
\operatorname{Var}(\partial_t f(y)\mid V_d)
\le Ca^2d_0^2.                                           \tag{3.3}
\]

On the cone, with \(0<d\le1\),

\[
d_0^2
=d^2\{4(2\lambda-1)^2+d^2\lambda^2(\lambda-1)^2\}
\le40d^2.                                                 \tag{3.4}
\]

Therefore the Gaussian half-Mahalanobis exponent satisfies

\[
\frac{\mu_t^2}
 {2\operatorname{Var}(\partial_tf(y)\mid V_d)}
\ge\frac{\kappa_-^2}{320Cd^2}.                            \tag{3.5}
\]

At \(\lambda=1/2\), \(d_0^2=d^4/16\), so the actual suppression is the
stronger \(e^{-c/d^4}\).

Combining (2.3)--(3.5) yields the punctured-cone density bound

\[
\boxed{
p_{\nabla f(y)\mid V_d}(0)
\le C|a|^{-3}d_0^{-1}\exp(-c\kappa_-^2/d^2).
}                                                         \tag{3.6}
\]

This is precisely the local (4.4)--(4.5)-class axial-suppression statement.

## 4. Relation to the \(5\%\) endpoint window

For \(|\rho|\le0.05d\), the leading Euclidean axis determinant ratio is
bounded between

\[
0.5954244314\ldots\quad\text{and}\quad1.6215157252\ldots,
\]

as derived independently in AXIAL_D7_MECHANISM_2026-08-02.md. Hence the
local normalized covariance remains comparable on both sides of \(M\) for
all sufficiently small \(d\). In a transverse Euclidean direction the
corresponding leading ratio is \(1+(\rho/d)^2\), so the linear term is zero
and the perturbation is quadratic.

The small window and the cone serve different overlaps:

- \(|\rho|\le0.05d\) controls the local two-sided endpoint Taylor chart;
- \(\epsilon d<\rho\le2d\) supplies the axial mean-mismatch suppression
  once the witness radius is comparable to the pair gap.

## 5. Remaining interfaces

This lemma does **not** by itself close RP-C/RP-S. In particular:

1. At \(\lambda=1\), \(y=S\) is a pinned point. The factorized density is
   interpreted on the punctured chart; integration across the collision
   needs the Hessian collision weights and radial measure.
2. The algebraic prefactor \(|a|^{-3}\) is not uniformly integrable at both
   endpoint divisors without those weights.
3. The argument absorbs every fixed power of \(d\), but it does not absorb
   an untracked nonprojective loss in \(|\lambda-1|\).
4. No full three-Hessian conditional-moment envelope or regional
   \(O(d^3)\) count is claimed here.
5. The existence of a sufficiently small \(d_*\) follows from analytic
   compactness after normalization; no explicit numerical \(d_*\) is
   computed.

The companion fail-closed verifier is
verify_axial_suppression_cone.py.

