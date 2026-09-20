# SIDE24 blind note: the endpoint-axis \(d^7\) mechanism

**Date:** 2026-08-02  
**Disposition:** **CONFIRMED**, with the normalization stated below  
**Method:** independent derivation from the covariance and corrected pair
pins; no Kimi artifact, Drive source, or web source was read

## 1. Scope and notation

Let

\[
M=0,\qquad S=dt,\qquad d>0,
\]

and condition on the two endpoint first jets

\[
(f,\nabla f)(M),\qquad (f,\nabla f)(S).
\]

For a unit direction \(u\), put

\[
\mathcal{G}_{d,u}(z)
=\frac{\nabla f(M+zu)-\nabla f(M)}{z},                    \tag{1.1}
\]

with its analytic extension at \(z=0\). Thus

\[
\mathcal{G}_{d,u}(z)=G_{d,u}+zG'_{d,u}+O_{L^2}(z^2),
\qquad
G_{d,u}=H_f(M)u,
\qquad
G'_{d,u}=\tfrac12D_u(H_f\,u)(M).                          \tag{1.2}
\]

The requested \(G_1,G_1'\) are interpreted as the axial scalar components
of (1.2) when \(u=t\):

\[
G_1=f_{tt}(M),\qquad G_1'=\tfrac12f_{ttt}(M).             \tag{1.3}
\]

Here \(z\) is a **signed physical witness displacement**. It is not the
angular \(\rho\) used in some V3.4 endpoint charts.

## 2. Why the apparent \(O(d^2)\) term disappears

Use the corrected axial pin combinations

\[
D=\frac{f_t(S)-f_t(M)}{d},\qquad
C=\frac{f(S)-f(M)-\frac d2(f_t(S)+f_t(M))}{d^3}.          \tag{2.1}
\]

Taylor expansion at \(M\) gives

\[
\begin{aligned}
D&=f_{tt}+\frac d2f_{ttt}+\frac{d^2}{6}f_{tttt}
   +O_{L^2}(d^3),\\
C&=-\frac1{12}f_{ttt}-\frac d{24}f_{tttt}
   +O_{L^2}(d^2).
\end{aligned}                                             \tag{2.2}
\]

Consequently, modulo the pair-pin span,

\[
\begin{aligned}
G_1&=\frac{d^2}{12}R_4+O_{L^2}(d^3),\\
G_1'&=-\frac d4R_4+O_{L^2}(d^2),                         \tag{2.3}
\end{aligned}
\]

where \(R_4\) is the residual of \(f_{tttt}(M)\) after projection onto the
contact pin space. Hence

\[
\boxed{
\operatorname{Cov}(G_1,G_1'\mid\text{pins})
=-\frac{d^3}{48}\operatorname{Var}(R_4\mid P_0)+O(d^4).
}                                                         \tag{2.4}
\]

Full Fourier support makes the residual variance strictly positive. On
the compact side-24 frame atlas it has a uniform floor and ceiling.
Therefore (2.4) is not merely \(O(d^3)\); it is uniformly
\(-\Theta(d^3)\).

This identifies the cancellation. Midpoint reflection splits the corrected
pin frame into even and odd sectors. The odd third-jet component is already
carried by \(C\), so it cannot leave an \(O(1)\) residual in \(G_1'\).
Both surviving leading terms come from the same even fourth-jet residual,
with coefficients \(d^2/12\) and \(-d/4\). Calling this only a
Cauchy--Schwarz improvement hides the mechanism: it is the corrected
odd-sector pin plus the exact Hermite remainder.

For the Euclidean Bargmann--Fock covariance,

\[
\operatorname{Var}(R_4\mid P_0)=24,
\]

so

\[
\operatorname{Cov}(G_1,G_1'\mid\text{pins})
=-\frac12d^3+\frac15d^5+O(d^7).                          \tag{2.5}
\]

## 3. The determinant coefficient

Put \(\eta=z/d\). Cubic Hermite division gives the leading axial and two
mixed residual coefficients

\[
c_a(\eta)=\frac{(\eta-1)(2\eta-1)}{12},\qquad
c_\perp(\eta)=\frac{\eta-1}{2}.                           \tag{3.1}
\]

For Bargmann--Fock,

\[
\operatorname{Var}(R_4\mid P_0)=24,\qquad
\operatorname{Var}(R_{ttv}\mid P_0)
=\operatorname{Var}(R_{ttw}\mid P_0)=2,                  \tag{3.2}
\]

and the three residuals are conditionally orthogonal. Thus, uniformly for
\(\eta\) in a compact set away from the deeper zeros when this leading
chart is used,

\[
\begin{aligned}
D_{\rm ax}(d,\eta d)
&:=\det\operatorname{Cov}\!\left(
  \mathcal{G}_{d,t}(\eta d)\mid\text{pins}\right)\\
&=\frac{d^8}{24}(1-\eta)^6(1-2\eta)^2+O(d^{10}).
\end{aligned}                                             \tag{3.3}
\]

At \(\eta=0\),

\[
D_{\rm ax}(d,0)=\frac{d^8}{24}+O(d^{10}),\qquad
\partial_zD_{\rm ax}(d,0)=-\frac5{12}d^7+O(d^9),          \tag{3.4}
\]

and hence

\[
\boxed{
\frac{\partial_zD_{\rm ax}(d,0)}{D_{\rm ax}(d,0)}
=-\frac{10}{d}+O(d).
}                                                         \tag{3.5}
\]

This explains the observed coefficient near \(9.93/d\): its limiting
magnitude is \(10/d\), not \(1/d^2\).

For the side-24 field, the same Hermite scalars multiply a compact,
positive normalized residual Gram determinant. Therefore

\[
D_{24,\rm ax}(d,\eta d)
=d^8(1-\eta)^6(1-2\eta)^2\{A_{24}(t)+o(1)\},             \tag{3.6}
\]

with \(0<c\le A_{24}(t)\le C\). In particular, its first \(z\)-coefficient
is uniformly \(\Theta(d^7)\), with the same leading logarithmic slope
\(-10/d\).

## 4. Transverse check and the \(5\%\) window

For \(u=v\perp t\), the leading soft row is

\[
\frac d2(-R_{ttv}+\eta R_{tvv}).                          \tag{4.1}
\]

Euclidean parity makes the two residuals orthogonal and gives variance two
for each. The other two rows have limiting variances two and one. Hence

\[
D_\perp(d,\eta d)=d^2(1+\eta^2)+o(d^2).                  \tag{4.2}
\]

The transverse first-order term is exactly zero in this Euclidean
acceptance model; the correction is benign at order \((z/d)^2\). Exact
vanishing for an arbitrary side-24 rotated frame would require the relevant
mirror symmetry. Full inversion alone does not force the sixth mixed
spectral moment to vanish, so no such stronger uniform statement is made
here.

On the signed window \(|z|\le0.05d\), the Euclidean leading axial ratio is

\[
(1-\eta)^6(1-2\eta)^2
\in\left[
\frac{3810716361}{6400000000},
\frac{10377700641}{6400000000}
\right]
=[0.5954244314\ldots,1.6215157252\ldots].                 \tag{4.3}
\]

Continuity of the normalized side-24 divided-difference Gram family then
gives a two-sided comparable window with the same fixed \(0.05\) aperture
for all sufficiently small \(d\). This is an existence statement; the
present note does not calculate an explicit numerical \(d_*\).

## 5. What is and is not closed

Confirmed here:

1. \(\operatorname{Cov}(G_1,G_1'\mid\text{pins})=O(d^3)\), in fact
   \(-\Theta(d^3)\).
2. The endpoint-axis determinant's first \(z\)-coefficient is
   \(\Theta(d^7)\).
3. The Euclidean constants are \(-1/2\), \(-5/12\), and logarithmic slope
   \(-10/d\).
4. The Euclidean transverse linear term vanishes and its first correction is
   \((z/d)^2\).

Not asserted:

- an explicit numerical side-24 threshold \(d_*\);
- transverse first-order cancellation for every rotated side-24 frame;
- endpoint collision integrability or the complete RP-C/RP-S Kac--Rice
  ledger. Those require the determinant weights and radial measure in
  addition to this covariance calculation.

The companion fail-closed verifier is
verify_axial_d7_mechanism.py.
