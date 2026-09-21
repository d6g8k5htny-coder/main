# Three-Hessian conditional moment envelope

**Date:** 2026-08-01  
**Status:** replacement for the invalid all-face pathwise interpolation claim  
**Scope:** collar and singular-near charts for the side-24 field

This note proves the sixth item in the RP-C/RP-S work order.  The common
monomial is

\[
 {\cal D}(r,s)=[r(r+s^2)]^2(r^2+s^3).                    \tag{H.1}
\]

The result is deliberately stated in the density-weighted form consumed by
Kac--Rice.  A pathwise bound by (H.1) is true on noncollinear charts but
false on the exact axial face.  The axial face is controlled by its
target-mean exponential, not by transferring type to a comparison cubic.

## 1. Noncollinear mixed-curvature interpolant

In the contact plane put

\[
 M=(-r/2,0),\qquad S=(r/2,0),\qquad
 y=s(c,\sigma),\qquad c^2+\sigma^2=1,\qquad\eta=r/s.
                                                               \tag{H.2}
\]

Scale the plane coordinates by \(s\), write \(a=\eta/2\), and set

\[
 F(X,Y)=s^{-3}\{f(sX,sY)-b\}.
\]

Let \(p\) be the normalized cubic matching the three exact normalized
values, the six exact plane gradients, and

\[
 \frac{p_{XY}(a,0)-p_{XY}(-a,0)}{2a}
 =
 \frac{F_{XY}(a,0)-F_{XY}(-a,0)}{2a},                    \tag{H.3}
\]

Uniformity uses normalized confluent rows, not the raw matrix.  For a test
polynomial \(R\), take

\[
\begin{aligned}
L_0&=[R(a,0)+R(-a,0)]/2,\\
L_1&=[R(a,0)-R(-a,0)]/(2a),\\
L_2&=[R_X(a,0)-R_X(-a,0)]/(2a),\\
L_3&=\frac3{a^2}\{[R_X(a,0)+R_X(-a,0)]/2-L_1\},\\
L_4&=[R_Y(a,0)+R_Y(-a,0)]/2,\\
L_5&=[R_Y(a,0)-R_Y(-a,0)]/(2a),\\
L_6&=[R_{XY}(a,0)-R_{XY}(-a,0)]/(2a).                    \tag{H.3b}
\end{aligned}
\]

Their limits are
\[
 R,R_X,R_{XX},R_{XXX},R_Y,R_{XY},R_{XXY}
\quad\hbox{at the coalesced endpoint}.
\]
Together with \(R,R_X,R_Y\) at \((c,\sigma)\), the normalized
ten-by-ten determinant is \(-24\sigma^6\), independent of \(\eta\) and
\(c\).

Define the physical cubic

\[
                   Q(z)=b+s^3p(z/s).                    \tag{H.3a}
\]

Thus \(H_Q=sH_p\) and
\(\det H_Q=s^2\det H_p\).  In physical coordinates (H.3) matches

\[
 \frac{D_{te}^2Q(S)-D_{te}^2Q(M)}r
 =
 \frac{D_{te}^2f(S)-D_{te}^2f(M)}r.                      \tag{H.4}
\]

To check unisolvence, kill the six endpoint value-gradient rows.  The
remaining cubic has the form

\[
 A(X^2-a^2)Y+Y^2(BX+C)+DY^3.                              \tag{H.5}
\]

The row (H.3) gives \(2A\).  After \(A=0\), the witness value and two
gradient rows on \(B,C,D\) have determinant \(\sigma^6\).  Hence the
divided-difference-normalized interpolation matrix and its inverse extend
through \(\eta=0\), uniformly
on every chart \(|\sigma|\ge\theta>0\).  This includes the perpendicular
face \(c=0\) and introduces no negative power of \(c\) or \(\eta\).  The
raw endpoint matrix has the expected \(\eta^5\) confluent Jacobian and is
not claimed to extend invertibly.

The exact value pins may be written

\[
 f(y)=b-\alpha\kappa r^3/6,\qquad0\le\alpha\le1.          \tag{H.6}
\]

All cubics satisfying the nine value-gradient pins form a one-dimensional
family.  With

\[
 v=\frac{D\sigma+\kappa(2c+\eta)}{2\kappa\eta},\qquad
 L=\frac{\kappa^2r^4}{\sigma^2s^2},                       \tag{H.7}
\]

where \(D\) is the normalized mixed-curvature datum in (H.4), direct
elimination gives the exact square completions

\[
\begin{aligned}
 -\det H_{Q,M}^{\Pi}&=L(v^2-\alpha),\\
 -\det H_{Q,S}^{\Pi}&=L((v-1)^2-(1-\alpha)),\\
  \det H_{Q,y}^{\Pi}&=-L((v-\alpha)^2+\alpha(1-\alpha)).
                                                               \tag{H.8}
\end{aligned}
\]

The cubic entries obey

\[
\begin{aligned}
 |(H_{Q,M/S})_{tt}|+|(H_{Q,M/S})_{te}|
   +|(H_{Q,M/S})_{ee}|&\le CrP(J)\theta^{-N},\\
 \|H^\Pi_{Q,y}\|&\le CsP(J)\theta^{-N},                   \tag{H.9}
\end{aligned}
\]

and \(L\le Cr^2\).

## 2. Exact finite-\(r\) remainder factors

Put \(g=f-Q\).  Besides its three value-gradient zeros, it satisfies

\[
 g_{te}(S)=g_{te}(M).                                    \tag{H.10}
\]

The ordinary Peano bound alone would give \(g_{te}=O(s^2)\), which is too
coarse.  The matched row (H.10) supplies the missing short-edge square.
Indeed, for

\[
 h(q)=D_e g(x+qt),\qquad -r/2\le q\le r/2,
\]

one has \(h(\pm r/2)=0\) and
\(h'(r/2)=h'(-r/2)=\beta\).  The confluent divided difference is

\[
 h[-r/2,-r/2,r/2,r/2]=\frac{2\beta}{r^2},
\]

so its Peano representation gives \(|\beta|\le Cr^2J\).  Together with
the double zeros of \(g(\,\cdot\,,0)\), exact divided differences yield

\[
 \Delta_{M,S}:=H_f^\Pi-H_Q^\Pi
 =
 \begin{pmatrix}
  r^2A_{M,S}&r^2B\\
  r^2B&s^2C_{M,S}
 \end{pmatrix},
 \qquad
 \Delta_y=s^2A_y,                                        \tag{H.11}
\]

where every coefficient has uniformly bounded polynomial moments, up to
the displayed fixed \(\theta^{-N}\) loss.

For an endpoint cubic Hessian
\(H_Q^\Pi=\left(\begin{smallmatrix}a&b\\b&q\end{smallmatrix}\right)\),
(H.9) and the exact identity

\[
 \det(H_Q+\Delta)-\det H_Q
 =q\Delta_{tt}+a\Delta_{ee}-2b\Delta_{te}+\det\Delta
                                                               \tag{H.12}
\]

give

\[
 |\det H_f^\Pi(M)-\det H_Q^\Pi(M)|
 +|\det H_f^\Pi(S)-\det H_Q^\Pi(S)|
 \le Cr(r+s^2)P(J)\theta^{-N}.                            \tag{H.13}
\]

Now use type only on the exact field.  Since \(H_f(M)\prec0\),
\(\det H_f^\Pi(M)>0\).  Equations (H.8) and (H.13) imply

\[
 Lv^2\le L+Cr(r+s^2)P(J)\theta^{-N}.                      \tag{H.14}
\]

All three cubic determinants in (H.8) are bounded by
\(CL(1+v^2)\).  At the witness, (H.9), (H.11), and the two-dimensional
determinant perturbation identity give an additional \(Cs^3P(J)\).
Therefore

\[
\begin{aligned}
 |\det H_f^\Pi(M)|+|\det H_f^\Pi(S)|
  &\le Cr(r+s^2)P(J)\theta^{-N},\\
 |\det H_f^\Pi(y)|
  &\le C(r^2+s^3)P(J)\theta^{-N}.                         \tag{H.15}
\end{aligned}
\]

This proof covers \(c=0\).  It also explains why the former
\(ee\)-curvature interpolant failed there: it did not capture the
\((X^2-a^2)Y\) mode, whereas (H.3) does.

## 3. Hard complement

Let \(n\perp\Pi\).  The scalar function \(D_nf|_\Pi\) vanishes at all
three nodes.  Its exact divided differences give, at an endpoint,

\[
 H_{tn}=O(rP(J)),\qquad H_{en}=O(sP(J)),                  \tag{H.16}
\]

and at the witness both mixed columns are \(O(sP(J))\).
At an endpoint, (H.15) and the entry bounds give

\[
 H_{tt},H_{te}=O(rP),\qquad H_{ee}=O((r+s^2)P).           \tag{H.17}
\]

Expanding the three-dimensional determinant by components, every term is
bounded by \(Cr(r+s^2)P(J)\).  At \(M\), the same conclusion also follows
directly from the negative Schur complement of the \((e,n)\) block.
At \(y\), the plane and hard-mixed columns are \(O(sP)\), so the Schur
correction is \(O(s^3P)\).  Consequently, on every noncollinear chart,

\[
\begin{aligned}
 |\det H_M|+|\det H_S|&\le Cr(r+s^2)P(J)\theta^{-N},\\
 |\det H_y|&\le C(r^2+s^3)P(J)\theta^{-N},                \tag{H.18}\\
 |\det H_M\det H_S\det H_y|&\le C{\cal D}(r,s)P(J)\theta^{-N}.
                                                               \tag{H.19}
\end{aligned}
\]

## 4. Why the axial face is density-weighted

There is no uniformly moment-bounded pathwise version of (H.19) at
\(\sigma=0\).  A one-dimensional quintic can have the three prescribed
critical values and endpoint types with
\[
 |\det H_M\det H_S\det H_y|\asymp r^3
\]
in the collar, while \({\cal D}(r,2r)\asymp r^6\).  Such a realization
has a normalized fourth or fifth jet of order \(r^{-1}\).  It is precisely
an ESIP-large-jet event.

The correct universal pathwise estimate on the collinear axis is the crude
soft-column bound

\[
 |\det H_M\det H_S\det H_y|
 \le Cr^2s\,P(J)                                         \tag{H.20}
\]

in the singular-near chart: the pair-gradient identity supplies one
\(O(r)\) column at each endpoint, and the witness-gradient identity supplies
one \(O(s)\) column at \(y\).  Since \(r\le\eta_0s\),

\[
 {\cal D}(r,s)\ge r^2s^7,\qquad
 r^2s\le {\cal D}(r,s)s^{-6}.                             \tag{H.21}
\]

In the collar the analogous crude product is \(O(r^3P(J))\), hence at most
\({\cal D}(r,s)r^{-3}P(J)\).

## 5. Gaussian density-weighted absorption

Let

\[
\begin{aligned}
 {\cal K}_r(y,u)
 :={}&p_{(f(y),\nabla f(y))\mid V_r}(u,0)\\
 &\times\mathbb E\!\left[
 |\det H_M\det H_S\det H_y|\mathbf1_{\rm types}
 \mid V_r,f(y)=u,\nabla f(y)=0\right].
                                                               \tag{H.22}
\end{aligned}
\]

For a finite jointly Gaussian vector, conditioning expresses every jet in
\(J\) as a Gaussian with regression mean and covariance rational in the
normalized covariance entries.  Wick's formula shows that any fixed
polynomial moment grows by at most a fixed projective power.  Multiplication
by the target density retains its Mahalanobis exponential.  Thus, whenever
a scalar corrected residual has target mismatch bounded below and variance
at most \(C\Theta^2\),

\[
 p_{\rm target}\,\mathbb E[P(J)\mid{\rm target}]
 \le C\,(\hbox{reference density})\,
       \Theta^{-N}e^{-c/\Theta^2}.                        \tag{H.23}
\]

This statement is fail-closed: no bounded conditional moment is asserted
after the density factor has been removed.

On the singular mixed/collinear chart the pair-pin-measurable frame in
Section 5 of `rp_c_rp_s_facewise_closure.md` uses \(V_1\) itself as the
scalar ESIP residual.  After shrinking the axis chart it satisfies

\[
\begin{aligned}
 |(V_1)_{\rm target}-\mathbb EV_1|
  &\ge c_K\kappa,\\
 \operatorname{Var}(V_1\mid V_r)
  &\le C(s^2+\varrho^2)\le C\Theta^2,\qquad
 \Theta^2=s^2+\varrho^2.                                 \tag{H.24}
\end{aligned}
\]

Combining (H.19)--(H.24) with the single four-row projective frame
\((\bar V_0/\Lambda,\bar W/\Lambda^2,V_2/\Lambda,V_3/\Lambda)\) gives the
singular all-face envelope

\[
 {\cal K}_r(y,u)
 \le Cs^{-7}{\cal D}(r,s)\,
       \Theta^{-N}e^{-c/\Theta^2}                         \tag{H.25}
\]

on the axis charts; away from them the exponential is omitted and the
normalized density has an eigenfloor.  When
\(\varrho\le Cs\), (H.21) costs only \(s^{-6}\), absorbed by (H.24).
When \(\varrho\ge cs\), use (H.19) with its fixed \(\varrho^{-N}\) loss,
again absorbed by (H.24).  These charts overlap compactly.

In the collar away from an endpoint collision, the axial-interior
projective frame (AI.1)--(AI.4) in
`rp_c_rp_s_facewise_closure.md` gives
\(e^{-c/\epsilon_a^2}\), while its midpoint frame (MP.1)--(MP.6) gives
\(e^{-c/\delta_m^2}\) and \(e^{-c/r^4}\) on the exact midpoint face.  After
the single value integration,

\[
 \int_h^b{\cal K}_r(y,u)\,du
 \le Cr^{-4}{\cal D}(r,s)\,
       \Theta^{-N}e^{-c/\Theta^2}                         \tag{H.26}
\]

on the non-endpoint axial charts, with \(\Theta=\epsilon_a\) on the
axial-interior chart and \(\Theta=\delta_m\) on the midpoint chart.  The
factor \(r^{-3}\) from the crude collar product, and every further finite
regression loss, are absorbed by the exponential.
On nonaxial charts (H.26) follows from (H.19) and the \(r^{-7}\) joint
density times the value width \(O(r^3)\).

There is no version of (H.26), with only an \(r\)-projective loss, that is
uniform as an axial witness collides with an endpoint.  The exact typed
family in `scripts/verify_hybrid_mixed_curvature_and_axis_absorption.py` has
Hessian product \(\asymp rd^2\), whereas the gradient density is
\(\asymp d^{-3}r^{-4}e^{-c/r^2}\).  A factor \(d^{-1}\) remains before the
physical radial measure is inserted.  The correct endpoint-axis statement
is the measure-level estimate

\[
 r^{-2}(d^{-3}r^{-4})(rd^2)d^2\,dd\,e^{-c/r^2}
 \le C r^{-5}d e^{-c/r^2}\,dd .                          \tag{H.27}
\]

Its integral over \(0<d<\delta r\) is
\(O(r^{-3}e^{-c/r^2})=O(r^3)\).  Every additional fixed Gaussian-regression
power is absorbed by the same exponential.  Thus (H.25)--(H.27) give the
requested hybrid control: the identical monomial \({\cal D}\) occurs in the
pointwise density-weighted bounds off the endpoint-axis divisor, and
(H.27) is its integrable endpoint-axis replacement.  They do not imply a
false all-face pathwise or pointwise density-weighted bound.
