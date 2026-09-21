# C099 Near Maximum-Window Cubic Closure

**Grade:** `DERIVED-EXACT-EXISTENCE`  
**Object:** exact periodized BF field, \(L=24\), typed six-pin
maximum–saddle pair-Palm law  
**Numerical coefficient:** not claimed

## Theorem

There exists a finite \(C_{\rm near}^{\max}\) such that

\[
\boxed{
E^{MS}_{r,b}
N_{\max}
\left(
(b-r^3/6,b);
B_3\setminus
\bigl(
B(M,2r)\cup B(S,2r)
\bigr)
\right)
\le
C_{\rm near}^{\max}r^3
}
\]

uniformly for \(0<r\le0.025\).

The theorem is qualitative. It proves a finite cubic coefficient exists; it
does not display that coefficient.

## 1. Fixed-distance annulus

For every fixed \(\delta>0\), the region

\[
\delta\le|x|\le3
\]

is separated from the collapsing pair. In the corrected pair
divided-difference frame:

- the exact conditional Gaussian law extends continuously to \(r=0\);
- the pair determinant numerator and pair-Palm normalizer carry the same
  \(r^2\) degeneracy;
- the third value/gradient covariance has a compact positive floor;
- determinant-weighted third-maximum moments are bounded.

The window has width \(r^3/6\). Hence the annulus contribution is
\(O(r^3)\).

## 2. Singular two-scale chart

On the remaining disk write

\[
t=|x|,
\qquad
\eta=\frac rt,
\qquad
\omega=(c,s),
\qquad
\alpha=\frac{b-u}{r^3/6}.
\]

The two radius-\(2r\) collar exclusions imply

\[
0\le\eta<\sqrt{\frac4{15}}<0.517.
\]

Thus \((\eta,\omega,\alpha)\) ranges over a compactified chart domain.

## 3. Cubic type no-go

The exact pair-pinned cubic normal form gives, for \(cs\ne0\),

\[
\det H_x^{(3)}
=
-\frac{\eta^2}{64c^2s^2}
\left[
(K+8\alpha c\eta)^2
+
64\alpha(1-\alpha)c^2\eta^2
\right]
\le0.
\]

The transverse chart gives

\[
\det H_x^{(3)}
=
-\frac{\eta^2}{4}
\left[
(a+(1-2\alpha)\eta)^2
+
4\alpha(1-\alpha)\eta^2
\right]
\le0.
\]

On the pair axis,

\[
P_X(X,0)=X^2-\frac{\eta^2}{4},
\]

so a third critical point at scaled radius one would require \(\eta=2\),
outside the collar domain.

Therefore a third local maximum cannot occur at leading cubic order. On the
finite-\(t\) maximum event,

\[
\boxed{
|\det H_x|
\le
Ct^3\mathcal P(J),
}
\]

because its positive determinant is supplied only by the quartic remainder.

## 4. Exact generic scaled frame

For the conditional residual field define

\[
Y_1
=
\frac{
f(t\omega)-f(0)
-\frac t2\omega\cdot\nabla f(t\omega)
}{t^3},
\]

\[
Y_2=\frac{f_x(t\omega)}{t^2},
\qquad
Y_3=\frac{f_y(t\omega)}{t}.
\]

After conditioning the coalesced pair jets

\[
f,\ f_x,\ f_y,\ f_{xx},\ f_{xy},\ f_{xxx},
\]

the free jets

\[
f_{yy},\ f_{xxy},\ f_{xyy},\ f_{yyy}
\]

are independent with variances

\[
2,\ 2,\ 2,\ 6.
\]

The limiting covariance of \(Y\) is

\[
\begin{pmatrix}
\dfrac{s^2(3c^4+3c^2s^2+s^4)}{24}
&
-\dfrac{cs^2(2c^2+s^2)}4
&
0\\[6pt]
-\dfrac{cs^2(2c^2+s^2)}4
&
\dfrac{s^2(4c^2+s^2)}2
&
0\\[6pt]
0&0&2s^2
\end{pmatrix},
\]

with determinant

\[
\boxed{
\frac{
s^8(c^2+s^2)(3c^2+s^2)
}{24}.
}
\]

It is positive on every compact off-axis chart.

## 5. Exact pair-axis frame

The pair residual satisfies

\[
g(x)
=
\left(
x^2-\frac{r^2}{4}
\right)^2
(C_0+C_1x+\cdots),
\]

and

\[
f_y(x,0)
=
\left(
x^2-\frac{r^2}{4}
\right)
(D_0+\cdots).
\]

Put

\[
A_\eta=1-\frac{\eta^2}{4}.
\]

The axis divided differences are

\[
X_1
=
\frac{
g(t)-\frac{A_\eta}{4}t g'(t)
}{t^5},
\]

\[
X_2=\frac{g'(t)}{t^3},
\qquad
X_3=\frac{f_y(t,0)}{t^2}.
\]

Their limiting covariance is diagonal:

\[
\boxed{
\operatorname{Cov}(X)
=
\operatorname{diag}
\left(
\frac{A_\eta^6}{1920},
\frac{2A_\eta^2}{3},
\frac{A_\eta^2}{2}
\right),
}
\]

with

\[
\det\operatorname{Cov}(X)
=
\frac{A_\eta^{10}}{5760}>0.
\]

## 6. Axis-overlap suppression

In the generic frame,

\[
E[Y_2]
\longrightarrow
c^2-\frac{\eta^2}{4}.
\]

On

\[
|s|\le\frac12,
\qquad
\eta^2\le\frac4{15},
\]

\[
E[Y_2]\ge\frac{41}{60},
\]

while

\[
\operatorname{Var}(Y_2)\le2s^2.
\]

Thus the density at the zero-gradient target contains the suppression

\[
\exp\left[
-\frac{(41/60)^2}{4s^2}
\right],
\]

which dominates every polynomial angular singularity. The generic and axis
frames therefore form an integrable overlap atlas.

## 7. Determinant and Palm powers

Under the third station constraints,

\[
|\det H_M|+|\det H_S|
\le
Cr(r+t^2)\mathcal P(J).
\]

The pair-Palm normalizer satisfies

\[
Z_r\ge c r^2.
\]

Combining pair determinants, the quartic third-maximum determinant, uniform
Gaussian jet moments, and the angularly integrated station-density scale
\(t^{-6}\) gives

\[
\boxed{
\int_{\mathbb S^1}
\lambda_{\max}^{MS}(t,\theta,u;r)
\,d\theta
\le
C\frac{(r+t^2)^2}{t^3}.
}
\]

## 8. Integration

After integrating the height window of width \(r^3/6\),

\[
\begin{aligned}
\frac1{r^3}
E N_{\max}^{\rm singular}
&\le
C
\int_{c r}^{\delta}
\left(
\frac{r^2}{t^3}
+
\frac{2r}{t}
+
t
\right)t\,dt\\
&=
C
\int_{c r}^{\delta}
\left(
\frac{r^2}{t^2}
+
2r
+
t^2
\right)dt\\
&\le
C(r+r\delta+\delta^3).
\end{aligned}
\]

This is uniformly finite. Adding the fixed annulus proves the theorem.

## 9. Diagnostic check

The endpoint-aware C099 scan used:

- three rungs;
- six geometric charts;
- 27 height fractions clustered at both endpoints;
- 400,000 pair-weight samples per rung;
- 120,000 triple-Hessian samples per chart/rung.

All nonzero fixed-\(\eta\) coefficients decreased under refinement. The
mesoscopic and transverse charts also had positive decay exponents. No
unresolved endpoint spike was observed. These facts support the proof
architecture but are not used as its logical basis.

## 10. Status

```text
GAMMA-MAXCOUNT-NEAR-CUBIC:
    CLOSED-QUALITATIVELY

explicit C_near^max:
    OPEN

interval numerical near certificate:
    OPEN-REFEREE-SHARPENING

4.35:
    BLOCKED
```
