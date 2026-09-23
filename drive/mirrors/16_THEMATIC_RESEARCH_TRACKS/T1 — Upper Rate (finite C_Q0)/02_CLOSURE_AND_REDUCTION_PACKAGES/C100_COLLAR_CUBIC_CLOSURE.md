# C100 Collar Cubic Closure

**Grade:** `DERIVED-EXACT-EXISTENCE`  
**Law:** exact periodized BF field, \(L=24\), typed six-pin
maximum–saddle pair-Palm  
**Numerical coefficient:** not claimed

## 1. Collar object

Let

\[
\mathcal C_r
=
B(M,2r)\cup B(S,2r),
\]

and let \(N_{\rm add}(\mathcal C_r)\) count critical points in
\(\mathcal C_r\) other than the two conditioned points \(M,S\).

This count dominates:

- every direct-gain terminal maximum lying in a collar;
- every collar residue whose mechanism requires an additional critical point.

The two pinned points themselves are excluded from the count. They form a
measure-zero set in the Kac–Rice integral.

## 2. Exact typed pair-Palm intensity

Under the six linear pins \(J_6\), define

\[
W_{MS}
=
|\det H_M\det H_S|
\mathbf 1_{\{H_M\prec0\}}
\mathbf 1_{\{\det H_S<0\}},
\]

and

\[
Z_r=E[W_{MS}\mid J_6].
\]

The third-critical-point intensity is

\[
\boxed{
\lambda_{\rm crit}^{MS}(y;r)
=
p_{\nabla f(y)\mid J_6}(0)
\frac{
E[
W_{MS}|\det H_y|
\mid
J_6,\nabla f(y)=0
]
}{
Z_r
}.
}
\]

No change of measure is made. The pair type marks remain inside
\(W_{MS}\), and the third point is deliberately counted over all critical
types.

## 3. Generic scaled gradient frame

Use midpoint coordinates

\[
y=r(X,Y),
\qquad
M=(-1/2,0),
\qquad
S=(1/2,0).
\]

Under the coalesced six-pin law, the free leading jets are

\[
q=f_{yy},\quad
a=f_{xxy},\quad
w=f_{xyy},\quad
z=f_{yyy},
\]

with means

\[
(-b,0,0,0)
\]

and covariance

\[
\operatorname{diag}(2,2,2,6).
\]

The scaled gradient converges to

\[
G_1
=
\frac{f_x(rX,rY)}{r^2}
\longrightarrow
X^2-\frac14+aXY+\frac w2Y^2,
\]

\[
G_2
=
\frac{f_y(rX,rY)}r
\longrightarrow
qY.
\]

Its covariance is

\[
\operatorname{Cov}(G)
=
\begin{pmatrix}
\dfrac{Y^2(4X^2+Y^2)}2&0\\
0&2Y^2
\end{pmatrix},
\]

so

\[
\det\operatorname{Cov}(G)
=
Y^4(4X^2+Y^2).
\]

Consequently,

\[
\boxed{
p_{\nabla f(rX,rY)\mid J_6}(0)
=
r^{-3}\widehat p_r(X,Y),
}
\]

with limiting density

\[
\widehat p_0(X,Y)
=
\frac{
\exp\left[
-\dfrac{(X^2-1/4)^2}{Y^2(4X^2+Y^2)}
-\dfrac{b^2}{4}
\right]
}{
2\pi |Y|^2\sqrt{4X^2+Y^2}
}.
\]

## 4. Three-Hessian determinant scale

Condition the third gradient to vanish. The leading cubic field is

\[
\begin{aligned}
P(x,y)
={}&
\frac{x^3}{3}
-\frac x4
-\frac1{12}\\
&+
\frac a2
\left(
x^2-\frac14
\right)y
+\frac Q2y^2
+\frac w2xy^2
+\frac z6y^3.
\end{aligned}
\]

Solving \(P_x(X,Y)=P_y(X,Y)=0\) gives a finite Gaussian
divided-difference frame for the remaining jets.

Every physical Hessian has the form

\[
H=rD^2P+O(r^2)
\]

under the third-station conditioning. Hence

\[
\boxed{
|\det H_M\det H_S\det H_y|
=
r^6\mathcal D_r(X,Y).
}
\]

The pair-Palm normalizer satisfies

\[
Z_r=r^2z_r,
\qquad
\inf_{0<r\le0.025}z_r>0.
\]

Thus, away from chart boundaries,

\[
\lambda_{\rm crit}^{MS}(rX,rY;r)
=
rF_r(X,Y).
\]

## 5. Pin-collision cancellation

Near \(M\), write

\[
(X,Y)
=
(-1/2,0)+\rho(c,s).
\]

Near \(S\), use the analogous coordinates.

The third-gradient density has the collision scale

\[
\widehat p_r\sim \rho^{-2}.
\]

The exact cubic determinant product factors as

\[
\boxed{
\det D^2P(M)\,
\det D^2P(S)\,
\det D^2P(X,Y)
=
\rho^2R_{M/S}(\rho,c,s,w,z).
}
\]

At \(\rho=0\), the generic leading factor is

\[
-\frac{
(-2c^2+s^2w)
(-4c^3+3cs^2w+s^3z)^2
}{
4s^6
}.
\]

The apparent angular powers are controlled by the collision density's
Gaussian exponent. Therefore the \(\rho^{-2}\) density and \(\rho^2\)
determinant zero cancel, and \(F_r\) has an integrable pin-collision limit.

## 6. Axis charts

Away from the pins, \(Y\to0\) carries

\[
\exp\left[
-\frac{(X^2-1/4)^2}{Y^2(4X^2+Y^2)}
\right]
\]

suppression.

At \(X=0\), the station equation forces

\[
w=\frac1{2Y^2},
\]

and the Gaussian cost is

\[
\exp\left[-\frac1{16Y^4}\right].
\]

The exact pair axis has leading gradient

\[
P_x(X,0)=X^2-\frac14,
\]

whose only zeros are the two pinned points. The pin-axis faces are therefore
covered by the collision chart and its exponential angular suppression.

## 7. Explicit coefficient functional

Let

\[
\Omega
=
B((-1/2,0),2)
\cup
B((1/2,0),2).
\]

Define

\[
F_*(\xi)
=
\sup_{0<r\le0.025}
\frac{
\lambda_{\rm crit}^{MS}(r\xi;r)
}{r}
\]

using the generic, transverse, axis, and collision atlas, and set

\[
\boxed{
C_{\rm col}
=
\int_\Omega F_*(\xi)\,d\xi.
}
\]

The preceding compact and boundary-chart bounds show

\[
C_{\rm col}<\infty.
\]

Since \(dy=r^2d\xi\),

\[
\begin{aligned}
E^{MS}N_{\rm add}(\mathcal C_r)
&=
\int_{r\Omega}
\lambda_{\rm crit}^{MS}(y;r)\,dy\\
&=
r^3
\int_\Omega F_r(\xi)\,d\xi\\
&\le
C_{\rm col}r^3.
\end{aligned}
\]

Therefore

\[
\boxed{
E^{MS}N_{\rm add}(\mathcal C_r)
\le
C_{\rm col}r^3.
}
\]

This also bounds every maximum-window count inside the collars.

## 8. Frozen diagnostic

The C100 scan used:

- four rungs from \(0.05\) to \(0.00625\);
- 26 stations per rung;
- nested collision radii down to \(\rho=0.02\);
- 300,000 pair-weight samples per rung;
- 50,000 triple-Hessian samples per station.

The sampled range was

\[
0
\le
\lambda_r/r
\le
0.093833.
\]

At \(\rho=0.02\),

\[
r^3\rho^2
p_{\nabla f\mid J_6}(0)
\approx
0.111
\]

across all four rungs, while

\[
\frac{
E[W_{MS}|\det H_y|\mid\nabla f(y)=0]
}{
r^6\rho^2
}
\]

remained finite and stable. The measured intensity exponents were
approximately one. These results falsification-check the exact scale proof;
they do not supply the theorem grade.

## 9. Status

```text
COLLAR-CUBIC-EXPLICIT:
    CLOSED-QUALITATIVELY

Gamma maximum-window collar count:
    CLOSED BY DOMINATION

explicit numerical C_col:
    OPEN

interval collar coefficient:
    OPEN-REFEREE-SHARPENING

4.35:
    BLOCKED
```

Any logically distinct typing-transfer residue must now be named separately;
it may not remain hidden under the word “collar.”
