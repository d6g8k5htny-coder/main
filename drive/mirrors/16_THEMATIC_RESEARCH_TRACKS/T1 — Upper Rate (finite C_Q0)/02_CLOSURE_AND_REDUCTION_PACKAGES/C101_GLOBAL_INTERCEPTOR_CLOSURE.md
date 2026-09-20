# C101 Global Interceptor Cubic Closure

**Grade:** `DERIVED-EXACT-EXISTENCE`  
**Scope:** exact periodized BF field, \(L=24\), \(b=6/5\), typed
maximum–saddle pair-Palm, \(\ell=r^3/6\)  
**Numerical coefficient:** not claimed

## 1. Interceptor event

Let

\[
W_r=(b-r^3/6,b),
\]

and let \(N_{\rm sad}(W_r;D)\) count saddles in \(D\) whose values lie in
\(W_r\).

If a window-valued saddle preempts the canonical pair, then at least one such
saddle exists. Hence

\[
\boxed{
\mathbf 1_{\Pi_r}
\le
N_{\rm sad}(W_r;\mathbb T_{24}^2)
\le
N_{\rm crit}(W_r;\mathbb T_{24}^2).
}
\]

Therefore

\[
P^{MS}(\Pi_r)
\le
E^{MS}N_{\rm crit}(W_r;\mathbb T_{24}^2).
\]

Counting all critical types is conservative and avoids a third-point type
boundary in the upper bound.

## 2. Exact same-measure Kac–Rice formula

Under the six linear pair pins, let

\[
W_{MS}
=
|\det H_M\det H_S|
\mathbf 1_{\{H_M\prec0\}}
\mathbf 1_{\{\det H_S<0\}},
\]

and \(Z_r=E[W_{MS}\mid J_6]\).

The third-critical-point height intensity is

\[
\boxed{
\lambda_{\rm crit}^{MS}(x,u;r)
=
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\frac{
E[
W_{MS}|\det H_x|
\mid
J_6,f(x)=u,\nabla f(x)=0
]
}{
Z_r
}.
}
\]

This is already under the typed pair-Palm law. The qualitative proof does not
divide a Gaussian-pinned count by an approximate typing probability.
Consequently, the old Gaussian-pinned “typing residue” is not a separate
successor obligation.

## 3. Regional partition

Partition the torus into:

1. the two radius-\(2r\) collars;
2. the singular near region inside \(B_3\) and outside the collars;
3. a fixed-distance annulus;
4. the exterior \(\mathbb T_{24}^2\setminus B_3\).

### Collars

C100 proves

\[
E^{MS}N_{\rm add}(\mathcal C_r)
\le
C_{\rm col}r^3.
\]

This dominates every window saddle in the collars.

### Fixed annulus and exterior

At fixed positive distance from the collapsing pair, the corrected pair
divided-difference Gaussian family extends continuously to \(r=0\).
The pair determinant numerator and \(r^2\) pair-Palm denominator have the same
degeneracy order. The third value/gradient covariance has a positive compact
floor, and determinant moments are bounded.

The height window has width \(r^3/6\). Both fixed-distance regions therefore
contribute \(O(r^3)\).

## 4. Singular near frame

Set

\[
t=|x|,
\qquad
\eta=\frac rt,
\qquad
\omega=(c,s),
\qquad
\alpha=\frac{b-u}{r^3/6}.
\]

Outside the two collars,

\[
t\ge\frac{\sqrt{15}}2r,
\qquad
\eta^2\le\frac4{15}.
\]

The C099 value/gradient divided-difference atlas gives

\[
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\le
Ct^{-6}
\]

after angular integration and axis-overlap control.

## 5. Exact \(\eta^6\) determinant factor

The leading pair-pinned cubic is

\[
\begin{aligned}
P(X,Y)
={}&
\frac{X^3}{3}
-\frac{\eta^2X}{4}
-\frac{\eta^3}{12}\\
&+
\frac a2
\left(
X^2-\frac{\eta^2}{4}
\right)Y
+\frac Q2Y^2
+\frac w2XY^2
+\frac z6Y^3.
\end{aligned}
\]

Impose

\[
P(c,s)
=
-\frac{\alpha\eta^3}{6},
\qquad
\nabla P(c,s)=0.
\]

Exact symbolic elimination shows that each leading Hessian determinant has
an \(\eta^2\) factor:

\[
\det D^2P(M)
=
\eta^2R_M,
\]

\[
\det D^2P(S)
=
\eta^2R_S,
\]

\[
\det D^2P(c,s)
=
\eta^2R_x.
\]

Thus

\[
\boxed{
\det D^2P(M)
\det D^2P(S)
\det D^2P(c,s)
=
\eta^6R.
}
\]

The physical Hessian product is therefore

\[
t^6\eta^6R
=
r^6R.
\]

For the generic chart, the third determinant also has the exact
sum-of-squares form

\[
\det D^2P(c,s)
=
-\frac{\eta^2}{64c^2s^2}
\left[
(K+8\alpha c\eta)^2
+
64\alpha(1-\alpha)c^2\eta^2
\right],
\]

so the third critical point is generically a saddle. The upper proof needs
only the absolute determinant.

## 6. Singular-region integration

The pair-Palm normalizer satisfies

\[
Z_r\ge c r^2.
\]

Hence the per-unit-height intensity obeys

\[
\lambda_{\rm crit}^{MS}(x,u;r)
\le
Cr^4t^{-6}.
\]

Multiplying by the height-window width \(r^3/6\) and polar area gives

\[
\begin{aligned}
E N_{\rm crit}^{\rm singular}
&\le
Cr^7
\int_{(\sqrt{15}/2)r}^{\delta}
t^{-5}\,dt\\
&=
Cr^7
\left[
-\frac1{4t^4}
\right]_{(\sqrt{15}/2)r}^{\delta}\\
&\le
C'r^3.
\end{aligned}
\]

All four regions therefore have finite cubic coefficients.

## 7. Global coefficient functional

Let

\[
C_\Pi
=
C_{\rm col}
+
C_{\rm singular}
+
C_{\rm annulus}
+
C_{\rm exterior}.
\]

Each term is the integral of an exact chartwise supremum and is finite.
Therefore

\[
\boxed{
P^{MS}(\Pi_r)
\le
E^{MS}N_{\rm sad}(W_r)
\le
C_\Pi r^3.
}
\]

This closes `GLOBAL_INTERCEPTOR_CUBIC_UNIFORM` qualitatively.

## 8. Frozen diagnostic

The C101 scan used three rungs, seven geometric charts, an endpoint-aware
height mesh, 350,000 pair-weight samples per rung, and 100,000
triple-Hessian samples per chart/rung.

At physical distance \(5\), the ratio to the stationary saddle benchmark
\(\rho_{\rm sad}(1.2)/6\) was

\[
0.997418670
\le
\text{ratio}
\le
1.009723342.
\]

The fixed-distance \(t=0.25\) chart remained finite. Every tested singular
fixed-\(\eta\) chart decreased under rung refinement rather than approaching
the conservative \(r^{-2}\) upper scale. The analytic power count is therefore
not asserted to be sharp.

The scan is a falsification check, not the proof.

## 9. Q0 consequence

C098–C101 now supply finite cubic coefficients for:

- the global interceptor count;
- the direct-gain Gamma event;
- the collar residues.

Therefore, within the inherited Q0 well-definedness scope,

\[
\boxed{
\exists C_{Q0}<\infty:
\quad
0\le1-q(r,6/5)\le C_{Q0}r^3,
\qquad
0<r\le0.025.
}
\]

By squeezing,

\[
\boxed{
\lim_{r\downarrow0}q(r,6/5)=1.
}
\]

This is an internal derived-exact existence theorem at the program/
verification level. External specialist acceptance of the inherited
R0/SARD-G well-definedness layer is not claimed.

## 10. What remains unclaimed

```text
4.35 or any other numerical upper coefficient:
    NOT RESTORED

finite lower decimal:
    NOT RESTORED

explicit C_Pi, C_Gamma, C_col:
    OPEN

external referee acceptance:
    OPEN TRACK
```
