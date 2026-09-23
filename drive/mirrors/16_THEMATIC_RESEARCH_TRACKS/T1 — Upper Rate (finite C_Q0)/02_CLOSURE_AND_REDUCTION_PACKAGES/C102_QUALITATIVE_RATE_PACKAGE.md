# Q0-C102 Referee-Grade Qualitative Rate Package

## Mandatory qualifier block

Every display of the theorem below carries the following scope:

```text
exact normalized periodized Bargmann–Fock field
torus side L=24
birth height b=6/5
typed maximum–saddle pair-Palm law
height gap ell=r^3/6
0<r<=0.025
finite-coefficient existence theorem
no numerical upper or lower coefficient claimed
```

## Theorem A — conditional external form

Assume the realized field is Morse–Smale and has distinct critical values.

Then there exists a finite constant

\[
C_{Q0}<\infty
\]

such that

\[
\boxed{
0\le1-q(r,6/5)\le C_{Q0}r^3,
\qquad
0<r\le0.025.
}
\]

Consequently,

\[
\boxed{
\lim_{r\downarrow0}q(r,6/5)=1.
}
\]

**Grade:** `PROVEN-HERE`, conditional on the displayed Morse–Smale
hypothesis.

## Theorem A′ — full Gaussian program form

The SARD-G package internally proves that the exact periodized BF field is
almost surely Morse–Smale. Consuming that program-grade result removes the
conditional hypothesis and gives the same qualitative rate and limit under
the Gaussian law.

**Internal grade:** `PROGRAM-GRADE-PROVEN`  
**External status:** `SPECIALIST-REVIEW-PENDING`

External referee acceptance is a status dependency, not a hidden
mathematical assumption. A conservative publication may state Theorem A and
present SARD-G as the separate appendix that upgrades it to Theorem A′.

## 1. Deterministic reduction

For a maximum \(M\) and a gradient-adjacent saddle \(S\), the C102
elder-rule dichotomy proves

\[
\{D(M)\ne S\}\subset\Pi\cup\Gamma.
\]

Here:

- \(\Pi\) is the existence of an earlier merge saddle whose value lies in
  \((f(S),f(M))\);
- \(\Gamma\) is the event that the other ascending branch of \(S\) terminates
  at a maximum whose value lies in the same open interval.

A loop/crater configuration is already contained in \(\Pi\): if the two arms
are connected above \(S\), their first connection after the birth of \(M\)
occurs at an index-one saddle in the open height window.

The union bound requires no independence.

## 2. Exact pair-Palm counting

Every regional upper bound is formed directly under the typed pair-Palm law:

\[
\lambda^{MS}(x,u;r)
=
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\frac{
E[
W_{MS}W_x
\mid
J_6,f(x)=u,\nabla f(x)=0
]
}{
E[W_{MS}\mid J_6]
}.
\]

No Gaussian-pinned coefficient is divided by an approximate typing
probability.

The denominator satisfies

\[
Z_r=E[W_{MS}\mid J_6]\ge c_Zr^2
\]

for a finite \(c_Z>0\).

## 3. Finite-jet and collision foundations

Strict positivity of every Fourier weight of the exact periodized BF kernel
implies that every finite family of distinct derivative evaluations has a
positive-definite Gaussian Gram matrix.

Corrected and nested divided differences extend analytically through the
collisions. The C102 atlas assigns every covariance rank-loss face to a
separate generic, axis, transverse, or pin-collision chart.

Polynomially weighted maximum/saddle moments are continuous because the type
boundaries are polynomial zero sets of nondegenerate Gaussian vectors and
therefore have probability zero.

## 4. Regional estimates

### Global interceptor

The expected count of window-valued critical points is split into collars,
the singular near region, a fixed annulus, and the exterior.

In the singular region the three leading Hessian determinants each contain
\(\eta^2\), so the physical product contributes \(r^6\). With the
value-gradient density, Palm denominator, height-window width, and polar
Jacobian,

\[
Cr^7
\int_{(\sqrt{15}/2)r}^{\delta}t^{-5}\,dt
=
O(r^3).
\]

Thus

\[
P(\Pi)\le C_\Pi r^3.
\]

### Direct-gain event

On \(\Gamma\), the terminal maximum lies in the height window
\((b-r^3/6,b)\), so \(\Gamma\) is bounded by a maximum-window count.

The exterior and fixed annulus follow by finite-jet compactness. In the
singular near region, the leading third-point cubic determinant is a
nonpositive sum of squares; a finite-scale maximum is a quartic
boundary-layer event. The resulting count is \(O(r^3)\).

Hence

\[
P(\Gamma)\le C_\Gamma r^3.
\]

### Collar residues

In scaled collar coordinates,

\[
\lambda_{\rm crit}^{MS}(r\xi;r)=rF_r(\xi).
\]

At either pinned point, a \(\rho^{-2}\) gradient-density collision is
cancelled by a \(\rho^2\) zero of the exact three-determinant product. Axis
faces are exponentially suppressed. Therefore

\[
E N_{\rm add}(\mathcal C_r)
\le C_{\rm col}r^3.
\]

## 5. Assembly

Set

\[
C_{Q0}=C_\Pi+C_\Gamma+C_{\rm col}.
\]

Each term is finite, so

\[
1-q(r,6/5)
\le
C_{Q0}r^3.
\]

The selection limit follows by squeezing.

## 6. Referee audit outcome

```text
finite-jet Fourier nondegeneracy:
    CLOSED

small-r analytic transfer:
    CLOSED QUALITATIVELY

full Gaussian chart atlas:
    CLOSED QUALITATIVELY

type-boundary continuity:
    CLOSED

pair-Palm normalizer positivity:
    CLOSED

deterministic loop/crater reduction:
    CLOSED HERE UNDER MORSE-SMALE

numerical coefficient:
    NOT-CLAIMED

independent SARD-G specialist acceptance:
    EXTERNAL-REVIEW-TRACK
```

## 7. Prohibited promotions

The package does not restore:

\[
4.3,\quad4.35,\quad0.8411,\quad0.84,\quad0.99,\quad1.01
\]

as theorem constants.

It also does not claim a thermodynamic-limit result, a critical-height
crossover theorem, or Theorem B’s normalized constant.

## 8. Terminal disposition

The internal referee package is complete. The project enters

```text
EXTERNAL-REVIEW-TRACK
```

with one named dependency: independent specialist review of SARD-G.
