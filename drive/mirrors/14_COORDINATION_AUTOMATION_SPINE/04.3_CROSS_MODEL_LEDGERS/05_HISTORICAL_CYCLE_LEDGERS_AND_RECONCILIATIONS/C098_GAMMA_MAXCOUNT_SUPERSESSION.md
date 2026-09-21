# C098 Gamma Maximum-Window Count Supersession

**Deterministic reduction:** `PROVEN`  
**Kac–Rice identity:** `DERIVED-EXACT`  
**Finite station atlas:** `MEASURED-DIAGNOSTIC`  
**Uniform cubic theorem:** `PROVEN-MODULO`

## 1. Load-bearing observation

The direct-gain event was defined as follows:

- the canonical saddle has value
  \[
  f(S)=b-\ell,
  \qquad
  \ell=\frac{r^3}{6};
  \]
- its second ascending branch reaches a local maximum \(X^\dagger\);
- the gain satisfies
  \[
  0<f(X^\dagger)-f(S)<\ell.
  \]

Therefore

\[
b-\ell<f(X^\dagger)<b.
\]

Let

\[
W_r=(b-\ell,b)
\]

and let \(N_{\max}(W_r;D)\) count local maxima in \(D\) whose values lie in
\(W_r\). Then, pathwise,

\[
\boxed{
\Gamma_r
\subset
\{N_{\max}(W_r;\mathbb T_{24}^2)\ge1\}.
}
\]

Consequently,

\[
\boxed{
P^{\rm Palm}(\Gamma_r)
\le
E^{\rm Palm}N_{\max}(W_r;\mathbb T_{24}^2).
}
\]

This reduction does not select the terminal maximum as a differentiable
functional. It does not need:

- a chart cover for \(X^\dagger\);
- an inverse-Hessian moment for \(X^\dagger\);
- a Malliavin divergence estimate for \(G_r\);
- a chart-switching residual.

The count deliberately includes every maximum in the height window, whether
or not that maximum is reached by the branch. The overcount is conservative.

## 2. Exact typed pair-Palm Kac–Rice object

Let \(\gamma_{r,b}\) be the exact periodized BF Gaussian law conditioned on

\[
f(M)=b,\quad \nabla f(M)=0,
\]

\[
f(S)=b-\ell,\quad \nabla f(S)=0.
\]

Define

\[
W_{MS}
=
|\det H_M\det H_S|
\,\mathbf 1_{\{H_M\prec0\}}
\,\mathbf 1_{\{\det H_S<0\}},
\]

and

\[
Z_{MS}
=
E_{\gamma_{r,b}}W_{MS}.
\]

The typed maximum–saddle pair-Palm law is

\[
dP^{MS}_{r,b}
=
\frac{W_{MS}}{Z_{MS}}\,d\gamma_{r,b}.
\]

At a third point \(x\), put

\[
W_x^{\max}
=
|\det H_x|\mathbf 1_{\{H_x\prec0\}}.
\]

The exact marked intensity is

\[
\boxed{
\lambda_{\max}^{MS}(x,u;r)
=
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\frac{
E_\gamma[
W_{MS}W_x^{\max}
\mid
J_6,f(x)=u,\nabla f(x)=0
]
}{
E_\gamma[W_{MS}\mid J_6]
}.
}
\]

Hence

\[
\boxed{
E^{MS}_{r,b}N_{\max}(W_r;D)
=
\int_D
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx.
}
\]

No Gaussian-pinned quantity is silently inserted under pair-Palm. The
determinant weight and all three critical-point type marks remain inside the
same Kac–Rice object.

## 3. Regional theorem obligations

Use the exact partition

\[
\mathbb T_{24}^2
=
\mathcal C_r
\cup
\mathcal N_r
\cup
\mathcal E,
\]

where

\[
\mathcal C_r
=
B(M,2r)\cup B(S,2r),
\]

\[
\mathcal N_r
=
B_3\setminus\mathcal C_r,
\]

\[
\mathcal E
=
\mathbb T_{24}^2\setminus B_3.
\]

### `GAMMA-MAXCOUNT-COLLAR-CUBIC`

Since maxima are critical points,

\[
N_{\max}(W_r;\mathcal C_r)
\le
N_{\rm crit}(\mathcal C_r).
\]

The existing COL architecture supplies the \(O(r^3)\) shape. An explicit
coefficient remains required.

### `GAMMA-MAXCOUNT-NEAR-CUBIC`

Prove

\[
\int_{\mathcal N_r}
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx
\le
C_{\rm near}^{\max}r^3
\]

uniformly for \(0<r\le0.025\).

This is a compact three-point Kac–Rice problem. It retains the collapsing
six-pin geometry but removes branch-selection charts and selected-Hessian
inverse moments.

### `GAMMA-MAXCOUNT-EXTERIOR-CUBIC`

Prove a finite exact transfer factor for the exterior maximum-height
intensity:

\[
\int_{\mathcal E}
\int_{b-\ell}^{b}
\lambda_{\max}^{MS}(x,u;r)
\,du\,dx
\le
C_{\rm ext}^{\max}r^3.
\]

The final coefficient is

\[
\boxed{
C_\Gamma
=
C_{\rm col}^{\max}
+
C_{\rm near}^{\max}
+
C_{\rm ext}^{\max}.
}
\]

## 4. Station-atlas execution

`C098_GAMMA_MAXCOUNT_BENCHMARK.py` evaluates the exact pair-Palm Kac–Rice
integrand using:

- exact periodized covariance;
- 80-decimal covariance assembly;
- high-precision covariance eigenpairs;
- three-node Gauss–Legendre integration over \(u\in(b-\ell,b)\);
- frozen Monte Carlo seeds;
- 500,000 pair-weight samples per rung;
- 80,000 triple-Hessian samples per station.

The pair determinant-weight normalizers were:

| \(r\) | measured normalizer | archived anchor | relative difference |
|---:|---:|---:|---:|
| 0.05 | 0.00808417960496 | 0.00808 | 0.0517% |
| 0.025 | 0.00201529242071 | 0.00202046 | -0.2558% |
| 0.0125 | 0.000503621144509 |  |  |

The two archived anchors are reproduced within the frozen Monte Carlo
uncertainty.

Across all exterior stations and all three rungs, the ratio to the stationary
maximum-height benchmark was

\[
0.984673226
\le
\frac{\lambda_{\max}^{MS}}{\rho_{\max}(1.2)}
\le
1.010298403.
\]

Pair-scaled and fixed-near stations were strongly anisotropic and mostly
suppressed. No sampled station outside the two radius-\(2r\) collars exceeded
the stationary benchmark by more than approximately \(1.1\%\).

This is not a continuum certificate. It is a frozen falsification and
localization atlas for the interval proof.

## 5. Coefficient scale

The recorded stationary maximum-height density is

\[
\rho_{\max}(1.2)=0.043685.
\]

For a full window of width \(\ell=r^3/6\) over area \(24^2=576\), the
stationary benchmark is

\[
\boxed{
\frac{576\,\rho_{\max}(1.2)}6
=
4.19376.
}
\]

The earlier nine-pin terminal count was \(2.03r^3\) for approximately a
half-window; linear full-window scaling gives the comparison value
\(4.06r^3\). That is a different conditioned law and is not used as a MEASURE
bridge.

The counting route is therefore naturally a **qualitative cubic-rate route**.
It should not be forced into the tiny residual margin left by the inherited
\(4.35\) total display.

## 6. Supersession of the C097 route

The C097 selected-gain density theorem is not false. It remains a valid,
sharper anti-concentration program.

For the Gamma upper bound, however, the following C097 objects become
non-load-bearing:

```text
GAIN-CHART-COVER
GAIN-RESIDUAL-VARIANCE-FLOOR
GAIN-HESSIAN-INVERSE-MOMENT
GAIN-KERNEL-DERIVATIVE-BOUND
GAIN-MALLIAVIN-DIVERGENCE
GAIN-PALM-FIBER-WEIGHT
GAIN-CHART-RESIDUAL
```

They are replaced in the weakest-link path by:

```text
GAMMA-MAXCOUNT-COLLAR-CUBIC
GAMMA-MAXCOUNT-NEAR-CUBIC
GAMMA-MAXCOUNT-EXTERIOR-CUBIC
```

The exact selected-value calculus remains useful for other problems and for a
future sharp Gamma constant.

## 7. Status

```text
Gamma event-to-count reduction:
    PROVEN

exact pair-Palm count formula:
    DERIVED-EXACT

station atlas:
    MEASURED-DIAGNOSTIC

uniform Gamma O(r^3):
    PROVEN-MODULO three regional count obligations

C097 selected-density route:
    STANDALONE / NON-LOAD-BEARING FOR THIS UPPER BOUND

4.35 total coefficient:
    BLOCKED

qualitative q0 route:
    substantially simplified
```
