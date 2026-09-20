# C097 Gamma-Gain Density Reduction

**Grade:** `DERIVED-EXACT-REDUCTION`  
**Target:** `GAMMA_GAIN_DENSITY_UNIFORM`  
**Promotion:** conditional only

## 1. Object

Under the six-pin conditional Gaussian law, followed by the typed canonical
pair-Palm determinant reweighting, let \(X^\dagger(f)\) be the maximum reached
by the second branch of the canonical saddle on a stable selection chart.

Define

\[
G_r(f)=f(X^\dagger(f))-f(S).
\]

The direct-gain event is

\[
\Gamma_r=\{0<G_r<\ell\},
\qquad
\ell=\frac{r^3}{6}.
\]

## 2. Exact chartwise derivatives

On a chart where \(X^\dagger\) is a nondegenerate critical point,

\[
\nabla f(X^\dagger)=0.
\]

For a conditional Cameron–Martin perturbation \(h\),

\[
DX^\dagger_f(h)
=
-[D^2f(X^\dagger)]^{-1}\nabla h(X^\dagger).
\]

Because the value at \(S\) is one of the six pins, every conditional
Cameron–Martin direction satisfies \(h(S)=0\). Therefore

\[
\boxed{
DG_r(f)[h]=h(X^\dagger).
}
\]

The mixed second derivative is

\[
\boxed{
D^2G_r(f)[h,k]
=
-\nabla h(X^\dagger)^T
[D^2f(X^\dagger)]^{-1}
\nabla k(X^\dagger).
}
\]

Under the six-pin Gaussian law, the Malliavin/RKHS gradient is the residual
kernel section

\[
DG_r=K_r^{(6)}(X^\dagger,\cdot),
\]

and

\[
\boxed{
\|DG_r\|_{\mathcal H}^2
=
K_r^{(6)}(X^\dagger,X^\dagger)
=
\operatorname{Var}\!\left(
f(X^\dagger)\mid\text{six pins}
\right).
}
\]

This is the key asymmetry: first-order anti-concentration does not contain a
Hessian inverse. The inverse Hessian enters only through second derivatives,
chart stability, and the divergence estimate.

## 3. Finite obligation set

### `GAIN-CHART-COVER`

The second-branch maximum selection is covered by measurable \(C^1\) charts
with uniform overlap \(N_{\rm chart}\), except for a separately bounded
residual event.

### `GAIN-RESIDUAL-VARIANCE-FLOOR`

\[
K_r^{(6)}(X^\dagger,X^\dagger)\ge v_0>0
\]

uniformly on active charts.

### `GAIN-HESSIAN-INVERSE-MOMENT`

A uniform inverse-Hessian moment controls the second Malliavin derivative and
chart motion.

### `GAIN-KERNEL-DERIVATIVE-BOUND`

The residual-kernel derivative norms at \(X^\dagger\) are uniformly bounded.

### `GAIN-MALLIAVIN-DIVERGENCE`

For every chart-local gain \(F_\alpha\),

\[
\mathbb E_\gamma
\left|
\delta\!\left(
\frac{DF_\alpha}{\|DF_\alpha\|_{\mathcal H}^2}
\right)
\right|
\le B_{\rm Mall}.
\]

This gives a chart-local Gaussian density ceiling.

### `GAIN-PALM-FIBER-WEIGHT`

For canonical-pair determinant weight \(W_r\),

\[
\sup_{u\in[0,\ell_0]}
\frac{
\mathbb E_\gamma[
W_r\,\mathbf 1_{\mathrm{chart}}\mid G_r=u
]
}{
\mathbb E_\gamma W_r
}
\le A_{\rm Palm}.
\]

This is the exact change from Gaussian pinning to pair-Palm density.

### `GAIN-CHART-RESIDUAL`

Chart switching, near-flat selected maxima, and uncovered selection
configurations contribute at most

\[
C_{\rm res}\ell
\]

inside the gain window.

## 4. Conditional theorem

Under these seven obligations,

\[
\|p_{G_r}^{\rm Palm}\|_{L^\infty([0,\ell_0])}
\le
N_{\rm chart}A_{\rm Palm}B_{\rm Mall}
+
C_{\rm res}
\equiv M_\Gamma.
\]

Hence

\[
\boxed{
P^{\rm Palm}(\Gamma_r)
\le
M_\Gamma\ell
=
\frac{M_\Gamma}{6}r^3.
}
\]

Thus the Gamma coefficient is

\[
\boxed{
C_\Gamma=\frac{M_\Gamma}{6}.
}
\]

## 5. What this closes and does not close

This reduction closes the **shape of the missing theorem**. It does not supply
the seven uniform constants.

In particular:

- the \(91.6\) station exponent is evidence at one geometric locus, not a
  density bound;
- chart switching remains load bearing;
- inverse-Hessian tails remain load bearing;
- pair-Palm determinant weighting remains load bearing;
- no \(4.35\) display is promoted.

The immediate executable target is now finite: certify the seven named
objects, rather than treating “Gamma is tiny” as one undifferentiated claim.
