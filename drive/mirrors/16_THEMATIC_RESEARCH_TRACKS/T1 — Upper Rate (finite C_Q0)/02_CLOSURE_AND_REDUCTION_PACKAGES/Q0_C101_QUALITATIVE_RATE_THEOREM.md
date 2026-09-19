# Q0-C101 Qualitative Cubic Rate and Selection Limit

## Canonical successor theorem

For the exact normalized periodized Bargmann–Fock field on
\(\mathbb T_{24}^2\), at

\[
b=\frac65,
\qquad
\ell=\frac{r^3}{6},
\]

under the typed maximum–saddle pair-Palm law, there exists a finite constant

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

## Grade and scope

```text
mathematical grade:
    internal derived-exact existence theorem

program dependency status:
    CORE-CLOSED

R0/SARD-G internal status:
    PROGRAM-GRADE-CLOSED

R0/SARD-G external status:
    SPECIALIST-REVIEW-PENDING

numerical upper coefficient:
    NOT-CLAIMED

finite numerical lower coefficient:
    NOT-CLAIMED
```

The theorem proves existence of a finite cubic coefficient. It does not claim
that the coefficient is \(4.35\), \(4.3\), or any other previously displayed
decimal.

## Proof chain

### 1. Defect decomposition

A pairing defect is contained in the union of:

1. a window-valued saddle interceptor;
2. the direct-gain Gamma event;
3. additional-critical-point collar residues.

The union bound is exact and does not require independence.

### 2. Global interceptor

The interceptor event is bounded by the expected number of all critical points
in the window

\[
(b-r^3/6,b).
\]

The exact typed pair-Palm Kac–Rice formula retains the determinant pair weight
and uses no Gaussian-pinned typing division.

The torus is partitioned into:

- radius-\(2r\) collars;
- singular pair-scaled near region;
- fixed annulus;
- exterior.

For the singular region, the value/gradient density has scale \(t^{-6}\), and
the exact third-station cubic law gives an \(\eta^2\) factor in each of the
three Hessian determinants. Their physical product is

\[
t^6\eta^6=r^6.
\]

After division by the \(r^2\) pair-Palm normalizer, height integration over a
window of width \(r^3/6\), and polar integration, the singular contribution is
\(O(r^3)\). The other regions follow by compactness or the collar theorem.

### 3. Direct-gain Gamma event

If the second branch reaches a maximum whose gain lies in \((0,r^3/6)\), then
that maximum has value in \((b-r^3/6,b)\).

Therefore Gamma is bounded by the maximum-window count.

The maximum-window count is \(O(r^3)\):

- exterior: fixed-distance pair-Palm compactness;
- near: exact cubic maximum-type no-go plus generic/axis scaled frames;
- collars: dominated by the all-critical collar count.

### 4. Collar residues

For \(y=r\xi\) in the two scaled radius-2 collars,

\[
p_{\nabla f(y)\mid J_6}(0)
=
r^{-3}\widehat p_r(\xi),
\]

the triple determinant moment is \(r^6\widehat m_r(\xi)\), and the pair-Palm
normalizer is \(r^2z_r\).

Thus

\[
\lambda_{\rm crit}^{MS}(r\xi;r)
=
rF_r(\xi).
\]

Near either conditioned point, the density has a \(\rho^{-2}\) collision
scale and the exact determinant product has a \(\rho^2\) zero. Axis
singularities are exponentially suppressed.

The scaled profile is integrable, giving

\[
E N_{\rm add}(\mathcal C_r)
\le
C_{\rm col}r^3.
\]

### 5. Squeeze

Adding the three finite cubic coefficients yields

\[
1-q(r,6/5)
\le
C_{Q0}r^3.
\]

Since \(r^3\to0\),

\[
q(r,6/5)\to1.
\]

## Machine roots

```text
Q0_CUBIC_RATE_EXISTENCE_C101
f92e4aae4c5b6cf7a3a34c41a43642f14f129543d03eceb82d73a4e77f49100f

Q0_LIMIT_C101
5ed20eafac7cffa923518fce06cae56822c0a2a62b2d8061e5cd90163ca47996
```

## What is not claimed

- No explicit numerical value of \(C_{Q0}\) is certified.
- The old \(4.3\) and \(4.35\) upper displays are not restored.
- The old \(0.8411\) and \(0.84\) finite lower displays are not restored.
- Direct measurement \(0.946\) at \(r=0.025\) remains rung-tagged measurement,
  not a scale-free truth constant.
- External specialist acceptance of SARD-G remains pending.
- No infinite-volume or critical-height theorem is implied by this
  fixed-\(L\), fixed-\(b\) result.
