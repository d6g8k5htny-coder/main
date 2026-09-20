# C098 E-LEDGER

**Cycle:** C098  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** Gamma direct-gain upper route and its Q0 dependency cone.

---

## E-C098-1 — Incorrect inherited C096 ZIP hash in the C097 freeze

**Finding.** `C097_FREEZE.json` recorded

```text
735913ddc3c6ecb5ff16411da21b028d4589f6fa1c63f983f81601629a3a8373
```

as the C096 release ZIP hash.

The immutable ZIP and `C096_ATTESTATION.json` agree on

```text
735913ddf4dd2a3dfd3bf5c076ce798e13711f16f600b3d71aa7b7046e4fc188
```

**Cause.** Transcription of the detached release hash during the C097 freeze.

**Impact.** No source or release byte changed. The C097 mathematical artifacts
remain intact, but the inherited provenance field was wrong.

**Disposition.** `CORRECTION`. C098 freezes the recomputed hash and preserves
the incorrect C097 record.

---

## E-C098-2 — The selected-maximum density route was unnecessarily load-bearing

**Prior route.** C097 selected the maximum reached by the second branch and
reduced Gamma anti-concentration to seven chart/Malliavin/Palm obligations.

**Finding.** On Gamma,

\[
f(S)=b-\ell,
\qquad
0<f(X^\dagger)-f(S)<\ell,
\]

so

\[
b-\ell<f(X^\dagger)<b.
\]

Therefore

\[
\Gamma_r
\subset
\{N_{\max}((b-\ell,b))\ge1\}.
\]

**Impact.** The upper bound can use

\[
P(\Gamma_r)
\le
E N_{\max}((b-\ell,b))
\]

without selecting \(X^\dagger\) as a differentiable functional.

**Disposition.** `SUPERSEDED-AS-LOAD-PATH`. The C097 density theorem is not
false; it is retained as a standalone sharper route.

---

## E-C098-3 — Gamma now has an exact same-measure Kac–Rice count object

**Finding.** The correct integrand keeps the maximum–saddle pair determinant
weight and the third-point maximum mark inside one typed pair-Palm object:

\[
\lambda_{\max}^{MS}(x,u;r)
=
p_{(f,\nabla f)(x)\mid J_6}(u,0)
\frac{
E[W_{MS}W_x^{\max}\mid J_6,f(x)=u,\nabla f(x)=0]
}{
E[W_{MS}\mid J_6]
}.
\]

**Impact.** No Gaussian-pinned coefficient is silently deployed under
pair-Palm. No mark or conditioning law is dropped.

**Disposition.** `DERIVED-EXACT`.

---

## E-C098-4 — Exterior maximum-window transfer is qualitatively closed

**Finding.** In the corrected pair frame,

\[
W_{MS}=r^2\widehat W_r,
\qquad
E[W_{MS}\mid J_6]=r^2z_r,
\qquad
z_r\to z_0>0.
\]

Outside a fixed ball around the collapsing pair, the exact conditional
Gaussian family extends continuously to \(r=0\), the third-point
value/gradient covariance has a uniform floor, and the scaled
determinant-weighted Hessian moment is bounded.

Therefore

\[
\sup\lambda_{\max}^{MS}(x,u;r)<\infty
\]

on the exterior, and integration over a window of width \(r^3/6\) gives a
finite \(C_{\rm ext}^{\max}r^3\) bound.

**Disposition.** `CLOSED-QUALITATIVELY`. An explicit exterior decimal remains
open.

---

## E-C098-5 — Station atlas validates the pair-weight and exterior mechanism

**Finding.** The independently assembled determinant-Palm normalizers were:

```text
r=0.05:   0.00808417960496
r=0.025:  0.00201529242071
r=0.0125: 0.000503621144509
```

Their \(r^{-2}\)-scaled values are approximately:

```text
3.23367
3.22447
3.22318
```

matching the archived \(3.230979\)-class limit.

Across twelve exterior station/rung combinations, the ratio of the
pair-Palm maximum-window intensity to the stationary maximum-height benchmark
was between approximately `0.98467` and `1.01030`.

**Disposition.** `MEASURED-DIAGNOSTIC`. This validates the mechanism but is not
a continuum interval certificate.

---

## E-C098-6 — The count route cannot preserve the inherited 4.35 total

**Finding.** The stationary all-torus maximum-window benchmark is

\[
\frac{24^2\rho_{\max}(1.2)}6
=
4.19376
\]

in units of \(r^3\).

This is the Gamma count alone, before the existing interceptor coefficient and
collar residue.

**Impact.** Global maximum counting is appropriate for a qualitative finite
cubic rate and \(q_0\), not for preserving the narrow 4.35 total display.

**Disposition.** `NO-DECIMAL-PROMOTION`. The 4.35 claim remains blocked.

---

## E-C098-7 — Q0 conditional dependency cone reduced

**Prior C098 v1 conditions:**

```text
GLOBAL_INTERCEPTOR_CUBIC_UNIFORM
COLLAR_CUBIC_EXPLICIT
GAMMA_MAXCOUNT_NEAR_CUBIC
GAMMA_MAXCOUNT_EXTERIOR_CUBIC
```

**After exterior closure:**

```text
GLOBAL_INTERCEPTOR_CUBIC_UNIFORM
COLLAR_CUBIC_EXPLICIT
GAMMA_MAXCOUNT_NEAR_CUBIC
```

The updated root hashes are:

```text
UB_G_CUBIC_MODULO_MAXCOUNT_v2
ca35af0d5b8920aa6824d0899f99e097cebb8cf299685757c4f4db6760cae00e

Q0_LIMIT_MODULO_MAXCOUNT_v2
df1597967aa1cb7666a55c74bb85d176ea0647195356f920f6e87290f61911f7
```

**Disposition.** `PROVEN-MODULO THREE CONDITIONS`.

---

**Entries:** 7  
**Homeless entries:** 0
