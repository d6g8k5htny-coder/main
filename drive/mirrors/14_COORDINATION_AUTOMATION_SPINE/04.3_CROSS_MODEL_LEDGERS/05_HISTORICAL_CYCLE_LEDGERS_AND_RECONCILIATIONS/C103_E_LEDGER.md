# C103 E-LEDGER

**Cycle:** C103  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** Q0-B near-diagonal persistence-density theorem.

---

## E-C103-1 — The approximate fold lock was unnecessary for the density Jacobian

The first C103 freeze inherited the older formulation

\[
\ell=\frac{\kappa r^3}{6}(1+o(1)).
\]

For the persistence-density pushforward, define the observed normalized gap
mark exactly by

\[
\boxed{
\kappa_{\rm gap}=\frac{6(f(M)-f(S))}{r^3}.
}
\]

Then

\[
\ell=\frac{\kappa_{\rm gap}r^3}{6}
\]

is an identity.

**Disposition:** `SHARPENING`. `C103_FREEZE_v2.json` supersedes the first
freeze. Pair pinning remains relevant to the limiting mark law, but no
approximate delta-family step is needed for the Jacobian.

---

## E-C103-2 — Exact fold Jacobian and exponent closed

The exact transformation is

\[
r\,dr
=
\frac{6^{2/3}}{3}
\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
\]

More generally, radial measure \(r^{d-1}dr\) and lock power \(p\) give
lifetime exponent

\[
\frac{d}{p}-1.
\]

At \(d=2,p=3\), this is \(-1/3\).

**Disposition:** `DERIVED-EXACT`.

---

## E-C103-3 — Fixed-mark selection could not support an integrated theorem

C101/C102 proved selection at

\[
b=\frac65,\qquad\kappa=1.
\]

A persistence-density coefficient over a birth and modulus window integrates
over \(b\) and \(\kappa\). A theorem at one mark point has measure zero in
that integral and cannot be silently promoted to compact-mark uniformity.

**Disposition:** `DOMAIN/UNIFORMITY-CORRECTION`.

---

## E-C103-4 — Contact neutrality closed at program grade

The exact corrected-frame power ledger is

\[
r^3\text{ mark Jacobian}
\times
r^{-5}\text{ six-pin density}
\times
r^2\text{ typed determinant moment}
=
r^0.
\]

Thus the unadjacent typed candidate-pair intensity has a finite positive
limit.

The adjacency factor converges by scaled-field convergence and
Morse–Smale/SARD-G structural stability. It is positive because the canonical
fold has an open trapping configuration with positive Gaussian small-ball
probability.

**Disposition:** `PROGRAM-GRADE-PROVEN`.

**Boundary:** external specialist acceptance of SARD-G remains pending.

---

## E-C103-5 — Compact-mark algebra and selection uniformity closed

The C099–C101 symbolic identities were rederived with arbitrary
\(\kappa>0\):

- the third-maximum determinant remains a negative sum of squares;
- every window-critical Hessian determinant retains an \(\eta^2\) factor;
- the three-determinant product retains \(\eta^6\);
- the collar collision product retains a \(\rho^2\) zero.

All additional \(b,\kappa\) factors are uniformly bounded on compact
\(B\times K\subset\mathbb R\times(0,\infty)\).

Therefore

\[
\sup_{\theta,b,\kappa}
(1-q(r,\theta,b,\kappa))
\le C_{B,K}r^3.
\]

**Disposition:** `PROGRAM-GRADE-PROVEN`.

---

## E-C103-6 — Modulus tails and off-fold separation closed

The corrected-frame Gaussian density gives

\[
I_r(\theta,b,\kappa)
\le
C(1+\kappa^m)e^{-c\kappa^2}.
\]

Consequently,

\[
\kappa^{-2/3}I_r
\]

is integrable near both \(0\) and \(\infty\).

Pair separations outside a fixed local chart have bounded density per unit
lifetime and contribute

\[
O(1)=o(\ell^{-1/3}).
\]

**Disposition:** `PROGRAM-GRADE-PROVEN`.

---

## E-C103-7 — Theorem B mathematical core obtained new roots

The Gate Kernel v2.0 contract reports both compact-mark and full-\(\kappa\)
roots as shell-valid, closable, and unconditionally promotable at internal
program grade:

```text
THEOREM_B_COMPACT_MARK_C103_v3
25ea3f354d485d8fe72669f64f1fd8687794b5034ce211a9994dd1fc01e50b59

THEOREM_B_FULL_KAPPA_C103_v3
b1be71324ec28c3bafea0263a0fad6b212d48906ad110bc32b5af34c90054d55
```

The full mathematical statement is

\[
\nu_B(\ell)
=
C_*\ell^{-1/3}(1+o(1)),
\qquad
0<C_*<\infty,
\]

for a compact birth-height window \(B\), under the exact model and the
program-grade SARD-G layer.

**Disposition:** `MATHEMATICAL-CORE-CLOSED-PROGRAM-GRADE`.

No numerical value of \(C_*\) is claimed.

---

## E-C103-8 — The pre-registered held-out simulation failed

The primary frozen gate used a periodic four-neighbor vertex filtration at
grid \(256\) and the window \([0.005,0.05]\).

Observed cumulative slope:

\[
0.1487,
\]

with field-bootstrap 95% interval

\[
[0.1370,0.1616].
\]

The target was

\[
\frac23.
\]

Every registered confidence interval at both grids excluded \(2/3\) in the
same downward direction. The frozen kill signal triggered.

**Disposition:** `FAILED-AS-WRITTEN`. The result may not be relabeled a pass.

---

## E-C103-9 — A discretization layer was detected but did not fully explain the failure

For grid spacing \(h=24/n\),

```text
n=192:
    q25(lifetime)/h^2 = 0.571

n=256:
    q25(lifetime)/h^2 = 0.654
```

A large population of bars therefore collapses at an \(h^2\)-class scale,
consistent with grid/interpolation artifacts contaminating the cumulative
count.

However, exploratory windows above the visible \(h^2\) layer still produced
slopes well below \(2/3\).

**Disposition:** `INSTRUMENT-OR-THEORY-UNRESOLVED`.

The discrete proxy does not automatically falsify the continuum theorem, but
it blocks empirical completion.

---

## E-C103-10 — Q0-B is not terminally closed

The mathematical derivation is internally closed at program grade, but the
project’s pre-registered held-out validation gate failed.

A new obligation is registered:

```text
CONTINUUM_PERSISTENCE_VALIDATION
```

The repair must use a continuum-consistent persistence approximation and
coupled refinement.

**Disposition:** `NOT TERMINAL`.

---

**Entries:** 10  
**Homeless entries:** 0
