# Q0-C104 Theorem B Package

## Mandatory qualifier block

```text
exact normalized periodized Bargmann–Fock field
torus side L=24
compact birth-height window B
non-essential superlevel H0 persistence bars
lifetime ell -> 0
candidate-pair intensity measured against r dr dtheta db dkappa
finite positive coefficient, no numerical value claimed
full Gaussian form at internal program grade
independent SARD-G specialist review pending
```

## Theorem B — conditional external form

Assume the realized field is Morse–Smale and has distinct critical values.

For any compact birth-height window

\[
B\subset\mathbb R,
\]

the expected density of non-essential superlevel \(H_0\) persistence
lifetimes satisfies

\[
\boxed{
\nu_B(\ell)
=
C_*\ell^{-1/3}(1+o(1)),
\qquad
\ell\downarrow0,
}
\]

where

\[
0<C_*<\infty.
\]

**Grade:** `PROVEN-HERE`, conditional on Morse–Smale.

## Theorem B′ — full Gaussian program form

Using the internally program-grade SARD-G theorem to discharge the
Morse–Smale condition, the same near-diagonal law holds for the exact
periodized BF Gaussian field.

**Internal grade:** `PROGRAM-GRADE-PROVEN`  
**External status:** `SPECIALIST-REVIEW-PENDING`

## 1. Exact mark and first-moment factorization

For an oriented, gradient-adjacent, typed maximum–saddle candidate pair, set

\[
\kappa
=
\frac{6(f(M)-f(S))}{r^3}
=
\frac{6\ell}{r^3}.
\]

This is an exact observed gap mark.

Let

\[
I_r(\theta,b,\kappa)
\]

be the candidate-pair intensity per unit base area with respect to

\[
r\,dr\,d\theta\,db\,d\kappa.
\]

The persistence-diagram first moment is the candidate-pair first moment
carrying the selection mark

\[
\mathbf 1_{\{D(M)=S\}}.
\]

Its pair-Palm expectation is

\[
q(r,\theta,b,\kappa).
\]

Hence the marked first-moment density is \(I_rq_r\).

## 2. Exact fold Jacobian

Since

\[
\ell=\frac{\kappa r^3}{6},
\]

\[
r
=
\left(
\frac{6\ell}{\kappa}
\right)^{1/3},
\]

and

\[
\boxed{
r\,dr
=
\frac{6^{2/3}}3
\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
}
\]

The exponent \(-1/3\) is therefore the exact consequence of:

\[
\text{two-dimensional polar measure}
+
\text{cubic fold scaling}.
\]

## 3. Contact-neutral candidate-pair intensity

The corrected pair-frame power ledger is

\[
r^3
\times r^{-5}
\times r^2
=
r^0,
\]

coming respectively from:

- the death-value to \(\kappa\) Jacobian;
- the six-pin Gaussian density;
- the typed pair determinant moment.

Thus the typed candidate-pair intensity has a finite positive contact limit.

The gradient-adjacency factor also has a positive limit:

- adjacency is locally stable at Morse–Smale fields;
- its discontinuity boundary is a separatrix incidence of SARD-G-null
  probability;
- the canonical fold has an open trapping configuration of positive Gaussian
  small-ball probability.

Therefore

\[
I_r(\theta,b,\kappa)
\longrightarrow
I_0(\theta,b,\kappa)
\in(0,\infty)
\]

on compact mark windows.

## 4. Uniform selection on compact marks

The C099–C101 obstruction estimates extend to arbitrary compact

\[
B\times K\subset\mathbb R\times(0,\infty).
\]

The general-\(\kappa\) symbolic identities preserve:

- the third-maximum cubic type no-go;
- the \(\eta^6\) window-critical determinant factor;
- the \(\rho^2\) collar collision cancellation.

Consequently,

\[
\sup_{\theta,b,\kappa}
\left[
1-q(r,\theta,b,\kappa)
\right]
\le
C_{B,K}r^3,
\]

so \(q_r\to1\) uniformly.

## 5. Modulus tails and off-fold localization

The corrected-frame Gaussian density and determinant moments satisfy

\[
I_r(\theta,b,\kappa)
\le
C(1+\kappa^m)e^{-c\kappa^2}.
\]

Therefore

\[
\kappa^{-2/3}I_r
\]

is integrable at both \(0\) and \(\infty\).

Separations outside a fixed local chart have bounded density per unit
lifetime and contribute

\[
O(1)=o(\ell^{-1/3}).
\]

Dominated convergence gives

\[
\boxed{
C_*
=
\frac{6^{2/3}}3
\int_{S^1}
\int_B
\int_0^\infty
I_0(\theta,b,\kappa)
\kappa^{-2/3}
\,d\kappa\,db\,d\theta.
}
\]

Contact neutrality makes the integral positive; Gaussian tail domination
makes it finite.

## 6. Machine roots

```text
THEOREM_B_COMPACT_MARK_C103_v3
25ea3f354d485d8fe72669f64f1fd8687794b5034ce211a9994dd1fc01e50b59

THEOREM_B_FULL_KAPPA_C103_v3
b1be71324ec28c3bafea0263a0fad6b212d48906ad110bc32b5af34c90054d55
```

## 7. Held-out validation

### C103 baseline

The raw four-neighbor proxy failed its pre-registered gate and triggered its
kill signal. That failure remains preserved.

### C104 continuum-consistent repair

C104 used:

- periodic Freudenthal triangulations;
- identical Fourier realizations on nested grids;
- explicit interpolation-error bounds;
- coarse/fine persistence-diagram matching;
- refinement-stability filtering;
- per-field left-truncated likelihood.

The frozen primary result was

\[
\widehat\alpha=-0.48144,
\]

with 95% interval

\[
[-0.71150,-0.24375],
\]

from 543 stable bars. The interval contains the theoretical density exponent

\[
-\frac13.
\]

The primary gate therefore **survives**, and the frozen kill signal does not
trigger.

A secondary \(U=0.3\) fit excludes \(-1/3\); this is retained as a
finite-window/refinement limitation. C104 supports the theorem under its
frozen criterion but is not a proof or a precise measurement of the exponent.

## 8. Project disposition

```text
Q0-B mathematical core:
    CORE-CLOSED AT PROGRAM GRADE

Q0-B held-out empirical gate:
    SURVIVES

Q0-B terminal state:
    EXTERNAL-REVIEW-TRACK

external dependency:
    INDEPENDENT SARD-G SPECIALIST REVIEW

numerical C_*:
    NOT-CLAIMED
```

## 9. What is not claimed

- No numerical value of \(C_*\).
- No exact power law at finite \(\ell\).
- No claim that individual bar lifetimes are independent.
- No infinite-volume theorem.
- No non-Gaussian universality result.
- No statement that the C104 point estimate equals \(-1/3\).
