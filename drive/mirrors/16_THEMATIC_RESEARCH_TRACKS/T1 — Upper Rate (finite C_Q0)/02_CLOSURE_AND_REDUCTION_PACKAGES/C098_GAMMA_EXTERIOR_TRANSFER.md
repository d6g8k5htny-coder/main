# C098 Gamma Maximum-Window Exterior Transfer

**Grade:** `DERIVED-EXACT-EXISTENCE-THEOREM`

## Theorem

For the exact periodized Bargmann–Fock field on \(\mathbb T_{24}^2\), under
the typed six-pin maximum–saddle pair-Palm law, there exists a finite
constant \(C_{\rm ext}^{\max}\) such that

\[
\boxed{
E^{MS}_{r,b}
N_{\max}
\left(
(b-r^3/6,b);
\mathbb T_{24}^2\setminus B_3
\right)
\le
C_{\rm ext}^{\max}r^3
}
\]

uniformly for \(0<r\le0.025\).

This is a qualitative finite-coefficient theorem. It does not assign a
numerical value to \(C_{\rm ext}^{\max}\).

## 1. Pair-weight scaling

Under the six pair constraints, the corrected pair frame gives

\[
H_{xx}(M),H_{xx}(S)=O(r),
\]

\[
H_{xy}(M),H_{xy}(S)=O(r),
\]

while the transverse curvatures remain \(O(1)\). Therefore

\[
\det H_M=O(r),
\qquad
\det H_S=O(r),
\]

and the typed pair weight has the exact scaled form

\[
W_{MS}=r^2\widehat W_r
\]

in the divided-difference variables.

The pair-Palm normalizer satisfies

\[
Z_r
=
E_\gamma[W_{MS}\mid J_6]
=
r^2z_r,
\]

where

\[
z_r\longrightarrow z_0>0.
\]

The archived exact limiting coefficient is in the
\(3.230979\)-class. The independent C098 simulation gives:

| \(r\) | \(Z_r\) | \(Z_r/r^2\) |
|---:|---:|---:|
| 0.05 | 0.00808417960496 | 3.23367184198 |
| 0.025 | 0.00201529242071 | 3.22446787314 |
| 0.0125 | 0.000503621144509 | 3.22317532486 |

These measurements are diagnostic; positivity of \(z_0\) is the load-bearing
mathematical input.

## 2. Exterior Gaussian family

For

\[
x\in\mathbb T_{24}^2\setminus B_3,
\]

the point \(x\) stays a fixed positive distance from the collapsing pair.
The exact periodized covariance is analytic. After replacing the collapsing
six-pin coordinates by the corrected divided differences, the joint
conditional Gaussian law of

\[
f(x),\quad \nabla f(x),\quad H_M,\quad H_S,\quad H_x
\]

extends continuously to \(r=0\), uniformly over the compact exterior.

KR-V supplies positive conditional covariance away from the pins. Therefore

\[
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\]

is uniformly bounded for \(u\) in a fixed compact neighborhood of \(b\).

## 3. Palm-weighted third-Hessian moment

Condition additionally on

\[
f(x)=u,
\qquad
\nabla f(x)=0.
\]

The corresponding Hessian vector remains Gaussian with uniformly bounded
mean and covariance in the corrected frame. Polynomial Gaussian moments are
therefore uniformly bounded.

Moreover,

\[
r^{-2}
E_\gamma[
W_{MS}W_x^{\max}
\mid
J_6,f(x)=u,\nabla f(x)=0
]
\]

is continuous and bounded. The index-indicator boundaries have Gaussian
measure zero, so the determinant-weighted typed moment is continuous under
parameter convergence.

Dividing by

\[
Z_r=r^2z_r,
\qquad
\inf z_r>0,
\]

cancels the pair-degeneracy factor.

Thus there exists \(M_{\rm ext}<\infty\) with

\[
\lambda_{\max}^{MS}(x,u;r)
\le
M_{\rm ext}
\]

uniformly on the exterior parameter set.

## 4. Window integration

Since

\[
|W_r|=\ell=\frac{r^3}{6},
\]

\[
\begin{aligned}
E^{MS}_{r,b}N_{\max}(W_r;\mathcal E)
&=
\int_{\mathcal E}\int_{W_r}
\lambda_{\max}^{MS}(x,u;r)\,du\,dx\\
&\le
|\mathcal E|M_{\rm ext}\frac{r^3}{6}.
\end{aligned}
\]

Hence

\[
C_{\rm ext}^{\max}
=
\frac{|\mathcal E|M_{\rm ext}}6
<\infty.
\]

For \(r\) bounded away from zero, the same conclusion follows directly from
ordinary compactness and nonsingularity. Combining the small-\(r\) and
closed-away-from-zero ranges proves the theorem on all \(0<r\le0.025\).

## 5. Falsification check

Across twelve frozen exterior station/rung combinations, the measured ratio
to the stationary maximum-height benchmark lay in

\[
0.984673226
\le
\frac{\lambda_{\max}^{MS}}{\rho_{\max}(1.2)}
\le
1.010298403.
\]

This agreement is not the proof. It checks the predicted limiting mechanism
and would have falsified a missing \(r^{-k}\) Palm amplification.

## 6. Status delta

```text
GAMMA-MAXCOUNT-EXTERIOR-CUBIC:
    CLOSED-QUALITATIVELY

explicit exterior coefficient:
    OPEN

GAMMA-MAXCOUNT-NEAR-CUBIC:
    OPEN

COLLAR-CUBIC-EXPLICIT:
    OPEN

4.35:
    BLOCKED
```
