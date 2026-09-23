# C102 Finite-Jet Fourier Nondegeneracy

**Grade:** `DERIVED-EXACT`  
**Model:** exact normalized periodized Bargmann–Fock field on
\(\mathbb T_L^2\)  
**Numerical eigenvalue floor:** not claimed

## 1. Exact Fourier representation

The covariance is

\[
K_L(x)
=
\frac{
\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
e^{2\pi i k\cdot x/L}
}{
\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
}.
\]

Every spectral weight

\[
w_k=e^{-2\pi^2|k|^2/L^2}
\]

is strictly positive.

Let

\[
\Lambda f
=
\sum_{j=1}^{m}
\sum_{\alpha\in A_j}
c_{j,\alpha}
\partial^\alpha f(x_j),
\]

where the points \(x_j\) are distinct modulo \(L\mathbb Z^2\). Its variance is

\[
\operatorname{Var}(\Lambda f)
=
\frac1{Z_L}
\sum_{k\in\mathbb Z^2}
w_k
\left|
\sum_{j,\alpha}
c_{j,\alpha}
\left(\frac{2\pi i k}{L}\right)^\alpha
e^{2\pi i k\cdot x_j/L}
\right|^2.
\]

Because every \(w_k>0\),

\[
\operatorname{Var}(\Lambda f)=0
\]

if and only if

\[
E(k_1,k_2)
=
\sum_j
p_j(k_1,k_2)
z_j^{k_1}y_j^{k_2}
=
0
\]

for every \((k_1,k_2)\in\mathbb Z^2\), where

\[
z_j=e^{2\pi i x_{j,1}/L},
\qquad
y_j=e^{2\pi i x_{j,2}/L},
\]

and each \(p_j\) is a polynomial.

## 2. One-dimensional exponential-polynomial lemma

### Lemma

Let \(z_1,\ldots,z_s\) be distinct nonzero complex numbers and let
\(p_1,\ldots,p_s\) be complex polynomials. If

\[
\sum_{j=1}^s p_j(n)z_j^n=0
\]

for every \(n\ge0\), then every \(p_j\) is the zero polynomial.

### Proof

For \(|t|\) sufficiently small,

\[
F(t)
=
\sum_{n\ge0}
\left(
\sum_j p_j(n)z_j^n
\right)t^n
=
\sum_j
\sum_{n\ge0}
p_j(n)(z_jt)^n.
\]

If \(\deg p_j=d_j\), the inner generating function is rational and has its
only possible pole at

\[
t=z_j^{-1},
\]

with order at most \(d_j+1\). Distinct \(z_j\) give distinct pole locations.

The assumed zero sequence gives \(F(t)\equiv0\) in a neighborhood of zero,
and hence as a rational function. Every principal part at every distinct pole
must vanish. The principal part at \(z_j^{-1}\) is zero only when
\(p_j=0\). Therefore all \(p_j\) vanish. \(\square\)

The confluent-Vandermonde example checked by the companion executable is

\[
\det V
=
z_1z_2(z_1-z_2)^4,
\]

the two-node, first-derivative instance of the same independence mechanism.

## 3. Two-dimensional independence

Group the terms of \(E(k_1,k_2)\) by equal first-coordinate phase \(z_j\).

For each fixed \(k_2\), the one-dimensional lemma in \(k_1\) shows that every
coefficient polynomial attached to each distinct \(z\)-group vanishes.

Within one \(z\)-group, the torus points are distinct, so their second phases
\(y_j\) are distinct. Apply the one-dimensional lemma again in \(k_2\).
Every polynomial \(p_j\) vanishes.

Finally,

\[
p_j(k_1,k_2)
=
\sum_\alpha
c_{j,\alpha}
\left(\frac{2\pi i}{L}\right)^{|\alpha|}
k_1^{\alpha_1}k_2^{\alpha_2}.
\]

A zero polynomial has all coefficients zero, so every
\(c_{j,\alpha}=0\).

Hence no nonzero linear combination of distinct finite derivative evaluations
has zero variance.

## 4. Theorem

### Finite-jet nondegeneracy theorem

For the exact periodized Bargmann–Fock field, every finite family

\[
\left\{
\partial^{\alpha_\nu}f(x_\nu)
\right\}_{\nu=1}^N
\]

with distinct point/multi-index pairs and distinct underlying points where
appropriate has a positive-definite Gaussian Gram matrix.

Equivalently, the derivative-evaluation functionals are linearly independent
in the Gaussian Hilbert space.

This proves the exact-model version of \(H2'\) for arbitrary finite derivative
order, not merely the order-four families required by the Q0 theorem.

## 5. Divided-difference corollary

Let \(J\) be a base finite-jet vector with positive-definite covariance
\(\Sigma_J\), and let

\[
Y=TJ
\]

for a deterministic matrix \(T\). Then

\[
\Sigma_Y=T\Sigma_JT^*.
\]

If \(T\) has full row rank, then for every nonzero \(v\),

\[
v^*\Sigma_Yv
=
(T^*v)^*\Sigma_J(T^*v)>0.
\]

Thus every divided-difference frame is nondegenerate once its coefficient
transform has full row rank.

The companion exact checker records:

- corrected six-pin pair-frame determinant
  \[
  \det T_r=-r^{-5}\ne0;
  \]
- coalesced six-jet covariance determinant
  \[
  12;
  \]
- generic near-frame covariance determinant
  \[
  \frac{s^8(c^2+s^2)(3c^2+s^2)}{24};
  \]
- axis-frame determinant
  \[
  \frac{(1-\eta^2/4)^{10}}{5760};
  \]
- collar-gradient determinant
  \[
  Y^4(4X^2+Y^2).
  \]

The zero sets of the last two generic formulas are precisely the boundary
faces assigned to separate axis, transverse, or collision charts.

## 6. Conditional Gaussian corollary

Let \(X\) and \(Y\) be finite jet families. If the joint Gram matrix of
\((X,Y)\) is positive definite, then the Schur complement

\[
\operatorname{Cov}(Y\mid X)
=
\Sigma_Y-\Sigma_{YX}\Sigma_X^{-1}\Sigma_{XY}
\]

is positive definite.

Therefore every fixed-\(r\), distinct-point conditional covariance used by the
Kac–Rice formulas is nonsingular. In collapsing limits, positivity is checked
after the named divided-difference transformation, where the limiting
determinants above apply.

## 7. Status

```text
H2' for exact periodized BF:
    CLOSED

fixed-r distinct-point jet Gram matrices:
    CLOSED

named C099-C101 scaled limit frames:
    CLOSED

outward numerical eigenvalue floors:
    NOT-CLAIMED
```
