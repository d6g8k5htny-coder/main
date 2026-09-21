# SIDE24 quartic-remainder diagnostic and V3.4 ledger audit

**Date:** 2026-08-01  
**Scope:** exact polynomial counterexample search and arithmetic-ledger audit;
not an audit of SIDE24 Gaussian realizability or analytic uniformity.

## 1. Exact perpendicular-axis family

Use radial-normalized plane coordinates with

\[
 M=(0,0),\qquad S=(\eta,0),\qquad P=(0,1),
 \qquad \eta=\varepsilon^p,quad p>1.
\]

Set \(\kappa=1\), \(\alpha=1/2\), and

\[
\begin{aligned}
P_3={}&\frac{x^3}{3}-\frac{\eta x^2}{2}
 -\frac{\eta^3y^2}{4}+\frac{\eta^3y^3}{6}
 -\varepsilon xy^2+2\varepsilon y^3-\varepsilon y^2,\\
Q_4={}&xy^3-y^4,\qquad F=P_3+\varepsilon Q_4.
\end{aligned}
\]

All coefficients of \(P_3\) and \(Q_4\) are uniformly bounded.  Direct
differentiation gives all nine pins

\[
 F(M)=0,\quad F(S)=-\eta^3/6,\quad F(P)=-\eta^3/12,
 \qquad \nabla F(M)=\nabla F(S)=\nabla F(P)=0,
\]

and

\[
 F_{yy}(M)-F_{yy}(S)=2\eta\varepsilon.
\]

The unique cubic with those three values, six gradients, and endpoint
curvature difference is

\[
C_3=\frac{x^3}{3}-\frac{\eta x^2}{2}
-\frac{\varepsilon}{\eta}x^2y+\varepsilon xy-\varepsilon xy^2
-\frac{\eta^3y^2}{4}+\frac{\eta^3y^3}{6}.
\]

The determinant of the ten-functional cubic interpolation matrix is
\(-2\eta^7\), so this comparison cubic is unique for \(\eta>0\).

## 2. Exact failure of the endpoint comparison weight

At \(M\),

\[
 H_F(M)=\operatorname{diag}(-\eta,-2\varepsilon-\eta^3/2)\prec0.
\]

The exact plane-determinant errors are

\[
\begin{aligned}
 \det H_F(M)-\det H_{C_3}(M)&=\varepsilon(\varepsilon+2\eta),\\
 \det H_F(S)-\det H_{C_3}(S)&=\varepsilon(\varepsilon-2\eta),\\
 \det H_F(P)-\det H_{C_3}(P)&=\eta\varepsilon(\eta+2).
\end{aligned}
\]

Hence, for every \(p>1\), each endpoint-error ratio against the claimed
normalized weight \(\eta(\eta+\varepsilon)\) grows like
\(\varepsilon/\eta=\varepsilon^{1-p}\).  The witness-error ratio against
\(\eta^2+\varepsilon\) tends to zero.

The mechanism is visible in

\[
 \det H_{C_3}(M)=\frac{\eta^4-2\varepsilon^2}{2}<0
\]

for small \(\varepsilon\), even though \(H_F(M)\prec0\).  Thus the exact
field's type does not transfer to the comparison cubic, and the cubic cone
inequality cannot be reused for the exact field without a separate argument.

## 3. Full three-dimensional type and product

Restore physical coordinates with \(r=\eta\varepsilon\):

\[
 f(X,Y,Z)=\varepsilon^3F(X/\varepsilon,Y/\varepsilon)-Z^2/2.
\]

Equivalently,

\[
\begin{aligned}
f={}&X^3/3-rX^2/2-(\eta^3\varepsilon/4)Y^2
 +(\eta^3/6)Y^3-\varepsilon XY^2+2\varepsilon Y^3
 -\varepsilon^2Y^2+XY^3-Y^4-Z^2/2.
\end{aligned}
\]

Thus the physical polynomial coefficients are also uniformly bounded; the
only blow-up is in the auxiliary comparison cubic, whose \(X^2Y\)
coefficient is \(-\varepsilon/\eta\).

The physical points are \((0,0,0)\), \((r,0,0)\), and
\((0,\varepsilon,0)\).  Their values are
\(0,-r^3/6,-r^3/12\), and all gradients vanish.  The hard eigenvalue
\(-1\) makes the full Hessian at \(M\) negative definite.  At \(S\), the
plane eigenvalue signs are \((+,-)\), so the full signs are \((+,-,-)\):
the index is exactly two.  The hard complement preserves every determinant
ratio.

This family does **not** refute the common product envelope.  Its normalized
plane determinants are

\[
\begin{aligned}
d_M&=\frac{\eta(\eta^3+4\varepsilon)}2,\\
d_S&=-\frac{\eta(\eta^3+4\eta\varepsilon+4\varepsilon)}2,\\
d_P&=-\frac{\eta^4-4\eta\varepsilon+2\varepsilon^2}2.
\end{aligned}
\]

The physical full-Hessian product is
\(\varepsilon^6d_Md_Sd_P\), while

\[
\mathcal D(r,\varepsilon)
=\varepsilon^6\eta^2(\eta+\varepsilon)^2(\eta^2+\varepsilon).
\]

For \(0<\eta,\varepsilon\le1\), elementary coefficient bounds give

\[
 |d_Md_Sd_P|
 \le32\eta^2(\eta+\varepsilon)^2(\eta^2+\varepsilon).
\]

For \(\eta=\varepsilon^p\), \(p=2,3,4\), the exact product ratio tends to
zero.  Therefore the finding is narrower but still load-bearing for the
printed proof route: the individual cubic-remainder comparison fails, while
the desired product happens to survive here by cancellation and needs a
different proof.

## 4. V3.4 verifier and ledger reconciliation

The current committed V3.4 facewise verifier has SHA-256
`bb4fdb739636e504d2c2eeafd8d8582dbf1dba12225bca88788927b049d58e5f`.
It reports 62 checks in normal and optimized Python, with byte-identical
output and no bare `assert` statements.

Its current ledger arithmetic is consistent:

- collar: raw density \(r^{-7}\), one value window \(r^3\), determinant
  product \(r^6\), volume \(r^3\), and Palm normalization \(r^{-2}\) give
  \(r^3\); the earlier double-counted \(r^6\) conclusion is absent;
- endpoint: product \(r^5d\), density \(d^{-3}r^{-1}\), volume
  \(d^2\,dd\), and Palm normalization \(r^{-2}\) give \(r^2\,dd\), hence
  \(O(r^3)\) after \(0<d<O(r)\);
- singular-near: value width \(r^3\), density \(s^{-7}\), volume
  \(s^2ds\), and Palm normalization \(r^{-2}\) give \(rs^{-5}ds\); the
  six displayed terms and their integral orders are correct.

The executable does not test the interpolation/type-transfer step in Lemma
6.1, so its PASS is compatible with the counterexample above.  It also only
checks the algebraic exponential-absorption lemma, not the analytic premise
that the required ESIP applies uniformly on every axis/projective overlap.

## 5. Disposition

- **Exact mismatch:** the endpoint cubic-remainder error does not carry the
  claimed radial weight with bounded \(P_3,Q_4\) coefficients and the full
  required \(M/S\) types.
- **Not established by this example:** failure of the final common product
  envelope.
- **Required repair:** prove the product directly with the type constraint
  and cancellation retained, or explicitly quantify and integrate the
  projective loss.  The exact-field type may not be assigned to the
  comparison cubic.
