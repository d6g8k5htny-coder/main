# THE q₀ RATE PROGRAM — C091 PROOF-GATE REDUCTION

## Marked Bonferroni repair, exact-torus matrix transfer, one-sided upper interpolation, and Palm-weighted Gaussian corridor bounds

**Date:** 2026-07-16  
**Status:** append-only correction and theorem reduction.  
**Precedence:** C091 preserves every verified C024–C090 computation that survives its object, measure, domain, and quantifier checks. It supersedes only proof rules shown here to be incomplete or false as general inequalities.

---

# 0. Executive adjudication

C090 correctly repaired the theorem contract at the level of finite versus asymptotic lower statements, rungwise versus continuous upper statements, periodized torus definition, measurement tags, and R0 precedence. C091 investigates the remaining proof gates rather than restating them.

The investigation produces four substantive conclusions.

## 0.1 Exact torus transfer is closed on the matrix domains actually tested

For the exact normalized periodized Bargmann–Fock covariance at `L=24`, the planar-to-torus covariance-entry error is bounded by

\[
1.855\times10^{-90}
\]

through derivative order four on the local chart \(|x|\le3\), and by

\[
1.839\times10^{-25}
\]

for pin-to-target derivative covariances with target distances in \([5,12]\).

Even when the smallest retrieved raw nine-pin floor

\[
3.0946\times10^{-14}
\]

is used instead of a regularized-frame floor, the propagated conditional-covariance error is

\[
1.7924\times10^{-56}
\]

locally and

\[
1.8608\times10^{-12}
\]

at far stations. The far Kac–Rice covariance floor therefore changes from

\[
1.4\times10^{-8}
\]

to at worst

\[
1.3998139\times10^{-8}.
\]

Thus `G-TORUS-LOCAL` and the declared far-station transfer are closed. Uses outside the declared derivative order, dimensions, or distance partition still require their own rows.

## 0.2 The upper interpolation gate is one-sided

C090 asked for \(\sup|U''|\). That is stronger than necessary. If

\[
U''(r)\ge-M,
\]

then

\[
\sup_{[a,b]}U
\le
\max\{U(a),U(b)\}
+
\frac{M(b-a)^2}{8}.
\]

Only the negative curvature can lift an analytic function above its endpoint chord. The sufficient gates remain numerically

\[
M\le1510.4
\quad\text{on }[0.0125,0.025],
\]

and

\[
M\le64
\quad\text{on }[0.025,0.05].
\]

An even simpler certificate is

\[
U'(r)\ge0
\quad(0<r\le0.05),
\]

which closes the between-rung and origin gates simultaneously. It would give the nominal maximum \(0.985\) and the banded maximum \(1.005\) at \(r=0.05\).

## 0.3 The C089 H4 Markov product is false as a general upper-bound rule

The one-step number

\[
q_{\rm step}
=
\Phi\!\left(
-
\frac{b-be^{-1/2}}{\sqrt{1-e^{-1}}}
\right)
=
0.276299\ldots
\]

is a valid one-dimensional Gaussian tail. It does not follow that an \(n\)-checkpoint corridor has probability at most \(q_{\rm step}^n\).

Under the exact six-pin Gaussian conditional law, the following diagnostics at \(r=0.025\) were obtained:

| skeleton | joint orthant probability | \(q_{\rm step}^n\) | ratio |
|---|---:|---:|---:|
| two two-cell branches | 0.009418 | 0.005828 | 1.616 |
| one three-cell east chain | 0.070560 | 0.021093 | 3.345 |
| two three-cell branches | 0.000900 | 0.000445 | 2.023 |

The same failures persist at \(r=0.05\). The numerical CDF evaluations are diagnostic rather than interval certificates, but the margins are far beyond numerical integration error. The inference rule itself is rejected.

The valid replacement is a finite-dimensional Gaussian orthant certificate, with a further exact correction for the Palm determinant weight.

## 0.4 The lower Bonferroni gate is a marked two-scale problem

The Master-v3.2 lower file writes the near-diagonal argument as

\[
\lambda_2(y_1,y_2)
\le
C_{\rm nd}|y_1-y_2|
\]

for a two-point intensity per unit height squared, then factors out \(\ell^2\). The cited spatial-repulsion literature does not by itself prove this marked, uniform-in-two-heights statement.

The relevant literature separates two effects:

1. generic critical-point counts in shrinking windows can lose one height-window factor near the diagonal;
2. same-type saddles repel much more strongly in space.

The missing bridge is the critical-value mark law. For two Bargmann–Fock critical points separated by \(d\), conditioned on both gradients being zero,

\[
A=\frac{f_-+f_+}{2},
\qquad
Z=\frac{f_+-f_-}{d^3}
\]

have a nondegenerate limiting Gaussian law:

\[
\operatorname{Var}(A)\to\frac23,
\qquad
\operatorname{Var}(Z)\to\frac1{24},
\qquad
\operatorname{Cov}(A,Z)=0.
\]

Therefore the critical-value gap lives on the \(d^3\) scale. Combining this mark law with same-type saddle repulsion yields a corrected two-scale Bonferroni theorem and again gives \(O(\ell^2)=O(r^6)\), but under a different, explicitly marked hypothesis.

The lower theorem is consequently **not promoted** until this new marked-repulsion gate is certified under the actual six-pin pair-Palm law.

---

# 1. Source-state and what remains trusted

The following are not disturbed by C091:

- the C024 independent estimator verification, including the 14/14 anchor pass;
- the exact fold normalization \(\ell=r^3/6\);
- the one-point \(\Lambda\) estimator and its factorization;
- the exact Palm normalizer architecture;
- the C043 one-point grid-continuum budget;
- the exact Bonferroni inclusion–exclusion identity;
- the C089 rungwise collar, outside, and corridor inputs as measurements;
- the C041–C046 SARD-G chain and the C090 R0 precedence decision;
- the exact periodized-torus definition introduced at C090.

Two inherited proof edges are quarantined:

```text
BONFERRONI_NEAR_POINTWISE_CND
    status: UNDISCHARGED AS A MARKED-WINDOW CLAIM

H4_QSTEP_PRODUCT
    status: KILLED AS A GENERAL PROBABILITY UPPER BOUND
```

The distinction is deliberate. The H4 product has direct counterexamples. The old Bonferroni statement may still be true under additional typed six-pin structure, but its cited spatial law does not establish it; it is therefore unproved rather than declared false.

---

# 2. Upper interval theorem: the correct one-sided gate

Let \(U(r)\) denote the normalized upper coefficient for one fixed epistemic track—nominal or nominal plus its one-sided uncertainty allowance.

## 2.1 One-sided interpolation lemma

**Lemma U-NEG.** Let \(U\in C^2([a,b])\). If

\[
U''(r)\ge-M
\quad\text{for every }r\in[a,b],
\]

then

\[
\boxed{
\sup_{r\in[a,b]}U(r)
\le
\max\{U(a),U(b)\}
+
\frac{M(b-a)^2}{8}.
}
\]

### Proof

Let \(L\) be the linear interpolant through the endpoint values. The Lagrange remainder gives, for every interior \(r\), some \(\xi_r\in(a,b)\) with

\[
U(r)-L(r)
=
\frac{U''(\xi_r)}2(r-a)(r-b).
\]

The product \((r-a)(r-b)\) is nonpositive. Because \(U''(\xi_r)\ge-M\),

\[
U(r)-L(r)
\le
\frac M2 |(r-a)(r-b)|.
\]

The maximum of the absolute product is \((b-a)^2/4\). Also, \(L(r)\) lies between its endpoint values. Combining the two facts proves the claim. \(\square\)

The positive part of \(U''\) bends the graph below the endpoint chord and is harmless for an upper bound. C090's \(|U''|\) gate is therefore replaced by a lower bound on \(U''\).

## 2.2 Numerical thresholds

The registered nominal values are

\[
U(0.0125)=0.9411,
\quad
U(0.025)=0.9605,
\quad
U(0.05)=0.9850.
\]

For the nominal ceiling \(0.99\), Lemma U-NEG requires

\[
M_1
\le
\frac{8(0.99-0.9605)}{(0.025-0.0125)^2}
=1510.4,
\]

and

\[
M_2
\le
\frac{8(0.99-0.985)}{(0.05-0.025)^2}
=64.
\]

If a one-sided \(+0.02\) uncertainty allowance is required, every endpoint and the target ceiling shift by the same amount, so the identical gates prove \(U\le1.01\).

## 2.3 Monotonicity shortcut

If an interval evaluator proves

\[
U'(r)\ge0
\quad(0<r\le0.05),
\]

then

\[
\sup_{0<r\le0.05}U(r)=U(0.05).
\]

This would close both the missing origin patch and the between-rung maximum.

The measured secant slopes are positive:

\[
\frac{0.9605-0.9411}{0.0125}=1.552,
\]

\[
\frac{0.9850-0.9605}{0.025}=0.980.
\]

The quadratic through the three rungs has

\[
\widehat U''=-30.5067,
\]

which lies inside the required \(-64\) floor. Neither observation is a certificate. They make monotonicity or one-sided curvature the correct numerical target.

## 2.4 Uncertainty semantics

The semantic part of the uncertainty gate is closed:

> A symmetric \(\pm0.02\) quantity used inside an upper theorem is consumed as a one-sided \(+0.02\) allowance unless its source is proved already to be a one-sided upper limit.

The calibration part remains open. The program must recover:

- whether \(0.02\) is deterministic or statistical;
- sample size;
- confidence level;
- simultaneous multiplicity across stations and rungs;
- whether the printed component values are central estimates or upper limits.

Until that source is recovered, the safe continuous target is \(1.01\), not \(0.99\).

---

# 3. Exact periodized-torus matrix transfer

C090 defined the exact covariance

\[
K_L(x)
=
\frac{\sum_{n\in\mathbb Z^2}e^{-|x+nL|^2/2}}
{\sum_{n\in\mathbb Z^2}e^{-|nL|^2/2}}.
\]

C091 propagates that perturbation through the matrix operations actually used by the local and far station calculations.

## 3.1 Block perturbation theorem

Let

\[
S=C-BA^{-1}B^T
\]

be a Gaussian conditional covariance. Let tildes denote the exact torus blocks and let

\[
\Delta A=\widetilde A-A,
\quad
\Delta B=\widetilde B-B,
\quad
\Delta C=\widetilde C-C.
\]

If

\[
\|A^{-1}\|\,\|\Delta A\|<1,
\]

then

\[
\|\widetilde A^{-1}\|
\le
\frac{\|A^{-1}\|}{1-\|A^{-1}\|\|\Delta A\|},
\]

and

\[
\|\widetilde A^{-1}-A^{-1}\|
\le
\|A^{-1}\|\|\Delta A\|\|\widetilde A^{-1}\|.
\]

Using

\[
\widetilde B\widetilde A^{-1}\widetilde B^T
-BA^{-1}B^T
=
\Delta B\widetilde A^{-1}\widetilde B^T
+B(\widetilde A^{-1}-A^{-1})\widetilde B^T
+BA^{-1}\Delta B^T,
\]

we obtain an explicit Schur-complement error bound.

## 3.2 Conservative declared scope

The executable certificate uses:

- at most nine pin coordinates;
- at most six target coordinates;
- covariance derivatives through total order four;
- the smallest retrieved raw nine-pin floor
  \[
  a_0=3.094564589647658\times10^{-14};
  \]
- local target region \(|x|\le3\);
- far pin-target distances \(5\le d\le12\).

The local result is

\[
\|\Delta S_{\rm local}\|
\le1.7924\times10^{-56},
\]

against a retrieved floor \(6.9\times10^{-13}\).

The far result is

\[
\|\Delta S_{\rm far}\|
\le1.8608\times10^{-12},
\]

against a retrieved floor \(1.4\times10^{-8}\).

The conditional-mean perturbation at far stations is bounded by

\[
7.411\times10^{-11}.
\]

These estimates already use the raw collapsing Gram floor; the divided-difference frame would improve the margins.

## 3.3 Adjudication

```text
G-TORUS-LOCAL:
    CLOSED in the declared scope.

G-TORUS-FAR-STATIONS:
    CLOSED in the declared scope.

G-TORUS-GLOBAL-INTEGRALS:
    PARTITION REQUIRED.
```

A global integral must identify its target dimension, derivative order, and distance partition. No use outside the declared scope inherits the certificate silently.

---

# 4. H4: exact Gaussian replacement for the Markov product

## 4.1 Why the product rule fails

The field values at successive corridor checkpoints are correlated. Six-pin conditioning also creates a strongly asymmetric mean ridge. A one-step conditional tail is therefore not a transition probability of a Markov chain.

The direct six-pin diagnostics give concrete product failures. For example, at \(r=0.025\), the east-chain checkpoint vector

\[
X=
(f(1.45,0),f(2.45,0),f(3.45,0))
\]

has

\[
P(X_i\ge b\ \forall i)
\approx0.07056,
\]

while

\[
q_{\rm step}^3
\approx0.02109.
\]

The product underestimates by a factor \(3.35\).

This does not prove that C089's final number \(0.0199\) is false; its ring masses and path geometry may provide enough slack. It proves that the stated Markov-comparison step cannot certify that number.

## 4.2 Unweighted Gaussian Chernoff theorem

Let

\[
X\sim N(\mu,\Sigma),
\]

and let \(t\) be a componentwise threshold. For every \(\theta\ge0\),

\[
\mathbf1_{\{X\ge t\}}
\le
\exp(\theta^T(X-t)).
\]

Taking expectation gives

\[
\boxed{
P(X\ge t)
\le
\exp\left(
-\theta^T(t-\mu)
+
\frac12\theta^T\Sigma\theta
\right).
}
\]

The exponent is a convex quadratic in \(\theta\). Minimizing over the nonnegative orthant gives a rigorous finite-dimensional corridor certificate once interval enclosures for \(\mu\) and \(\Sigma\) are supplied.

For a finite path family \(\mathcal P\),

\[
P(\text{some allowed corridor})
\le
\sum_{p\in\mathcal P}B_p.
\]

Path entropy must be included explicitly. It cannot be hidden in one representative ray.

## 4.3 Palm-weighted Gaussian Chernoff theorem

The q₀ theorem uses a determinant-weighted Palm law, so an unweighted orthant bound alone is insufficient.

Let \((H,X)\) be jointly Gaussian. Let \(H\) contain the Hessian coordinates entering a nonnegative Palm weight \(W(H)\), and let \(X\) contain corridor checkpoint values. Define

\[
\frac{dP_W}{dP}
=
\frac{W(H)}{E[W(H)]}.
\]

For \(\theta\ge0\),

\[
\begin{aligned}
P_W(X\ge t)
&=
\frac{E[W(H)\mathbf1_{\{X\ge t\}}]}{E[W(H)]}\\
&\le
\frac{E[W(H)e^{\theta^T(X-t)}]}{E[W(H)]}.
\end{aligned}
\]

Exponential tilting of the joint Gaussian shifts its mean by

\[
\operatorname{Cov}((H,X),X)\theta
\]

and leaves covariance unchanged. Therefore the marginal law of \(H\) under the tilt is

\[
H_\theta
\sim
N(\mu_H+\Sigma_{HX}\theta,\Sigma_{HH}).
\]

Hence

\[
\boxed{
P_W(X\ge t)
\le
\exp\left(
-\theta^T(t-\mu_X)
+
\frac12\theta^T\Sigma_{XX}\theta
\right)
\frac{E[W(H_\theta)]}{E[W(H)]}.
}
\]

This is exact. The determinant weight is not discarded; it becomes a shifted Gaussian determinant moment. The program already possesses the required class of Hessian regression and determinant-moment evaluators.

## 4.4 New H4 gate

The replacement gate is:

```text
H4-PALM-ORTHANT

1. Freeze the actual path/checkpoint family.
2. Interval-enclose μ_X, Σ_XX, and Σ_HX under the exact periodized six-pin law.
3. Evaluate the unshifted and shifted typed determinant moments.
4. Supply a nonnegative θ witness for every path class.
5. Multiply by certified ring and sector masses.
6. Sum over path entropy.
7. Carry one-sided uncertainty.
```

The theorem-level Palm identity is closed. The numerical path certificate is open.

---

# 5. Bonferroni: why the marked problem is different

## 5.1 The exact lower identity survives

The exact inclusion–exclusion statement remains

\[
1-q
\ge
E[N_{\rm qual}]
-
\frac12E[N_w(N_w-1)].
\]

The issue is only the proof that the second factorial moment is \(O(r^6)\).

## 5.2 Generic shrinking-window warning

For critical points of a smooth planar Gaussian field in a domain of area \(A\) and height-window width \(w\), the general second-moment theory has the form

\[
E[N_w^2]
\le
C(A^2w^2+Aw).
\]

The near-diagonal Kac–Rice bound uses one explicit height variable, not two independent window factors. This is precisely what one expects when two nearby critical values are highly correlated.

Therefore a spatial pair-correlation bound cannot simply be multiplied by \(\ell^2\) uniformly to \(d=0\) without a mark argument.

## 5.3 Same-type saddle repulsion

For isotropic planar Gaussian fields, the unmarked saddle–saddle two-point correlation has the strong short-distance form

\[
K_{ss}(d)
=
O\!\left(d^3\log\frac1d\right).
\]

Equivalent ball-count statements give

\[
E[N_s(B_\rho)(N_s(B_\rho)-1)]
\asymp
\rho^7\log\frac1\rho
\]

under their hypotheses.

This is stronger spatial repulsion than the old linear law, but it is unmarked: it does not by itself say that both saddle heights fall in the same \(\ell\)-window with probability \(O(\ell^2)\).

## 5.4 Exact two-critical-value gap law

Place two points at \((-d/2,0)\) and \((d/2,0)\) in the planar Bargmann–Fock field and condition on both gradients being zero. Let

\[
V_-=f(-d/2,0),
\qquad
V_+=f(d/2,0).
\]

Set

\[
A=\frac{V_-+V_+}{2},
\qquad
Z=\frac{V_+-V_-}{d^3}.
\]

Exact Gaussian regression yields

\[
\operatorname{Cov}(A,Z)=0
\]

by reflection symmetry, and

\[
\operatorname{Var}(A)
=
\frac23-rac{d^2}{18}+rac{d^4}{108}+O(d^6),
\]

\[
\operatorname{Var}(V_+-V_-)
=
\frac{d^6}{24}+rac{d^8}{96}+O(d^{10}),
\]

so

\[
\operatorname{Var}(Z)
=
\frac1{24}+rac{d^2}{96}+O(d^4).
\]

The limiting joint-density maximum of \((A,Z)\) is

\[
\frac{1}{2\pi\sqrt{(2/3)(1/24)}}
=
\frac3\pi.
\]

Thus the average critical value has an ordinary \(O(1)\) scale while the value difference has scale \(d^3\).

## 5.5 Window geometry

The inverse transformation is

\[
V_-=A-\frac{d^3Z}{2},
\qquad
V_+=A+\frac{d^3Z}{2}.
\]

Its Jacobian is

\[
\left|
\frac{\partial(V_-,V_+)}{\partial(A,Z)}
\right|
=d^3.
\]

For a height interval \(W\) of width \(\ell\), the preimage of \(W^2\) in \((A,Z)\)-space has area

\[
\frac{\ell^2}{d^3}.
\]

Also, if both values lie in \(W\), their average lies in \(W\). Therefore, if the joint density of \((A,Z)\) and the marginal density of \(A\) are bounded by \(C_{\rm mark}\),

\[
\boxed{
P(V_-,V_+\in W\mid\text{saddle pair})
\le
C_{\rm mark}\ell
\min\left(1,\frac\ell{d^3}\right).
}
\]

This is the missing two-scale mark law.

---

# 6. Corrected marked-repulsion Bonferroni theorem

## 6.1 Hypotheses

Work under the six-pin pair-Palm base law. Let \(K_{ss,r}(y,z)\) be the unmarked spatial intensity of an ordered pair of extra saddles. Let \(d=|y-z|\).

Assume uniformly for the relevant \(r\), base parameters, and locations:

### BR1 — same-type spatial repulsion

\[
K_{ss,r}(y,z)
\le
C_{\rm rep}d^3L(d),
\qquad
L(d)=\log(e/d),
\quad0<d\le\delta.
\]

### BR2 — critical-value mark density

Under the corresponding saddle-pair Palm law, the joint density of

\[
A=\frac{u_1+u_2}{2},
\qquad
Z=\frac{u_2-u_1}{d^3}
\]

and the marginal density of \(A\) are bounded by \(C_{\rm mark}\).

These hypotheses must be proved for the actual typed, determinant-weighted six-pin conditional family. The planar calculation supplies the limiting coordinate system and a benchmark constant, not the uniform transfer by itself.

## 6.2 Near-diagonal bound

The marked pair intensity in the window is at most

\[
C_{\rm rep}C_{\rm mark}
\,d^3L(d)\,
\ell\min\left(1,\frac\ell{d^3}\right).
\]

Integrating in polar coordinates and bounding the first location by \(|\Omega|\),

\[
\begin{aligned}
F_{2,\rm near}
\le{}&
2\pi|\Omega|C_{\rm rep}C_{\rm mark}
\int_0^\delta
 d^4L(d)\,
\ell\min\left(1,\frac\ell{d^3}\right)
\,dd.
\end{aligned}
\]

Split at

\[
d_*=\ell^{1/3}.
\]

Then

\[
\boxed{
\begin{aligned}
F_{2,\rm near}
\le{}&
2\pi|\Omega|C_{\rm rep}C_{\rm mark}
\left[
\ell\int_0^{d_*}d^4L(d)\,dd
+
\ell^2\int_{d_*}^{\delta}dL(d)\,dd
\right].
\end{aligned}
}
\]

The first term is \(O(\ell^{8/3}\log(1/\ell))\); the second is \(O(\ell^2)\). Hence

\[
F_{2,\rm near}=O(\ell^2)=O(r^6).
\]

This recovers the required Bonferroni order without assuming a uniform two-height density bound down to the diagonal.

## 6.3 Numerical gate for the conservative lower theorem

For \(\delta=0.1\) and the deliberately worst-case area \(|\Omega|=24^2=576\), the normalized near contribution has the form

\[
\mathfrak B_{2,\rm near}(r)
\le
c_{\rm near}(r)
C_{\rm rep}C_{\rm mark},
\]

with

\[
\begin{array}{c|c|c}
r&d_*=\ell^{1/3}&c_{\rm near}(r)\\
\hline
0.05&0.027516&0.895262\\
0.025&0.013758&0.938607\\
0.0125&0.006879&0.950928.
\end{array}
\]

Using the preserved far scenario \(C_{\rm dec}=4.051\) gives

\[
\mathfrak B_{2,\rm far}=2.450855.
\]

The finite \(0.84\) lower target through \(r=0.05\) permits total

\[
\overline{\mathfrak B}_2
\le78.895776.
\]

Therefore it is enough to prove approximately

\[
\boxed{
C_{\rm rep}C_{\rm mark}\le80.
}
\]

The exact rungwise allowable products are \(85.39\), \(81.45\), and \(80.39\).

This is a realistic interval target. The planar reference mark constant is below one. The remaining burden is the uniform six-pin typed spatial-repulsion constant.

## 6.4 Status change

The old dependency

```text
BONFERRONI
    -> ND_NOTE
    -> pointwise lambda_2 <= C_nd d per unit height^2
```

is replaced by

```text
BONFERRONI
    -> FAR_DECORRELATION
    -> SADDLE_SADDLE_SPATIAL_REPULSION
    -> CRITICAL_VALUE_GAP_MARK_LAW
    -> SIX_PIN_PALM_UNIFORMITY
```

Until these are certified, the decimal lower theorem remains conditional. The one-point \(\Lambda\) coefficient is unaffected.

---

# 7. Revised theorem status after C091

## 7.1 Lower side

The exact reduction remains:

\[
\frac{1-q}{r^3}
\ge
C_\Lambda(r)AO(r)
-
\mathfrak B_2(r)r^3.
\]

The one-point coefficient calculations remain valid at their recorded grades.

The statements

\[
\liminf_{r\downarrow0}
\frac{1-q}{r^3}
\ge0.8501
\]

and

\[
1-q\ge0.84r^3
\]

are now explicitly conditional on the marked-repulsion Bonferroni gate. The old citation to unmarked spatial repulsion is not enough.

## 7.2 Upper side

The rungwise arithmetic remains:

\[
0.9411,
\quad0.9605,
\quad0.9850.
\]

The continuous nominal target \(0.99\) and banded target \(1.01\) remain conditional on:

- `G-U-SHAPE`;
- the calibrated uncertainty rule;
- a valid H4 replacement;
- the already-closed exact-torus transfer in every consumed matrix scope.

Because the H4 Markov product is killed, the C089 \(0.0199\) corridor input cannot remain theorem-grade until it is re-established by a joint Gaussian/Palm certificate.

## 7.3 R0

No status change. R0 remains program-grade closed through SARD-G and externally unrefereed. C091 found no failure in that chain.

## 7.4 No replacement decimal is promoted

C091 does not replace one unsupported decimal theorem with another. The safe theorem statement is structural:

\[
1-q(r,b)=\Theta(r^3)
\]

**conditional on** the live one-point, AO, marked-Bonferroni, upper-shape, and H4-Palm-orthant gates, with each numerical tier named separately.

---

# 8. Revised dependency graph

```text
THEOREM_A_RATE
│
├── LOWER_EXACT_REDUCTION
│   ├── LAMBDA_SIDE                         [preserved]
│   ├── AO_SIDE                             [preserved]
│   └── BONFERRONI_MARKED                   [reopened]
│       ├── FAR_DECORRELATION               [measured/compactness]
│       ├── SADDLE_SADDLE_REPULSION         [new explicit node]
│       ├── VALUE_GAP_D3_MARK_LAW           [planar exact; six-pin transfer open]
│       └── PALM_UNIFORMITY                 [open]
│
├── UPPER_FIRST_INTERCEPTOR
│   ├── COLLAR_RUNG_VALUES                  [preserved measurements]
│   ├── OUTSIDE_CLASSIFICATION              [preserved]
│   ├── H4_PALM_ORTHANT                     [replaces Markov product]
│   │   ├── GAUSSIAN_CHERNOFF               [exact]
│   │   ├── PALM_WEIGHT_SHIFT_IDENTITY       [exact]
│   │   ├── PATH_FAMILY                     [open]
│   │   ├── INTERVAL_GAUSSIAN_BLOCKS         [open]
│   │   └── SHIFTED_DETERMINANT_MOMENTS      [open]
│   ├── UPPER_UNCERTAINTY_SEMANTICS          [closed]
│   ├── UPPER_UNCERTAINTY_CALIBRATION        [open]
│   └── UPPER_SHAPE                         [open]
│       ├── MONOTONICITY or
│       └── NEGATIVE_CURVATURE
│
├── EXACT_PERIODIZED_TORUS
│   ├── LOCAL_MATRIX_TRANSFER               [closed in declared scope]
│   ├── FAR_STATION_TRANSFER                [closed in declared scope]
│   └── GLOBAL_PARTITION_COVERAGE           [open bookkeeping]
│
└── R0 / SARD-G                             [program-grade closed]
```

Killed or quarantined edges:

```text
H4_QSTEP_PRODUCT                            [KILLED]
BONFERRONI_POINTWISE_CND_PER_HEIGHT2        [QUARANTINED]
```

---

# 9. Immediate next computations

## 9.1 BR-MARK: six-pin mark-density certificate

For each box in

\[
r\in(0,0.05],
\quad y,z\in\Omega,
\quad0<|y-z|\le0.1,
\]

construct the divided-difference coordinates

\[
A=\frac{u_1+u_2}{2},
\qquad
Z=\frac{u_2-u_1}{d^3}.
\]

Interval-enclose the covariance determinant under the two-saddle Palm conditioning. The target is a density bound sufficiently small that

\[
C_{\rm rep}C_{\rm mark}<80.
\]

## 9.2 BR-REP: typed saddle spatial repulsion

Evaluate the unmarked determinant-weighted pair intensity in the same boxes and certify

\[
K_{ss,r}(y,z)
\le
C_{\rm rep}d^3\log(e/d).
\]

A direct normalized ratio

\[
\frac{K_{ss,r}(y,z)}{d^3\log(e/d)}
\]

is the correct interval object.

## 9.3 H4-PATH

Recover the actual corridor skeleton and sector/ring partition from C086–C087. For every path class:

1. build the exact periodized six-pin Gaussian block;
2. choose a nonnegative Chernoff witness;
3. evaluate shifted Palm determinant moments;
4. interval-bound the result;
5. multiply by the class mass;
6. sum the classes.

The target is to re-establish or replace

\[
\varepsilon_{\rm far}=0.0199r^3.
\]

## 9.4 U-SHAPE

Differentiate the complete upper evaluator, not a fit through three printed values. Attempt the gates in this order:

1. prove \(U'\ge0\);
2. failing that, prove \(U''\ge-64\) on \([0.025,0.05]\);
3. prove \(U''\ge-1510.4\) on \([0.0125,0.025]\);
4. extend to \(r=0\) by the same regularized evaluator.

## 9.5 uncertainty provenance

Locate the source of the C089 \(0.02\) band. Until then, retain the \(+0.02\) theorem consumption and do not call the nominal values upper confidence limits.

---

# 10. Broader methodological conclusion

C091 demonstrates why theorem verification needs more than arithmetic and source hashes.

The lower failure mode was **type mismatch**:

```text
unmarked spatial correlation
    was consumed as
marked two-height-window intensity.
```

The upper failure mode was **dependence mismatch**:

```text
one-dimensional Gaussian tail
    was consumed as
multi-step transition probability.
```

Both errors preserve plausible exponents and plausible decimals. Neither is caught by a constants linter.

The required proof-object fields are therefore extended by:

```text
mark_space
conditioning_sigma_field
dependence_model
weight_measure
factorization_justification
path_entropy
```

This is directly relevant to verified LLM reasoning. A collection of individually calibrated verifier scores cannot be multiplied unless their dependence structure is certified; a detector calibrated on unmarked errors cannot be applied to a more selective error subtype without a mark-transfer theorem.

---

# 11. References and source map

## Internal q₀ archive

- Master v3.2, files 00, 01, 02a, 03, and 04.
- C024 lower reduction and independent estimator verification.
- C089 theorem compendium supplied in the conversation.
- C090 canonical theorem repair and exact periodized BF contract.

## External primary literature used in the Bonferroni audit

- Stephen Muirhead, *A Second Moment Bound for Critical Points of Planar Gaussian Fields in Shrinking Height Windows*, arXiv:1901.11336.
- Dmitry Beliaev, Valentina Cammarota, Igor Wigman, *Two Point Function for Critical Points of a Random Plane Wave*, arXiv:1911.03455.
- Safa Ladgham, Raphaël Lachièze-Rey, *Local repulsion of planar Gaussian critical points*, arXiv:2209.04150.

---

# 12. Final C091 status table

| object | status after C091 |
|---|---|
| exact periodized ensemble | CLOSED |
| local torus matrix transfer | CLOSED in declared scope |
| far-station torus transfer | CLOSED in declared scope |
| upper uncertainty semantics | CLOSED |
| upper uncertainty calibration | OPEN |
| one-sided interpolation theorem | CLOSED |
| upper monotonicity/curvature certificate | OPEN |
| C089 H4 Markov product | KILLED |
| Gaussian orthant replacement | CLOSED structurally |
| Palm-weighted Chernoff identity | CLOSED structurally |
| H4 numerical path assembly | OPEN |
| exact Bonferroni identity | CLOSED |
| planar \(d^3\) value-gap law | CLOSED |
| old pointwise marked \(C_{nd}d\) argument | QUARANTINED |
| marked saddle-repulsion theorem | DERIVED CONDITIONAL |
| six-pin constants \(C_{rep},C_{mark}\) | OPEN |
| finite lower \(0.84\) | BLOCKED on marked Bonferroni gate |
| asymptotic lower \(0.8501\) | BLOCKED on bounded marked Bonferroni coefficient |
| R0/SARD-G internal status | PROGRAM-GRADE CLOSED |
| R0 external status | SPECIALIST REVIEW PENDING |

**END OF C091 PROOF-GATE REDUCTION**
