# Thermodynamic stabilization and critical-height crossover  
## A proposed local-Palm extension using Bargmann–Fock percolation

## Abstract

This note asks whether the configured maximum–saddle selection problem stabilizes as the torus size grows.

The proposed mechanism is local rather than extensive. The elder-rule decision is determined by the two superlevel components incident to the saddle. At any fixed positive level, published Bargmann–Fock percolation results give exponential decay of long connections. A finite-rank Gaussian conditioning argument is then used to transfer this decay to the local value/gradient-pinned and determinant-weighted pair-Palm laws.

If the transfer is valid, the influence region has an exponential radius tail, a selected-set Campbell bound controls window saddles in that random region, and

\[
|q_L-q_\infty|\le Ce^{-cL},
\qquad
1-q_\infty\le Cr^3.
\]

The main new argument is the conditional-to-unconditioned transfer. It is written as a proof sketch and is the central review target.

---

# 1. Planar and torus fields

The planar Bargmann–Fock field is the centered stationary Gaussian field \(F\) on \(\mathbb R^2\) with covariance

\[
K_\infty(x-y)=e^{-|x-y|^2/2}.
\]

The periodized field \(F_L\) on \(\mathbb T_L^2\) has covariance

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

On a ball of radius \(o(L)\), \(K_L\) and its derivatives converge rapidly to \(K_\infty\).

Fix a typed gradient-adjacent maximum–saddle pair \(M,S\), with

\[
f(M)=b,
\qquad
f(S)=s=b-\frac{\kappa r^3}{6}.
\]

Assume

\[
b\in B\Subset(0,\infty),
\qquad
\kappa\in K\Subset(0,\infty),
\]

and \(r\) small enough that

\[
s\ge b_{\min}/2>0.
\]

---

# 2. Influence region

Consider the strict superlevel set

\[
E_s^+=\{x:f(x)>s\}.
\]

The two ascending separatrices of \(S\) enter two components of \(E_s^+\). Let:

- \(C_M\) be the component incident to the branch terminating at \(M\);
- \(C_O\) be the component incident to the other branch.

Define

\[
A_{\min}
=
\overline{C_M\cup C_O}\cup\{S\}.
\]

### Proposition 2.1 — information localization

Under the Morse–Smale and distinct-critical-value hypotheses, the event

\[
D(M)=S
\]

is determined by the restriction of \(f\) to \(A_{\min}\).

### Proof

The restriction determines every critical point and critical value in the two incident components. It therefore determines:

1. whether \(M\) is the highest maximum in \(C_M\), equivalently whether \(M\) survived to level \(s\);
2. the elder birth maximum in \(C_O\);
3. which component is older when they merge at \(S\).

The component born at \(M\) dies at \(S\) exactly when \(M\) is still alive and the elder birth of \(C_O\) is higher than \(f(M)\). \(\square\)

For cross-fit counting, it is convenient to pad the region by a deterministic unit ball centered at the pair midpoint:

\[
A=A_{\min}\cup\overline{B_1(m)}.
\]

Padding does not change the information available about the death partner. It only guarantees

\[
|A|\ge\pi.
\]

Define the influence radius

\[
R(A)=\inf\{R:A\subset B_R(m)\}.
\]

---

# 3. Published percolation input

For the planar Bargmann–Fock field, the critical excursion level is zero.

In the sign convention used here:

- the positive superlevel set
  \[
  \{F\ge u\},\qquad u>0,
  \]
  is subcritical;
- its long connection probabilities decay exponentially;
- at level zero, Russo–Seymour–Welsh-type crossing probabilities remain nontrivial across scales.

The relevant external results are:

1. A. Rivera and H. Vanneuville, *The critical threshold for Bargmann–Fock percolation*, arXiv:1711.05012.
2. S. Muirhead and H. Vanneuville, *The sharp phase transition for level set percolation of smooth planar Gaussian fields*, arXiv:1806.11545.

The argument below consumes their exponential positive-level connection decay. It does not reprove it.

---

# 4. Finite-rank Gaussian conditioning

Let \(Y\) be the finite vector of local pair pins: values and gradients at \(M,S\), and later possibly a third saddle pin.

In a Gaussian Hilbert space, decompose the unconditioned field as

\[
F=G+P,
\]

where:

- \(P\) is the orthogonal Gaussian projection onto the span of the covariance representers of \(Y\);
- \(G\) is independent of \(Y\).

Conditioning on \(Y=y\) replaces \(P\) by the deterministic regression mean:

\[
F^{\mathrm{pin}}=G+m_y.
\]

Thus

\[
F^{\mathrm{pin}}-F=m_y-P.
\]

The difference is finite dimensional.

For value and derivative pins of the Bargmann–Fock field, each covariance representer is a polynomial times

\[
e^{-|x-x_i|^2/2}.
\]

On an annulus at distance \(R\) from the pair, the \(C^2\) norm of each representer is bounded by

\[
C(1+R)^d e^{-cR^2}.
\]

The Gaussian coefficients of \(P\) have sub-Gaussian tails. Therefore, outside an event of very small probability, the conditional and unconditioned fields differ on a far annulus by less than a fixed sprinkling margin, for example \(b_{\min}/4\).

A connection in the conditioned field at level

\[
s\ge b_{\min}/2
\]

then implies a connection in the unconditioned field at some still-positive level, for example \(b_{\min}/4\).

Applying the published subcritical estimate suggests

\[
P^{\mathrm{pin}}\{R(A)>R\}
\le
Ce^{-cR}.
\]

---

# 5. Pair-Palm determinant reweighting

The typed pair-Palm law has density

\[
\frac{W_{MS}}{E[W_{MS}\mid J_6]}
\]

with respect to the six-pin Gaussian law.

The denominator is of order \(r^2\) and has a positive normalized limit. The numerator is a polynomial in conditioned Gaussian Hessian entries. On compact \(b,\kappa\) sets, the normalized weight

\[
\widetilde W_r=\frac{W_{MS}}{E[W_{MS}\mid J_6]}
\]

is claimed to have uniformly bounded \(L^p\) moments for some \(p>1\).

Then Hölder’s inequality gives

\[
P^{MS}(E)
=
E^{\mathrm{pin}}[\widetilde W_r\mathbf 1_E]
\le
\|\widetilde W_r\|_{L^p}
P^{\mathrm{pin}}(E)^{1-1/p}.
\]

Exponential decay is preserved, with a smaller exponent.

Thus the proposed influence-tail theorem is

\[
\boxed{
P^{MS}\{R(A)>R\}
\le
Ce^{-cR}.
}
\]

All polynomial moments of \(R(A)\) and \(|A|\) then follow.

### Main point for review

The finite-rank coupling is plausible, but a publishable proof should specify:

1. the precise annulus and sprinkling event;
2. the uniform covariance bounds for the finite-dimensional coefficients;
3. the relation between an influence-region connection and a standard rectangle or annulus crossing event;
4. the \(L^p\) bound for the normalized determinant weight;
5. uniformity after an additional third-saddle pin.

---

# 6. Same-field and cross-fit Campbell counts

Let

\[
W_r=(s,b)
\]

be the thin height window, and let \(\Xi_{W_r}(F)\) be the point process of saddles of \(F\) whose values lie in \(W_r\).

Define the same-field selected count

\[
N_{\mathrm{same}}
=
E\bigl[\Xi_{W_r}(F)(A(F))\bigr].
\]

Let \(G\) be an independent stationary Bargmann–Fock field. Define

\[
N_{\mathrm{cross}}
=
E\bigl[\Xi_{W_r}(G)(A(F))\bigr].
\]

By independence and Campbell’s theorem,

\[
\boxed{
N_{\mathrm{cross}}
=
\rho_{\mathrm{sad}}(W_r)E|A|.
}
\]

Here \(\rho_{\mathrm{sad}}(W_r)\) is the stationary saddle intensity integrated over the window.

Since the saddle-height density is continuous and positive at finite \(b\),

\[
\rho_{\mathrm{sad}}(W_r)
=
\rho_{\mathrm{sad}}(b)|W_r|(1+o(1)).
\]

Because \(|A|\ge\pi\),

\[
N_{\mathrm{cross}}\ge c|W_r|.
\]

Define the exact selection-amplification ratio

\[
\lambda_r=\frac{N_{\mathrm{same}}}{N_{\mathrm{cross}}}.
\]

No independence assumption is made for \(N_{\mathrm{same}}\).

---

# 7. Selected Kac–Rice intensity

The same-field count can be written as

\[
N_{\mathrm{same}}
=
\int_{\mathbb R^2}n_r(x)\,dx,
\]

where \(n_r(x)\) is a pair/third-saddle Kac–Rice intensity carrying the additional global mark

\[
\mathbf 1_{\{x\in A(F)\}}.
\]

On a fixed neighborhood of the pair, finite-jet compactness and the window width give

\[
n_r(x)\le C|W_r|.
\]

For \(|x|\) large, the event \(x\in A(F)\) requires a positive-level connection from the pair neighborhood to the neighborhood of \(x\).

If the local-Palm percolation transfer remains valid after additionally conditioning \(x\) to be a window saddle, then

\[
n_r(x)
\le
C|W_r|e^{-c|x|}.
\]

Therefore

\[
N_{\mathrm{same}}\le C|W_r|.
\]

Combining with the cross-fit lower bound gives

\[
\boxed{
\lambda_r\le C.
}
\]

Selection may change the coefficient, but not the thin-window order.

### Main point for review

The additional third-saddle conditioning changes the Gaussian law and determinant weight. The proposed proof treats it as another finite-rank local Palm perturbation. A reviewer should check whether the exponential connection estimate is uniform in the third-point location after the Kac–Rice density is factored out.

---

# 8. Thermodynamic stabilization

Let \(q_L(r,b,\kappa)\) be the configured-pair selection probability on \(\mathbb T_L^2\).

Let \(q_\infty(r,b,\kappa)\) be the corresponding planar pair-Palm probability, assuming the planar event is defined through the finite influence region.

If

\[
R(A)<L/4,
\]

the torus and planar fields can be coupled in \(C^2(B_{L/4}(m))\) with rapidly decaying error, because the periodized covariance differs from the planar covariance by distant image terms.

On a Morse–Smale configuration with a positive structural-stability margin, sufficiently small \(C^2\) error preserves:

- the two incident superlevel components;
- their elder maxima;
- the death-partner decision.

The influence-tail bound then yields

\[
\boxed{
|q_L-q_\infty|
\le
Ce^{-cL}.
}
\]

The fixed-\(r\) pairing analysis gives

\[
1-q_\infty\le Cr^3.
\]

Hence

\[
\boxed{
1-q_L
\le
Cr^3+Ce^{-cL}.
}
\]

---

# 9. Consequence for the global-hazard hypothesis

The total number of window saddles on a torus is proportional to

\[
L^2|W_r|.
\]

A naive global hazard would therefore suggest

\[
-\log q_L\asymp L^2r^3.
\]

The influence-region mechanism predicts otherwise. Distant window saddles affect the configured pair only if they belong to or connect into the localized merge-tree influence region.

If the exponential localization and selected-set Campbell estimates are valid, the relevant count remains \(O(r^3)\) as \(L\to\infty\).

Thus the \(L^2r^3\) law is rejected as the leading mechanism for a fixed configured pair.

This does not say that the total number of window saddles fails to grow with area. It says that the vast majority are topologically irrelevant to the selected pair.

---

# 10. Critical-height crossover

At every fixed positive level \(b>0\), the positive superlevel set is subcritical and has exponential connection decay.

At level zero, RSW-type crossing probabilities remain bounded away from zero across scales.

Let \(c(b)\) denote an admissible exponential decay rate:

\[
P\{\text{connection to distance }R\}
\le
C(b)e^{-c(b)R}.
\]

If \(c(b)\) failed to tend to zero as \(b\downarrow0\), a uniform positive lower bound on \(c(b)\) would imply exponential decay at or arbitrarily close to the critical level, contradicting scale-uniform critical crossings.

Therefore

\[
\boxed{
c(b)\to0
\qquad
(b\downarrow0).
}
\]

The localization length

\[
\xi(b)=1/c(b)
\]

diverges.

This proves a qualitative crossover. It does not identify the divergence exponent of

\[
E^{MS}|A|
\]

or of any pair-Palm susceptibility.

---

# 11. Numerical diagnostics

Two earlier computations are relevant but not part of the proof.

## 11.1 Single-value-pin susceptibility

A finite-size cluster experiment conditioned only on one field value suggested:

\[
\widehat\gamma\approx2.439,
\qquad
\widehat\nu\approx1.372,
\qquad
\widehat{\gamma/\nu}\approx1.778.
\]

These are close to standard two-dimensional percolation exponents, but the law is not the configured pair-Palm law.

The same experiment showed wrap probability tending to zero at sampled positive heights down to \(b=0.2\) by \(L=96\), suggesting large but finite positive-height susceptibility.

## 11.2 Cross-fit selection coefficient

A simpler selected-set experiment measured an order-one same-field/cross-fit amplification, roughly \(4.5\)–\(5.5\), over its resolved thin-window range.

It did not show divergence as the window narrowed, but it used a simpler pinning law and cannot certify the pair-Palm theorem.

---

# 12. Precise status

## Deterministic and exact

- definition of the two-component influence region;
- proof that it determines the elder-rule decision;
- cross-fit Campbell identity;
- implication of exponential influence tails for finite moments and torus stabilization;
- qualitative contradiction between uniform positive-height exponential decay and critical RSW.

## Published external input

- positive-level subcritical exponential connection decay;
- critical level zero;
- critical RSW behavior.

## New proof sketches requiring review

1. finite-rank conditioning transfer on far annuli;
2. uniform \(L^p\) moments of the normalized determinant Palm weight;
3. extension of the transfer after a third saddle pin;
4. selected Kac–Rice intensity with the random membership mark \(x\in A(F)\);
5. structural-stability coupling between torus and planar merge trees.

## Conservative theorem

A defensible statement before review is:

> Conditional on the local-Palm percolation-transfer and selected-intensity lemmas above, the configured-pair selection probability has a planar thermodynamic limit with exponentially small finite-size error at every fixed positive height.

---

# References

1. A. Rivera and H. Vanneuville, “The critical threshold for Bargmann–Fock percolation,” arXiv:1711.05012.
2. S. Muirhead and H. Vanneuville, “The sharp phase transition for level set percolation of smooth planar Gaussian fields,” arXiv:1806.11545.
3. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer, 2007.
4. J.-M. Azaïs and M. Wschebor, *Level Sets and Extrema of Random Processes and Fields*, Wiley, 2009.
