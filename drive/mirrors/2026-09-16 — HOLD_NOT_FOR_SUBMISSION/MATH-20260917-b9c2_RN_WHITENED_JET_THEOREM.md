# Whitened Gaussian chi-square: exact fourth-order jets and conditional uniform enclosures

Date: 2026-09-17. Parent claim: DQ-MATH-20260917-b9c2.

**Disposition:** Author-side mathematical reconstruction and executable conditional enclosure. This supplies the functional-calculus layer missing from the RN-UNIF whitened-envelope stub. It does not certify the SIDE24 field's domain-wide input jets, the full κ_far composition, Piece 2, or uniformity in r. **D3-LEMMA-RN-UNIF remains OPEN; lemma_closed=false.**

Provenance: same-family author-side work after shared external reconnaissance. Memo `MATH-20260917-b9c2_EXTERNAL_RECON.md`, SHA-256 `29a117b8702e590e0a34005497bda19409c167c31d7d8ec291cb1b4452badb9a`, and its verified custody receipt were read before substantive derivation. No blind or independent-provider credit is claimed. The formulas below are applications of standard Gaussian integration, matrix differentiation, and Bell polynomials; no novelty claim is made for those identities. The useful application here is an explicit covariance/mean-jet pipeline replacing residual-form proxy division.

## 1. Objects, convention, and exact domain

Let Q=N(0,I_d), P_t=N(m(t),S(t)), with S(t) real symmetric positive definite, and set A(t)=I_d−S(t). All displayed jets X_j mean **raw derivatives** d^jX/dt^j, not Taylor coefficients. The parameter t can be a Cartesian coordinate or a straight unit-direction path through a spatial domain. A nonlinear polar path requires its own chain-rule input jets.

Write q=∫(dP/dQ)² dQ and χ²=q−1. Then q is finite if and only if

\[
0<S<2I_d,\qquad\text{equivalently}\qquad -I_d<A<I_d.
\]

Inside that domain,

\[
\boxed{\ell:=\log q=-\tfrac12\log\det(I_d-A^2)
                   +m^T(I_d+A)^{-1}m.}\tag{1}
\]

The covariance S must be positive definite to define the nonsingular Gaussian density. Given that hypothesis, finiteness is equivalent to positive definiteness of M=S^{-1}−I/2. At a zero eigenvalue of M the Lebesgue Gaussian integral diverges even if its linear exponent vanishes along the null direction; if that linear term is nonzero one tail diverges. Negative eigenvalues also cause divergence. Thus a positive determinant alone is insufficient.

**Proof of (1).** Completing the square in p²/q₀ gives

\[
\ell=-\tfrac d2\log2-\log\det S-\tfrac12\log\det M
 +(S^{-1}m)^TM^{-1}(S^{-1}m)-m^TS^{-1}m.
\]

Because M=½S^{-1}(2I−S), the determinant contribution reduces to −½log det[S(2I−S)], and the mean contribution reduces to mᵀ(2I−S)^{-1}m. Substitution S=I−A gives (1). These commutations involve rational functions of the same matrix S and are legitimate. **No commutation of derivative matrices is used later.**

Both terms in (1) are nonnegative. Indeed the eigenvalues of I−A² lie in (0,1], and I+A is positive definite. Consequently ℓ=0 iff A=0 and m=0. A negative numerical χ² close to zero is a precision/rounding issue, never a permitted replacement of the exact identity.

For a fixed nonidentity reference law N(μ₀,S₀), choose one y-independent W with WS₀Wᵀ=I. Then S=WS_pairWᵀ and m=W(μ_pair−μ₀). Changing coordinates in both laws proves the same formula. A y-dependent whitening matrix requires its derivatives and is outside the simplified input convention.

## 2. Exact matrix recurrences through every finite order

For a differentiable invertible matrix X with inverse R,

\[
R_0=X_0^{-1},\qquad
R_n=-R_0\sum_{j=1}^{n}\binom nj X_jR_{n-j}.\tag{2}
\]

The products remain in the displayed order. Equation (2) follows by differentiating XR=I n times and solving for R_n. In particular,

\[
\begin{aligned}
R_1={}&-RX_1R,\\
R_2={}&2RX_1RX_1R-RX_2R,\\
R_3={}&-RX_3R+3RX_1RX_2R+3RX_2RX_1R
          -6RX_1RX_1RX_1R,\\
R_4={}&-RX_4R+4RX_1RX_3R+4RX_3RX_1R+6RX_2RX_2R\\
 &-12(RX_1RX_1RX_2R+RX_1RX_2RX_1R+RX_2RX_1RX_1R)\\
 &+24RX_1RX_1RX_1RX_1R.
\end{aligned}\tag{3}
\]

Define B=(I+A)^{-1} and C=(I−A)^{-1}. Their jets come from (2) using X₀=I±A₀ and X_j=±A_j for j≥1. Jacobi's formula gives, for n≥1,

\[
\boxed{\ell_n=-\frac12\sum_{j=0}^{n-1}\binom{n-1}{j}
       \operatorname{tr}[(B_j-C_j)A_{n-j}]
 +\sum_{i+j+k=n}\frac{n!}{i!j!k!}\,m_i^TB_jm_k.}\tag{4}
\]

This formula, implemented without commutative shortcuts, includes every covariance, mean, inverse, and cross derivative. It applies at orders 1–4 and beyond.

For explicit checking write ℓ_n=H_n+T_n, where H=−½log det(I−A²) and T=mᵀBm. With undecorated B,C,m denoting order zero,

\[
\begin{aligned}
2H_2={}&\operatorname{tr}\{(C-B)A_2+(BA_1)^2+(CA_1)^2\},\\
2H_3={}&\operatorname{tr}\{(C-B)A_3+3BA_1BA_2+3CA_1CA_2
                              -2(BA_1)^3+2(CA_1)^3\},\\
2H_4={}&\operatorname{tr}\{(C-B)A_4+4BA_1BA_3+4CA_1CA_3
                 +3(BA_2)^2+3(CA_2)^2\\
 &\qquad-12BA_1BA_1BA_2+12CA_1CA_1CA_2
                 +6(BA_1)^4+6(CA_1)^4\}.
\end{aligned}\tag{5}
\]

Only cyclicity of the trace is used to combine words in (5). It does not justify dropping one of the distinct untraced products in (3).

Since all B_j are symmetric,

\[
\begin{aligned}
T_2={}&2m_2^TBm+2m_1^TBm_1+4m_1^TB_1m+m^TB_2m,\\
T_3={}&2m_3^TBm+6m_2^TBm_1+6m_2^TB_1m+6m_1^TB_1m_1
                   +6m_1^TB_2m+m^TB_3m,\\
T_4={}&2m_4^TBm+8m_3^TBm_1+6m_2^TBm_2
       +8m_3^TB_1m+24m_2^TB_1m_1\\
 &+12m_2^TB_2m+12m_1^TB_2m_1+8m_1^TB_3m+m^TB_4m.
\end{aligned}\tag{6}
\]

For mixed spatial derivatives, replace n by a multi-index α, binomial coefficients by multinomial componentwise binomials, and j by β≤α. In the log-determinant part first choose e with α_e>0 and use ∂^α logdetX=∂^(α−e) tr(X^{-1}∂_eX). This removes ambiguity about the first differentiated coordinate. Ordinary smoothness ensures equality for another choice of e.

## 3. Complete exponential and square-root composition

Let X=χ²=exp(ℓ)−1. The full derivatives are

\[
\begin{aligned}
X_1&=e^\ell\ell_1,\\
X_2&=e^\ell(\ell_2+\ell_1^2),\\
X_3&=e^\ell(\ell_3+3\ell_1\ell_2+\ell_1^3),\\
\boxed{X_4}&=\boxed{e^\ell(\ell_4+4\ell_1\ell_3+3\ell_2^2
                       +6\ell_1^2\ell_2+\ell_1^4).}
\end{aligned}\tag{7}
\]

An envelope for ℓ₄ by itself is not an envelope for X₄. The implementation uses the complete recurrence E₀=e^ℓ, E_n=Σ_{j=0}^{n−1} binom(n−1,j) ℓ_{j+1}E_{n−1−j}.

For g=√X at X>0,

\[
\begin{aligned}
g_1={}&X_1/(2\sqrt X),\\
g_2={}&X_2/(2\sqrt X)-X_1^2/(4X^{3/2}),\\
g_3={}&X_3/(2\sqrt X)-3X_1X_2/(4X^{3/2})+3X_1^3/(8X^{5/2}),\\
g_4={}&X_4/(2\sqrt X)-(X_1X_3+\tfrac34X_2^2)/X^{3/2}
             +\tfrac94X_1^2X_2/X^{5/2}-\tfrac{15}{16}X_1^4/X^{7/2}.
\end{aligned}\tag{8}
\]

The exact alternative recurrence g₀=√X₀ and g_n=[X_n−Σ_{j=1}^{n−1}binom(n,j)g_jg_{n−j}]/(2g₀) is implemented.

**Zero-set hazard.** The function √χ² need not even be C¹ at a zero. For A=0,m=t in dimension one, χ²=e^{t²}−1 and √χ²=|t|(1+O(t²)), with one-sided derivatives ±1 at zero. Thus uniform use of (8) requires a proved X≥δ>0 on the entire domain, or a separate argument avoiding division by √X. A positive value at a center, finite-difference samples, or a fitted scale is insufficient. The root workstream's separate Hilbert-space modulus is not incorporated into this fourth-order result.

The SIDE24 pair term is κ_pair=(hL2/Z_LO)√χ², where the source variable `hL2=(sqrt(dM4)*sqrt(dS4))**0.5` denotes the fixed reference L² bound; its name does not denote the square of a variable h_L. Its prefactor is y-independent at the fixed r,v under discussion. Equations (8) therefore apply to that term after multiplication by this prefactor, provided its uniform zero-set issue is resolved. No conclusion about κ_y, κ_cross, R_pg, or R_wm follows from these pair calculations alone.

## 4. A rigorous norm envelope retaining small-covariance cancellation

Let Ω be a common domain on which the jets exist and A is symmetric. Suppose actual input estimates

\[
\sup_\Omega\|A_j\|_2\le a_j,\qquad
\sup_\Omega\|m_j\|_2\le c_j,\qquad j=0,1,2,3,4,
\quad 0\le a_0<1
\tag{9}
\]

hold. All quantities may also be uniform over a specified family of unit directions. Norms of all covariance derivatives refer to their actual ordered Schur/inverse construction, not to a residual form divided by √λ.

For U=A², set

\[
u_n=\sum_{j=0}^n\binom nj a_j a_{n-j},\qquad
h_0=(1-a_0^2)^{-1},\quad b_0=(1-a_0)^{-1},
\]

\[
h_n=h_0\sum_{j=1}^n\binom nj u_jh_{n-j},\qquad
b_n=b_0\sum_{j=1}^n\binom nj a_jb_{n-j}.\tag{10}
\]

They bound the derivative norms of (I−A²)^{-1} and (I+A)^{-1}. From Jacobi's formula on I−U, the determinant term admits the useful quadratic-cancellation bound

\[
L_n=\frac d2\sum_{j=0}^{n-1}\binom{n-1}{j}h_j u_{n-j}
     +\sum_{i+j+k=n}\frac{n!}{i!j!k!}c_i b_j c_k,
\quad 1\le n\le4.\tag{11}
\]

Then |ℓ_n|≤L_n. In particular u₁=2a₀a₁ retains the vanishing first derivative of the determinant term at A=0, unlike an indiscriminate separate logdet bound.

A rational bound for ℓ itself is

\[
0\le\ell\le L_0:=\frac{d a_0^2}{2(1-a_0^2)}+\frac{c_0^2}{1-a_0},\tag{12}
\]

using −log(1−x)≤x/(1−x). One can tighten (12) using a certified log evaluation, but this is unnecessary for validity.

Choose an exact rational E≥exp(L₀). The executable module uses N=max(1,ceil(32L₀)) and

\[
E=(1-L_0/N)^{-N};\tag{13}
\]

it is valid because −log(1−u)≥u for 0≤u<1. At L₀=0 set E=1. This operation is exact rational arithmetic. Very large input bounds are rejected as computationally unsuitable rather than silently rounded.

Set X̄₀=E−1 and replace e^ℓ by E and each |ℓ_j| by L_j in the nonnegative polynomial on the right of (7). This gives rigorous X̄_j≥sup|X_j|, j≤4. If a strictly positive uniform floor δ is supplied, replacing negative signs by plus signs in (8), X_j by X̄_j, and reciprocal powers of X by those of δ gives valid √χ² derivative envelopes. The executable code implements these coefficients exactly, with a rational upper bound for 1/√δ.

**Proof.** The spectral theorem gives the zeroth inverse bounds in (10). Submultiplicativity, the product rule, and (2) prove the remaining bounds by induction. For the trace use |tr Z|≤d‖Z‖₂; for the quadratic terms use |xᵀRy|≤‖x‖₂‖R‖₂‖y‖₂. Equations (11)–(13), followed by (7) and (8), complete the argument. None of these inequalities requires derivative matrices to commute.

## 5. Actual SIDE24 residual-covariance propagation

The source's two-stage conditioning gives, with the pair rows selected,

\[
D=YY_6,\quad T=TY_{6,\mathrm{pair}},\quad
F=WT,\quad z=(v,0,0)^T-YC\,w_6.
\]

The y-independent reference covariance is S_pair,0; the reference mean is μ₆=X₆w₆. Therefore

\[
\boxed{A=FD^{-1}F^T,\qquad m=FD^{-1}z.}\tag{14}
\]

This is an exact Schur-complement consequence, not an assumed approximation. D must be positive definite, and S_pair=S_pair,0−TD^{-1}Tᵀ must remain positive definite. Hence (14) also implies A≥0 in this particular application. The general theorem allows negative eigenvalues of A as well.

Given true jets of F,D,z, compute J_j=(D^{-1})_j by (2), then

\[
A_n=\sum_{i+j+k=n}\frac{n!}{i!j!k!}F_iJ_jF_k^T,\qquad
m_n=\sum_{i+j+k=n}\frac{n!}{i!j!k!}F_iJ_jz_k.\tag{15}
\]

These are the required whitened residual-**covariance and mean** jets through order four. In particular they include all derivatives of D^{-1}, the second residual factor Fᵀ, and the residual mean z. The original first-gradient mean-fix term is part of this construction; it cannot be dropped for higher orders.

For a stationary field let Y_j be the actual jets of YC and V_j those of unconditioned TY_pair. The fixed matrices G₆^{-1},X₆,W,w₆ give

\[
F_j=W(V_j-X_6G_6^{-1}Y_j^T),\quad
D_0=YY-Y_0G_6^{-1}Y_0^T,
\]

\[
D_n=-\sum_{j=0}^n\binom nj Y_jG_6^{-1}Y_{n-j}^T\ (n\ge1),\quad
z_0=(v,0,0)^T-Y_0w_6,\quad z_n=-Y_nw_6\ (n\ge1).\tag{16}
\]

Here YY is independent of y because all its arguments move together. Fixed-r assumptions matter: r-derivatives must also differentiate the pins, whitening, and reference data, and are not supplied by (16).

`conditional_interval_jets` implements (14)–(15) with exact rational interval arithmetic; `whiten_interval_jets` also accepts direct actual covariance and mean jets. The interval inverse uses Gauss–Jordan elimination with every pivot interval excluding zero. A separate interval LDL test certifies D>0 for a symmetric exact matrix enclosed by a symmetric interval array. An inconclusive pivot or LDL test fails rather than asserting positivity.

Natural interval arithmetic encloses the exact operations for every point in the common domain, including dependencies it overestimates. Pointwise algebraic elimination identities permit setting an eliminated entry to exactly zero and a normalized pivot to one. No roundoff allowance is needed for the rational operations. Operator-norm bounds use the minimum of the Frobenius bound and √(‖A‖₁‖A‖∞), with exact upward rational square-root rounding. Numerical `mpmath` functions are explicitly **point evaluation only**, not certified enclosures.

## 6. What is established, and what remains

Established mathematical content: the exact domain and reduction (1), all ordered inverse/logq derivatives through four, complete χ² and √χ² composition, a proved norm-envelope map from actual uniform input jets, and an executable interval Schur-jet construction. Validation checks noncommuting matrices against independent scalar symbolic differentiation, one-dimensional Gaussian integrals against the probability definition, actual polynomial-law interval input jets, exact rational interval conditional propagation, and rejection cases.

Still required for the research theorem:

1. Valid all-cell enclosures of F,D,z through order four for the exact SIDE24 covariance, including all periodization tails and fixed-data conditioning/whitening errors. Existing high-precision probes and finite image kernels are numerical evidence until these errors are enclosed.
2. Domain-wide D and covariance positivity, or a sharper local spectral criterion when the sufficient a₀<1 norm check is inconclusive.
3. A proved positive χ² floor for the fourth-order square-root route, or a separately justified route through its zero set.
4. The complete κ_far fourth-order composition, including Wick, Bures/square-root matrices, cross terms, conditional normal probabilities, denominators, and their possible zero sets.
5. Accepted-cell coverage, exact source linkage, certified Piece-2 integration, canonical mutations and freeze discipline, and the separate all-small-r obligations.

A successful conditional enclosure on a polynomial fixture validates the machinery and constitutes a proof for that fixture. It is not a certificate for the research field. No scientific gate or historical disposition changes here.

## 7. Source attribution

- Shared reconnaissance source: Rasmussen and Williams, *Gaussian Processes for Machine Learning*, chapter 2, standard Gaussian density/conditioning formulas: https://gaussianprocess.org/gpml/chapters/RW2.pdf .
- Shared reconnaissance source: Petersen and Pedersen, *The Matrix Cookbook* (2012), §§2.1–2.2, standard inverse and log-determinant differentiation. This is a standard-identity reference, not evidence for novelty or for this field's uniform bounds.
- Exact supplied implementation: `intake/rnu_chi2_white_v2.py`; base engine `intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py`, SHA-256 `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`.
- Input-contract scope and unresolved obligations: `intake/RN_UNIF_INTAKE_CONTRACT.md`; original proxy status: `intake/MATH_PUSH_WHITENED_ENV_FORM_2_4.md`.
