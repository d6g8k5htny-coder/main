# A square-root-safe modulus for the RN Gaussian comparison

Status: author-side mathematical lemma and reusable bound, under DQ-MATH-20260917-b9c2. This is a local analytic advance; RN-UNIF, its full spatial cover, and the parent theorem remain open. All arguments below are supplied explicitly. No novelty or organizational independence is claimed.

Reconnaissance: shared memo `MATH-20260917-b9c2_EXTERNAL_RECON.md`, SHA256 `29a117b8702e590e0a34005497bda19409c167c31d7d8ec291cb1b4452badb9a`, verified raw Drive carrier `1HMwqF8Z0k8JOAjg7MPpvuflOm4jCr-cz`. Standard Gaussian completion of squares is consistent with Rasmussen–Williams, *Gaussian Processes for Machine Learning*, Appendix A, https://gaussianprocess.org/gpml/chapters/RWA.pdf. The following modulus is derived here for the actual RN functional; source attribution does not discharge field-specific bounds.

## 1. Setting and the obstruction

Let q be the density of N(0,I_n), and p_t the density of N(m(t),S(t)), where m and the symmetric matrix S are C1 on a compact parameter interval and 0<S(t)<2I. Write

\[
Q(t)=\int p_t(x)^2/q(x)\,dx,\quad
g(t)=\sqrt{Q(t)-1}=\|p_t/q-1\|_{L^2(q)}.
\]

The square root is not generally differentiable at zero. For S=I and m(t)=t e_1, Q=e^{t^2}, so g(t)=|t|+O(|t|^3). A blanket fourth derivative bound for g is therefore invalid without additional hypotheses. A modulus based on the Hilbert norm is available at these points.

## 2. Exact speed identity

Set A=I-S, H=(I+A)^{-1}=(2I-S)^{-1}, T=(I-A^2)^{-1}, D=S', and v=m'. Define

\[
\ell=-\tfrac12\log\det(I-A^2)+m^THm,
\]
\[
\mu_*=-\tfrac12\operatorname{tr}(DTA)
+\tfrac12 m^THDHm+v^THm,
\]
\[
V_*=\tfrac12\operatorname{tr}(DTDT)
+(DHm+v)^TT(DHm+v).
\]

Then Q=e^ell and

\[
\left\|\frac{\partial_t p_t}{q}\right\|_{L^2(q)}^2
=Q(\mu_*^2+V_*).
\tag{1}
\]

Consequently, for any s,t in the interval,

\[
|g(t)-g(s)|\le\int_{\min(s,t)}^{\max(s,t)}
\sqrt{Q(u)(\mu_*(u)^2+V_*(u))}\,du.
\tag{2}
\]

**Proof.** Completing squares gives Q=det(S(2I-S))^{-1/2} exp(m^T(2I-S)^{-1}m). Under the normalized tilted density p_t^2/(qQ), the random vector z=x-m has covariance V=SH and mean d=SHm. The score is

\[
s_t(x)=\partial_t\log p_t(x)
=z^TBz+l^Tz+c,
\quad B=\tfrac12S^{-1}DS^{-1},\quad l=S^{-1}v,
\quad c=-\tfrac12\operatorname{tr}(S^{-1}D).
\]

For a Gaussian with mean d and covariance V, expanding z=d+epsilon and using the vanishing of odd centered moments gives

\[
\mathbb E s_t=\operatorname{tr}(BV)+d^TBd+l^Td+c,
\]
\[
\operatorname{Var}(s_t)=2\operatorname{tr}(BVBV)
+(2Bd+l)^TV(2Bd+l).
\]

Only S,H,A,T commute with one another; D need not commute with them. Substituting V,d and cyclically moving trace factors gives exactly mu_* and V_* above. Since partial_t p_t=p_t s_t, (1) follows. On the compact parameter interval the strict spectral inequalities supply uniform positive gaps at 0 and 2; completing squares bounds both the densities and their derivatives in L2(q) by integrable Gaussian-polynomial envelopes. Hence t maps to p_t/q as a C1 Hilbert-space curve. Its fundamental theorem of calculus, followed by the reverse triangle inequality for its norm after subtracting 1, proves (2), including g=0.

## 3. A computable uniform scalar majorant

Suppose throughout a path

\[
\|A\|_2\le a<1,\quad \|m\|_2\le M,
\quad\|D\|_F\le D_0,\quad\|v\|_2\le V_0,
\]

with all four bounds nonnegative. Put h=1-a, delta=1-a^2, and

\[
Q_0=\delta^{-n/2}\exp(M^2/h),
\]
\[
E_0=\frac{\sqrt n\,aD_0}{2\delta}
+\frac{D_0M^2}{2h^2}+\frac{V_0M}{h},
\]
\[
V_1=\frac{D_0^2}{2\delta^2}
+\frac{(D_0M/h+V_0)^2}{\delta}.
\]

Then

\[
|g(t)-g(s)|\le L|t-s|,
\qquad L=\sqrt{Q_0(E_0^2+V_1)}.
\tag{3}
\]

**Proof of the estimates.** The spectral bounds give ||H||<=1/h, ||T||<=1/delta and ||TA||_F<=sqrt(n)a/delta. Bound the three terms in mu_* by trace Cauchy–Schwarz and operator norms. Also tr(DTDT)=||T^(1/2)DT^(1/2)||_F^2<=D_0^2/delta^2. Bound the final quadratic form by ||T||(D_0||H||M+V_0)^2. Finally det(I-A^2)>=delta^n and m^THm<=M^2/h. Apply (2).

Unlike a formula containing 1/sqrt(chi2), (3) remains finite at A=m=0. At a=M=0 it becomes sqrt(D_0^2/2+V_0^2), agreeing with the exact infinitesimal score norm.

## 4. Spatial cell and RN adapter

On a convex spatial cell, suppose the same bounds hold in every unit direction, with D_0 bounding the Frobenius norm of the directional covariance derivative and V_0 bounding the directional mean derivative. Apply (3) along the straight segment from the cell center to any point. A covering radius rho therefore gives g(y)<=g(center)+L rho. For nonconvex polar cells, establish the bounds on a containing convex region, or integrate along a path inside the verified domain and use its length. Endpoint bounds alone do not establish the hypotheses.

The inspected RN source has kappa_pair=(hL2/Z_LO)g, with hL2 and the certified positive Z_LO fixed as y varies at its fixed r and mark values. Multiply L by this fixed nonnegative factor. If r, marks, or the factor vary, their derivatives must also be included; this statement does not supply uniformity in r.

For full assembly F=P W[(1+u)(1+v)+c]-1, where P,W,u,v,c are nonnegative and have uniform bounds P0,W0,U0,V0,C0 and Lipschitz constants LP,LW,LU,LV,LC on the same domain, the product rule gives

\[
L_F\le (L_PW_0+P_0L_W)((1+U_0)(1+V_0)+C_0)
+P_0W_0[L_U(1+V_0)+(1+U_0)L_V+L_C].
\]

The pair modulus supplies LU only. The actual RN ratios, y-block Bures term, cross term and all required uniform bounds still need certification. Thus this is a mathematically valid alternative to differentiating sqrt(chi2) four times, but no claim is made that the resulting cell count or numerical margin is adequate.

## 5. Implementation boundary

`sqrt_chi2_modulus.py` evaluates the scalar formula using python-flint Arb ball arithmetic. The upper endpoint of the output ball bounds the formula, conditional on validated uniform input bounds. Its matrix evaluator uses numpy only for diagnostics. No random samples or quadrature comparisons certify a field domain. The accompanying tests check algebraic special cases, a noncommuting case against independent Gaussian quadrature, cusp behavior, the scalar majorant, and rejection of invalid domains.

Remaining: populate certified cell-wide A,m,S',m' bounds from the exact conditioned periodized field; certify all other assembly factors; run a complete cover with the repaired leaf aggregation; perform the separately required mutation/FREEZE and external review gates. No parent obligation is relabeled CLOSED by this note.
