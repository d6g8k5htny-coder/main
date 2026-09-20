# LPW-CAND-20260912 — A local-path candidate proof of an existential cubic pairing-defect lower bound

**Version:** 1.0, 2026-09-12. **Authoring line:** ChatGPT / OpenAI, in response to Dylan Roy's authorization to proceed with the next research phase.

**Disposition:** COMPLETE AUTHOR-SIDE ANALYTIC CANDIDATE; EXACT ALGEBRA TESTED; INDEPENDENT REVIEW OPEN; NO PROGRAM PROMOTION. This is a new route, not a repaired or promoted K3-THM-001. Neither novelty nor human peer-review acceptance is claimed. The executable companion checks algebra and finite regression cases, not the full analysis below.

## 0. What changes in the proof strategy

The earlier lower construction counts candidate third saddles, conditions again, estimates their reliability, and subtracts a second factorial moment. Its surviving abstract architecture is useful, but its concrete weighted-Palm inputs remain open (GP-LB-STAT-004, Sections 3, 7, 10).

This candidate instead constructs a positive-probability **local field event directly inside the original six-pin law**. On that event, a fixed continuous path joins the pinned maximum to a higher point while staying above the pinned saddle level. This alone prevents the maximum from dying at the pinned saddle. No third-point count is introduced.

The proposed power ledger is:

    one shrinking transverse-jet interval:       r
    pair determinant weight on that event:      r^4
    upper bound for the Palm normalizer:         r^2
    resulting probability lower bound:          r^(1+4-2) = r^3.

The route does not claim any of the old WP, H-B3, exit, Bonferroni, or sharp-coefficient obligations have been proved. It does not consume them. Its conclusion is only existence of some positive c and r0, not the advertised 0.9144-class constant.

## 1. Exact object and proposed theorem

Let f be the centered, unit-variance, normalized periodized Bargmann–Fock Gaussian field on the flat two-torus of side 24. Its exact covariance is

\[
K_{24}(z)=\frac{\sum_{n\in\mathbb Z^2}\exp(-|z+24n|^2/2)}{\sum_{n\in\mathbb Z^2}\exp(-|24n|^2/2)}.
\]

Equivalently, every dual-lattice frequency k in (pi/12) Z^2 has a strictly positive weight proportional to exp(-|k|^2/2). No planar-kernel substitution or assertion of exact planar Gaussian spectral moments is used.

Fix b=6/5, r>0, M_r=(-r/2,0), S_r=(r/2,0). Let Q_r be the **continuous Gaussian regression version** of the conditional law with pins

\[
f(M_r)=b,\quad f(S_r)=b-r^3/6,\quad \nabla f(M_r)=\nabla f(S_r)=0.
\]

The choice of regression version matters: these pins are probability-zero events, so an arbitrary almost-everywhere version of a conditional probability is not a suitable definition at a prescribed pin value.

Set

\[
W_r=|\det H_f(M_r)\det H_f(S_r)|
\mathbf 1\{H_f(M_r)\prec0,\ \det H_f(S_r)<0\},
\qquad Z_r=\mathbb E_{Q_r}W_r,
\]

and define P_r by dP_r=W_r dQ_r/Z_r. These are the same six-pin and determinant-weight conventions used in GP-DER-118-v1.10, Section 0, and the typed merge-tree interpretation specified by W10, Section 1. There is **no additional conditioning on gradient adjacency**.

Let q(r,b)=P_r{D_f(M_r)=S_r}, where D_f is the superlevel elder-rule death partner under the program's usual well-defined persistence interpretation. The only fact about that interpretation used below is: a maximum at height b cannot die at level s if its component is already connected to a point of height greater than b at a level strictly larger than s. Equivalently, one may formulate the conclusion directly as a lower bound for the path-defined preemption event, avoiding dependence on any tie-breaking convention.

**Candidate theorem LPW.** There exist c>0 and r0>0 such that

\[
1-q(r,6/5)\ge c r^3\qquad (0<r\le r_0).
\]

The proof supplies positive analytic expressions for c and r0 in terms of compact Gaussian density and conditional-moment bounds. It does **not** provide evaluated numerical constants. Its scope is fixed b=6/5, fixed side 24, dimension two, and the exact typed pair-Palm law above.

## 2. Deterministic topological criterion

Let a continuous function on the torus have a strict local maximum M of height b and a candidate saddle S of height s<b. Suppose a continuous path gamma starts at M, ends at a point z with f(z)>b, and satisfies

\[
\min_t f(\gamma(t))>s.
\]

Choose a level h strictly between s and that minimum. M and z are in the same superlevel component at h. That component contains a point above b, and hence an older birth than the maximum M under the superlevel elder rule. Thus M's component has already lost its separate surviving identity by level h; it cannot first die at S at level s.

The conclusion is D_f(M) != S, **not** identification of a particular third saddle as the death partner. M might have died even earlier. The argument also covers a path which joins a component already containing an older maximum. No self-attachment exclusion or assertion that a specified third saddle performs the first merge is needed.

This criterion is pathwise. It needs neither gradient trajectories nor a Morse–Smale hypothesis. It does not establish any claim about gradient adjacency of M and S. When using the program's q symbol, its merge-tree interpretation must be retained; no adjacent-conditioned q is substituted.

## 3. An explicit rational cubic witness

In rescaled coordinates define

\[
F_*(x,y)=\frac{x^3}{3}-\frac{x}{4}-\frac1{12}
+2\left(x^2-\frac14\right)y-5y^2.
\]

At m=(-1/2,0) and s=(1/2,0), direct differentiation gives

\[
F_*(m)=0,\quad F_*(s)=-\frac16,
\qquad \nabla F_*(m)=\nabla F_*(s)=0,
\]

\[
H_*(m)=\begin{pmatrix}-1&-2\\-2&-10\end{pmatrix},
\quad \det H_*(m)=6,
\qquad
H_*(s)=\begin{pmatrix}1&2\\2&-10\end{pmatrix},
\quad \det H_*(s)=-14.
\]

The first is negative definite and the second is a saddle matrix.

Consider the path, parameterized with x decreasing from -1/2 to -2,

\[
\gamma_*(x)=\left(x,\frac{x^2-1/4}{5}\right).
\]

It lies in D=[-2,1/2] x [-1,1], starts at m, and ends at (-2,3/4). Its height is

\[
R(x)=F_*(\gamma_*(x))
=\frac{(2x+1)^2(12x^2+8x-17)}{240}.
\]

The exact identity

\[
R(x)+\frac{99}{1280}
=\frac{(4x+5)^2(48x^2-40x+1)}{3840}
\]

shows R(x)>=-99/1280 on this interval: x is negative, so the second factor is strictly positive. Equality occurs at x=-5/4. Also R(-2)=9/16. Consequently

\[
\min R=-\frac{99}{1280}>-\frac16,
\qquad R(-2)=\frac9{16}>0.
\]

The saddle at the ridge minimum is not needed as a counted random object. The fixed path and its strict level clearance are sufficient.

## 4. A robust C2 neighborhood

Use the norm

\[
\|G\|_{C^2(D)}=\max_{|\alpha|\le2}\sup_{D}|\partial^\alpha G|.
\]

Suppose G has the exact pinned values and gradients at m and s, and

\[
\|G-F_*\|_{C^2(D)}\le\varepsilon,\qquad \varepsilon=1/16.
\]

Along the same path,

\[
\min G(\gamma_*)\ge-99/1280-1/16=-179/1280>-1/6,
\]

with remaining clearance

\[
\frac16-\frac{99}{1280}-\frac1{16}=\frac{103}{3840}>0.
\]

At the endpoint, G(-2,3/4)>=1/2>0.

For the maximum Hessian, entrywise error at most epsilon gives a negative first diagonal entry and

\[
\det H_G(m)\ge(1-\varepsilon)(10-\varepsilon)-(2+\varepsilon)^2
=81/16>5.
\]

For the saddle Hessian, the first diagonal entry is positive, the second negative, and

\[
\det H_G(s)\le-(1-\varepsilon)(10-\varepsilon)-(2-\varepsilon)^2
=-1673/128<-13.
\]

Thus every such G supplies the topological defect witness and

\[
|\det H_G(m)\det H_G(s)|\ge65.
\]

These are interval-safe entrywise bounds, not eigenvalues inferred from a numerical sample.

## 5. Six-pin Hermite normalization

Put the raw pin vector in the order

    P6_r=(f(M_r), f_x(M_r), f_y(M_r), f(S_r), f_x(S_r), f_y(S_r)).

Let V be the 6x6 matrix of value, x-derivative, and y-derivative evaluations at m and s on the polynomial basis

    (1, x, y, x^2, xy, x^3).

Its determinant is exactly -1. Set

\[
D_r=\operatorname{diag}(1,r,r,r^2,r^2,r^3),\quad
R_r=\operatorname{diag}(1,r,r,1,r,r),\quad
T_r=D_r^{-1}V^{-1}R_r.
\]

The explicit matrix (columns follow P6_r) is

    [ 1/2,      r/8,       0,     1/2,     -r/8,      0 ]
    [-3/(2r),  -1/4,       0,  3/(2r),    -1/4,      0 ]
    [   0,        0,     1/2,       0,        0,    1/2 ]
    [   0,  -1/(2r),       0,       0,   1/(2r),      0 ]
    [   0,        0,    -1/r,       0,        0,    1/r ]
    [2/r^3,   1/r^2,       0,  -2/r^3,    1/r^2,      0 ].

Then det T_r=-r^{-5}; T_r is invertible for every r>0. The transformed pin vector U_r=T_r P6_r therefore defines the same Gaussian conditioning, not another law.

For the prescribed values,

\[
u_r=T_r(b,0,0,b-r^3/6,0,0)^T
=(b-r^3/12,-r^2/4,0,0,0,1/3)^T.
\]

Here the displayed symbol u_r is a six-dimensional observed vector, not the persistence density nu. As r tends to zero, U_r converges in Gaussian L2 to

\[
U_0=(f(0),f_x(0),f_y(0),f_{xx}(0)/2,f_{xy}(0),f_{xxx}(0)/6).
\]

To see the y-containing coordinates, the y-derivative data interpolate f_y along the x-axis by its constant and linear terms. Higher x powers contribute errors tending to zero. For the remaining four coordinates the cubic Hermite interpolation is exact through degree three. Taylor remainder estimates prove the stated convergence, also after taking covariance against any fixed derivative evaluation of f.

In particular, u_r stays in a compact set. The rare pin event does not produce an unbounded observed vector in this normalized frame.

## 6. Nondegenerate conditional jet density

Define the remaining four jet coordinates

\[
J=(q,A,B,D_3)
=\left(f_{yy}(0),\frac12 f_{xxy}(0),\frac12 f_{xyy}(0),\frac16 f_{yyy}(0)\right).
\]

The pair (U_0,J) is an invertible rescaling/reordering of the complete ten-dimensional third-order jet at the origin.

That jet has positive-definite covariance for the exact periodized field. Indeed, the variance of a nonzero linear combination of its derivatives is a positive weighted lattice sum of |P(ik)|^2 for a nonzero polynomial P of degree at most three. A nonzero polynomial cannot vanish on all of Z^2: fix one coordinate, use infinitely many zeros in the other, and repeat on the coefficient polynomials. Every spectral weight is positive. The variance is therefore strictly positive.

It follows by continuity that the covariance of (U_r,J) has a uniform positive eigenvalue lower bound on some [0,r_G]. Its Schur complement gives a uniformly positive-definite conditional covariance for J given U_r=u_r; the conditional means and covariance matrices vary continuously down to r=0.

Fix delta=1/1024. The density of J under Q_r therefore has a positive lower bound m on the fixed compact box

\[
K_J=[-11,1]\times[2-\delta,2+\delta]\times[-\delta,\delta]^2
\]

for 0<=r<=min(r_G,1). This follows directly from the Gaussian density formula: its determinants are bounded above and below, its inverse covariance is bounded, and its means and arguments are bounded. No sampled finite-difference or H-B3 assumption is involved.

Define the thin event E_r by

\[
\left|\frac{q}{2r}+5\right|\le\delta,
\quad |A-2|\le\delta,
\quad |B|\le\delta,
\quad |D_3|\le\delta.
\]

For r<=1 it is contained in K_J. In J coordinates its volume is

\[
(4\delta r)(2\delta)^3=32\delta^4r,
\]

so

\[
Q_r(E_r)\ge32m\delta^4r.
\]

Only the transverse second derivative is restricted to a width of order r. The three free cubic coefficients remain in fixed-width intervals. The r-dependent change q/(2r) must not be mistaken for a fixed-density coordinate.

## 7. Uniform conditional remainder control, inside the thin event

Let

\[
M_4(f)=\max_{|\alpha|=4}\sup_{\mathbb T_{24}^2}|\partial^\alpha f|.
\]

We need a bound **conditional on J=j as well as the six pins**. An unconditional tail subtraction could exceed the O(r) mass of E_r and would invalidate the argument.

The field's Gaussian Fourier coefficients have exponentially decaying standard deviations. For every finite p and derivative order k, Minkowski's inequality applied to the absolutely summable Fourier majorant gives

\[
\mathbb E\|f\|_{C^k}^p<\infty.
\]

For completeness, choose independent real standard Gaussian sine/cosine coefficients. A C^k norm is bounded by a constant times

\[
\sum_{n\in\mathbb Z^2}\sqrt{a_n}(1+|n|)^k |\xi_n|,
\]

and the sum of Lp norms is finite because a_n decays Gaussianly. This gives both almost-sure smoothness and all finite moments of the required norm.

Let V_r=(U_r,J), with covariance Gamma_r. Conditional on V_r=v, the entire field has the regression representation

\[
f_{r,v}=f-C_r\Gamma_r^{-1}V_r+C_r\Gamma_r^{-1}v,
\qquad C_r(z)=\operatorname{Cov}(f(z),V_r).
\]

The residual term in this expression is independent of V_r. On [0,r_G], Gamma_r^{-1} is bounded. The C^4 norms of the deterministic representers C_r are bounded. In fact, no C^4 convergence of the representers is required: covariance Cauchy–Schwarz gives, for every |alpha|<=4 and coordinate j,

\[
\sup_z|\partial^\alpha C_{r,j}(z)|
\le\sup_z\big(\mathbb E|\partial^\alpha f(z)|^2\big)^{1/2}
\big(\mathbb E|V_{r,j}|^2\big)^{1/2}.
\]

The first factor is finite and uniform on the torus; the second is bounded by the covariance convergence already established. Differentiation under covariance is justified by the smooth field and its derivative moment bounds. The Gaussian vector V_r has bounded moments. For v=(u_r,j), j in K_J, v is bounded.

Taking C^4 norms in the regression formula therefore yields a finite uniform number

\[
B_4=\sup_{0<r\le\min(r_G,1),\ j\in K_J}
\mathbb E[M_4(f)\mid U_r=u_r,J=j]<\infty.
\]

Use this continuous Gaussian regression version for every j, not an arbitrarily redefined conditional version. Choose K=max(1,2B_4). Markov's inequality gives

\[
Q_r(M_4\le K\mid J=j)\ge1/2
\]

uniformly for j in K_J. Integrating **over the thin jet box** gives

\[
Q_r(E_r\cap\{M_4\le K\})\ge16m\delta^4r.
\]

No independence between E_r and the remainder event has been asserted.

## 8. Deterministic Taylor lifting of the witness

For a field satisfying the six pins, define

\[
F_r(x,y)=\frac{f(rx,ry)-b}{r^3}.
\]

For r<=1 the rectangle rD lies inside one torus coordinate chart. Under M_4<=K, Taylor expansion and the two endpoint gradient equations give

\[
|f_{xx}(0)|,\ |f_{xy}(0)|\le Kr^2/24,
\]

\[
\left|\frac{f_x(0)}{r^2}+\frac{f_{xxx}(0)}8\right|\le Kr/48,
\quad
\left|\frac{f_y(0)}{r^2}+\frac{f_{xxy}(0)}8\right|\le Kr/48.
\]

The pinned value difference gives

\[
|f_{xxx}(0)-2|\le5Kr/16,
\quad
\left|\frac{f(0)-b}{r^3}+\frac1{12}\right|\le Kr/128,
\quad
\left|\frac{f_x(0)}{r^2}+\frac14\right|\le23Kr/384.
\]

Derivation of the first cubic bound: the value difference is

    -r^3/6 = r f_x(0) + r^3 f_xxx(0)/24 + e_v,
    |e_v| <= Kr^4/192;

whereas the average x-gradient equation gives

    f_x(0) = -r^2 f_xxx(0)/8 + e_x,
    |e_x| <= Kr^3/48.

Substitution yields the coefficient 5/16 above. The midpoint value bound uses the average of the two value equations and |f_xx(0)|<=Kr^2/24.

Set C=q/(2r). The cubic determined by the four free coefficients is

\[
F_J(x,y)=g(x)+A(x^2-1/4)y+Bxy^2+Cy^2+D_3y^3,
\quad g(x)=x^3/3-x/4-1/12.
\]

On D, |x|+|y|<=3. The rescaled multivariate Taylor remainder and its derivatives through order two are bounded by (9/2)Kr: for |alpha|<=2, use

\[
Kr\,\frac{(|x|+|y|)^{4-|\alpha|}}{(4-|\alpha|)!}.
\]

The low-order coefficient discrepancies multiply C2 monomial bounds 1,2,1,4,2,12 for 1,x,y,x^2,xy,x^3. The resulting total, including the Taylor remainder, is

\[
\|F_r-F_J\|_{C^2(D)}\le\frac{2089}{384}Kr<8Kr.
\]

The four basis functions (x^2-1/4)y, xy^2, y^2, y^3 have C2 norms at most 4,4,2,6 on D. Therefore E_r implies

\[
\|F_J-F_*\|_{C^2(D)}\le16\delta.
\]

Choose

\[
r_0=\min\{r_G,1,(256K)^{-1}\}.
\]

Then for 0<r<=r0, on E_r intersect {M_4<=K},

\[
\|F_r-F_*\|_{C^2(D)}\le16/1024+8/256=3/64<1/16.
\]

Section 4 applies. The physical path r gamma_* joins M_r to a point whose field value exceeds b, and stays strictly above b-r^3/6. Thus this event forces D_f(M_r)!=S_r.

The Hessian scaling is exact:

\[
H_f(rz)=r H_{F_r}(z).
\]

In dimension two each determinant scales by r^2, so the pair weight on the event satisfies

\[
W_r\ge65r^4.
\]

## 9. An independent O(r2) upper bound for the normalizer

This proof needs only Z_r<=C_Z r^2, not the sharper asymptotic constant or a numerical value 15 from a previous package.

Let M_3=max(1,||f||_{C^3}). Along the x-axis between the pins, the x-gradient has equal endpoint values zero. Consequently the average of f_xx on the interval is zero. Its variation is at most r M_3, so

\[
|f_{xx}(M_r)|,|f_{xx}(S_r)|\le rM_3.
\]

The same argument applied to f_y, also zero at both endpoints, gives

\[
|f_{xy}(M_r)|,|f_{xy}(S_r)|\le rM_3.
\]

The remaining Hessian entry is at most M_3. For r<=1,

\[
|\det H_f(M_r)|,|\det H_f(S_r)|\le rM_3^2+r^2M_3^2\le2rM_3^2.
\]

Therefore W_r<=4r^2 M_3^4. The six-pin version of the uniform Gaussian regression argument in Section 7 gives

\[
B_3=\sup_{0<r\le\min(r_G,1)}\mathbb E_{Q_r}M_3^4<\infty,
\qquad Z_r\le C_Zr^2,\quad C_Z=4B_3.
\]

Positivity of Z_r for these small r follows already from the strictly positive weighted event constructed above. Thus the Palm quotient is finite and well-defined on the whole chosen interval.

## 10. Final composition

Write G_r=E_r intersect {M_4<=K}. Sections 2–8 give G_r subset {D_f(M_r)!=S_r} and W_r>=65r^4 on G_r. Hence

\[
\begin{aligned}
1-q(r,b)
&\ge \frac{\mathbb E_{Q_r}[W_r\mathbf1_{G_r}]}{Z_r}\\
&\ge \frac{65r^4\,16m\delta^4r}{C_Zr^2}\\
&=\frac{1040m\delta^4}{C_Z}r^3.
\end{aligned}
\]

All constants were fixed before r was allowed to vary on (0,r0]. In particular m>0, K<infinity, C_Z<infinity, and r_G>0. Taking c=1040m delta^4/C_Z proves the stated conclusion **within this author-side candidate derivation**.

This is an existential all-small-r argument, not an interpolation between finite rungs. It does not identify a limiting coefficient, an optimal lower bound, a useful numerical radius, or a two-sided law. It does not transfer the result to dimension three or to an adjacent-conditioned pairing probability.

## 11. Why the old bottlenecks are not premises of this route

**WP:** no random third-saddle reliability calculation is made.

**Lambda/H-B3:** no spatial third-point intensity is integrated. The rare-jet event has an ordinary four-dimensional Gaussian density under the six-pin law.

**Triple-Palm reweighting:** only the original pair weight is used, explicitly.

**Bonferroni:** no count is converted into an occurrence probability. We lower-bound one field event directly, so there is no second factorial moment to subtract.

**Far-field terminal floor:** the path endpoint has f>b within an O(r) neighborhood. The existence of an older connected birth follows without following a separatrix to the far field.

**Gamma-LOC and exit:** no trajectory is integrated or required to remain in a tube.

**eta_r:** the event directly implies failure of typed elder pairing. It is not obtained from failure of gradient adjacency. Any later adjacent-law restatement must still carry the exact eta_r identity; this proof neither estimates nor sets eta_r to zero.

**SARD-G:** the deterministic witness is a superlevel path. No Morse–Smale connection theorem is used to construct it. Compatibility with the precise canonical persistence interpretation is still an explicit review item; no separate global genericity theorem is silently marked closed.

The old obligations remain open for the old quantitative/limiting-coefficient route. A successful independent review of this candidate would support a different, narrower closure request.

## 12. Adversarial review boundary and falsifiers

The most important scrutiny targets are not the polynomial identities; those are directly checked. They are:

1. **Exact estimand:** confirm that the program's q is the typed six-pin merge-tree probability and carries no extra adjacency or shape conditioning.
2. **Conditioning version:** verify the continuous Gaussian regression version and the limiting six-/ten-coordinate covariance arguments at the prescribed, measure-zero pin values.
3. **Full jet rank:** verify positive spectral support for the exact normalized periodic field, not a finite truncation or a PDE-constrained monochromatic model.
4. **Uniform conditional C4 moment:** check the regression-representer C4 bounds and the conditional-in-J tail step. A bound only before conditioning on J is insufficient.
5. **Taylor lift:** verify every coefficient and the rescaled derivative remainder on the entire fixed rectangle.
6. **Topology:** try to construct a valid elder-rule interpretation in which M remains paired with S despite a path to a point above b whose minimum is strictly above f(S).
7. **Palm power ledger:** confirm the width-r jet box, the r^4 pair weight on that box, and the O(r^2) denominator without swapping probability laws.
8. **Uniform quantifiers:** every constant must be independent of all r in (0,r0], rather than chosen separately at each r.

If a reviewer finds any failure, preserve the failed candidate and issue a numbered correction. Do not use the exact-algebra PASS as a substitute for these checks. No independent reviewer was run in the authoring session, and normal/-O reruns are the same code lineage.

## 13. Provenance and related-work boundary

Exact target source: GP-DER-118-v1.10, Drive ID 1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9, Section 0. Whole-file SHA-256 c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564; frozen body 9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014. Both identities were recomputed in this phase.

Latest prior lower disposition: GP-LB-STAT-004, Drive ID 1lfH7g57LcshqpLfNbrx-H7gbJckpzPqr, Sections 3, 4, 7, 10. It remains unchanged.

Alternative-architecture source: W10_REPORT.md, raw Drive ID 1Ls7kwptRf7XGjmy2y4fZV3lbCtznzsK7, Section 1 and Sections 2–6. Raw SHA-256 67c7fe63e168060b9db1f2279de5d88e7f804508a0e4e98a90afbec468ee21d1, freshly recomputed. The present route does not adopt W10's representative-height integrand as an exact integral or its numerical grades.

Limited external screening located work on near-diagonal critical-point correlations and on Gaussian Palm couplings. It did not constitute a theorem-by-theorem novelty audit. See RELATED_WORK_SCREENING.md. No priority or originality claim is licensed by this package.

**Final author-side disposition:** the deterministic witness, conditional-Gaussian local-mass argument, and Palm normalization compose into a complete candidate proof at the stated existential scope. The program's accepted lower-theorem status stays OPEN until independent mathematical review and the applicable control action.
