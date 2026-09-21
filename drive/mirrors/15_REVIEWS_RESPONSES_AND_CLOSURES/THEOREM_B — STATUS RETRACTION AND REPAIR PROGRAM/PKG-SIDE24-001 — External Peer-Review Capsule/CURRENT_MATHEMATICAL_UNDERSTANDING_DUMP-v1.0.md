# Current Mathematical Understanding and Findings

**Author:** Dylan Roy  
**Snapshot:** 2026-07-30, America/Chicago  
**Status:** technical research dump; not a canonical promotion, release authorization, or substitute for the frozen sources

## 0. Executive state

The current mathematical record contains one especially strong, coherent result:

> For the normalized side-\(24\) periodized three-dimensional Bargmann–Fock field, the first-moment density of finite nonessential superlevel \(H_0\) persistence lifetimes has a short-lifetime singularity
> \[
> \nu_{3,24}(\ell)
> =c_{3,24}\ell^{-1/3}(1+o(1)),
> \qquad \ell\downarrow0,
> \]
> with \(0<c_{3,24}<\infty\), an explicit finite-dimensional Gaussian-cone formula for \(c_{3,24}\), and a rigorously bounded side-\(24\) correction to the planar coefficient.

This fixed-\(d=3\), fixed-\(L=24\) theorem is mathematically closed at the reviewed finite-side scope recorded by `LS-CLS-077-v1.0`. Its pair-counting convention, selection normalization, Kac–Rice factor, lifetime Jacobian, planar coefficient, and analytic side-\(24\) correction have all passed the project’s terminal review chain.

This does **not** restore the broader historical object called “Theorem B.” That older global/full-scope route remains `RETRACTED / NOT RESTORED`. It also does not establish:

- the theorem for arbitrary torus side \(L\);
- a theorem for arbitrary stationary or isotropic Gaussian fields;
- higher persistent homology;
- an infinite-volume persistence process;
- an interchange of \(L\to\infty\) and \(\ell\downarrow0\);
- a directed-rounding certificate for the full nonlinear quadrature.

The Drive contains a broader three-dimensional periodic-field-class theorem candidate, `LS-DER-070-v1.0`, and a separate two-dimensional compact-selection chain led by `LS-DER-023-v1.1` and `LS-DER-036-v1.1`. These are mathematically substantial but are not at the same review status as the fixed side-\(24\), \(d=3\) result.

## 1. Version and authority rules used for this dump

The current sources were rechecked for numbered successors before use.

- `LS-DER-073-v1.3` is the controlling successor to `v1.0`, `v1.1`, and `v1.2`.
- `LS-DER-023-v1.1` is the controlling source for the TB-G3A compact two-dimensional result.
- `LS-DER-036-v1.1` is the controlling source for the TB-G3B same-maximum loop result.
- `LS-CLS-077-v1.0`, `LS-DER-070-v1.0`, `LS-DER-075-v1.0`, and `LS-DER-076-v1.0` have no located `.1`–`.9` successors.
- `SIDE24_MANUSCRIPT_CORE.md` and `SIDE24_TECHNICAL_APPENDIX_EXACT.md` are unique current files under those names.

Administrative manifests and registers are used only to determine identity, status, and supersession. They do not create mathematical credit.

## 2. The fixed side-\(24\) field and persistence object

Let
\[
\mathbb T_{24}^3=(\mathbb R/24\mathbb Z)^3
\]
and define the normalized periodized Bargmann–Fock covariance
\[
K_{24}(z)
=Z_{24}^{-1}\sum_{n\in\mathbb Z^3}
\exp\!\left(-\frac{|z+24n|^2}{2}\right),
\qquad
Z_{24}=\sum_{n\in\mathbb Z^3}e^{-288|n|^2}.
\tag{2.1}
\]
Let \(f_{24}\) be a centered stationary Gaussian field with covariance \(K_{24}\).

For the superlevel filtration
\[
X_a=\{x\in\mathbb T_{24}^3:f_{24}(x)\ge a\},
\tag{2.2}
\]
each finite nonessential \(H_0\) bar has a creating local maximum \(M\) and an index-two merging death saddle \(S\). Its lifetime is
\[
\ell=f_{24}(M)-f_{24}(S)>0.
\tag{2.3}
\]
The quantity \(\nu_{3,24}(\ell)\,d\ell\) is the expected number per unit torus volume of such bars with lifetime in \(d\ell\).

The manuscript draft proves that \(f_{24}\) has real-analytic sample paths and is almost surely Morse with pairwise distinct critical values. The argument uses:

1. positivity and Gaussian decay of every Fourier coefficient;
2. nondegeneracy of finite derivative-evaluation distributions;
3. a Bulinskaya/Adler–Taylor-type Morse argument;
4. an overdetermined Gaussian zero-set argument for equality of critical values at distinct points.

The mathematical mechanism is plausible and explicit, but the exact external theorem numbering and hypotheses used in this genericity paragraph should still be checked during the final literature/citation pass.

## 3. Main fixed-side theorem

### Theorem 3.1 — Side-\(24\), three-dimensional short-lifetime law

For the field \(f_{24}\) with covariance (2.1),
\[
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0,
\tag{3.1}
\]
where \(0<c_{3,24}<\infty\).

Fix \(t\in S^2\) and an orthonormal transverse frame \(n_1,n_2\). The controlling technical source uses the odd contact variable
\[
u_t=-\frac{\partial_t^3 f_{24}}{12},
\tag{3.2}
\]
not \(\nu_t\). Define
\[
g_t=
(\partial_t f_{24},
 \partial_{n_1}f_{24},
 \partial_{n_2}f_{24}),
\tag{3.3}
\]
\[
h_t=
(\partial_{tt}f_{24},
 \partial_{tn_1}f_{24},
 \partial_{tn_2}f_{24}),
\tag{3.4}
\]
and
\[
q_t=
(\partial_{n_1n_1}f_{24},
 \partial_{n_1n_2}f_{24},
 \partial_{n_2n_2}f_{24}),
\qquad
Q_t(q)=
\begin{pmatrix}
q_1&q_2\\
q_2&q_3
\end{pmatrix}.
\tag{3.5}
\]
Put
\[
\Sigma_g(t)=\operatorname{Cov}(g_t),
\qquad
\Sigma_h(t)=\operatorname{Cov}(h_t),
\tag{3.6}
\]
\[
\sigma_t^2
=\operatorname{Var}(u_t)
-\operatorname{Cov}(u_t,g_t)
\Sigma_g(t)^{-1}
\operatorname{Cov}(g_t,u_t),
\tag{3.7}
\]
\[
\Gamma_t
=\operatorname{Cov}(q_t)
-\operatorname{Cov}(q_t,h_t)
\Sigma_h(t)^{-1}
\operatorname{Cov}(h_t,q_t),
\tag{3.8}
\]
and
\[
D_t
=\mathbb E\!\left[
(\det Q_t(q_t))^2
\mathbf 1_{\{Q_t(q_t)<0\}}
\mid h_t=0
\right].
\tag{3.9}
\]
Then
\[
c_{3,24}
=B_3\int_{S^2}
\frac{(\sigma_t^2)^{2/3}D_t}
{\sqrt{\det\Sigma_g(t)\det\Sigma_h(t)}}\,dt,
\qquad
B_3=
\frac{3\Gamma(7/6)}
{2^{4/3}\pi^{7/2}}.
\tag{3.10}
\]
This formula is independent of the transverse-frame choice.

If \(A_tA_t^\top=\Gamma_t\) and
\[
\mathcal Q_t(v)=Q_t(A_tv),
\]
then
\[
D_t
=\frac{15}{4\pi}
\int_{S^2}
[\det\mathcal Q_t(v)]^2
\mathbf 1_{\{
\operatorname{tr}\mathcal Q_t(v)<0,\,
\det\mathcal Q_t(v)>0
\}}\,dv.
\tag{3.11}
\]
Thus \(c_{3,24}\) is a deterministic nested angular integral depending only on the order-six covariance jet at the origin.

### Exact planar value and side-\(24\) correction

The planar coefficient is
\[
c_{3,\infty}
=2^{-23/6}3^{-8/3}
(29\sqrt6-36)
\Gamma(1/6)\pi^{-5/2},
\tag{3.12}
\]
numerically
\[
c_{3,\infty}
=0.0417759318405983433429366654285755564666815196\ldots.
\tag{3.13}
\]
The fixed-side correction satisfies
\[
\frac{c_{3,24}}{c_{3,\infty}}-1
=-\frac{620813376}{35}e^{-288}
+\varepsilon_{24},
\qquad
|\varepsilon_{24}|<10^{-180}.
\tag{3.14}
\]
Consequently,
\[
c_{3,24}<c_{3,\infty},
\]
and the relative discrepancy is of order \(10^{-118}\).

## 4. Why the exponent is \(-1/3\)

The exponent comes from an exact local fold power balance, not from a fitted numerical law.

For a typed maximum–saddle pair at separation \(r\), write
\[
M=x-\frac r2t,
\qquad
S=x+\frac r2t,
\qquad t\in S^2,
\tag{4.1}
\]
and impose
\[
f_{24}(M)=b,
\qquad
f_{24}(S)=b-\frac{\kappa r^3}{6},
\qquad
\nabla f_{24}(M)=\nabla f_{24}(S)=0.
\tag{4.2}
\]

The exact factors in dimension three are:

- displacement measure: \(r^2\,dr\,d\sigma_2(t)\);
- value-to-gap Jacobian: \(r^3\,d\kappa/6\);
- corrected six-pin density: \(r^{-6}\);
- two soft endpoint determinant factors: \(r^2\).

Therefore the contact-scale radial measure is
\[
\frac{r^3}{6}\,r^{-6}\,r^2\,r^2\,dr
=\frac16 r\,dr.
\tag{4.3}
\]
There is no additional unordered-pair, antipodal, unstable-branch, or merge multiplicity factor.

The lifetime relation is
\[
\ell=\frac{\kappa r^3}{6}.
\tag{4.4}
\]
For fixed \(\kappa>0\),
\[
r\,dr
=\frac{6^{2/3}}3
\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
\tag{4.5}
\]
Equations (4.3)–(4.5) force the exponent \(-1/3\). The remaining work is to prove that the selected-persistence mass converges to the all-typed contact mass and that the coefficient is finite and positive.

## 5. Pair counting and multiplicity

The local change of variables
\[
(x,u)\longmapsto(M,S)=(x-u/2,x+u/2)
\tag{5.1}
\]
has determinant of absolute value one, so
\[
dM\,dS=dx\,du
=dx\,r^2dr\,d\sigma_2(t).
\tag{5.2}
\]

The direction \(-t\) exchanges the typed maximum and saddle. It is not a second representation of the same typed ordered pair. Hence the full directed sphere \(S^2\) is correct and no factor \(1/2\) is introduced.

For a Morse function with distinct critical values, a finite nonessential superlevel \(H_0\) bar has one creating maximum and one merging death saddle. An index-two merge lowers \(\beta_0\) by one, so the elder pairing has unit multiplicity. Counting critical-point pairs rather than individual unstable branches creates no branch factor.

These conventions are not cosmetic: they control the absolute coefficient.

## 6. Elder-rule selection

Let \(p_r(t,b,\kappa)\) be the determinant-weighted Palm probability that the typed contact pair is the actual elder persistence pair. The fixed side-\(24\) chain establishes, uniformly on compact positive \(b,\kappa\) sets,
\[
1-p_r(t,b,\kappa)\le Cr^3.
\tag{6.1}
\]

The proof decomposes selection failure into local and remote obstructions.

### 6.1 Collar and singular-near third-critical-point charts

In collar variables \(y=x+r(Xt+\rho v)\), the scaled conditional gradient covariance has the structural factorization
\[
\det C_{\mathrm{col}}
=\rho^6(X^2+\rho^2)A_{\mathrm{col}},
\qquad
A_{\mathrm{col}}>0.
\tag{6.2}
\]
In the singular-near chart,
\[
\det C_{\mathrm{sing}}
=\rho^{10}(3c^2+\rho^2)A_{\mathrm{sing}},
\qquad
A_{\mathrm{sing}}>0.
\tag{6.3}
\]
The determinant numerator carries the anisotropic envelope
\[
[r(r+s^2)]^2(r^2+s^3).
\tag{6.4}
\]
The collision zeros in (6.2)–(6.4) compensate the apparent covariance singularities and yield Palm expectations of order \(O(r^3)\).

### 6.2 Fixed-distance witnesses

Complete positive Fourier support implies strict covariance positivity for every linearly independent finite family of derivative-evaluation distributions. On compact separated configuration sets this gives uniform covariance eigenvalue floors.

The endpoint determinant factor is \(r^2\), the remote critical value lies in a window of width \(O(r^3)\), and the Palm normalizer is comparable to \(r^2\). Consequently,
\[
\mathbb E_r^{MS}N_{\mathrm{far}}
\le
Cr^3
[1+(|b|+\kappa)^{10}]
e^{-c(b^2+\kappa^2)}.
\tag{6.5}
\]

### 6.3 Local branch geometry

The controlling transverse matrix is
\[
T_r=-\frac{Q_\perp}{\kappa r}.
\tag{6.6}
\]
The final repair uses \(T_r\)-energy-adapted cone and strip regions. It does not require an artificial upper bound on \(\lambda_{\max}(T_r)\). The inward branch is captured by the maximum, while the outward branch reaches a forward section at a level above the birth value. Shallow-eigenvalue, finite-jet, and fifth-derivative failures have Palm probability \(O(r^3)\) or \(o(r^3)\).

Together, the collar, singular-near, remote, and same-maximum self-attachment estimates prove (6.1).

## 7. All-mark domination and coefficient reduction

The corrected pair target is injective in \((b,\kappa)\). Gaussian regression and conditional polynomial-moment bounds give the integrable majorant
\[
C[1+(|b|+\kappa)^N]
e^{-c(b^2+\kappa^2)}
\kappa^{-2/3}
\tag{7.1}
\]
on
\[
S^2\times\mathbb R\times(0,\infty).
\]
Since \(0\le p_r\le1\), equation (6.1) and dominated convergence identify the selected and all-typed contact limits.

Even and odd covariance derivatives are independent because \(K_{24}\) is even. Birth-height integration removes the field-value coordinate:
\[
\int_{\mathbb R}
p_{(f,h_t)}(b,0)
\mathbb E[F(q_t)\mid f=b,h_t=0]\,db
=p_{h_t}(0)\mathbb E[F(q_t)\mid h_t=0],
\tag{7.2}
\]
where
\[
F(q)=(\det Q_t(q))^2\mathbf 1_{\{Q_t(q)<0\}}.
\]

Conditioned on \(g_t=0\),
\[
u_t\sim N(0,\sigma_t^2).
\]
The normalized-gap integral is
\[
\int_0^\infty
p_{(u_t,g_t)}(-\kappa/6,0)
\kappa^{4/3}\,d\kappa
=p_{g_t}(0)
\frac{\Gamma(7/6)72^{7/6}
(\sigma_t^2)^{2/3}}
{2\sqrt{2\pi}}.
\tag{7.3}
\]
Combining the contact factor \(1/6\), the lifetime Jacobian, (7.2), and (7.3) gives (3.10).

At the planar Bargmann–Fock jet,
\[
\Sigma_g=I_3,
\qquad
\Sigma_h=\operatorname{diag}(3,1,1),
\qquad
\sigma_t^2=\frac1{24},
\tag{7.4}
\]
and
\[
Q\mid h_t=0
\stackrel d=
G_2+\sqrt{\frac23}ZI_2,
\qquad
D_t=\frac{29}{6}-\sqrt6,
\tag{7.5}
\]
where \(G_2\) is standard \(2\times2\) GOE and \(Z\sim N(0,1)\) is independent. This yields the closed form (3.12).

## 8. Analytic side-\(24\) correction

Let \(J_0\) be the planar even covariance jet through order six, \(J_{24}\) the corresponding periodized jet, and
\[
F(J)=\frac{\Phi(J)}{\Phi(J_0)},
\tag{8.1}
\]
where \(\Phi\) is the coefficient functional in (3.10).

On the explicit jet ball
\[
\|J-J_0\|_{\max}\le10^{-110},
\tag{8.2}
\]
the coefficient chain gives
\[
\sup\|DF\|<10^{16},
\qquad
\sup\|D^2F\|<10^{42}.
\tag{8.3}
\]
With \(q=e^{-288}\),
\[
J_{24}-J_0=J_{\mathrm{near}}^{(1)}+R_{\mathrm{jet}},
\tag{8.4}
\]
where the linear term comes from the six nearest images \(\pm24e_i\), and
\[
\|J_{24}-J_0\|_{\max}<10^{-113},
\qquad
\|R_{\mathrm{jet}}\|_{\max}<10^{-230}.
\tag{8.5}
\]
Rotational projection gives
\[
DF(J_0)[J_{\mathrm{near}}^{(1)}]
=P_3(24)e^{-288},
\tag{8.6}
\]
where
\[
P_3(L)
=-\frac{L^2(10L^4-147L^2+315)}{105},
\qquad
P_3(24)
=-\frac{620813376}{35}.
\tag{8.7}
\]
The tail and quadratic terms obey
\[
|DF(J_0)[R_{\mathrm{jet}}]|<10^{-210},
\qquad
|R_{\mathrm{quad}}|<5\cdot10^{-185}.
\tag{8.8}
\]
This proves (3.14).

The full nonlinear coefficient has also been evaluated at high precision and cross-checked through Poisson-dual and Gaussian-cone reductions. Those calculations support the analytic result, but the theorem’s finite-side enclosure is not being justified by a black-box floating-point quadrature.

## 9. Ancillary exact results that are genuinely closed

### 9.1 Fixed-\(r\) exact adjacency positivity

At the single separation
\[
r=\frac1{20},
\]
the exact determinant-weighted typed-Palm law for the side-\(24\) periodized Bargmann–Fock field satisfies
\[
\mathbb P_{1/20}^{MS}(A_{1/20})>0.
\tag{9.1}
\]
No numerical lower bound and no uniform-in-\(r\) conclusion is claimed.

### 9.2 Exact qualitative Palm normalizer

For the frozen side-\(24\) exact six-pin Domain-G object,
\[
\frac{Z_r^{\mathrm{exact}}}{r^2}
\longrightarrow
\mathbb E\!\left[
Q_L^2\mathbf 1_{\{Q_L<0\}}
\right]
\in(0,\infty).
\tag{9.2}
\]
Hence, for sufficiently small \(r\),
\[
c_Zr^2\le Z_r^{\mathrm{exact}}\le C_Zr^2.
\tag{9.3}
\]
This is a qualitative normalizer theorem, not a numerical determination of \(c_Z,C_Z\), or the small-\(r\) threshold.

### 9.3 Closed exact interfaces in the two-dimensional P0.2 chain

The following interfaces are terminal within their exact stated scopes:

- the deterministic degree-four capture majorant;
- the \(r^4\) determinant-weight second-moment bound;
- the uniform conditional Gaussian density and moment interface;
- the determinant-weighted Palm-tail transfer inequality.

The wider thirteen-interface P0.2 composition remains nonterminal because the remaining interfaces still require qualifying exact-object reviews.

## 10. Separate two-dimensional compact-selection candidate

This line is mathematically distinct from the reviewed \(d=3,L=24\) theorem above.

For the normalized periodized Bargmann–Fock field on \(\mathbb T_{24}^2\), define
\[
B_*=[23/20,5/4],
\qquad
K_*=[3/4,5/4],
\qquad
P_*=S^1\times B_*\times K_*.
\tag{10.1}
\]

The current TB-G3A source `LS-DER-023-v1.1` proves at same-family candidate level that there exist \(r_A>0\) and \(C_A<\infty\) such that
\[
\sup_{\lambda\in P_*}
\bigl[1-a_{r,\lambda}\bigr]
\le C_A r^3,
\qquad 0<r\le r_A.
\tag{10.2}
\]
The base law is the exact all-typed determinant-weighted maximum–saddle Palm law; it is not adjacency-conditioned.

The current TB-G3B loop source `LS-DER-036-v1.1`, conditional on its named Gaussian interfaces, proves
\[
\sup_{\lambda\in P_*}
\mathbb P_{r,\lambda}^{MS}
(L_{\mathrm{same},r,\lambda})
\le C_{\mathrm{loop}}r^3.
\tag{10.3}
\]
Together with the regional count inherited from `LS-DER-024-v1.0`, this gives the corrected compact defect estimate
\[
\sup_{\lambda\in P_*}
\mathbb P_{r,\lambda}^{MS}
(E_{r,\lambda}^c\cap A_{r,\lambda})
\le (C_B+C_{\mathrm{loop}})r^3.
\tag{10.4}
\]

The corrected compact synthesis is carried by `LS-DER-038-v1.0`. Its external review remains open, as do dependency-aware reviews of `LS-DER-023-v1.1` and `LS-DER-036-v1.1`. This line therefore remains a serious candidate rather than a peer-review-closed theorem.

## 11. Broader three-dimensional periodic-field candidate

`LS-DER-070-v1.0` proposes a genuine field-class extension.

Let \(f\) be a centered, unit-variance, stationary real-analytic Gaussian field on \(\mathbb T_L^3\) with covariance
\[
K(x)
=\sum_{m\in\mathbb Z^3}
w_m e^{2\pi i m\cdot x/L},
\tag{11.1}
\]
where
\[
w_m=w_{-m}>0,
\qquad
\sum_m w_m=1,
\tag{11.2}
\]
and the weights have sufficient exponential decay. Assume
\[
\|J_6(K)-J_6(K_{\mathrm{BF}})\|_{\max}
\le10^{-110},
\tag{11.3}
\]
\[
\|K-K_{\mathrm{BF}}\|_{C^8(B(0,1))}
\le10^{-100}.
\tag{11.4}
\]

The candidate theorem states that the complete finite nonessential superlevel \(H_0\) first-moment lifetime density satisfies
\[
\nu_K(\ell)
=c_K\ell^{-1/3}(1+o(1)),
\qquad
0<c_K<\infty,
\tag{11.5}
\]
and that \(c_K\) is the exact all-typed finite-torus Gaussian-cone coefficient. It also claims continuity of
\[
K\longmapsto c_K
\tag{11.6}
\]
on compact full-spectrum families with a common analytic decay envelope.

The conceptual advance is the separation of:

- **local jet proximity**, which controls fold geometry, collar charts, and singular-near collisions; from
- **global positive Fourier support**, which controls remote-point nondegeneracy, conditional moments, and all-mark domination.

This extension is a same-family theorem candidate with no organizationally distinct review. It should not be presented as an established generalization until its hypotheses, especially the common high-derivative spectral envelope needed for uniform family arguments, receive a dedicated external audit.

## 12. What remains retracted or open

### 12.1 Historical Theorem B

The broader object historically labeled “Theorem B” remains
\[
\texttt{RETRACTED / NOT RESTORED / NOT PROMOTED}.
\]
The fixed \(d=3,L=24\) theorem does not automatically restore that broader canonical object.

### 12.2 Generalization

Open:

- other torus sides \(L\);
- arbitrary dimensions at selected-persistence theorem level;
- arbitrary isotropic or stationary Gaussian fields;
- fields with spectral gaps or finite spectral support;
- higher homological degrees;
- non-Gaussian universality.

### 12.3 Limits

Open:

- a uniform-in-\(L\) lifetime remainder;
- interchange of \(L\to\infty\) and \(\ell\downarrow0\);
- construction and control of an infinite-volume persistence process.

### 12.4 Certification and formalization

Open:

- directed-rounding certification of the full nonlinear nested angular integral;
- clean Lean/Mathlib compilation of the recovered formal source;
- independent formal-methods review after a successful build.

### 12.5 Literature and external peer review

Open:

- systematic MathSciNet/zbMATH/citation-network novelty search;
- human expert review of the local elder-selection topology;
- human expert review of the energy-adapted matrix escape lemma;
- final verification of every imported genericity citation;
- journal-specific notation, bibliography, and style normalization.

## 13. Concrete manuscript corrections already visible

The following should be fixed before submission:

1. In `SIDE24_MANUSCRIPT_CORE.md`, the odd contact variable is displayed as `\nu_t` in one definition but is used as \(u_t\) afterward. The exact technical appendix and controlling source use
   \[
   u_t=-\partial_t^3f/12.
   \]
2. Source-local equation numbers in the exact appendix must be globally renumbered.
3. Governance labels, source hashes, review-routing terms, and correction-history prose should remain outside the submitted mathematical body.
4. The genericity proof needs exact bibliographic theorem-number verification.
5. The theorem must continue to say “per unit volume” and “finite nonessential superlevel \(H_0\) bars.”
6. The scope firewall must continue to exclude arbitrary \(L\), arbitrary fields, higher homology, limit interchange, and infinite volume.

## 14. Current source map

| Role | Current object |
|---|---|
| Closed mathematical disposition | `LS-CLS-077-v1.0` |
| Consolidated fixed-side manuscript | `SIDE24_MANUSCRIPT_CORE.md` |
| Exact proof-module appendix | `SIDE24_TECHNICAL_APPENDIX_EXACT.md` |
| Contact scaling and lifetime pushforward | `LS-DER-046-v1.0` |
| Planar \(d=3\) local charts | `LS-DER-053-v1.0`, with superseded transfer paragraph excluded |
| Exact finite-side coefficient | `LS-DER-056-v1.0` |
| Coefficient-map derivative bounds | `LS-DER-064-v1.0` |
| Side-\(24\) structural transfer | `LS-DER-065-v1.0` |
| Full-spectrum nondegeneracy | `LS-DER-067-v1.0` |
| Exact image-tail ledger | `LS-DER-068-v1.0` |
| Correction and G1–G7 ledger | `LS-DER-071-v1.0` |
| Palm normalization and selection recombination | `LS-DER-072-v1.0` |
| Controlling energy-adapted escape repair | `LS-DER-073-v1.3` |
| Absolute normalization | `LS-DER-075-v1.0` |
| Pair Jacobian and unit multiplicity | `LS-DER-076-v1.0` |
| Broader periodic-field candidate | `LS-DER-070-v1.0` |
| Current two-dimensional adjacency candidate | `LS-DER-023-v1.1` |
| Current two-dimensional loop candidate | `LS-DER-036-v1.1` |
| Current corrected compact synthesis candidate | `LS-DER-038-v1.0` |

## 15. Bottom line

The strongest defensible claim is no longer merely that the fold Jacobian suggests a \(-1/3\) exponent. The current reviewed chain supports a complete fixed-side theorem:
\[
\boxed{
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1))
}
\]
for finite nonessential superlevel \(H_0\) persistence of the normalized side-\(24\) periodized three-dimensional Bargmann–Fock field, with explicit \(c_{3,24}>0\), exact normalization, and
\[
\boxed{
\frac{c_{3,24}}{c_{3,\infty}}-1
=-\frac{620813376}{35}e^{-288}
+\varepsilon_{24},
\qquad
|\varepsilon_{24}|<10^{-180}.
}
\]

The broad scientific message is that the short-lifetime singularity is produced by a dimension-cancelling generic-fold mechanism, while elder selection alters the all-typed contact law only at order \(r^3\). The fixed side-\(24\), \(d=3\) instance is reviewed and closed within its declared scope. The broader field-class, all-\(L\), and infinite-volume programs remain separate research fronts.
