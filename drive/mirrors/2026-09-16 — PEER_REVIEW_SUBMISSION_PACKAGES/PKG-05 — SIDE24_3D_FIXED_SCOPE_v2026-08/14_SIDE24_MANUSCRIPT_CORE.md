# Short-lifetime \(H_0\) persistence for the three-dimensional side-24 periodized Bargmann–Fock field

**Author:** Dylan Roy

**Manuscript status.** Consolidated noncanonical submission draft, revised 2026-07-30. This text states only the fixed-side result supported by the current closed proof chain. It does not assert a theorem for other side lengths, arbitrary Gaussian fields, an infinite-volume persistence process, or an interchange of the limits \(L\to\infty\) and \(\ell\downarrow0\).

## Abstract

Let \(f_{24}\) be the normalized periodization to the cubic torus \(\mathbb T_{24}^3\) of the real Bargmann–Fock field with covariance \(\exp(-|x-y|^2/2)\). We consider finite, nonessential superlevel-set \(H_0\) persistence bars and their first-moment lifetime density per unit volume. We prove the short-lifetime law
\[
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0,
\]
where \(c_{3,24}>0\) is given by an explicit finite-dimensional Gaussian-cone integral. The proof identifies the dimension-cancelling local fold mechanism, proves that elder-rule selection changes the all-typed local contact count by only \(O(r^3)\), and fixes all ordered-pair, directed-sphere, Kac–Rice, and lifetime-Jacobian normalizations. The coefficient is related to its planar Bargmann–Fock value by
\[
\frac{c_{3,24}}{c_{3,\infty}}-1
=-\frac{620813376}{35}e^{-288}+\varepsilon_{24},
\qquad |\varepsilon_{24}|<10^{-180},
\]
with
\[
c_{3,\infty}
=2^{-23/6}3^{-8/3}(29\sqrt6-36)
\Gamma(1/6)\pi^{-5/2}.
\]

## 1. Field, filtration, and lifetime intensity

Write \(\mathbb T_{24}^3=(\mathbb R/24\mathbb Z)^3\). The normalized periodized Bargmann–Fock covariance is
\[
K_{24}(z)
=Z_{24}^{-1}\sum_{n\in\mathbb Z^3}
\exp\!\left(-\frac{|z+24n|^2}{2}\right),
\qquad
Z_{24}=\sum_{n\in\mathbb Z^3}e^{-288|n|^2}.
\tag{1.1}
\]
Let \(f_{24}\) be a centered stationary Gaussian field with covariance \(K_{24}\).

### Proposition 1.1 (genericity of the side-24 field)

The field \(f_{24}\) has real-analytic sample paths and, with probability one, is Morse and has pairwise distinct critical values.

**Proof.**
In the Fourier basis \(e_m(z)=\exp(2\pi i\,m\cdot z/24)\), Poisson summation gives
\[
\widehat K_{24}(m)
=\frac{(2\pi)^{3/2}}{24^3Z_{24}}
\exp\!\left(-\frac{2\pi^2|m|^2}{24^2}\right)>0,
\qquad m\in\mathbb Z^3.
\tag{1.2}
\]
Write the sample series using standard Gaussian Fourier coefficients with amplitudes \(\widehat K_{24}(m)^{1/2}\). For every \(R>0\),
\[
\sum_{m\in\mathbb Z^3}
\mathbb E|\xi_m|\,\widehat K_{24}(m)^{1/2}
\exp(2\pi R|m|/24)<\infty.
\]
Fubini therefore gives almost-sure absolute and locally uniform convergence on every complex strip \(\{z\in\mathbb C^3:|\operatorname{Im}z|\le R\}\). The sample series has an entire periodic extension, so its restriction to the real torus is real analytic.

If \(T\) is a finite linear combination of derivative evaluations, then
\[
\operatorname{Var}(Tf_{24})
=\sum_{m\in\mathbb Z^3}
\widehat K_{24}(m)|\widehat T(m)|^2.
\tag{1.3}
\]
Because every \(\widehat K_{24}(m)\) is positive and Fourier coefficients determine distributions on the torus, this variance vanishes only when the corresponding finite derivative distribution is zero. At one point, the derivatives of the Dirac distribution indexed by distinct multi-indices are linearly independent. Hence the gradient and symmetric-Hessian coordinate vector is nondegenerate at every point. The smoothness, density, and modulus conditions in Adler–Taylor, Theorem 11.3.4, follow from the analytic covariance, this nondegeneracy, and compactness. That theorem implies that \(f_{24}\) is Morse almost surely.

For distinct critical values, fix \(\delta>0\) and consider the compact six-dimensional configuration set
\[
\mathcal T_\delta
=\{(x,y)\in(\mathbb T_{24}^3)^2:
\operatorname{dist}(x,y)\ge\delta\}.
\tag{1.4}
\]
Define the seven-dimensional Gaussian field
\[
\mathcal H(x,y)
=\bigl(\nabla f_{24}(x),\nabla f_{24}(y),
f_{24}(x)-f_{24}(y)\bigr).
\tag{1.5}
\]
For \(x\ne y\), the derivative distributions in (1.5) are linearly independent: distributions supported at distinct points have disjoint supports, and the coefficient of \(\delta_x-\delta_y\) must vanish separately from all derivative coefficients. Equation (1.3) therefore makes the covariance of \(\mathcal H(x,y)\) positive definite. On \(\mathcal T_\delta\), compactness gives a uniform covariance eigenfloor and hence uniformly bounded one-point densities. Apply Adler–Taylor, Lemma 11.2.10, in each member of a finite coordinate cover of \(\mathcal T_\delta\). Since \(\mathcal H\) is a \(C^1\) field from a six-dimensional compact set to \(\mathbb R^7\),
\[
\mathbb P\{\mathcal H^{-1}(0)\cap\mathcal T_\delta
\ne\varnothing\}=0.
\tag{1.6}
\]
Taking the countable union over \(\delta=1/n\) shows that no two distinct critical points have the same value. \(\square\)

For the superlevel filtration
\[
X_a=\{x\in\mathbb T_{24}^3:f_{24}(x)\ge a\},
\tag{1.7}
\]
each non-global local maximum creates an \(H_0\) class and each finite class dies at a unique index-two merging saddle. If \(M\) is the creating maximum and \(S\) its death saddle, its lifetime is
\[
\ell=f_{24}(M)-f_{24}(S)>0.
\tag{1.8}
\]

### Proposition 1.2 (existence of the first-moment lifetime density)

Let \(M_*\) be the global maximum and let \(D_f(M)\) be the elder-rule death saddle of a non-global maximum \(M\). The finite measure
\[
\Lambda_{3,24}(A)
=24^{-3}\,
\mathbb E\!\sum_{M\ne M_*}
\mathbf 1_{\{f_{24}(M)-f_{24}(D_f(M))\in A\}},
\qquad A\subset(0,\infty),
\tag{1.9}
\]
is absolutely continuous with respect to Lebesgue measure.

**Proof.**
Proposition 1.1 makes the elder pairing unambiguous almost surely. A Morse function on the compact torus has finitely many critical points, and the one-point Kac–Rice formula gives finite expected critical-point count. Hence (1.9) is a finite measure.

For \(\delta>0\), restrict the ordered maximum–index-two-saddle pair process to \(\operatorname{dist}(M,S)\ge\delta\). The elder-pair indicator is a bounded measurable mark of the finite Morse filtration. On that compact configuration set, complete positive Fourier support makes
\[
\bigl(f_{24}(x),f_{24}(y),
\nabla f_{24}(x),\nabla f_{24}(y)\bigr)
\]
nondegenerate uniformly in \(x\ne y\). The two-point Kac–Rice formula, with the bounded measurable mark that the ordered pair is the elder pair, therefore expresses the expected birth–death measure as an integral against \(db\,dh\). Thus the restricted measure is absolutely continuous on \(\{(b,h):b>h\}\).

Every distinct critical pair belongs to one of the sets \(\{\operatorname{dist}(M,S)\ge1/n\}\). Monotone convergence over this countable cover shows that the full expected birth–death measure is absolutely continuous. Finally, if \(A\subset\mathbb R\) has one-dimensional Lebesgue measure zero, then
\[
\{(b,h):b-h\in A\}
\tag{1.10}
\]
has two-dimensional Lebesgue measure zero by Fubini after the linear change of variables \((b,h)\mapsto(b,b-h)\). The lifetime pushforward is therefore absolutely continuous. \(\square\)

Let \(\nu_{3,24}(\ell)\,d\ell=d\Lambda_{3,24}(\ell)\). In the proof below, \(\nu_{3,24}\) denotes the pointwise Kac–Rice representative obtained from the pair coordinates and lifetime pushforward.

## 2. Main theorem

### Theorem 2.1 (fixed-side short-lifetime law)

For the normalized side-24 field (1.1), the finite nonessential superlevel \(H_0\) first-moment lifetime density satisfies
\[
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0,
\tag{2.1}
\]
where \(0<c_{3,24}<\infty\).

Fix \(t\in S^2\) and an orthonormal transverse frame \(n_1,n_2\). Define
\[
u_t=-\frac{\partial_t^3f_{24}}{12},
\qquad
g_t=(\partial_tf_{24},\partial_{n_1}f_{24},
\partial_{n_2}f_{24}),
\tag{2.2}
\]
\[
h_t=(\partial_{tt}f_{24},\partial_{tn_1}f_{24},
\partial_{tn_2}f_{24}),
\tag{2.3}
\]
and
\[
q_t=(\partial_{n_1n_1}f_{24},
\partial_{n_1n_2}f_{24},
\partial_{n_2n_2}f_{24}),\qquad
Q_t(q)=
\begin{pmatrix}q_1&q_2\\q_2&q_3\end{pmatrix}.
\tag{2.4}
\]
Put
\[
\Sigma_g(t)=\operatorname{Cov}(g_t),\qquad
\Sigma_h(t)=\operatorname{Cov}(h_t),
\tag{2.5}
\]
\[
\sigma_t^2
=\operatorname{Var}(u_t)
-\operatorname{Cov}(u_t,g_t)\Sigma_g(t)^{-1}
\operatorname{Cov}(g_t,u_t),
\tag{2.6}
\]
\[
\Gamma_t
=\operatorname{Cov}(q_t)
-\operatorname{Cov}(q_t,h_t)\Sigma_h(t)^{-1}
\operatorname{Cov}(h_t,q_t),
\tag{2.7}
\]
and
\[
D_t
=\mathbb E\!\left[
(\det Q_t(q_t))^2\mathbf 1_{\{Q_t(q_t)<0\}}
\mid h_t=0
\right].
\tag{2.8}
\]
Then
\[
c_{3,24}
=B_3\int_{S^2}
\frac{(\sigma_t^2)^{2/3}D_t}
{\sqrt{\det\Sigma_g(t)\det\Sigma_h(t)}}\,dt,
\qquad
B_3=\frac{3\Gamma(7/6)}
{2^{4/3}\pi^{7/2}}.
\tag{2.9}
\]
The formula is independent of the transverse-frame choice.

Moreover,
\[
\frac{c_{3,24}}{c_{3,\infty}}-1
=-\frac{620813376}{35}e^{-288}+\varepsilon_{24},
\qquad |\varepsilon_{24}|<10^{-180},
\tag{2.10}
\]
where
\[
c_{3,\infty}
=2^{-23/6}3^{-8/3}(29\sqrt6-36)
\Gamma(1/6)\pi^{-5/2}
\tag{2.11}
\]
\[
\hspace{2.8cm}
=0.0417759318405983433429366654285755564666815196\ldots.
\]

In particular, \(c_{3,24}<c_{3,\infty}\), and the relative discrepancy has order \(10^{-118}\).

## 3. Proof of Theorem 2.1

The technical lemmas used below are reproduced, with their current amendments, in the accompanying exact technical appendix.

### 3.1. Ordered pair coordinates and unit multiplicity

For a maximum \(M\) and an index-two saddle \(S\) at distance \(r<12\), let
\[
x=\frac{M+S}{2},\qquad
S-M=rt,\qquad t\in S^2.
\tag{3.1}
\]
The inverse local Euclidean map is
\[
(x,u)\longmapsto(M,S)=(x-u/2,x+u/2).
\tag{3.2}
\]
Its determinant has absolute value one, and therefore
\[
dM\,dS=dx\,du=dx\,r^2dr\,d\sigma_2(t).
\tag{3.3}
\]
The antipodal direction \(-t\) exchanges \(M\) and \(S\); it does not represent the same typed ordered pair. Thus the full directed sphere \(S^2\) counts each maximum–saddle pair once, with no factor \(1/2\).

For a Morse function with distinct critical values, each finite nonessential superlevel \(H_0\) bar has one creating maximum and one merging death saddle. An index-two merge reduces \(\beta_0\) by one, so the elder pairing has unit multiplicity. Counting critical-point pairs, rather than unstable-branch incidences, introduces no branch factor.

### 3.2. Contact scaling and the universal radial measure

Put
\[
M=x-\frac r2t,\qquad S=x+\frac r2t,
\tag{3.4}
\]
and impose
\[
f_{24}(M)=b,\qquad
f_{24}(S)=b-\frac{\kappa r^3}{6},\qquad
\nabla f_{24}(M)=\nabla f_{24}(S)=0,
\tag{3.5}
\]
with \(\kappa>0\). The change from the saddle value \(h=f_{24}(S)\) to \(\kappa\) has the exact Jacobian
\[
|dh|=\frac{r^3}{6}\,d\kappa.
\tag{3.6}
\]

Use the corrected divided-difference pin frame
\[
V^+=\frac{f(S)+f(M)}2,\qquad
V^-_{\rm corr}
=\frac{f(S)-f(M)-\frac r2(\partial_tf(S)+\partial_tf(M))}
{r^3},
\tag{3.7}
\]
\[
G_e^+=\frac{\partial_ef(S)+\partial_ef(M)}2,\qquad
G_e^-=\frac{\partial_ef(S)-\partial_ef(M)}r,
\tag{3.8}
\]
for \(e\in\{t,n_1,n_2\}\). The exact determinant of the raw-to-corrected pin map is \(r^{-6}\). Hence the raw pin density contributes \(r^{-6}\).

At a generic fold, one eigenvalue of each endpoint Hessian is soft. Conditionally on the pins,
\[
\det H_M=rD_M,\qquad \det H_S=rD_S,
\tag{3.9}
\]
with uniformly bounded conditional Gaussian moments; consequently the two endpoint determinant weights contribute \(r^2\).

Combining (3.3), (3.6), the corrected-pin density, and (3.9) gives
\[
\frac{r^3}{6}\,r^{-6}\,r^2\,r^2dr
=\frac16\,r\,dr.
\tag{3.10}
\]
No additional ordering, antipodal, branch, or merge multiplicity occurs.

### 3.3. Lifetime pushforward

The lifetime in (3.5) is
\[
\ell=\frac{\kappa r^3}{6}.
\tag{3.11}
\]
For fixed \(\kappa>0\),
\[
r\,dr
=\frac{6^{2/3}}3\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
\tag{3.12}
\]
Equations (3.10)–(3.12) already determine the exponent \(-1/3\). It remains to prove that elder selection has a nonzero limiting mass and to evaluate the coefficient.

### 3.4. Elder selection

Let \(p_r(t,b,\kappa)\) be the Palm probability that the typed maximum–index-two-saddle contact pair in (3.5) is the actual elder persistence pair. The side-24 selection analysis proves, uniformly for \(t\in S^2\) and compact \(b,\kappa>0\),
\[
1-p_r(t,b,\kappa)\le Cr^3.
\tag{3.13}
\]

The proof has three components.

First, the collar and singular-near third-critical-point charts retain their exact vanishing factors under the side-24 periodized jet. In collar variables \(y=x+r(Xt+\rho v)\), the scaled conditional gradient covariance factors as
\[
\det C_{\rm col}
=\rho^6(X^2+\rho^2)A_{\rm col},
\qquad A_{\rm col}>0.
\tag{3.14}
\]
In the singular-near chart, the corrected covariance factors as
\[
\det C_{\rm sing}
=\rho^{10}(3c^2+\rho^2)A_{\rm sing},
\qquad A_{\rm sing}>0.
\tag{3.15}
\]
Together with the anisotropic triple-determinant envelope
\[
[r(r+s^2)]^2(r^2+s^3),
\tag{3.16}
\]
these give Palm expectations \(O(r^3)\) for collar and singular-near witnesses.

Second, complete positive Fourier support of (1.1) implies strict covariance positivity for every linearly independent finite family of derivative-evaluation distributions. Compactness then gives uniform eigenvalue floors for the corrected pair pins joined to a fixed-distance remote value-gradient block. The endpoint determinant factor \(r^2\), the remote value window of width \(\kappa r^3/6\), and the Palm normalizer \(Z_r\asymp r^2\) give
\[
\mathbb E_r^{MS}N_{\rm far}
\le Cr^3[1+(|b|+\kappa)^{10}]
e^{-c(b^2+\kappa^2)}.
\tag{3.17}
\]

Third, the local unstable-manifold comparison is performed in the correctly scaled transverse matrix
\[
T_r=-\frac{Q_\perp}{\kappa r}.
\tag{3.18}
\]
The outward trajectory is controlled by \(T_r\)-energy-adapted cone and strip regions, so no upper bound on \(\lambda_{\max}(T_r)\) is required. The inward branch is captured by the maximum, while the outward branch crosses a forward section at a level strictly above \(f(M)\). Shallow-eigenvalue, finite-jet, and fifth-derivative failures have Palm probability \(O(r^3)\) or \(o(r^3)\).

The elder-rule trichotomy now implies that selection failure produces either a collar witness, a singular-near witness, a fixed-distance witness, or a same-maximum self-attachment. Each event has Palm probability \(O(r^3)\), proving (3.13).

### 3.5. All-mark domination and selected/all-typed equality

The corrected pair target is uniformly injective in \((b,\kappa)\). Gaussian regression and bounded conditional polynomial moments give the common majorant
\[
C[1+(|b|+\kappa)^N]
e^{-c(b^2+\kappa^2)}\kappa^{-2/3},
\tag{3.19}
\]
which is integrable on
\[
S^2\times\mathbb R\times(0,\infty).
\]
Since \(0\le p_r\le1\), (3.13) and dominated convergence show that the selected and all-typed contact integrals have the same limit.

### 3.6. Exact coefficient reduction

Even and odd covariance derivatives are independent because \(K_{24}\) is even. Integrating over the birth height removes the field-value coordinate:
\[
\int_{\mathbb R}
p_{(f,h_t)}(b,0)
\mathbb E[F(q_t)\mid f=b,h_t=0]\,db
=p_{h_t}(0)\mathbb E[F(q_t)\mid h_t=0],
\tag{3.20}
\]
where
\[
F(q)=(\det Q_t(q))^2\mathbf 1_{\{Q_t(q)<0\}}.
\]
Conditionally on \(g_t=0\), the variable \(u_t\) is centered Gaussian with variance \(\sigma_t^2\). The remaining normalized-gap integral is
\[
\int_0^\infty
p_{(u_t,g_t)}(-\kappa/6,0)
\kappa^{4/3}\,d\kappa
=p_{g_t}(0)
\frac{\Gamma(7/6)72^{7/6}(\sigma_t^2)^{2/3}}
{2\sqrt{2\pi}}.
\tag{3.21}
\]
Combining (3.10), (3.12), (3.20), and (3.21) gives exactly (2.9).

Equivalently, if \(A_tA_t^\top=\Gamma_t\) and
\(\mathcal Q_t(v)=Q_t(A_tv)\), then
\[
D_t=\frac{15}{4\pi}\int_{S^2}
[\det\mathcal Q_t(v)]^2
\mathbf 1_{\{\operatorname{tr}\mathcal Q_t(v)<0,\,
\det\mathcal Q_t(v)>0\}}\,dv.
\tag{3.22}
\]
Thus \(c_{3,24}\) is a deterministic nested angular integral involving only the order-six covariance jet at the origin.

### 3.7. Planar value

At the planar Bargmann–Fock jet,
\[
\Sigma_g=I_3,\qquad
\Sigma_h=\operatorname{diag}(3,1,1),\qquad
\sigma_t^2=\frac1{24},
\tag{3.23}
\]
and
\[
Q\mid h_t=0
\stackrel d=G_2+\sqrt{\frac23}ZI_2,
\qquad
D_t=\frac{29}{6}-\sqrt6,
\tag{3.24}
\]
where \(G_2\) is standard \(2\times2\) GOE and
\(Z\sim N(0,1)\) is independent. Substitution in (2.9) gives (2.11).

### 3.8. Side-24 correction

Let \(J_0\) be the planar even covariance jet through order six,
\(J_{24}\) the corresponding jet of (1.1), and
\[
F(J)=\frac{\Phi(J)}{\Phi(J_0)},
\tag{3.25}
\]
where \(\Phi\) is the coefficient functional in (2.9). On the explicit jet ball
\[
\|J-J_0\|_{\max}\le10^{-110},
\tag{3.26}
\]
the covariance and Schur-complement matrices have uniform positive lower and finite upper bounds. Fixed-cone Gaussian differentiation gives
\[
\sup\|DF\|<10^{16},\qquad
\sup\|D^2F\|<10^{42}.
\tag{3.27}
\]

Put \(q=e^{-288}\). The normalized periodized jet has the decomposition
\[
J_{24}-J_0=J_{\rm near}^{(1)}+R_{\rm jet},
\tag{3.28}
\]
where the linear term comes from the six images
\(\pm24e_i\), and exact rational Hermite bounds give
\[
\|J_{24}-J_0\|_{\max}<10^{-113},\qquad
\|R_{\rm jet}\|_{\max}<10^{-230}.
\tag{3.29}
\]
Rotational projection of the six-image orbit yields
\[
DF(J_0)[J_{\rm near}^{(1)}]=P_3(24)e^{-288},
\tag{3.30}
\]
where
\[
P_3(L)
=-\frac{L^2(10L^4-147L^2+315)}{105},
\qquad
P_3(24)=-\frac{620813376}{35}.
\tag{3.31}
\]
Taylor’s theorem, (3.27), and (3.29) give
\[
|DF(J_0)[R_{\rm jet}]|<10^{-210},
\qquad
|R_{\rm quad}|<5\cdot10^{-185}.
\tag{3.32}
\]
Therefore
\[
F(J_{24})-1
=P_3(24)e^{-288}+\varepsilon_{24},
\qquad |\varepsilon_{24}|<10^{-180},
\tag{3.33}
\]
which is (2.10). This completes the proof. \(\square\)

## 4. Exact scope

Theorem 2.1 is restricted to:

- ambient dimension \(3\);
- torus side \(24\);
- the normalized periodized Bargmann–Fock covariance (1.1);
- finite nonessential superlevel \(H_0\) bars;
- the first-moment lifetime density as \(\ell\downarrow0\);
- the ordered maximum–index-two-saddle, directed-\(S^2\), and Palm conventions used above.

It does not imply:

- a theorem for other side lengths;
- a theorem for arbitrary isotropic or stationary Gaussian fields;
- a result for \(H_k\), \(k>0\);
- an \(L\to\infty\), \(\ell\downarrow0\) limit interchange;
- an infinite-volume persistence process;
- a directed-rounding certificate for the full nonlinear quadrature.

## 5. Literature position and remaining submission checks

The closest located literature separates into three groups. Adler–Bobrowski–Borman–Subag–Weinberger and Pranav study persistent homology of random or Gaussian fields, principally through general structure and computation. Chazal–Divol prove existence of expected persistence-diagram densities for broad classes of random filtrations, but not the present smooth-field near-diagonal asymptotic. Klein–Agam and Ancona–Gass–Letendre–Stecconi analyze critical-point correlations and Kac–Rice singularities without imposing the elder persistence pairing. The targeted search through 2026-07-30 did not locate a prior theorem giving the fixed-side three-dimensional Bargmann–Fock near-diagonal \(H_0\) density (2.1), the exact coefficient reduction (2.9), or the correction (2.10). This is evidence of novelty, not a substitute for a systematic MathSciNet, zbMATH, and citation-network search.

Before journal submission:

1. complete the systematic database and backward/forward citation search for short-lifetime persistence asymptotics of smooth Gaussian fields and local maximum–saddle Kac–Rice pair laws;
2. convert the exact technical appendix into conventionally numbered lemmas and proofs, eliminating provenance labels from the submitted version;
3. have a human specialist independently check the local elder-selection topology and the energy-adapted matrix escape lemma;
4. complete a journal-specific notation, bibliography, and style pass;
5. optionally produce directed-rounding bounds for the nonlinear integral (2.9). This would strengthen numerical certification but is not used in the analytic estimate (2.10).

## References for the literature audit

1. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer Monographs in Mathematics, Springer, 2007.
2. R. J. Adler, O. Bobrowski, M. S. Borman, E. Subag, and S. Weinberger, “Persistent homology for random fields and complexes,” *IMS Collections* **6** (2010), 124–143. <https://projecteuclid.org/ebooks/institute-of-mathematics-statistics-collections/Borrowing-Strength--Theory-Powering-Applications--A-Festschrift-for/chapter/Persistent-homology-for-random-fields-and-complexes/10.1214/10-IMSCOLL609>
3. F. Chazal and V. Divol, “The density of expected persistence diagrams and its kernel based estimation,” *Journal of Computational Geometry* **10**(2) (2019), 127–153. <https://jocg.org/index.php/jocg/article/download/3090/2817/>
4. A. Klein and O. Agam, “Critical point correlations in random Gaussian fields,” *Journal of Physics A: Mathematical and Theoretical* **45** (2012), 025001. <https://arxiv.org/abs/1111.5286>
5. P. Pranav, “Topology and geometry of Gaussian random fields II: on critical points, excursion sets, and persistent homology,” arXiv:2109.08721 (2021). <https://arxiv.org/abs/2109.08721>
6. C. Hirsch and R. Lachièze-Rey, “Functional central limit theorem for topological functionals of Gaussian critical points,” arXiv:2411.11429 (2024). <https://arxiv.org/abs/2411.11429>
7. M. Ancona, L. Gass, T. Letendre, and M. Stecconi, “Zeros and critical points of Gaussian fields: cumulants asymptotics and limit theorems,” arXiv:2501.10226 (2025), revised 2025. <https://arxiv.org/abs/2501.10226>
