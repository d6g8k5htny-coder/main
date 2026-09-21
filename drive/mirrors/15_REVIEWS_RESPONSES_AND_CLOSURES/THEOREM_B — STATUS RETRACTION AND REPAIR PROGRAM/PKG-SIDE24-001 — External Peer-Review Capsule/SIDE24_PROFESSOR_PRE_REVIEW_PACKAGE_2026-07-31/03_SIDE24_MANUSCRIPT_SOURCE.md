# Short-lifetime \(H_0\) persistence for the three-dimensional side-24 periodized Bargmann–Fock field

**Author:** Dylan Roy

**Manuscript status.** Pre-peer-review working manuscript, revised 2026-07-31. This document records a fixed-scope candidate theorem, several completed analytic reductions, and the remaining proof obligations. It is transfer-self-contained but is not represented as a closed proof or a submission-ready paper. The component-based elder-failure exhaustion, deterministic matrix capture/escape argument, determinant-weighted exceptional-set transfer, and fixed-distance witness bound are now written out. The uniform finite-\(r\) collar and singular-near triple-contact estimates—and the quantitative side-24 chart transfer needed for them—remain subject to independent verification or completion. No claim is made for other side lengths, arbitrary Gaussian fields, an infinite-volume persistence process, or an interchange of the limits \(L\to\infty\) and \(\ell\downarrow0\).

## Abstract

Let \(f_{24}\) be the normalized periodization to the cubic torus \(\mathbb T_{24}^3\) of the real Bargmann–Fock field with covariance \(\exp(-|x-y|^2/2)\). We consider finite, nonessential superlevel-set \(H_0\) persistence bars and their first-moment lifetime density per unit volume. The candidate short-lifetime law is
\[
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0,
\]
where \(c_{3,24}>0\) is given by an explicit finite-dimensional Gaussian-cone integral. The completed local calculation identifies the dimension-cancelling fold mechanism and fixes the ordered-pair, directed-sphere, Kac–Rice, and lifetime-Jacobian normalizations. This revision inserts for review a Borel construction of the elder mark, an exact near-density pushforward, a uniform all-mark majorant, and a separate off-diagonal bound. The passage from the all-typed contact law to the elder-selected law remains conditional on the unresolved estimate \(1-p_r=O(r^3)\). Subject also to the stated quantitative jet-transfer bounds, the coefficient is related to its Euclidean (nonperiodized) Bargmann–Fock value by
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
Write the sample series using standard Gaussian Fourier coefficients with amplitudes \(\widehat K_{24}(m)^{1/2}\). For each positive integer \(R\), put
\[
S_R=\sum_{m\in\mathbb Z^3}
|\xi_m|\,\widehat K_{24}(m)^{1/2}
\exp(2\pi R|m|/24).
\]
The Gaussian decay in (1.2) gives \(\mathbb E S_R<\infty\). Tonelli's theorem therefore implies \(S_R<\infty\) almost surely. Intersecting these probability-one events over \(R\in\mathbb N\), the Weierstrass M-test gives absolute and uniform convergence on every compact complex strip. Termwise differentiation is valid there, so the sample series has an entire periodic extension and its restriction to the real torus is real analytic.

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

### Lemma 1.2a (a Borel elder-pair mark)

On the set of \(C^2\) Morse functions on \(\mathbb T_{24}^3\) with distinct critical values, the indicator
\[
\mathfrak e(f;M,S)
=\mathbf 1_{\{(M,S)\text{ is the finite elder }H_0\text{ pair}\}}
\]
has a Borel extension as a function of \((f,M,S)\in C^2(\mathbb T_{24}^3)\times(\mathbb T_{24}^3)^2\).

**Proof.** For \(a\in\mathbb R\), write \(U(f,a)=\{f>a\}\). Fix a countable basis \(\mathcal B\) of relatively compact, path-connected coordinate balls, chosen so that every point of every open set \(V\) lies in some \(B\in\mathcal B\) with \(\overline B\subset V\). For \(x,y\in U(f,a)\), membership in the same component of \(U(f,a)\) is equivalent to the existence of a finite chain \(B_1,\ldots,B_k\in\mathcal B\) such that \(x\in B_1\), \(y\in B_k\), \(B_i\cap B_{i+1}\ne\varnothing\), and \(\min_{\overline B_i}f>a\) for every \(i\). This is a countable union of Borel conditions in \((f,a,x,y)\), so the component relation is Borel.

Let \((q_j)_{j\ge1}\) be a fixed countable dense subset of the torus. Define \(O(f,M,a)\) to mean that \(M\in U(f,a)\) and that the component of \(U(f,a)\) containing \(M\) contains some \(q_j\) with \(f(q_j)>f(M)\). The relation \(O\) is Borel. If \(h=f(S)\), define \(\mathfrak e(f;M,S)=1\) exactly when \(M\) is a local maximum, \(S\) is an index-two critical point, \(f(M)>h\), and for some \(n\ge1\), for every rational \(q\in(0,1/n)\),
\[
O(f,M,h+q)=0,
\qquad
O(f,M,h-q)=1.
\tag{1.8a}
\]
All conditions are Borel; criticality and Morse index are Borel conditions on the first two jets. For a Morse function with distinct critical values, choose \(n\) so large that \(1/n<f(M)-h\) and \(2/n<|c-h|\) for every critical value \(c\ne h\).
The Morse deformation lemma and the local index-two handle model show that (1.8a) holds precisely when the component born at \(M\) first joins a component with a strictly higher birth maximum at \(S\). This is the finite elder-rule pairing. The locus of Morse functions with pairwise distinct critical values is open in \(C^2\): nondegenerate critical points persist uniquely under small \(C^2\) perturbations, no new critical points occur away from their isolating neighborhoods, and the finite positive gap between critical values persists. In particular, this locus is Borel. Setting the mark to zero off it gives the required Borel extension. \(\square\)

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

For \(\delta>0\), restrict the ordered maximum–index-two-saddle pair process to \(\operatorname{dist}(M,S)\ge\delta\). On that compact configuration set, complete positive Fourier support makes
\[
\bigl(f_{24}(x),f_{24}(y),
\nabla f_{24}(x),\nabla f_{24}(y)\bigr)
\]
nondegenerate uniformly in \(x\ne y\). Ordinary two-point Kac–Rice applied to all typed maximum–index-two-saddle pairs therefore gives an absolutely continuous expected value-pair measure on \(\{(b,h):b>h\}\). The selected elder-pair measure is a submeasure of this all-typed measure and is consequently absolutely continuous. Lemma 1.2a supplies a Borel version of the mark when a marked formulation is used.

Every distinct critical pair belongs to one of the sets \(\{\operatorname{dist}(M,S)\ge1/n\}\). Monotone convergence over this countable cover shows that the full expected birth–death measure is absolutely continuous. Finally, if \(A\subset\mathbb R\) has one-dimensional Lebesgue measure zero, then
\[
\{(b,h):b-h\in A\}
\tag{1.10}
\]
has two-dimensional Lebesgue measure zero by Fubini after the linear change of variables \((b,h)\mapsto(b,b-h)\). The lifetime pushforward is therefore absolutely continuous. \(\square\)

Let \(\nu_{3,24}(\ell)\,d\ell=d\Lambda_{3,24}(\ell)\). Throughout, \(\nu_{3,24}\) means a **first-moment intensity density per unit ordinary torus volume**, not the probability density of a randomly selected bar. We choose the pointwise representative obtained by adding the near- and far-pair Kac–Rice representatives constructed below; this representative agrees almost everywhere with the Radon–Nikodym derivative in Proposition 1.2.

## 2. Candidate theorem and verification status

### Candidate Theorem 2.1 (fixed-side short-lifetime law)

The proposed conclusion for the normalized side-24 field (1.1) is that the finite nonessential superlevel \(H_0\) first-moment lifetime density satisfies
\[
\nu_{3,24}(\ell)
=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0,
\tag{2.1}
\]
where \(0<c_{3,24}<\infty\).

**Verification status.** Sections 3.1--3.3 and 3.5 supply the exact local coordinates, Palm normalization, near-density pushforward, domination, and the bounded off-diagonal contribution. Lemma 3.4a gives the deterministic component-based exhaustion of elder-selection failure; Appendix Module J and Lemma 3.4b give the deterministic capture/escape theorem and its determinant-weighted exceptional-set transfer; Lemma 3.4c gives the fixed-distance witness bound. Sections 3.6--3.8 give the coefficient reduction and its computationally checked side-24 expansion. The unconditional implication to (2.1) still requires Proposition 3.4: the uniform finite-\(r\) collar and singular-near triple-contact estimates, including the quantitative side-24 chart transfer and angular integrations, are explicitly unresolved. Thus (2.1) is a candidate theorem, not a theorem claimed proved in this version.

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
{\sqrt{\det\Sigma_g(t)\det\Sigma_h(t)}}\,d\sigma_2(t),
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
and numerically
\[
=0.0417759318405983433429366654285755564666815196\ldots.
\]

In particular, \(c_{3,24}<c_{3,\infty}\), and the relative discrepancy has order \(10^{-118}\).

## 3. Derivation and remaining proof obligations

The technical material used below is reproduced, with current amendments and explicit status labels, in the accompanying review appendix. Completed lemmas are distinguished from the unresolved selection proposition.

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

Indeed, order the raw variables by the two values followed by the three endpoint-gradient pairs, and order the corrected variables by \((V^+,V^-_{\rm corr})\) followed by the three \((G_e^+,G_e^-)\) pairs. The derivative matrix has the block-triangular form
\[
\frac{\partial Y_r}{\partial X_r}
=\begin{pmatrix}A_r&B_r\\0&C_r\end{pmatrix},
\qquad
|\det A_r|=r^{-3},
\qquad
|\det C_r|=r^{-3}.
\tag{3.8a}
\]
The block \(B_r\) contains the tangential-gradient correction in \(V^-_{\rm corr}\) and is a shear; it does not change the determinant. Thus \(|\det(\partial Y_r/\partial X_r)|=r^{-6}\) exactly. Taylor expansion of the even covariance shows that the apparent negative powers in \(\operatorname{Cov}(Y_r)\) are removable. Its limit is positive definite by derivative-distribution independence, and convergence is uniform in \(t\) and on compact \((b,\kappa)\)-sets.

At a generic fold, one eigenvalue of each endpoint Hessian is soft. Conditionally on the pins,
\[
\det H_M=rD_M,\qquad \det H_S=rD_S,
\tag{3.9}
\]
with uniformly bounded conditional Gaussian moments; consequently the two endpoint determinant weights contribute \(r^2\).

For later use, let \(\Pi_{r,x,t}\) be the raw value-gradient pin vector in (3.5), let \(z_r(b,\kappa)=(b,b-\kappa r^3/6,0,0)\), and write
\[
\mathbb E^0_{r,t,b,\kappa}[\,\cdot\,]
=\mathbb E[\,\cdot\mid\Pi_{r,x,t}=z_r(b,\kappa)].
\tag{3.9a}
\]
Put
\[
W_r=|\det H_M|\,|\det H_S|
\mathbf 1_{\{H_M\prec0\}}
\mathbf 1_{\{\operatorname{ind}(H_S)=2\}},
\qquad
Z_r=\mathbb E^0_{r,t,b,\kappa}[W_r].
\tag{3.9b}
\]
The regular conditional law exists because the pin covariance is nondegenerate for \(r>0\). In fact, the joint Gaussian vector consisting of the value-gradient pins and all endpoint Hessian coordinates is nondegenerate: the corresponding derivative distributions at the two distinct support points are linearly independent. Its Hessian Schur complement is therefore positive definite. Consequently the conditional endpoint-Hessian law has full support, the open maximum--index-two type region has positive conditional probability, and
\[
Z_r(t,b,\kappa)>0
\]
for every \(r>0\), \(t\), \(b\), and \(\kappa>0\).

On the pinned affine space, the exact endpoint identities in (3.9) give
\[
Z_r=r^2\zeta_r.
\tag{3.9d}
\]
The corrected-frame contact limit gives, locally uniformly on compact positive mark sets,
\[
\zeta_r(t,b,\kappa)\longrightarrow
\zeta_0(t,b,\kappa)
=\kappa^2\mathbb E\!\left[(\det Q)^2
\mathbf 1_{\{Q\prec0\}}\mid\text{limiting corrected pins}\right].
\tag{3.9e}
\]
The conditional covariance of \(Q\) is positive definite, so its law has full support on \(\operatorname{Sym}_2\). The negative-definite cone is nonempty and open, and hence \(\zeta_0>0\) for every \(\kappa>0\). Continuity and compactness therefore give \(0<c\le\zeta_r\le C<\infty\) on compact positive mark sets for all sufficiently small \(r\).

The determinant-weighted all-typed Palm law and selection probability are therefore well defined by
\[
\mathbb E^{MS}_{r,t,b,\kappa}[F]
=\frac{\mathbb E^0_{r,t,b,\kappa}[W_rF]}{Z_r},
\qquad
p_r(t,b,\kappa)
=\mathbb E^{MS}_{r,t,b,\kappa}[\mathfrak e(f;M,S)].
\tag{3.9c}
\]
Neither adjacency nor persistence is built into the base Palm law.

Combining (3.3), (3.6), the corrected-pin density, and (3.9) gives
\[
\frac{r^3}{6}\,r^{-6}\,r^2\,r^2dr
=\frac16\,r\,dr.
\tag{3.10}
\]
No additional ordering, antipodal, branch, or merge multiplicity occurs.

### 3.3. Exact near-density pushforward

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
Define
\[
\mathcal K_r^{\mathrm{sel}}(t,b,\kappa)
=p_{\Pi_{r,x,t}}(z_r(b,\kappa))
\mathbb E^0_{r,t,b,\kappa}
[W_r\mathfrak e(f;M,S)]
\tag{3.12a}
\]
and define \(\mathcal K_r^{\mathrm{all}}\) by omitting \(\mathfrak e\). Stationarity makes these factors independent of the midpoint \(x\), which is why \(x\) is absent from their arguments. Equation (3.9c) gives \(\mathcal K_r^{\mathrm{sel}}=p_r\mathcal K_r^{\mathrm{all}}\).

**Lemma 3.3 (near-pair density representative).** Fix \(0<\rho<12\). For every nonnegative Borel test function \(\varphi\), the contribution from elder pairs with \(d(M,S)<\rho\) satisfies
\[
\begin{aligned}
\int_0^\infty\varphi(\ell)\nu_\rho^{\mathrm{near}}(\ell)d\ell
={}&\int_{S^2}\int_{\mathbb R}\int_0^\rho\int_0^\infty
\varphi\!\left(\frac{\kappa r^3}{6}\right)\\
&\quad\times\frac{r^5}{6}\,
\mathcal K_r^{\mathrm{sel}}(t,b,\kappa)
\,d\kappa\,dr\,db\,d\sigma_2(t).
\end{aligned}
\tag{3.12b}
\]
Consequently, with \(r_\ell(\kappa)=(6\ell/\kappa)^{1/3}\), a pointwise representative is
\[
\nu_\rho^{\mathrm{near}}(\ell)
=\int_{S^2}\int_{\mathbb R}
\int_{6\ell/\rho^3}^{\infty}
\frac{2\ell}{\kappa^2}
\mathcal K_{r_\ell(\kappa)}^{\mathrm{sel}}(t,b,\kappa)
\,d\kappa\,db\,d\sigma_2(t).
\tag{3.12c}
\]
If \(\mathcal G_r^{\mathrm{sel}}=r^4\mathcal K_r^{\mathrm{sel}}\), then
\[
\begin{aligned}
\nu_\rho^{\mathrm{near}}(\ell)
={}&\frac{6^{2/3}}{18}\ell^{-1/3}
\int_{S^2}\int_{\mathbb R}
\int_{6\ell/\rho^3}^{\infty}
\kappa^{-2/3}\\
&\quad\times
\mathcal G_{r_\ell(\kappa)}^{\mathrm{sel}}(t,b,\kappa)
\,d\kappa\,db\,d\sigma_2(t).
\end{aligned}
\tag{3.12d}
\]

**Proof.** Apply marked two-point Kac–Rice with the Borel mark from Lemma 1.2a. Stationarity cancels the midpoint integral against the ordinary-volume normalization \(24^{-3}\). Equations (3.3) and (3.6) give the factor \(r^2(r^3/6)=r^5/6\), proving (3.12b). At fixed \(\kappa\), change variables from \(r\) to \(\ell=\kappa r^3/6\); direct differentiation gives (3.12c). Substitution of \(r_\ell^4=(6\ell/\kappa)^{4/3}\) yields (3.12d). \(\square\)

Equations (3.10)–(3.12d) determine the exponent \(-1/3\) for the all-typed contact law. Passing to the elder-selected law requires the unresolved selection proposition below.

### 3.4. Elder selection: unresolved proposition

The candidate theorem requires the following statement.

**Proposition 3.4 (selection estimate; not yet proved in this manuscript).** Uniformly for \(t\in S^2\) and compact \(b,\kappa>0\),
\[
1-p_r(t,b,\kappa)\le Cr^3.
\tag{3.13}
\]

The topological exhaustion required by this proposition can be stated without
any Morse--Smale or branch-convergence hypothesis. The following lemma closes
that deterministic part of the argument; the probability estimates for its
events remain separate.

**Lemma 3.4a (elder-failure exhaustion conditional on component adjacency).**
Let \(f\) be a Morse function with pairwise distinct critical values on a
compact connected three-manifold. Let \(M\) be a local maximum with
\(b=f(M)\), and let \(S\) be an index-two critical point with
\(h=f(S)<b\). Choose \(\varepsilon>0\) so that
\(\varepsilon<b-h\) and \(h\) is the only critical value in
\([h-\varepsilon,h+\varepsilon]\). Let \(C_1,C_2\) be the components of
\(\{f>h+\varepsilon\}\) containing the two local upper sectors at \(S\).
Assume that \(M\) belongs to one of \(C_1,C_2\). If \((M,S)\) is not the
finite elder \(H_0\) pair, then exactly one of the following topological cases
occurs:

1. the class born at \(M\) died earlier at its unique merging saddle \(R\),
   with \(h<f(R)<b\);
2. that class is alive immediately above \(h\), but \(C_1=C_2\), so the
   index-two handle at \(S\) is a same-component attachment and kills no
   \(H_0\) class;
3. that class is alive immediately above \(h\), \(C_1\ne C_2\), and the
   other component has a unique elder representative maximum \(N\) satisfying
   \(h<f(N)<b\).

**Proof.** If the class born at \(M\) is not alive immediately above \(h\),
its unique elder-rule death occurs at an index-two merging saddle \(R\).
The class is born at level \(b\), and it is already dead at level
\(h+\varepsilon\), so distinct critical values give
\(h<f(R)<b\). This is case 1.

Suppose instead that the class is alive immediately above \(h\). If
\(C_1=C_2\), the local index-two handle attaches within one component, which
is case 2. Otherwise the handle merges two distinct components. The component
containing \(M\) is represented by \(M\), because the class born at \(M\) is
still alive. Let \(N\) be the unique elder representative maximum of the
other component. Necessarily \(f(N)>h\). If \(f(N)>b\), the component
represented by \(M\) is younger and dies at \(S\), contrary to the hypothesis
that \((M,S)\) is not the elder pair. Equality is excluded by distinct
critical values. Hence \(h<f(N)<b\), which is case 3. The three cases are
disjoint by construction and exhaustive. \(\square\)

For the probabilistic decomposition, define the **component-adjacency event**
\(A_r\) by the hypothesis of Lemma 3.4a, rather than by an infinite-time flow
event. Cases 1 and 3 supply a canonical third critical witness \(y\) in the
open value window \((h,b)\): respectively \(y=R\) and \(y=N\). Fix
\(C>2\), \(\delta_0>0\), and \(r<\delta_0/C\), and partition that canonical
witness using torus geodesic distance into
\[
\begin{aligned}
E_{\rm collar}&=\{d(y,\{M,S\})\le Cr\},\\
E_{\rm singular}&=\{Cr<d(y,\{M,S\})<\delta_0\},\\
E_{\rm fixed}&=\{d(y,\{M,S\})\ge\delta_0\}.
\end{aligned}
\tag{3.13a}
\]
With
\[
E_{\rm self}
=\{\text{the class born at }M\text{ is alive above }h, C_1=C_2\},
\]
Lemma 3.4a gives the measurable inclusion
\[
\{\mathfrak e=0\}
\subset A_r^c\cup E_{\rm collar}\cup E_{\rm singular}
\cup E_{\rm fixed}\cup E_{\rm self}.
\tag{3.13b}
\]
The component relations are Borel by Lemma 1.2a, and the regional witness
events are Borel critical-point count events. Other, noncanonical witnesses
may occur in more than one region, so regional existence events need not be
mutually exclusive; only the union inclusion and a union bound are used.

The current appendix contains calculations intended to support three components of a proof, summarized here so that the missing interfaces are auditable.

First, the collar and singular-near third-critical-point charts are asserted to retain their vanishing factors under the side-24 periodized jet. In collar variables \(y=x+r(Xt+\rho v)\), the scaled conditional gradient covariance is intended to factor as
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
Together with the proposed anisotropic triple-determinant envelope
\[
[r(r+s^2)]^2(r^2+s^3),
\tag{3.16}
\]
these would give Palm expectations \(O(r^3)\) for collar and singular-near witnesses. The symbolic annex verifies the limiting elimination conditional on the chosen chart, but the manuscript still needs a uniform side-24 statement, an explicit angular integration, and a proof that every geometric subregime is covered.

Second, complete positive Fourier support of (1.1) implies strict covariance positivity for every linearly independent finite family of derivative-evaluation distributions. Compactness then gives uniform eigenvalue floors for the corrected pair pins joined to a fixed-distance remote value-gradient block. The endpoint determinant factor \(r^2\), the remote value window of width \(\kappa r^3/6\), and the Palm normalizer \(Z_r\asymp r^2\) give the bound
\[
\mathbb E_r^{MS}N_{\rm far}
\le Cr^3[1+(|b|+\kappa)^{10}]
e^{-c(b^2+\kappa^2)}.
\tag{3.17}
\]

Third, the local unstable-manifold comparison is formulated in the scaled transverse matrix
\[
T_r=-\frac{Q_\perp}{\kappa r}.
\tag{3.18}
\]
Module J gives a contiguous deterministic proof of the inward tapered-tube
capture, the energy-adapted outward cone and strip, and the strict forward
level crossing without an upper bound on \(\lambda_{\max}(T_r)\). The next
lemma supplies the determinant-weighted exceptional-set estimate needed to
apply that deterministic result.

**Lemma 3.4b (Palm transfer of the matrix capture/escape theorem).** Fix a
compact set
\(\mathcal C\subset S^2\times\mathbb R\times(0,\infty)\) of marks
\((t,b,\kappa)\). Write \(\mathbb P^0_{r,t,b,\kappa}\) for the conditional
probability associated with (3.9a). Under the all-typed Palm law (3.9c), let
\(\lambda_*\) be the least eigenvalue of \(-Q_{\perp,r}\), the negative
transverse saddle block, and let \(\mathcal D_r\) be the event on which the
three hypotheses of Module J hold:
\[
\begin{gathered}
rR_r^5\le\varepsilon_0,\\
\frac{\lambda_*}{\kappa r}\ge 2KR_r^5+2R_r,\\
\|F_{\rm exact}-F_r\|_{C^1}\le\eta_{R_r}.
\end{gathered}
\tag{3.18a}
\]
Here \(R_r\) is one plus the norm of the finite normalized coefficient vector
in the matrix normal form and
\(\eta_R=(8192R^3)^{-1}\). There are constants \(C,r_0>0\), uniform on
\(\mathcal C\), such that
\[
\mathbb P^{MS}_{r,t,b,\kappa}(\mathcal D_r^c)\le Cr^3,
\qquad 0<r<r_0.
\tag{3.18b}
\]
On \(\mathcal D_r\), the inward upper half-branch lies in the component of
\(M\), and the other upper half-branch reaches a point of value strictly
larger than \(b\). Consequently
\[
\mathbb P^{MS}_{r,t,b,\kappa}
\bigl(A_r^c\cup E_{\rm self}\cup E_{\rm young}\bigr)
\le Cr^3,
\tag{3.18c}
\]
where \(E_{\rm young}\) is case 3 of Lemma 3.4a.

**Proof.** Choose a maximal independent finite Gaussian coordinate vector
\(U_r\) containing the derivatives through order four used by the normalized
matrix form. Corrected-pin nondegeneracy and Gaussian regression give,
uniformly on \(\mathcal C\), sub-Gaussian tails for \(U_r\), bounded
conditional moments of every order, and a nondegenerate conditional density
for the transverse \(2\times2\) block. The conditioned fifth-derivative
process on a fixed physical chart has a uniformly bounded affine mean and a
covariance dominated by the unconditional derivative covariance. Borell--TIS
therefore gives, for its supremum \(H_{5,r}\),
\[
\mathbb P^0_{r,t,b,\kappa}
\{\|U_r\|+H_{5,r}>u\}\le C e^{-cu^2}
\tag{3.18d}
\]
after increasing \(C\) to cover bounded \(u\).

The exact endpoint factorization (3.9), Taylor expansion of the two transverse
blocks, and a determinant expansion give, on the typed cone,
\[
W_r\le Cr^2
 [\lambda_*+rP(U_r)]^2(1+\|U_r\|)^N,
\qquad
\mathbb E^0[W_r^2]\le Cr^4,
\tag{3.18e}
\]
for a fixed nonnegative polynomial \(P\). To see the first inequality, write
each endpoint Hessian in axial/transverse block form. Its axial row and column
have an exact factor \(r\), while its transverse block differs from
\(Q_{\perp,r}\) by \(r\) times a polynomially bounded finite jet. Expansion
along the axial row gives one factor
\(r[\lambda_*+rP(U_r)]\) per endpoint, up to polynomially bounded hard
factors. The second inequality follows from the same expansion and Gaussian
moments.

Failure of the coercivity inequality in (3.18a) implies
\[
0<\lambda_*\le rP_0(U_r)
\tag{3.18f}
\]
for another fixed polynomial, because \(\kappa\) is bounded on
\(\mathcal C\). The Gaussian density of the transverse block is uniformly
bounded with a Gaussian polynomial envelope. The eigenvalue coarea formula
on the negative-definite \(2\times2\) cone and (3.18e) therefore give
\[
\begin{aligned}
\mathbb E^0[W_r\mathbf1_{\{\text{(3.18f)}\}}]
&\le Cr^2\int e^{-c\|u\|^2}(1+\|u\|)^N
 \int_0^{rP_0(u)}[s+rP(u)]^2\,ds\,du\\
&\le Cr^5.
\end{aligned}
\tag{3.18g}
\]
The possible double-eigenvalue locus has lower dimension and is already
covered by the coarea integral. Division by \(Z_r\ge cr^2\) yields a
\(Cr^3\) Palm bound for the shallow layer.

It remains to justify the other two inequalities in (3.18a). The normalized
coefficient maps and the Taylor interpolation constant are polynomially
bounded functions of \(\|U_r\|+H_{5,r}\) on \(\mathcal C\). Choose
\(\alpha>0\) smaller than the reciprocal of all finite polynomial degrees
appearing there and smaller than \(1/5\). On
\(\{\|U_r\|+H_{5,r}\le r^{-\alpha}\}\), reducing \(r_0\) gives both
\(rR_r^5\le\varepsilon_0\) and
\(\|F_{\rm exact}-F_r\|_{C^1}\le\eta_{R_r}\). By (3.18d) the complement
has conditional probability at most \(Ce^{-c r^{-2\alpha}}\). Cauchy--Schwarz,
(3.18e), and \(Z_r\ge cr^2\) imply for either tail event \(B\)
\[
\mathbb P^{MS}_r(B)
\le C\,[\mathbb P^0_r(B)]^{1/2}=o(r^3).
\tag{3.18h}
\]
Together with (3.18g), this proves (3.18b).

On \(\mathcal D_r\), Module J's deterministic theorem sends one upper
half-branch to \(M\); hence \(A_r\) holds. The other half-branch remains in
the other upper sector and reaches a point with value greater than \(b\). If
the class born at \(M\) were alive and \(C_1=C_2\), that same component would
contain a point above \(b\), contradicting that its elder representative is
\(M\). If case 3 of Lemma 3.4a held, the other component would likewise
contain a point above \(b\), contradicting \(f(N)<b\). Thus neither
\(E_{\rm self}\) nor \(E_{\rm young}\) occurs on \(\mathcal D_r\), and
(3.18c) follows from (3.18b). \(\square\)

**Lemma 3.4c (fixed-distance witness bound).** Fix \(\delta>0\). Let
\(N_{\rm fixed}\) count critical points \(y\) satisfying
\(d(y,\{M,S\})\ge\delta\) and \(h<f(y)<b\), without a type restriction.
Uniformly on compact positive mark sets,
\[
\mathbb E^{MS}_{r,t,b,\kappa}N_{\rm fixed}
\le Cr^3[1+(|b|+\kappa)^{10}]e^{-c(b^2+\kappa^2)}.
\tag{3.18i}
\]

**Proof.** Join the corrected pair pins to \((f(y),\nabla f(y))\). At
contact the pair-pin distributions are supported at the midpoint while the
remote value-gradient distributions are supported at \(y\); full Fourier
support makes the combined family nondegenerate. Continuity and compactness
for \(d(y,\{M,S\})\ge\delta\) give a uniform covariance eigenfloor. Gaussian
regression therefore bounds the joint target density by
\(Ce^{-c(b^2+\kappa^2)}\) and the conditional product of the three Hessian
determinants by
\(Cr^2[1+(|b|+\kappa)^9]\); the \(r^2\) is the exact pair-endpoint soft
factor. Kac--Rice integration over the remote value window of length
\(b-h=\kappa r^3/6\) and over the finite-volume separated configuration
region gives an unnormalized bound
\(Cr^5[1+(|b|+\kappa)^{10}]e^{-c(b^2+\kappa^2)}\). Divide by
\(Z_r\ge cr^2\). \(\square\)

Lemma 3.4b controls adjacency failure, the live same-component case, and the
younger-other-component case. Lemma 3.4c controls the fixed-distance canonical
witness. Thus, up to an \(O(r^3)\) Palm event, failure of elder selection is
reduced to the earlier-death saddle of case 1 lying in the collar or
singular-near region. To complete Proposition 3.4 it remains to prove the
uniform finite-\(r\) side-24 triple-contact bounds and carry out those two
regional integrations. Accordingly (3.13) remains the principal open
proposition rather than being cited as established.

### 3.5. Domination, the conditional selected limit, and far pairs

**Lemma 3.5a (uniform all-mark majorant).** There are \(r_0,c,C>0\) and an integer \(N\) such that, for \(0<r<r_0\),
\[
0\le \mathcal G_r^{\mathrm{sel}}
\le \mathcal G_r^{\mathrm{all}}
\le C[1+(|b|+\kappa)^N]e^{-c(b^2+\kappa^2)}
\tag{3.19}
\]
uniformly in \(t\in S^2\), \(b\in\mathbb R\), and \(\kappa>0\). Consequently \(\kappa^{-2/3}\mathcal G_r^{\mathrm{sel}}\) and \(\kappa^{-2/3}\mathcal G_r^{\mathrm{all}}\) have a common integrable majorant on \(S^2\times\mathbb R\times(0,\infty)\).

**Proof.** Let \(p_{Y_r}\) denote the corrected-pin density and let \(y_r(b,\kappa)\) be its target. The raw-to-corrected Jacobian and (3.9d) give
\[
p_{\Pi_r}(z_r(b,\kappa))
=r^{-6}p_{Y_r}(y_r(b,\kappa)),
\qquad
Z_r=r^2\zeta_r.
\tag{3.19e}
\]
Therefore
\[
\mathcal G_r^{\mathrm{all}}
=r^4\mathcal K_r^{\mathrm{all}}
=p_{Y_r}(y_r(b,\kappa))\,\zeta_r(t,b,\kappa).
\tag{3.19f}
\]
In the corrected frame, covariance matrices and their inverses are uniformly bounded for small \(r\) on a finite frame atlas over \(S^2\). Uniform injectivity of \((b,\kappa)\mapsto y_r(b,\kappa)\) gives
\[
p_{Y_r}(y_r(b,\kappa))\le Ce^{-c(b^2+\kappa^2)}.
\]
Gaussian regression and the exact endpoint determinant factorizations give
\[
\zeta_r(t,b,\kappa)
=r^{-2}\mathbb E^0_{r,t,b,\kappa}[W_r]
\le C[1+(|b|+\kappa)^N].
\]
Combining these inequalities proves (3.19). The inequality between selected and all-typed factors is \(0\le\mathfrak e\le1\). Finally,
\[
\int_0^1\kappa^{-2/3}d\kappa=3,
\]
while the Gaussian factor controls the tails in \(b\) and \(\kappa\). Compactness of the frame atlas gives uniformity in \(t\). \(\square\)

Assume Proposition 3.4 and fix \(0<\rho<\min\{12,r_0\}\). For each fixed \((t,b,\kappa)\in S^2\times\mathbb R\times(0,\infty)\),
\[
r_\ell(\kappa)\longrightarrow0,
\qquad
\mathbf 1_{\{\kappa\ge6\ell/\rho^3\}}
\mathcal G_{r_\ell(\kappa)}^{\mathrm{sel}}(t,b,\kappa)
\longrightarrow\mathcal G_0^{\mathrm{all}}(t,b,\kappa).
\]
Indeed, the all-typed corrected-frame covariance expansion gives \(\mathcal G_r^{\mathrm{all}}\to\mathcal G_0^{\mathrm{all}}\), while Proposition 3.4 gives \(p_r\to1\) on compact positive mark sets. Lemma 3.5a and dominated convergence therefore yield
\[
\begin{aligned}
&\lim_{\ell\downarrow0}
\int_{S^2}\int_{\mathbb R}\int_0^\infty
\mathbf 1_{\{\kappa\ge6\ell/\rho^3\}}
\kappa^{-2/3}
\mathcal G_{r_\ell(\kappa)}^{\mathrm{sel}}(t,b,\kappa)
\,d\kappa\,db\,d\sigma_2(t)\\
&\qquad=
\int_{S^2}\int_{\mathbb R}\int_0^\infty
\kappa^{-2/3}\mathcal G_0^{\mathrm{all}}(t,b,\kappa)
\,d\kappa\,db\,d\sigma_2(t).
\end{aligned}
\tag{3.19a}
\]
This is the exact conditional point at which the all-typed coefficient becomes the elder-selected coefficient. All pairs with separation at least \(\rho\) are handled by Lemma 3.5b.

**Lemma 3.5b (off-diagonal elder pairs).** Fix \(\rho>0\) and \(\ell_0<\infty\). Let \(\nu_\rho^{\mathrm{far}}\) be the lifetime-density contribution from elder pairs \((M,S)\) with \(d(M,S)\ge\rho\). Then
\[
\sup_{0<\ell\le\ell_0}\nu_\rho^{\mathrm{far}}(\ell)<\infty,
\qquad
\nu_\rho^{\mathrm{far}}(\ell)=o(\ell^{-1/3}).
\tag{3.19b}
\]

**Proof.** Work directly on the compact separated configuration set
\(D_\rho=\{(x,y):d(x,y)\ge\rho\}\); no logarithm chart or restriction \(r<12\) is used. By (1.3), the vector
\[
(f(x),f(y),\nabla f(x),\nabla f(y))
\]
is nondegenerate for every \((x,y)\in D_\rho\), and compactness gives a uniform covariance eigenfloor. Two-point Kac–Rice and Lemma 1.2a give the representative
\[
\begin{aligned}
\nu_\rho^{\mathrm{far}}(\ell)
=24^{-3}\int_{D_\rho}\int_{\mathbb R}
&p_{x,y}(b,b-\ell,0,0)\\
&\times\mathbb E[W_{x,y}\mathfrak e(f;x,y)
\mid f(x)=b,f(y)=b-\ell,\nabla f(x)=\nabla f(y)=0]
\,db\,dx\,dy.
\end{aligned}
\tag{3.19c}
\]
Here \(p_{x,y}\) is the joint density of \((f(x),f(y),\nabla f(x),\nabla f(y))\), and
\[
W_{x,y}=|\det H_x|\,|\det H_y|
\mathbf 1_{\{H_x\prec0\}}
\mathbf 1_{\{\operatorname{ind}(H_y)=2\}}.
\]
Uniform Gaussian regression bounds the integrand, for \(0<\ell\le\ell_0\), by
\[
C_\rho(1+|b|+\ell_0)^N e^{-c_\rho b^2},
\tag{3.19d}
\]
because the conditional Hessian means are affine in \((b,b-\ell)\), their covariances are uniformly bounded, and \(0\le\mathfrak e\le1\). The bound is integrable in \(b\), and \(D_\rho\) has finite volume. This proves the first assertion; multiplying by \(\ell^{1/3}\) proves the second. \(\square\)

### 3.6. Exact coefficient reduction

The periodized kernel is even: replacing \(n\) by \(-n\) in (1.1) gives \(K_{24}(-z)=K_{24}(z)\). Hence all odd-order derivatives at the origin are uncorrelated with all even-order derivatives, and the corresponding Gaussian blocks are independent. Integrating over the birth height removes the field-value coordinate:
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
Identity (3.20) is the disintegration formula followed by Tonelli's theorem; it does not assume that \(q_t\) is independent of \(f\).
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
\det\mathcal Q_t(v)>0\}}\,d\sigma_2(v).
\tag{3.22}
\]
Thus \(c_{3,24}\) is a deterministic nested angular integral involving only the order-six covariance jet at the origin.

### 3.7. Euclidean (nonperiodized) value

At the Euclidean Bargmann–Fock jet,
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
where \(Z\sim N(0,1)\) is independent and the convention for standard \(G_2\) is: the two diagonal entries are independent \(N(0,2)\), the off-diagonal entry is \(N(0,1)\), and these three variables are independent. Substitution in (2.9) gives (2.11). The symbolic annex verifies this value under exactly this convention; a journal proof should retain the full cone integration rather than cite a pass marker.

For completeness, here is the cone calculation. Write
\[
G_2=\begin{pmatrix}X&Y\\Y&W\end{pmatrix},
\]
where \(X,W\sim N(0,2)\) and \(Y\sim N(0,1)\) are independent, and put
\[
m=\frac{Q_{11}+Q_{22}}2,
\qquad d=\frac{Q_{11}-Q_{22}}2,
\qquad R=\sqrt{d^2+Q_{12}^2}.
\tag{3.24a}
\]
Then \(m\sim N(0,5/3)\), while \(d,Q_{12}\) are independent standard
Gaussians, independent of \(m\). Thus \(R\) is Rayleigh with density
\(re^{-r^2/2}\), and the two eigenvalues of \(Q\) are \(m-R\) and \(m+R\).
Consequently
\[
Q\prec0\quad\Longleftrightarrow\quad m<-R,
\qquad \det Q=m^2-R^2.
\tag{3.24b}
\]
If \(\phi_\tau\) denotes the density of \(N(0,\tau^2)\), with
\(\tau^2=5/3\), symmetry and the substitution \(x=-m\) give
\[
D_t=\int_0^\infty\phi_\tau(x)
\int_0^x(x^2-r^2)^2r e^{-r^2/2}\,dr\,dx.
\tag{3.24c}
\]
The inner integral is
\[
\int_0^x(x^2-r^2)^2r e^{-r^2/2}\,dr
=x^4-4x^2+8-8e^{-x^2/2}.
\]
Using \(\mathbb E X^4=3\tau^4\) and
\(\mathbb E e^{-X^2/2}=(1+\tau^2)^{-1/2}\) for
\(X\sim N(0,\tau^2)\), (3.24c) becomes
\[
D_t
=\frac12(3\tau^4-4\tau^2+8)
-\frac4{\sqrt{1+\tau^2}}
=\frac{29}{6}-\sqrt6.
\tag{3.24d}
\]
This supplies the analytic derivation; the symbolic annex is an independent
arithmetic check of the same value and convention.

### 3.8. Side-24 correction

Let \(J_0\) be the Euclidean even covariance jet through order six,
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
Here is the exact first-variation calculation. For the rotationally projected
even jet, let
\[
\begin{gathered}
a=\mathbb E\xi_1^2,\\
m_4=\mathbb E\xi_1^4,\\
\chi=\mathbb E\xi_1^6,\\
\Omega=a\chi-m_4^2,
\end{gathered}
\tag{3.31a}
\]
where \(\xi\) is the normalized spectral variable. At the Euclidean
Bargmann--Fock jet,
\((a,m_4,\chi,\Omega)=(1,3,15,6)\). The isotropic specialization of (2.9)
has jet dependence
\[
c_3\ \propto\ \Omega^{2/3}m_4^{1/2}a^{-13/6},
\tag{3.31b}
\]
so at that jet
\[
\delta\log c_3
=\frac{\delta\Omega}{9}+\frac{\delta m_4}{6}
-\frac{13}{6}\delta a,
\qquad
\delta\Omega=15\delta a+\delta\chi-6\delta m_4.
\tag{3.31c}
\]
After factoring out \(e^{-L^2/2}\), the six axial images
\(\{\pm Le_i:1\le i\le3\}\) have rotationally averaged moment variations
\[
\begin{gathered}
\delta a=-2L^2,\\
\delta m_4=\frac65L^2(L^2-10),
\end{gathered}
\tag{3.31d}
\]
\[
\delta\chi
=\frac{30}{35}L^2(-L^4+21L^2-105).
\tag{3.31e}
\]
These identities follow directly by differentiating
\(e^{-|z\pm Le_i|^2/2}\) at \(z=0\) and using the spherical averages of
second, fourth, and sixth powers. Because \(F\) is \(O(3)\)-invariant and
\(J_0\) is fixed by \(O(3)\), \(DF(J_0)\) depends only on this rotational
projection. Substitution of (3.31d)--(3.31e) into (3.31c) gives
\[
\delta\log c_3
=-\frac{2}{21}L^6+\frac75L^4-3L^2=P_3(L),
\tag{3.31f}
\]
which proves (3.30)--(3.31) under the same normalized jet convention as
\(F\). The symbolic ledger is an exact independent expansion check.

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
which is (2.10), conditional on the quantitative jet-stability bounds (3.27)--(3.29). Together with Lemmas 3.3 and 3.5, this completes the analytic reduction of Candidate Theorem 2.1 to Proposition 3.4 and the stated jet-transfer envelope. It does not close those remaining obligations. \(\square\)

## 4. Exact scope

Candidate Theorem 2.1 is restricted to:

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

1. complete the finite-\(r\) collar and singular-near triple-contact bounds, including the joint side-24 blow-ups and angular integrations;
2. have a human specialist independently check Lemmas 3.4a--3.4c and the energy-adapted matrix theorem;
3. complete the systematic database and backward/forward citation search for short-lifetime persistence asymptotics of smooth Gaussian fields and local maximum–saddle Kac–Rice pair laws;
4. convert the review technical appendix into conventionally numbered lemmas and proofs, eliminating provenance labels from the submitted version;
5. complete a journal-specific notation, bibliography, and style pass;
6. optionally produce directed-rounding bounds for the nonlinear integral (2.9). This would strengthen numerical certification but is not used in the analytic estimate (2.10).

## References for the literature audit

1. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer Monographs in Mathematics, Springer, 2007.
2. R. J. Adler, O. Bobrowski, M. S. Borman, E. Subag, and S. Weinberger, “Persistent homology for random fields and complexes,” *IMS Collections* **6** (2010), 124–143. <https://projecteuclid.org/ebooks/institute-of-mathematics-statistics-collections/Borrowing-Strength--Theory-Powering-Applications--A-Festschrift-for/chapter/Persistent-homology-for-random-fields-and-complexes/10.1214/10-IMSCOLL609>
3. F. Chazal and V. Divol, “The density of expected persistence diagrams and its kernel based estimation,” *Journal of Computational Geometry* **10**(2) (2019), 127–153. <https://jocg.org/index.php/jocg/article/download/3090/2817/>
4. A. Klein and O. Agam, “Critical point correlations in random Gaussian fields,” *Journal of Physics A: Mathematical and Theoretical* **45** (2012), 025001. <https://arxiv.org/abs/1111.5286>
5. P. Pranav, “Topology and geometry of Gaussian random fields II: on critical points, excursion sets, and persistent homology,” arXiv:2109.08721 (2021). <https://arxiv.org/abs/2109.08721>
6. C. Hirsch and R. Lachièze-Rey, “Functional central limit theorem for topological functionals of Gaussian critical points,” arXiv:2411.11429 (2024). <https://arxiv.org/abs/2411.11429>
7. M. Ancona, L. Gass, T. Letendre, and M. Stecconi, “Zeros and critical points of Gaussian fields: cumulants asymptotics and limit theorems,” arXiv:2501.10226 (2025), revised 2025. <https://arxiv.org/abs/2501.10226>
