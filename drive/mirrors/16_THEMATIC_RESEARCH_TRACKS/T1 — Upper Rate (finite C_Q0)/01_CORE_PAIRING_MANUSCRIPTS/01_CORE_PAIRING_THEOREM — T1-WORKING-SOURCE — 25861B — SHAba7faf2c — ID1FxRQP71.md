# A cubic upper bound for elder-rule mispairing near a maximum–saddle fold  
## Candidate theorem for the periodized Bargmann–Fock field

## Abstract

Let \(f\) be the exact normalized periodized Bargmann–Fock Gaussian field on the flat torus \(\mathbb T_{24}^2\). Fix a local maximum \(M\) and an index-one saddle \(S\) at separation \(r\), with

\[
f(M)=\frac65,\qquad
f(S)=\frac65-\frac{r^3}{6}.
\]

For superlevel-set \(H_0\) persistence, let \(D(M)\) denote the saddle at which the component born at \(M\) dies under the elder rule. The proposed conclusion is that, for a configured gradient-adjacent maximum–saddle pair and under the Morse–Smale hypothesis,

\[
P\{D(M)\neq S\}\le Cr^3
\]

for some finite \(C\) and \(0<r\le0.025\).

The proof has three layers:

1. a deterministic elder-rule dichotomy;
2. exact pair-Palm Kac–Rice formulas;
3. a regional analysis of third critical points near a collapsing fold.

Several algebraic and finite-dimensional Gaussian lemmas are written in full. The remaining delicate issues are explicitly identified: the typed-versus-adjacent Palm measure, uniform remainder estimates in the singular charts, and the passage from the exact leading cubic identities to integrable determinant-weighted bounds.

No numerical value of \(C\) is claimed.

---

# 1. The Gaussian field

## 1.1 The torus and covariance

Write

\[
\mathbb T_L^2=\mathbb R^2/(L\mathbb Z)^2,
\qquad L=24.
\]

The field is centered and stationary, with covariance

\[
K_L(x)
=
\frac{
\displaystyle
\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
e^{2\pi i k\cdot x/L}
}{
\displaystyle
\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
}.
\]

All Fourier weights are strictly positive. The field is real analytic almost surely.

For multi-indices \(\alpha,\beta\),

\[
\operatorname{Cov}
\left(
\partial^\alpha f(x),
\partial^\beta f(y)
\right)
=
(-1)^{|\beta|}
\partial^{\alpha+\beta}K_L(x-y).
\]

The planar covariance approached as \(L\to\infty\) is

\[
K_\infty(x)=e^{-|x|^2/2}.
\]

The fixed-torus theorem in this file uses \(L=24\) only.

---

## 1.2 Finite-jet nondegeneracy

The following fact is used repeatedly.

### Lemma 1.1 — finite derivative evaluations are nondegenerate

Let \(x_1,\ldots,x_m\) be distinct points of \(\mathbb T_L^2\), and let \(A_j\) be finite sets of multi-indices. Then the Gaussian vector

\[
\left(
\partial^\alpha f(x_j)
\right)_{j,\alpha\in A_j}
\]

has a positive-definite covariance matrix, provided duplicate functionals are removed.

### Proof

For coefficients \(c_{j,\alpha}\), define

\[
\Lambda f
=
\sum_{j,\alpha}
c_{j,\alpha}\partial^\alpha f(x_j).
\]

The Fourier representation gives

\[
\operatorname{Var}(\Lambda f)
=
\frac1{Z_L}
\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
\left|
\sum_{j,\alpha}
c_{j,\alpha}
\left(\frac{2\pi i k}{L}\right)^\alpha
e^{2\pi i k\cdot x_j/L}
\right|^2.
\]

Every weight is positive. Zero variance therefore implies

\[
\sum_j p_j(k_1,k_2)z_j^{k_1}y_j^{k_2}=0
\]

for every \((k_1,k_2)\in\mathbb Z^2\), where \(p_j\) is a polynomial and

\[
z_j=e^{2\pi i x_{j,1}/L},
\qquad
y_j=e^{2\pi i x_{j,2}/L}.
\]

A one-dimensional exponential-polynomial sequence

\[
\sum_j p_j(n)z_j^n
\]

cannot vanish for all integers \(n\ge0\) unless all \(p_j\) vanish, when the \(z_j\) are distinct. One proof uses the generating function: each term has a pole only at \(z_j^{-1}\), and distinct bases produce distinct poles. Applying this first in \(k_1\), grouped by the first-coordinate phases, and then in \(k_2\), forces every polynomial \(p_j\) to vanish. Hence every coefficient \(c_{j,\alpha}\) is zero. \(\square\)

### Conditional consequence

If a joint finite-jet covariance is positive definite, every Schur complement obtained by conditioning on a subvector is positive definite. Thus all fixed-\(r\), distinct-point Kac–Rice densities used below are well defined.

At \(r\downarrow0\), raw coordinates become ill conditioned; corrected divided differences are introduced in Section 6.

---

# 2. Superlevel persistence and the configured pair

## 2.1 Elder-rule \(H_0\) persistence

For \(t\in\mathbb R\), define the superlevel set

\[
E_t=\{x\in\mathbb T_L^2:f(x)\ge t\}.
\]

At a local maximum, a connected component is born. At an index-one saddle, two components may merge. Under the elder rule, the component with the lower birth maximum dies.

Assume for now:

1. \(f\) is Morse;
2. its critical values are distinct;
3. its gradient flow is Morse–Smale.

Then each non-essential component has a unique birth maximum and death saddle. Denote the death saddle of the component born at \(M\) by \(D(M)\).

The transversality manuscript in `02_GAUSSIAN_TRANSVERSALITY.md` addresses whether the third assumption holds almost surely for the Bargmann–Fock field.

---

## 2.2 Pair geometry

By stationarity, put the pair midpoint at the origin and orient the pair along the \(x\)-axis:

\[
M=(-r/2,0),
\qquad
S=(r/2,0).
\]

Fix

\[
b=\frac65,
\qquad
\ell=\frac{r^3}{6},
\]

and impose

\[
f(M)=b,\qquad \nabla f(M)=0,
\]

\[
f(S)=b-\ell,\qquad \nabla f(S)=0.
\]

The event that \(M\) is a maximum and \(S\) a saddle is

\[
\mathcal T_r
=
\{H_M\prec0,\ \det H_S<0\}.
\]

A topologically configured pair also requires that one ascending separatrix of \(S\) terminates at \(M\). Denote this adjacency event by \(\mathcal A_r\).

There are therefore two related Palm laws.

---

## 2.3 Typed pair-Palm law

Let \(\gamma_r\) be the Gaussian law conditioned on the six linear value and gradient constraints above. Define

\[
W_{MS}
=
|\det H_M\det H_S|
\mathbf 1_{\mathcal T_r},
\]

and

\[
Z_r=E_{\gamma_r}W_{MS}.
\]

The typed maximum–saddle pair-Palm law is

\[
dP_r^{MS}
=
\frac{W_{MS}}{Z_r}\,d\gamma_r.
\]

This is the law naturally produced by the two-point Kac–Rice formula.

---

## 2.4 Adjacent configured-pair law

The intended topological object is

\[
dP_r^{\mathrm{adj}}
=
\frac{
W_{MS}\mathbf 1_{\mathcal A_r}
}{
E_{\gamma_r}[W_{MS}\mathbf 1_{\mathcal A_r}]
}
\,d\gamma_r.
\]

Define

\[
q(r)=P_r^{\mathrm{adj}}\{D(M)=S\}.
\]

### Important measure issue

Most explicit local Gaussian calculations below are first written under \(P_r^{MS}\), because type indicators and determinant weights are local jet marks. Passing to \(P_r^{\mathrm{adj}}\) additionally requires control of

\[
a_r
=
P_r^{MS}(\mathcal A_r)
\]

and of regional counts reweighted by \(\mathbf 1_{\mathcal A_r}\).

The available argument is that adjacency is \(C^1\)-stable away from saddle connections, the limiting fold has an open trapping configuration, and the saddle-connection boundary is null under the transversality theorem. This suggests

\[
0<c\le a_r\le1
\]

for small \(r\), and hence only a bounded change of constants.

That transfer is not developed here as a full quantitative lemma. A reviewer should decide whether it is immediate from the stated stability and compactness arguments or whether it is a genuine missing step.

To isolate the issue, the remainder of the manuscript proves the regional \(O(r^3)\) estimates under the typed Palm law. The adjacent theorem follows if the adjacency reweighting has a uniform bounded density ratio on the relevant marked events.

---

# 3. Deterministic elder-rule reduction

Define two defect events.

### Earlier-interceptor event

Let \(\Pi_r\) be the event that the component born at \(M\) merges at some index-one saddle \(S'\) with

\[
f(S)<f(S')<f(M)
\]

before the filtration reaches \(S\).

### Direct-gain event

Let \(\Gamma_r\) be the event that the second ascending branch of \(S\) terminates at a maximum \(N\) satisfying

\[
f(S)<f(N)<f(M).
\]

### Proposition 3.1 — defect dichotomy

Under the Morse, distinct-critical-value, and Morse–Smale hypotheses,

\[
\boxed{
\{D(M)\neq S\}\subset \Pi_r\cup\Gamma_r.
}
\]

### Proof

If \(M\) dies before level \(f(S)\), its death saddle lies in the open value interval, so \(\Pi_r\) occurs.

Suppose \(M\) is alive immediately above \(f(S)\).

If the two local arms of \(S\) already belong to the same superlevel component, then that connection appeared after the birth of \(M\) and before the level of \(S\). For a Morse function with distinct critical values, superlevel connectivity changes only at index-one saddle values. The first such connection therefore occurs at a saddle \(S'\) with

\[
f(S)<f(S')<f(M),
\]

so \(\Pi_r\) occurs.

Otherwise the two arms belong to distinct components immediately above \(S\). Crossing \(S\) merges them. If \(M\) does not die at this merge, its component survives by the elder rule. Hence the elder birth maximum of the other component is lower than \(f(M)\). The maximum \(N\) at which the second ascending branch terminates belongs to that component, and

\[
f(S)<f(N)<f(M).
\]

Thus \(\Gamma_r\) occurs. \(\square\)

Consequently,

\[
1-q(r)
\le
P(\Pi_r)+P(\Gamma_r),
\]

up to the typed-to-adjacent measure issue isolated in Section 2.4.

---

# 4. Kac–Rice count objects

## 4.1 Third maxima

For a Borel set \(D\subset\mathbb T_L^2\) and interval \(I\), let

\[
N_{\max}(I;D)
\]

be the number of local maxima \(x\in D\) with \(f(x)\in I\).

At a third point \(x\), define

\[
W_x^{\max}
=
|\det H_x|\mathbf 1_{\{H_x\prec0\}}.
\]

The typed pair-Palm maximum intensity is

\[
\lambda_{\max}^{MS}(x,u;r)
=
p_{(f(x),\nabla f(x))\mid J_6}(u,0)
\frac{
E[
W_{MS}W_x^{\max}
\mid
J_6,f(x)=u,\nabla f(x)=0
]
}{
Z_r
},
\]

where \(J_6\) denotes the six pair pins.

Therefore

\[
E_r^{MS}N_{\max}(I;D)
=
\int_D\int_I
\lambda_{\max}^{MS}(x,u;r)\,du\,dx.
\]

On \(\Gamma_r\), the terminal maximum satisfies

\[
b-\ell<f(N)<b.
\]

Hence, pathwise,

\[
\mathbf 1_{\Gamma_r}
\le
N_{\max}((b-\ell,b);\mathbb T_L^2),
\]

and

\[
P_r^{MS}(\Gamma_r)
\le
E_r^{MS}N_{\max}((b-\ell,b);\mathbb T_L^2).
\]

This overcounts all maxima in the window and requires no differentiable selection of the branch terminal.

---

## 4.2 Third saddles or all critical points

Let

\[
W_x^{\mathrm{sad}}
=
|\det H_x|\mathbf 1_{\{\det H_x<0\}},
\]

or, for a coarser upper bound,

\[
W_x^{\mathrm{crit}}=|\det H_x|.
\]

The same Kac–Rice formula defines the corresponding intensities.

If \(\Pi_r\) occurs, at least one saddle has value in \((b-\ell,b)\). Thus

\[
\mathbf 1_{\Pi_r}
\le
N_{\mathrm{sad}}((b-\ell,b);\mathbb T_L^2)
\le
N_{\mathrm{crit}}((b-\ell,b);\mathbb T_L^2).
\]

It is therefore enough to show that the expected number of window-valued critical points is \(O(r^3)\) under the pair-Palm law.

---

# 5. Pair-Palm normalizer

The pair-Palm denominator loses powers as the two critical points coalesce. Those powers must be identified before any third-point count is estimated.

### Proposition 5.1

There exists \(c_Z>0\) such that

\[
Z_r\ge c_Zr^2
\]

for \(0<r\le0.025\).

Moreover,

\[
\frac{Z_r}{r^2}\longrightarrow
z_0,
\]

where

\[
z_0
=
E[Q^2\mathbf 1_{\{Q<0\}}],
\qquad
Q\sim N(-b,2).
\]

The closed form is

\[
z_0
=
(b^2+2)\Phi\!\left(\frac b{\sqrt2}\right)
+
\sqrt2\,b\,\phi\!\left(\frac b{\sqrt2}\right).
\]

At \(b=6/5\),

\[
z_0\approx3.230978535.
\]

### Argument

The corrected fold coordinates force the longitudinal curvatures to have opposite first-order signs:

\[
H_M=
\begin{pmatrix}
-r+O(r^2)&O(r)\\
O(r)&Q+O(r)
\end{pmatrix},
\]

\[
H_S=
\begin{pmatrix}
+r+O(r^2)&O(r)\\
O(r)&Q+O(r)
\end{pmatrix}.
\]

The limiting transverse curvature has conditional law

\[
Q\sim N(-b,2).
\]

On an event

\[
Q\le-\varepsilon
\]

with the normalized mixed and remainder jets bounded, \(M\) is a maximum, \(S\) is a saddle, and

\[
|\det H_M\det H_S|
\ge c\varepsilon^2r^2.
\]

The event has positive limiting Gaussian probability, giving the lower bound. Dominated convergence in the corrected Gaussian frame gives the stated limit.

For \(r\) bounded away from zero, positivity follows from finite-jet nondegeneracy, positivity of the open type region, and compactness.

### Point for review

The displayed Hessian expansions are derived from corrected divided differences. A fully formal proof should state the coordinate map and uniform remainder bounds explicitly. The limiting formula itself is an exact finite-dimensional Gaussian calculation.

---

# 6. Corrected collapsing-pair coordinates

The raw six pin vector is

\[
X_r=
\left(
f(M),f(S),
f_x(M),f_y(M),
f_x(S),f_y(S)
\right).
\]

A useful corrected frame consists of:

\[
V_+=\frac{f(M)+f(S)}2,
\]

\[
V_-=
\frac{f(S)-f(M)}{r^3}
-
\frac{f_x(M)+f_x(S)}{2r^2},
\]

\[
G_{x,+}=\frac{f_x(M)+f_x(S)}2,
\qquad
G_{x,-}=\frac{f_x(S)-f_x(M)}r,
\]

\[
G_{y,+}=\frac{f_y(M)+f_y(S)}2,
\qquad
G_{y,-}=\frac{f_y(S)-f_y(M)}r.
\]

The linear transformation from the raw pins has determinant

\[
-r^{-5},
\]

so it is invertible for every \(r>0\).

Its limiting covariance is the covariance of a six-jet vector such as

\[
(f,f_{xxx},f_x,f_{xx},f_y,f_{xy}),
\]

whose determinant is positive. Analyticity of the covariance and the Hermite–Genocchi representation of divided differences imply analytic extension through \(r=0\).

This gives two standard consequences:

1. corrected conditional means and covariances remain bounded and continuous;
2. a positive limiting covariance matrix gives a positive covariance floor for small \(r\).

For the remaining closed interval away from zero, finite-jet nondegeneracy and compactness give a positive floor.

---

# 7. Type-indicator continuity

Let \(X_\theta\sim N(m_\theta,\Sigma_\theta)\) be a compact continuous family with

\[
0<\lambda_*
\le
\lambda_{\min}(\Sigma_\theta)
\le
\lambda_{\max}(\Sigma_\theta)
\le
\Lambda_*.
\]

If \(P\) is a polynomial and \(A\) is defined by finitely many strict polynomial inequalities, then

\[
\theta\mapsto
E[P(X_\theta)\mathbf 1_A(X_\theta)]
\]

is continuous whenever the expectation is absolutely integrable.

Indeed, write

\[
X_\theta=m_\theta+\Sigma_\theta^{1/2}Z.
\]

The indicator converges almost surely except when a boundary polynomial vanishes. A nonzero polynomial has zero probability of vanishing under a nondegenerate Gaussian law. Uniform polynomial Gaussian moments give dominated convergence.

For a two-dimensional Hessian,

\[
H\prec0
\iff
H_{11}<0,\quad\det H>0,
\]

and

\[
H\text{ saddle}
\iff
\det H<0.
\]

Thus maximum and saddle marks fit this lemma.

This justifies compactness of typed determinant moments on each positive-definite corrected chart. It does not justify crossing a rank-loss face without a separate chart.

---

# 8. Regional decomposition

Use

\[
\mathcal C_r
=
B(M,2r)\cup B(S,2r),
\]

\[
\mathcal N_r
=
B_3(0)\setminus\mathcal C_r,
\]

\[
\mathcal E
=
\mathbb T_{24}^2\setminus B_3(0).
\]

The radius \(3\) is fixed in physical units. The proof is regional because the third-point Gaussian law has different degeneracies in the collars, in the collapsing near region, and away from the pair.

---

# 9. Exterior and fixed-annulus bounds

If \(x\) stays a fixed positive distance from \(M,S\), the corrected joint Gaussian family

\[
\left(
f(x),\nabla f(x),H_M,H_S,H_x
\right)\mid J_6
\]

extends continuously to \(r=0\) and remains nondegenerate.

The pair determinant weight has the form

\[
W_{MS}=r^2\widehat W_r,
\]

and

\[
Z_r=r^2z_r,
\qquad
\inf z_r>0.
\]

After additionally conditioning on

\[
f(x)=u,\qquad\nabla f(x)=0,
\]

the Hessian vector has uniformly bounded Gaussian moments. Therefore

\[
r^{-2}
E[
W_{MS}W_x^{\max}
\mid J_6,f(x)=u,\nabla f(x)=0
]
\]

is bounded, and so is the pair-Palm intensity.

Since the window width is

\[
\ell=\frac{r^3}{6},
\]

integration over a compact fixed-distance region gives \(O(r^3)\).

### Status

This is a standard compactness proof once the corrected frame, finite-jet nondegeneracy, and type-continuity lemma are accepted. No explicit numerical covariance floor is supplied.

---

# 10. Collar count

Inside \(\mathcal C_r\), use scaled coordinates

\[
y=r(X,Y),
\qquad
M=(-1/2,0),
\qquad
S=(1/2,0).
\]

Under the coalesced six-pin law, the free leading jets are

\[
q=f_{yy},\qquad
a=f_{xxy},\qquad
w=f_{xyy},\qquad
z=f_{yyy},
\]

with means

\[
(-b,0,0,0)
\]

and covariance

\[
\operatorname{diag}(2,2,2,6).
\]

The scaled third gradient is

\[
\frac{f_x(rX,rY)}{r^2}
\to
X^2-\frac14+aXY+\frac w2Y^2,
\]

\[
\frac{f_y(rX,rY)}r
\to
qY.
\]

The limiting mean is

\[
\left(X^2-\frac14,-bY\right),
\]

and the covariance is

\[
\begin{pmatrix}
Y^2(4X^2+Y^2)/2&0\\
0&2Y^2
\end{pmatrix}.
\]

Hence the raw third-gradient density has the scale

\[
r^{-3}
\frac{
\exp\left[
-\dfrac{(X^2-1/4)^2}{Y^2(4X^2+Y^2)}
-\dfrac{b^2}{4}
\right]
}{
2\pi |Y|^2\sqrt{4X^2+Y^2}
}.
\]

The apparent singularity on \(Y=0\) is exponentially suppressed away from \(X=\pm1/2\).

At \(X=0\), the third-station equation forces

\[
w=\frac1{2Y^2}.
\]

Because \(w\sim N(0,2)\) at leading order, this produces

\[
\exp\left(-\frac1{16Y^4}\right)
\]

suppression.

---

## 10.1 Three-determinant collision factor

The leading cubic field is

\[
P(x,y)
=
\frac{x^3}{3}-\frac x4-\frac1{12}
+
\frac a2\left(x^2-\frac14\right)y
+
\frac Q2y^2
+
\frac w2xy^2
+
\frac z6y^3.
\]

Impose a third station at \((X,Y)\):

\[
P_x(X,Y)=0,
\qquad
P_y(X,Y)=0.
\]

For \(XY\neq0\), solving for \(a,Q\) gives

\[
a=
-\frac{4X^2+2Y^2w-1}{4XY},
\]

\[
Q
=
-\frac{
-16X^4+24X^2Y^2w+8X^2+16XY^3z+2Y^2w-1
}{
32XY^2
}.
\]

Each physical Hessian equals \(rD^2P+O(r^2)\), so the product of three Hessian determinants is \(r^6\) times a scaled polynomial-rational expression.

Near \(M\), write

\[
(X,Y)=(-1/2,0)+\rho(c,s).
\]

Direct algebra gives

\[
\det D^2P(M)\det D^2P(S)\det D^2P(X,Y)
=
\rho^2R_M(\rho,c,s,w,z).
\]

The same factorization holds near \(S\).

At \(\rho=0\), the generic leading factor is

\[
R_M(0,c,s,w,z)
=
-\frac{
(-2c^2+s^2w)
(-4c^3+3cs^2w+s^3z)^2
}{
4s^6
}.
\]

The gradient density has a \(\rho^{-2}\) collision scale. The determinant product has a \(\rho^2\) zero, so the radial collision powers cancel. The remaining angular poles are paired with the Gaussian suppression displayed above.

Combining:

- gradient density \(r^{-3}\);
- triple determinant moment \(r^6\);
- Palm normalizer \(r^2\);

gives a third-critical-point intensity of the form

\[
\lambda_{\mathrm{crit}}^{MS}(rX,rY;r)
=
rF_r(X,Y),
\]

where \(F_r\) is claimed to have an integrable uniform envelope on the scaled two-collar domain.

Since \(dy=r^2\,dX\,dY\),

\[
E^{MS}N_{\mathrm{crit}}(\mathcal C_r)
\le
Cr^3.
\]

### Point for review

The exact density and determinant factorizations are explicit. The uniform envelope for the full finite-\(r\) remainders is argued by analytic continuation, chart subdivision, and Gaussian moment bounds; it is not accompanied by explicit constants.

---

# 11. Singular near maximum count

Let

\[
t=|x|,
\qquad
\eta=\frac rt,
\qquad
x=t(c,s),
\qquad
c^2+s^2=1.
\]

Outside the two radius-\(2r\) collars,

\[
\eta^2\le\frac4{15}.
\]

Let

\[
\alpha=\frac{b-u}{\ell}\in[0,1].
\]

Write the local field as

\[
f(tX,tY)=b+t^3P(X,Y)+O(t^4),
\]

where the most general cubic satisfying the pair pins is

\[
P
=
\frac{X^3}{3}
-\frac{\eta^2X}{4}
-\frac{\eta^3}{12}
+
\frac a2\left(X^2-\frac{\eta^2}{4}\right)Y
+
\frac Q2Y^2
+
\frac w2XY^2
+
\frac z6Y^3.
\]

Impose a third critical point at \((c,s)\) with

\[
P(c,s)=-\frac{\alpha\eta^3}{6},
\qquad
\nabla P(c,s)=0.
\]

For \(cs\neq0\),

\[
a
=
-\frac{4c^2-\eta^2+2s^2w}{4cs},
\]

\[
Q
=
-\frac{
\eta^2
(8\alpha c\eta-4c^2-4c\eta-\eta^2+2s^2w)
}{
8cs^2
}.
\]

Define

\[
K=-4c^2-4c\eta-\eta^2+2s^2w.
\]

Then the third Hessian determinant is

\[
\boxed{
\det D^2P(c,s)
=
-\frac{\eta^2}{64c^2s^2}
\left[
(K+8\alpha c\eta)^2
+
64\alpha(1-\alpha)c^2\eta^2
\right]
\le0.
}
\]

For \(0<\alpha<1\), it is strictly negative on the generic chart.

On the transverse chart \(c=0,s=1\),

\[
\boxed{
\det D^2P(0,1)
=
-\frac{\eta^2}{4}
\left[
(a+(1-2\alpha)\eta)^2
+
4\alpha(1-\alpha)\eta^2
\right]
\le0.
}
\]

On the pair axis,

\[
P_X(X,0)=X^2-\eta^2/4,
\]

and an additional station at scaled radius one would require \(\eta=2\), excluded by the collar geometry.

Thus a third point in the maximum-value window cannot be a nondegenerate maximum at cubic order. A finite-\(t\) third maximum must be created by quartic remainders near a cubic degeneracy.

The claimed consequence is the determinant bound

\[
|\det H_x|
\mathbf 1_{\{H_x\prec0\}}
\le
Ct^3\mathcal P(J),
\]

where \(\mathcal P(J)\) is a polynomial in normalized Gaussian jets with uniformly bounded moments.

The pair determinants satisfy

\[
|\det H_M|+|\det H_S|
\le
Cr(r+t^2)\mathcal P(J).
\]

The three-point value-gradient density has scale

\[
Ct^{-6}
\]

after the generic/axis chart decomposition. Dividing by \(Z_r\gtrsim r^2\) gives

\[
\lambda_{\max}^{MS}(x,u;r)
\le
C\frac{(r+t^2)^2}{t^3}.
\]

Integrating the height window and polar area:

\[
\frac1{r^3}
E N_{\max}^{\mathrm{near}}
\le
C\int_{cr}^{\delta}
\left(
\frac{r^2}{t^2}+2r+t^2
\right)\,dt,
\]

which is bounded.

### Point for review

The negative sum-of-squares identity is exact. The implication from “no cubic maximum” to the uniform finite-\(r\) determinant bound is the most delicate step in the maximum-count argument. It uses Taylor remainders, localization near the degeneracy sets, and Gaussian small-coordinate estimates. The available exposition gives the power count but not a fully quantified partition of those boundary layers.

---

# 12. Singular window-critical count

The earlier-interceptor count allows all third critical types, so no maximum-type boundary depletion is needed. Instead, the exact third value and gradient constraints produce an \(\eta^2\) factor in each leading Hessian determinant.

With the same cubic \(P\), direct elimination gives

\[
\det D^2P(M)=\eta^2R_M,
\]

\[
\det D^2P(S)=\eta^2R_S,
\]

\[
\det D^2P(c,s)=\eta^2R_x.
\]

Hence

\[
\det D^2P(M)
\det D^2P(S)
\det D^2P(c,s)
=
\eta^6R.
\]

The physical product is

\[
t^6\eta^6R=r^6R.
\]

The three-point value-gradient density is \(O(t^{-6})\), and the Palm denominator is \(O(r^2)\). Thus the per-unit-height intensity is bounded by

\[
Cr^4t^{-6}.
\]

Multiplying by the height-window width \(r^3/6\) and the polar element \(t\,dt\,d\theta\) gives

\[
Cr^7\int_{(\sqrt{15}/2)r}^{\delta}t^{-5}\,dt.
\]

The exact integral is

\[
r^7\int_{(\sqrt{15}/2)r}^{\delta}t^{-5}\,dt
=
\frac{4}{225}r^3
-
\frac{r^7}{4\delta^4}.
\]

Therefore the singular near interceptor contribution is \(O(r^3)\).

### Point for review

The \(\eta^6\) factor is exact algebra. The uniform bound on the reduced polynomial Gaussian moment over the complete generic/axis/transverse atlas is justified by corrected-frame compactness and exponential axis suppression. A reviewer should check whether all overlap regions are genuinely covered.

---

# 13. Assembly

Under the typed pair-Palm law:

- the direct-gain event is bounded by a maximum-window count;
- the earlier-interceptor event is bounded by a window-critical count;
- each count is split into collars, singular near region, fixed annulus, and exterior;
- every regional contribution is \(O(r^3)\).

Therefore there is a finite constant \(C_{MS}\) such that

\[
P_r^{MS}\{D(M)\neq S\}
\le
C_{MS}r^3,
\]

provided the deterministic configured-pair interpretation is made under the same law.

If the adjacency event satisfies a uniform positive probability and the regional count estimates remain uniformly bounded after adjacency reweighting, then

\[
\boxed{
1-q(r)\le Cr^3.
}
\]

Since \(r^3\to0\),

\[
q(r)\to1.
\]

---

# 14. What is fully written and what is not

## Fully written in this manuscript

- the exact model and covariance;
- finite-jet nondegeneracy;
- the deterministic elder-rule dichotomy;
- the pair-Palm Kac–Rice identities;
- corrected-frame invertibility;
- type-indicator continuity;
- the Palm-normalizer limiting law;
- the collar leading density;
- the exact collar collision factor;
- the exact cubic maximum-type no-go;
- the exact \(\eta^6\) window-critical factor;
- the final power integrals.

## Standard inputs not reproved from first principles

- Kac–Rice formulas for marked critical-point processes;
- Bulinskaya-type absence of degenerate critical points and critical-value ties;
- smooth dependence of hyperbolic critical points and invariant manifolds;
- the elder-rule description of Morse superlevel persistence.

## Steps presented as proof sketches requiring specialist scrutiny

1. a uniform typed-to-adjacent Palm transfer;
2. uniform finite-\(r\) remainder control near the cubic maximum-degeneracy sets;
3. full chartwise envelopes for the reduced determinant moments;
4. the assertion that the generic, transverse, axis, and collision charts cover every singular regime without overlap loss.

The proposed theorem should be accepted only if those points are judged complete or repaired.

---

# 15. Numerical checks, not proof

The following computations were used as falsification checks:

- pair-Palm normalizers:
  \[
  Z_{0.05}=0.00808418,\quad
  Z_{0.025}=0.00201529,\quad
  Z_{0.0125}=0.000503621,
  \]
  giving \(Z_r/r^2\approx3.22\)–\(3.23\);

- exterior pair-Palm maximum-window intensities were between \(0.9847\) and \(1.0103\) times the stationary maximum-height benchmark at sampled stations;

- exterior saddle-window intensities were within about one percent of the stationary saddle-height benchmark;

- nested collar collision scans showed stable \(r^{-3}\rho^{-2}\) gradient-density and \(r^6\rho^2\) determinant-moment scalings.

These computations do not establish continuum uniformity and should not be used to replace the missing analytic estimates.

---

# References

1. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer, 2007.
2. J.-M. Azaïs and M. Wschebor, *Level Sets and Extrema of Random Processes and Fields*, Wiley, 2009.
3. H. Edelsbrunner and J. Harer, *Computational Topology: An Introduction*, AMS, 2010.
4. V. I. Bogachev, *Gaussian Measures*, AMS, 1998.
5. M. W. Hirsch, C. C. Pugh, and M. Shub, *Invariant Manifolds*, Springer Lecture Notes in Mathematics 583, 1977.
