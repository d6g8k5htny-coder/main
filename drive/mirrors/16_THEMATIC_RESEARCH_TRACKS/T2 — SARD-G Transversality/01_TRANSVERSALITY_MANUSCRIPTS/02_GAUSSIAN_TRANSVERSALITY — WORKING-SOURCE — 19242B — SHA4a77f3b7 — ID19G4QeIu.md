# A Cameron–Martin transversality argument for the Bargmann–Fock gradient flow  
## Proposed almost-sure exclusion of saddle–saddle heteroclinic connections

## Abstract

Let \(f\) be the exact normalized periodized Bargmann–Fock Gaussian field on a two-dimensional torus. The persistence arguments in the companion manuscripts require that the gradient flow of \(f\) have no trajectory connecting two distinct index-one saddles.

This note gives the proposed proof.

For a locally tracked ordered pair of saddles and a chosen pair of branches, a transverse section defines a scalar mismatch \(D(f)\). A heteroclinic connection is equivalent, on the chart, to \(D(f)=0\). The derivative of \(D\) in a Cameron–Martin direction is a compactly supported distribution consisting of:

- a nonzero curve contribution along the regular interior orbit segment;
- finitely many endpoint jet atoms caused by motion of the saddles and local invariant manifolds.

The Bargmann–Fock spectral density is strictly positive. Therefore the RKHS embedding of compactly supported finite-order distributions is injective. Support separation between the interior curve and the endpoint atoms implies that the derivative functional is never identically zero at a connection.

A dense countable set of Cameron–Martin directions is then used to reduce the connection event to a countable union of regular zeros along one-dimensional Gaussian coordinates. Each such zero set is discrete and hence conditionally Gaussian-null.

The derivative and support-separation arguments are explicit. The part most in need of expert review is the countable chart construction and the one-dimensional disintegration across moving saddle pairs.

---

# 1. Statement

Let \(f\) be a real-analytic centered Gaussian field on \(\mathbb T_L^2\) with the periodized Bargmann–Fock covariance

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

Let \(\varphi_f^t\) be the flow of

\[
\dot x=\nabla f(x).
\]

### Proposed theorem

With probability one, there is no gradient trajectory whose \(\alpha\)- and \(\omega\)-limits are two distinct index-one saddles.

Equivalently, for every ordered pair of distinct saddles \(p,q\),

\[
W^u(p)\pitchfork W^s(q),
\]

and in dimension two this excludes saddle–saddle heteroclinic orbits.

The theorem is only about saddle–saddle connections. Ordinary transverse intersections of one-dimensional stable and unstable manifolds are allowed.

---

# 2. Preliminary Gaussian facts

## 2.1 Almost-sure local Morse properties

The standard Bulinskaya/Kac–Rice hypotheses hold for the Bargmann–Fock field:

- the field is \(C^\infty\), indeed analytic;
- finite jets at distinct points are nondegenerate;
- almost surely all critical points are nondegenerate;
- almost surely critical values are distinct.

The finite-jet nondegeneracy can be proved directly from the positive Fourier weights, as in File 01.

Thus the unresolved issue is global: exclusion of a connecting gradient orbit.

---

## 2.2 Cameron–Martin space

Let \(\mathcal H\) be the Cameron–Martin space of the field. For the planar Bargmann–Fock covariance, and analogously for the periodized field, the RKHS kernel is \(K\), and

\[
\langle h,K(\cdot,y)\rangle_{\mathcal H}=h(y).
\]

For derivatives,

\[
\langle h,\partial_y^\alpha K(\cdot,y)\rangle_{\mathcal H}
=
\partial^\alpha h(y).
\]

The spectral density is strictly positive at every Fourier frequency. Consequently, if \(T\) is a compactly supported finite-order distribution and

\[
h_T(\cdot)=T_yK(\cdot,y),
\]

then

\[
\|h_T\|_{\mathcal H}^2
=
\int |\widehat T(\xi)|^2\widehat K(\xi)\,d\xi.
\]

Because \(\widehat K(\xi)>0\), \(h_T=0\) implies \(\widehat T=0\), hence \(T=0\).

This injectivity is the central noncancellation fact used below.

---

# 3. Local connection charts

Fix a field \(f_0\) with two distinct nondegenerate index-one saddles \(p_0,q_0\), a chosen unstable branch of \(p_0\), and a chosen stable branch of \(q_0\).

Assume the chosen branches form a connecting orbit for \(f_0\).

Choose:

1. small disjoint neighborhoods \(U_p,U_q\) in which the saddles and their local invariant manifolds can be tracked;
2. a compact regular orbit segment
   \[
   \gamma_0:[t_-,t_+]\to\mathbb T_L^2
   \]
   lying outside \(U_p\cup U_q\);
3. a small transverse section \(\Sigma\) through an interior point of the segment;
4. a narrow tube \(T\) around the segment.

For fields in a sufficiently small \(C^2\) neighborhood of \(f_0\):

- the saddles persist as \(p(f),q(f)\);
- the chosen local branches persist;
- the forward branch from \(p(f)\) and backward branch from \(q(f)\) have first crossings of \(\Sigma\) inside the tube.

Let \(z_u(f),z_s(f)\in\Sigma\) be those crossings. Choose a signed coordinate \(\sigma\) on \(\Sigma\) and define

\[
D(f)=\sigma(z_u(f))-\sigma(z_s(f)).
\]

On this chart,

\[
D(f)=0
\]

if and only if the tracked branches coincide through the tube, hence form the tracked connection.

The smooth dependence theorem for ODEs and hyperbolic invariant manifolds gives \(C^1\) dependence of \(D\) on \(f\) in a sufficiently strong Banach topology, and in particular along Cameron–Martin directions.

### Review issue

The intended global event is covered by countably many such charts, obtained from rational boxes, rational tubes, rational sections, rational branch labels, and rational lower bounds on hyperbolicity and transversality. The detailed countability argument is given in Section 8.

---

# 4. Derivative of the mismatch

Let

\[
f_\varepsilon=f+\varepsilon h,
\qquad
h\in\mathcal H.
\]

We compute

\[
dD_f[h]
=
\left.\frac d{d\varepsilon}\right|_{\varepsilon=0}
D(f_\varepsilon).
\]

There are two kinds of contributions.

1. The vector field changes along the interior orbit.
2. The saddles and local invariant-manifold launch data move.

The result has the form

\[
\boxed{
dD_f[h]
=
\int_{t_-}^{t_+}
a_f(t)\,
n_f(t)\cdot\nabla h(\gamma_f(t))\,dt
+
A_{p,f}[j^2_ph]
+
A_{q,f}[j^2_qh].
}
\]

Here:

- \(\gamma_f\) is the regular connecting segment between the local saddle neighborhoods;
- \(n_f(t)\) is a continuous nonzero transverse covector or transverse unit vector;
- \(a_f(t)\) is a continuous nonzero scalar weight;
- \(A_{p,f}\) and \(A_{q,f}\) are finite linear functionals of the endpoint two-jets.

The endpoint functionals arise because

\[
\delta p=-H_f(p)^{-1}\nabla h(p),
\qquad
\delta q=-H_f(q)^{-1}\nabla h(q),
\]

and because the local stable and unstable eigendirections vary through the Hessian perturbations.

---

## 4.1 Interior adjoint formula

Let

\[
X_f=\nabla f,
\]

and let \(\gamma(t)\) be the reference orbit. The first variation \(\eta\) of the orbit satisfies

\[
\dot\eta(t)
=
H_f(\gamma(t))\eta(t)
+
\nabla h(\gamma(t)).
\]

Let \(w(t)\) solve the adjoint equation

\[
-\dot w(t)
=
H_f(\gamma(t))^\top w(t),
\]

with terminal or section normalization chosen so that

\[
w(t_+)\cdot\eta(t_+)
\]

equals the transverse variation of the section crossing.

Then

\[
\frac d{dt}\bigl(w(t)\cdot\eta(t)\bigr)
=
w(t)\cdot\nabla h(\gamma(t)).
\]

Integrating gives the curve contribution. Since the Hessian is symmetric,

\[
H_f^\top=H_f.
\]

The weight \(a_f(t)n_f(t)\) is a representation of the nonzero adjoint covector \(w(t)\).

A nonzero solution of the adjoint linear equation cannot vanish at an isolated time without vanishing identically. Thus the curve density is nonzero throughout the regular segment.

---

## 4.2 Endpoint atoms

The orbit is not launched from the critical point itself; it is launched from a local section in the saddle neighborhood. The launch point depends on:

- the saddle location;
- the Hessian eigenvectors and eigenvalues;
- the local invariant manifold.

Differentiation produces a finite combination of

\[
h(p),\quad \nabla h(p),\quad H_h(p),
\]

and analogous terms at \(q\).

The archived derivation checked these atoms against finite differences in an explicit testbed, but the complete symbolic coefficient formulas are not reproduced here because they are lengthy and not needed for the nonvanishing argument. Their only structural properties used below are:

- finite order;
- support contained in \(\{p,q\}\).

A reviewer who regards the endpoint differentiation as a potential gap should classify the theorem as conditional on the displayed derivative formula.

---

# 5. Riesz representative in the RKHS

The curve term has RKHS representative

\[
G_{\mathrm{curve}}(\cdot)
=
\int_{t_-}^{t_+}
a_f(t)\,
n_f(t)\cdot
\nabla_2K_L(\cdot,\gamma_f(t))\,dt.
\]

The endpoint terms have representatives that are finite linear combinations of

\[
\partial_2^\alpha K_L(\cdot,p),
\qquad
\partial_2^\alpha K_L(\cdot,q),
\qquad
|\alpha|\le2.
\]

Thus

\[
dD_f[h]
=
\langle h,G_f\rangle_{\mathcal H},
\]

where

\[
G_f=G_{\mathrm{curve}}+G_{\mathrm{endpoints}}.
\]

Equivalently, \(G_f\) is the kernel embedding of the distribution

\[
T_f
=
\int_{t_-}^{t_+}
a_f(t)\,
n_f(t)\cdot\nabla\delta_{\gamma_f(t)}\,dt
+
T_{p,f}+T_{q,f}.
\]

---

# 6. Nonvanishing of the derivative

### Proposition 6.1

At every nondegenerate saddle–saddle connection chart,

\[
G_f\neq0.
\]

### Proof

Suppose \(G_f=0\). By injectivity of the Bargmann–Fock kernel embedding,

\[
T_f=0
\]

as a distribution.

Choose a smooth test function \(\psi\) supported in a small neighborhood of an interior point of the orbit segment, disjoint from \(p\) and \(q\). Then the endpoint atoms vanish on \(\psi\), so

\[
T_f(\psi)
=
\int
a_f(t)\,
n_f(t)\cdot\nabla\psi(\gamma_f(t))\,dt.
\]

Because the density \(a_f(t)n_f(t)\) is continuous and nonzero, one can choose \(\psi\) so that this integral is nonzero. This contradicts \(T_f=0\).

Therefore \(G_f\neq0\). \(\square\)

### What this argument avoids

It does not require finding a Cameron–Martin function that prescribes one interior gradient while holding all endpoint jets fixed. Such finite-jet steering would not control the perturbation along the rest of the orbit and could allow cancellation. The distribution/RKHS argument rules out cancellation globally.

---

# 7. Endpoint behavior and convergence of the curve integral

The full orbit approaches the saddles exponentially in time. In time parametrization:

- the adjoint weight decays exponentially toward each endpoint;
- the RKHS norm of
  \[
  \nabla_2K(\cdot,\gamma(t))
  \]
  remains bounded.

Therefore the defining curve integral converges absolutely in \(\mathcal H\).

In arclength \(s\) from an endpoint, the weight may behave like

\[
s^{\nu-1},
\qquad
\nu
=
\frac{|\lambda_{\mathrm{transverse}}|}
{\lambda_{\mathrm{departure}}}
>0.
\]

This is integrable but numerically difficult when \(\nu\ll1\). An explicit testbed gave \(\nu\approx1/6\), explaining why finite arclength truncations missed an order-one fraction of the integral.

This endpoint singularity affects quadrature, not the existence of the RKHS element.

---

# 8. Countable chart covering

The goal is to express the global connection event as a countable union of chart events to which one-dimensional slicing can be applied.

A proposed construction is as follows.

## 8.1 Rational saddle neighborhoods

Use a countable basis of rational coordinate boxes. For each ordered pair of disjoint boxes and rational \(\delta>0\), consider fields having exactly one critical point in each box with Hessian eigenvalues bounded away from zero by \(\delta\).

The implicit function theorem gives measurable tracked saddles on this event, for example by choosing the first rational Newton seed whose iteration converges inside the box.

## 8.2 Branch labels

Each index-one saddle has two local stable and two local unstable branches. Label them by their first intersection with a rationally chosen small circle or local section.

## 8.3 Rational tubes and transverse sections

Use a countable family of polygonal tubes with rational vertices and rational-width bounds. Inside a tube, choose the first rational line segment satisfying a quantitative transversality condition.

Require:

- the tracked branch remains inside the tube until the first section crossing;
- the speed is bounded below by a rational constant;
- the section angle is bounded away from tangency;
- the crossing time is bounded by a rational constant.

Every compact regular segment of a genuine connection satisfies such bounds and is captured by at least one rational chart.

## 8.4 Measurability

The field-to-flow map is measurable in the \(C^1\) topology. Hitting times of a fixed closed section by continuous paths are measurable on the event of a unique transverse first crossing. The first rational chart satisfying the stated open inequalities is a measurable countable selection.

Therefore each chart mismatch \(D_\chi(f)\) is measurable.

### Review issue

This section gives the intended covering argument but does not cite a single theorem that packages all moving-saddle, moving-branch, first-crossing, and measurable-selection statements. A specialist may prefer a different construction, for example a finite-dimensional universal moduli space and a parametric transversality theorem.

---

# 9. Countable Cameron–Martin directions

Let

\[
\{h_j\}_{j\ge1}
\]

be dense in the unit ball of \(\mathcal H\).

At a charted connection \(f\), Proposition 6.1 gives a direction \(h\in\mathcal H\) with

\[
dD_f[h]\neq0.
\]

Because the derivative is a continuous linear functional of \(h\), some \(h_j\) satisfies

\[
dD_f[h_j]\neq0.
\]

Thus every connection belongs to an event of the form

\[
\{D_\chi(f)=0,\ dD_{\chi,f}[h_j]\neq0\}.
\]

The global connection event is contained in a countable union over:

- saddle boxes;
- branch labels;
- tubes;
- sections;
- quantitative bounds;
- directions \(h_j\).

---

# 10. One-dimensional Gaussian slicing

Fix a chart \(\chi\) and a direction \(h_j\).

Decompose the Gaussian field into a Cameron–Martin coordinate and an orthogonal complement:

\[
f=g+\xi h_j,
\]

where, after normalization, \(\xi\) is a nondegenerate one-dimensional Gaussian independent of \(g\).

For fixed \(g\), define

\[
F_g(\xi)=D_\chi(g+\xi h_j)
\]

on each interval of \(\xi\) for which the chart remains valid.

If

\[
F_g(\xi_0)=0,
\qquad
F_g'(\xi_0)\neq0,
\]

then \(\xi_0\) is an isolated zero. Hence the set of regular zeros is discrete, therefore countable and Lebesgue-null.

The conditional law of \(\xi\) has a density, so the conditional probability of hitting the regular zero set is zero. Integrating over \(g\) gives probability zero for the fixed chart and direction.

Taking the countable union gives zero probability for any saddle–saddle connection.

---

# 11. Exceptional shift parameters and chart boundaries

The slicing proof only treats regular zeros inside a chart. The following exceptional situations must be excluded or covered separately:

1. a tracked saddle becomes degenerate;
2. two critical values coincide;
3. a branch leaves the rational tube before the section crossing;
4. the section becomes tangent;
5. the derivative in the chosen direction vanishes.

Items 3–5 are handled by chart refinement and the countable direction union.

Items 1–2 are intended to be null by Bulinskaya-type finite-jet arguments. Along a one-dimensional shift \(f+\xi h\), degeneracy is described by simultaneous equations

\[
\nabla(f+\xi h)(x)=0,
\qquad
\det H_{f+\xi h}(x)=0.
\]

The proposed proof sketches an analytic-set argument for the projection of this set to the \(\xi\)-axis, together with Bulinskaya at rational \(\xi\).

### Main review question

Is that analytic-set argument sufficient? A cleaner repair might be:

- use a parametric Kac–Rice estimate for the three-component field
  \[
  (\partial_x f_\xi,\partial_y f_\xi,\det Hf_\xi)
  \]
  on \(\mathbb T^2\times I\); or
- work on a full-measure set on which all fields along almost every line parameter are Morse, then apply slicing only there.

This is one of the two most important unresolved points in the proposed proof.

---

# 12. Engineered deterministic testbed

The derivative architecture was tested on

\[
f_0(x,y)
=
-\frac12y^2(1-2x)+3x^2-2x^3.
\]

The \(x\)-axis is invariant and contains an exact saddle–saddle connection from \((0,0)\) to \((1,0)\).

A perturbation

\[
h(x,y)
=
y\,e^{-8(x-1/2)^2}
\]

breaks the connection.

For perturbation amplitude \(\mu\), the measured section mismatch satisfied

\[
D(0)=0,
\]

\[
D(0.02)\approx0.01488,
\qquad
D(-0.02)\approx-0.01488,
\]

and

\[
\left.\frac{dD}{d\mu}\right|_{\mu=0}
\approx0.7441.
\]

Additional comparisons gave:

- finite difference versus variational ODE: agreement to displayed precision;
- adjoint pairing: about \(1.96\%\) discrepancy in an early implementation;
- endpoint atom formulas: \(0.01\%\) and \(0.32\%\) discrepancies in localized tests;
- full section-to-section variational evaluation versus finite difference: \(0.37\%\) discrepancy.

Three direct kernel-quadrature attempts failed by \(22\%\), \(33\%\), and \(36\%\), because the \(s^{-5/6}\)-type endpoint behavior makes finite truncation ineffective. Those failures are evidence about the numerical method, not about the abstract derivative.

The testbed supports the derivative architecture but cannot prove the probabilistic theorem.

---

# 13. Precise status of the proof

## Strongest parts

1. The RKHS injectivity statement is standard and exact for the Bargmann–Fock spectral density.
2. The support-separation proof of \(G_f\neq0\) is short and robust if the derivative formula is accepted.
3. Regular zeros along a fixed Cameron–Martin line are isolated.
4. A countable dense family of directions is enough to witness every nonzero derivative.

## Parts requiring expert confirmation

1. The full endpoint-motion derivative formula.
2. The countable chart construction for moving saddle pairs and branches.
3. Measurability of the chart mismatch and its derivative.
4. The treatment of degenerate saddle parameters along the shift line.
5. The conditional disintegration when chart-validity intervals depend on the orthogonal complement.

## Conservative conclusion

A safe theorem statement at present is:

> Conditional on the charting/measurability and exceptional-parameter lemmas in Sections 8–11, the Bargmann–Fock gradient flow has almost surely no saddle–saddle heteroclinic connection.

The requested external review is specifically intended to determine whether those lemmas are standard consequences of existing transversality machinery, require a local repair, or contain a substantive gap.

---

# References

1. V. I. Bogachev, *Gaussian Measures*, American Mathematical Society, 1998.
2. S. Janson, *Gaussian Hilbert Spaces*, Cambridge University Press, 1997.
3. M. W. Hirsch, C. C. Pugh, and M. Shub, *Invariant Manifolds*, Lecture Notes in Mathematics 583, Springer, 1977.
4. J. Palis, Jr. and W. de Melo, *Geometric Theory of Dynamical Systems*, Springer, 1982.
5. R. Abraham and J. Robbin, *Transversal Mappings and Flows*, Benjamin, 1967.
6. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer, 2007.
7. J.-M. Azaïs and M. Wschebor, *Level Sets and Extrema of Random Processes and Fields*, Wiley, 2009.
