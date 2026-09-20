# RN-JOINT-001 — Uniform joint comparison on the complete far region

2026-09-17 · DQ-RN3-20260917-b9c2 · author-side proof and interval certificate.

At the fixed rung **r = 1/20**, the complete far-region marked saddle intensity has relative upper correction **less than 0.12058**, below the requested **0.6798** budget. The proof uses one joint Gaussian comparison, including the pair Hessians, third Hessian, and their dependence. It yields the conditional expected raw far saddle count

\[
2.22542\,r^3<I_{\rm far}<2.83312\,r^3.
\]

The near-annulus integral, all-small-r extension, and independent acceptance remain open. This is an alternative proof of the fixed-r far-region target; it does not certify the numerical value of the older five-factor expression or close the parent D3-LEMMA-RN-UNIF.

## 1. Exact model, pins, and comparison object

The actual field is the centered Gaussian field on the side-24 square torus with normalized covariance

\[
K(x)=k(x_1)k(x_2),\qquad
k(s)=\frac{\sum_{j\in\mathbb Z}e^{-(s+24j)^2/2}}
{\sum_{j\in\mathbb Z}e^{-(24j)^2/2}}.
\]

Set b=6/5, M=(-r/2,0), S=(r/2,0), and condition on the six endpoint pins

\[
c=(b,0,0,b-r^3/6,0,0)^T.
\]

The pin order at each endpoint is (f,fx,fy). The Hessian order is (fxx,fyy,fxy). The entire certified domain is

\[
5\le |y|\le17,\qquad b-r^3/6\le v\le b,
\]

in the declared coordinate lift, with all directions and both mark endpoints included. The covariance remains periodic across the fundamental-domain boundary. Neither rotation invariance of the square torus nor monotonic decay in every direction is assumed.

Let Q6 be the exact six-pin law of the pair Hessians, and define

\[
W=|\det H_M\det H_S|\,1_{\{H_M\prec0\}}1_{\{\det H_S<0\}},
\quad Z=E_{Q6}W,
\quad g(H)=(-\det H)_+.
\]

The normalizer input is the H3 certificate

\[
Z\ge Z_{\rm lo}=0.0077592917375327855.
\]

Its exact source bytes and decimal are checked at runtime. This round imports that result; it does not independently re-prove the H3 floor.

Let P(y,v) be the exact joint law of all nine Hessian coordinates after also pinning Y=(f(y),fx(y),fy(y))=(v,0,0). Introduce a reference law

\[
Q(v)=Q6\otimes Q_y^0(v),\quad
Q_y^0(v)=N((-v,-v,0)^T,\operatorname{diag}(2,2,1)).
\]

The third-Hessian reference is deliberately planar. The actual field, conditioning, covariance and density are still those of the exact torus. Every discrepancy from this reference is retained in the joint mean and covariance below. Q_y^0 is not asserted to equal the exact torus one-point conditional law.

## 2. Joint Gaussian chi-square lemma

After an invertible affine whitening of Q(v), write Q=N(0,I), P=N(m,I+E), with E symmetric. If delta=||E||F<1, both I+E and I-E are positive definite, and

\[
\log(1+\chi^2(P\Vert Q))
=-\tfrac12\log\det(I-E^2)+m^T(I-E)^{-1}m
\le\frac{\delta^2}{2(1-\delta^2)}+
\frac{\|m\|^2}{1-\delta}.
\tag{1}
\]

Proof: integrate p(x)^2/q(x). Its quadratic precision is 2(I+E)^-1-I, which is positive exactly when every eigenvalue of E is below 1, given I+E positive. Completing the square gives the equality. For eigenvalues e_i, use -log(1-e_i^2) <= e_i^2/(1-delta^2), sum e_i^2=||E||F^2, and ||(I-E)^-1||op <= (1-delta)^-1. This also proves finiteness before using Cauchy–Schwarz. No factor equal to the dimension is needed in the determinant bound.

## 3. Exact typed saddle second moment and the weighted comparison

Under Q_y^0(v), put U=(Hxx+Hyy)/2, V=(Hxx-Hyy)/2 and B=Hxy. Then U~N(-v,1), V,B~N(0,1) are independent, T=V^2+B^2~chi-square(2), and g=(T-U^2)+. For any c>=0, direct integration against the exponential density gives

\[
E(T-c)_+=2e^{-c/2},\qquad
E(T-c)_+^2=8e^{-c/2}.
\]

Integrating U therefore yields

\[
m_0(v):=E_Qg=\sqrt2 e^{-v^2/4},\qquad E_Qg^2=4m_0(v).
\tag{2}
\]

Define

\[
h_2=\big(E_{Q6}(\det H_M)^4\ E_{Q6}(\det H_S)^4\big)^{1/4}.
\]

Dropping the typing indicators and applying Cauchy–Schwarz gives ||W||2 <= h2. Independence in the reference law gives ||Wg||2 <= 2h2 sqrt(m0(v)). Thus, with L=dP/dQ,

\[
|E_P(Wg)-Z m_0(v)|
=|E_Q[(L-1)Wg]|
\le2h_2\sqrt{m_0(v)}\sqrt{\chi^2(P\Vert Q)}.
\]

Consequently

\[
\left|\frac{E_P(Wg)}{Z m_0(v)}-1\right|
\le\eta:=\frac{2h_2\sqrt{\chi^2(P\Vert Q)}}
{Z_{\rm lo}\sqrt{m_{0,\min}}},\qquad
m_{0,\min}=\sqrt2 e^{-b^2/4}.
\tag{3}
\]

The last minimum is valid because the entire mark interval is positive. Fourth determinant moments are exact Gaussian polynomial moments evaluated by a finite recurrence with interval inputs, not estimated from samples. No differentiation of an indicator or smoothing of the saddle typing is required.

## 4. Source-specific covariance construction

Let C be the six pins, H the pair Hessian vector, G=Cov(C), X=Cov(H,C), and S0=Cov(H)-XG^-1 X^T. Positive interval Cholesky pivots certify both matrices. Use their exact positive-diagonal factors Lc,Lh to define

\[
U=L_c^{-1}C,\qquad R=L_h^{-1}(H-XG^{-1}C).
\]

Then Cov(U)=Cov(R)=I and Cov(U,R)=0. The code encloses these exact fixed forms, not a freely variable collection of rounded coefficients. Set

\[
B=\operatorname{Cov}(U,Y),\ F=\operatorname{Cov}(R,Y),\
T=\operatorname{Cov}(U,H_y),\ J=\operatorname{Cov}(R,H_y),
\]

\[
D=\operatorname{Cov}(Y)-B^TB,\quad
C_y=\operatorname{Cov}(H_y,Y)-T^TB,\quad
W_y=\operatorname{diag}(1/\sqrt2,1/\sqrt2,1).
\]

With muY=E[Y|C=c], muH=E[Hy|C=c], and z=(v,0,0)^T-muY, the blocks of I+E are

\[
\begin{aligned}
\Sigma_{pp}&=I-FD^{-1}F^T,\\
\Sigma_{py}&=(J-FD^{-1}C_y^T)W_y,\\
\Sigma_{yy}&=W_y[\operatorname{Cov}(H_y)-T^TT-C_yD^{-1}C_y^T]W_y.
\end{aligned}
\tag{4}
\]

The mean displacement is

\[
m_p=FD^{-1}z,\qquad
m_y=W_y[\mu_H-C_yD^{-1}\mu_Y+
v(C_yD^{-1}e_0+(1,1,0)^T)].
\tag{5}
\]

Grouping the affine expression in v before interval evaluation preserves its small cancellation over the whole mark window. Gaussian conditioning identities follow, for example, from [GPML Appendix A](https://gaussianprocess.org/gpml/chapters/RWA.pdf); equations (4)–(5) instantiate them for the exact derivative field.

## 5. Joint Y-density ratio

Let E0=Cov(Y)=diag(1,a2,a2), with a2=-k''(0)>0, and e*=min(1,a2). Put beta=||B||F^2/e*<1 and u=||muY||. For w=(v,0,0), compare the actual six-pin density p_D,muY(w) to p_E0,0(w).

The determinant ratio is between 1 and (1-beta)^(-3/2). Since D<=E0, replacing the quadratic form of w-muY by that under E0 gives the upper bound

\[
R_Y\le(1-\beta)^{-3/2}\exp(bu/e_*).
\tag{6}
\]

Also ||D^-1-E0^-1||op <= beta/[e*(1-beta)] and ||D^-1||op <= 1/[e*(1-beta)], whence

\[
R_Y\ge\exp\left[-\frac{b^2\beta+2bu+u^2}
{2e_*(1-\beta)}\right].
\tag{7}
\]

These bounds include value and both gradient densities at once. They avoid subtraction of nearly equal interval quadratic forms.

## 6. Pointwise intensity and integrated far count

The source Kac–Rice integrand under the W-weighted endpoint Palm law is

\[
\rho(y,v)=p_{Y\mid C=c}(v,0,0)\ E_{P(y,v)}(Wg)/Z.
\]

Use the explicitly defined reference

\[
\rho_{\rm ref}(v)=p_{E0,0}(v,0,0)m_0(v)
=\frac{e^{-3v^2/4}}{2\pi^{3/2}a_2}.
\]

Combining (3), (6), and (7) gives pointwise lower and upper factors RY,lo(1-eta) and RY,hi(1+eta). The interval run proves the following outward-rounded statements on every spatial and mark point in the domain:

| Quantity | Certified bound |
|---|---:|
| ||E||F | < 0.043544631 |
| ||m|| | < 0.008864422 |
| chi-square(P || Q) | < 0.001032557 |
| eta | < 0.116819988 |
| Y-density ratio, upper | < 1.003365693 |
| Complete marked-intensity ratio | 0.88021 < rho/rho_ref < 1.12058 |
| Smallest eigenvalue of D | > 0.99999340406 |
| Pair covariance reduction norm | < 0.000147382258 |
| Pair/third-Hessian cross block norm | < 0.030790505 |
| Third-Hessian covariance shift norm | < 0.000054889063 |
| h2 | < 0.014009957 |

The torus fundamental square [-12,12]^2 is inside |y|<17. Its far part |y|>=5 has area 576-25pi. The mark integral is evaluated without quadrature:

\[
J_{\rm ref}=\frac{\operatorname{erf}(\sqrt3b/2)-
\operatorname{erf}(\sqrt3(b-r^3/6)/2)}{2\pi\sqrt3\,a_2}
\approx6.3529298594549898874\times10^{-7}.
\]

The certificate's unrounded factors yield

\[
0.00027817807867916<I_{\rm far}<0.00035413986017516,
\]

and therefore the bounds stated at the start. The conservative decimal upper 2.83312 is rounded outward. This is an expected raw saddle count under the declared conditioned law. It is not itself a probability or a complete chart-plus-remote theorem.

## 7. Interval coverage and arithmetic contract

`rn_joint.py` imports the exact prior interval kernel `closure_round2/rn_field.py`, SHA-256 `d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8`. Both files and their source contracts are included. The prior proof documents the image-tail and spectral-tail inequalities in full.

At 384 bits, derivative covariances use normalized image sums with both infinite image tails enclosed. The x dependence uses grouped Taylor polynomials of degree 28, with remainders bounded by spectral absolute derivative moments, including the infinite spectral tail. y dependence is enclosed directly on each interval. The exact convention is Cov(d^a f(p),d^b f(q))=(-1)^|b| d^(a+b)K(p-q).

The cover consists of every quarter-unit Cartesian box intersecting the closed annulus. The minimum squared radius of a box is the sum of coordinatewise squared distances from zero; its maximum is the sum of the squared largest absolute endpoints. Membership uses exact integer inequalities against 25 and 289. All **13,604** selected boxes are evaluated in full, including their parts outside the annulus. This overcoverage is safe. The entire mark interval is an Arb ball in each evaluation. No sampled station is substituted for a box, and no quadrant is omitted.

Each inversion has a proven covariance floor; the joint finite-chi-square condition, positive two-sided factor, exact normalizer pin, complete box count, and final 0.6798 target are runtime gates that remain active under Python `-O`.

## 8. Verification and limits

`audit_rn_joint.py` passed **294 checks**, including **24 full Gaussian laws** (8 sites times 3 marks). A separate spectral kernel with an infinite-tail enclosure constructs the raw full nine-pin Gram matrix and nine-Hessian covariance. Its conditioning, mean, covariance, exact Gaussian integral and Y-density are compared to the grouped image construction. Controls cover zero, pure mean, pure variance, and a single correlated pair; singular chi-square boundaries and an inflated normalizer floor are rejected. Omitting cross dependence or mean displacement is detected by analytic controls. Exact-grid checks include closed annulus boundaries and the torus square.

At the torus corner the correction is approximately 2.4e-112. The first audit implementation evaluated log(det(I-E^2)) directly, whose absolute roundoff interval was too wide to verify the very small norm bound. The final audit uses ten terms of the trace series plus the rigorous remainder delta^22/[22(1-delta^2)]. It still compares to the raw unwhitened Gaussian integration formula by interval overlap. This was a verification-precision repair, not evidence that the failed wider interval proved a false inequality. An interval-equality check for the normalizer literal was likewise replaced by exact rational decimal identity and interval overlap.

Normal and optimized full-cover/audit runs agree exactly in all mathematical result fields; elapsed time and the resulting certificate-file hash are the only allowed metadata differences. `RN_MODE_REPLAY.json` records both.

This is same-provider verification with a separately constructed numerical path, **zero organizational-independence credit**. The H3 floor and prior kernel enclosures remain explicit dependencies. The numerical certificate is an author-side computational proof, not a formal proof-assistant result.

## 9. Consumption boundary and next exact work

The prior D3 v2 amendment leaves zone uniformity and the certified near-annulus integral as separate obligations. This proof supplies the complete far-region uniform comparison at r=1/20 using a stronger alternative. It neither edits that frozen amendment nor retrospectively validates its pointwise five-factor assembly.

The unresolved near part is **0.1 <= |y| <= 5**. Its displayed floor-consistent target is 17.6804r^3, but this round has not supplied the required certified interval integral. Conditional arithmetic only: if that near bound is later certified for this same law and geometry, then I_near+I_far < 20.51352r^3. That conditional sum is not presently a full remote certificate.

The next computational target is a normalized, cancellation-preserving near-annulus enclosure with explicit covariance floors, mark integration, exact coverage, outward quadrature error, and a fail-closed consumer. Uniformity in r, other endpoint orientations, chart integration, and independent acceptance require their own evidence. No prize, parent theorem, review gate, or external-release status is promoted here.
