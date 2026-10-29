# D5 fixed scaled annulus: a height-window stitching candidate

**Object:** D5-FIXED-ANNULUS-STITCH-20260925-v1. **Author:** OpenAI / ChatGPT.
**Disposition:** AUTHOR-SIDE ANALYTIC CANDIDATE; independent analytic review OPEN.
**Scientific effect:** NONE. No global RN, elder-selection, lifetime, or historical numerical gate is closed by this file. The earlier PROOF.md and TWO_SCALE_ADDENDUM.md are unchanged.

## 1. Exact target, and what is stronger and weaker

Fix L>0 and the centered variance-one Gaussian field on R^2/(L Z^2) with FULL covariance

    K_L(h)=sum_{n in Z^2} exp(-|h+Ln|^2/2)
                         /sum_{n in Z^2} exp(-|Ln|^2/2).

For any orthonormal frame E write f(x,z)=F(x e1+z e2). Let b range over a fixed compact interval and 0<k_-<=k<=k_+<infinity. At M=(-r/2,0), S=(r/2,0) prescribe the six ORIGINAL pins

    f(M)=b, f(S)=b-k*r^3, grad f(M)=grad f(S)=0.            (A1)

Q_r is the continuous Gaussian regression law at (A1), not an elder/adjacency law. Define F_j(H)=|det H| times the indicator of nonsingular negative index j, zero on singular matrices, and

    W_r=F_2(H_M) F_1(H_S), Z_r=E_Qr W_r, dQ_r^W=(W_r/Z_r)dQ_r.

Fix 1<A0<B0<infinity and the entire scaled annulus

    K={ (u,t): A0<=sqrt(u^2+t^2)<=B0 }.

For a Borel E0 subset K let N_j(r E0) count index-j critical points in r E0 whose heights lie STRICTLY between b-k*r^3 and b. Here r E0 is embedded in the torus via the frame, with r sufficiently small.

**Candidate theorem.** There are C,r_*>0, uniform in the frame, compact marks, E0 and j=0,1,2, such that

    E_Qr^W N_j(r E0) <= C*k*r^3*area(E0),  0<r<=r_*.       (A2)

The all-index sum satisfies the same statement after changing C; Markov gives the corresponding existence-probability upper bound. Equivalently, the HEIGHT-RESTRICTED weighted intensity per physical area is at most C*k*r throughout the annulus, up to the usual density representative.

This removes a fixed-angle exclusion for this fixed scaled annulus at candidate scope. Unlike the earlier thin-tube result, (A2) is NOT an all-height assertion. The shrinking height window is essential in the outer part of the proof. Constants depend on L,A0,B0 and the mark compacts. There is no uniform-L, k down to zero, d>=3, pin-collision, r<<distance<<rho, multiple-witness, sharp coefficient, or numerical24-jet conclusion.

## 2. Stable Gaussian regression and the original normalizer

Let a=r/2, g(x)=f(x,0), h(x)=f_z(x,0). Let H_r interpolate the two values and two longitudinal slopes by a cubic, and L_r interpolate the transverse slopes by an affine function. For unconditioned observations put

    S0=(g(-a)+g(a))/2, D0=(g(a)-g(-a))/(2a),
    S1=(g'(-a)+g'(a))/2, D1=(g'(a)-g'(-a))/(2a),
    U_r=(S0-a^2 D1/2, (3D0-S1)/2, D1, 3(S1-D0)/a^2,
         (h(-a)+h(a))/2, (h(a)-h(-a))/(2a)).              (A3)

Evaluation of the interpolants recovers the six observations, proving invertibility. The exact target is

    v_r=(b-k*r^3/2, -(3/2)k*r^2, 0, 12k, 0, 0).          (A4)

In L^p for every fixed finite p, U_r=U_0+O(r^2), with

    U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)(0), v_0=(b,0,0,12k,0,0).

All Fourier weights of K_L are positive and decay Gaussianly. Their square roots times every fixed polynomial in frequency are summable, so every fixed derivative supremum has finite L^p moments. Distinct one-site derivative monomials have positive-definite covariance: a zero-variance combination would give a polynomial vanishing on every rotated lattice frequency, hence the zero polynomial. The compact set of frames gives uniform positive eigenvalue floors at fixed jet order.

Thus Cov(U_r)^-1 is uniformly bounded. The common-field Gaussian coupling

    f^(r)(s)=f(s)+Cov(f(s),U_r) Cov(U_r)^-1 (v_r-U_r)       (A5)

has law Q_r and uniformly bounded C^m moments at every fixed order on the torus. Covariance kernels and their derivatives in (A5) converge at order O(r^2); hence f^(r) couples to f^(0) in L^p(C^m) at that order. This follows directly by differentiating the finite-dimensional regression coefficients and using the Fourier-sum bounds. It does not impose a deterministic bound on a random derivative norm.

The finite-r pin identities, before scaling, include

    f_x(0)=-(r^2/8)f_xxx(0)+O_Lp(r^4),
    f_xx(0)=-(r^2/24)f_xxxx(0)+O_Lp(r^4),
    f_xxx(0)=12k+O_Lp(r^2), f(0)=b-k*r^3/2+O_Lp(r^4),
    f_z(0)=-(r^2/8)f_xxz(0)+O_Lp(r^4),
    f_xz(0)=-(r^2/24)f_xxxz(0)+O_Lp(r^4).                (A6)

They follow by adding/subtracting the symmetric endpoint slope/height Taylor formulas; the derivative moments in (A5) justify their remainders. In particular, with S_r=f_zz(0), under endpoint-only conditioning,

    det H_M/r=-6k*S_r+O_Lp(r), det H_S/r=6k*S_r+O_Lp(r),
    f_xx(M)/r=-6k+O_Lp(r).                               (A7)

S_r is Gaussian with mean bounded and variance uniformly bounded above and below. Therefore P_Qr(-2<=S_r<=-1)>=p0>0. On this event, excluding determinant errors exceeding 3k_- and longitudinal-diagonal errors exceeding 3k_- removes only O(r^p) probability. On the remaining positive-probability event H_M is negative definite, det H_S<0, and both absolute determinants are >=3k_- r. Consequently

    Z_r>=c_Z*r^2.                                       (A8)

This is the FULL normalizer under (A1), not after witness conditioning. Its upper finiteness follows from the same moment bounds. No numerical eigenfloor or prior acceptance is imported.

## 3. Retain the height: the three-row contact vector

All variables in this section are under Q_r; no zero pin target is substituted into an UNCONDITIONED covariance. At X=(ru,rt) set

    J_r=( f_x(X)/r^2, f_z(X)/r,
          [f(X)-b-(r*t/2)f_z(X)]/r^3 ).                 (A9)

For bounded u,t, (A6) and multivariate Taylor imply uniform L^p expansions

    f_x(X)/r^2=6k(u^2-1/4)+u*t*T+(t^2/2)*C+O_Lp(r),
    f_z(X)=r*t*S+r^2[(u^2-1/4)T/2+u*t*C+t^2*D/2]+O_Lp(r^3),
    f(X)-b=r^2*t^2*S/2+r^3[k(2u^3-3u/2-1/2)
             +(u^2-1/4)t*T/2+u*t^2*C/2+t^3*D/6]+O_Lp(r^4),

where S=f_zz(0), T=f_xxz(0), C=f_xzz(0), D=f_zzz(0). The quadratic height cancels in (A9), as does the C contribution to its third row. In the coupling (A5),

    J_r = d(u,k)+M(u,t) V_0+O_L2(r),                    (A10)
    d=(6k(u^2-1/4),0,k(2u^3-3u/2-1/2)),
    V_0=(S,T,C,D) under U_0=v_0,
    M=[ 0        u*t            t^2/2     0      ]
      [ t         0               0       0      ]
      [ 0  (u^2-1/4)*t/4          0      -t^3/12 ].

The four V_0 jets are independent modulo the six U_0 derivatives; their conditional covariance G obeys lambda I<=G<=Lambda I uniformly in frame. Means are uniformly bounded. Hence J_r has uniformly bounded L^p norms, and

    ||Cov_Qr(J_r)-M G M^T|| <= C*r                      (A11)

uniformly on the entire bounded scaled region. The O(r) comparison is absolute, not falsely asserted relative to a vanishing covariance.

The minor of M on columns (S,C,D) is exactly t^6/24. Cauchy-Binet gives det(M M^T)>=t^12/576. Its largest eigenvalue is uniformly bounded for |u|,|t|<=B0. Therefore, with changed constants,

    lambda_min(M G M^T)>=c*|t|^12.                     (A12)

The exponent 12 is deliberately conservative. Optimizing it is unnecessary for this proof. It is positive for every nonzero t, including u=0; it does NOT cover t=0.

## 4. An explicit shrinking cutoff makes perturbation uniform

Choose q=1/24 and h_r=r^q. For |t|>=h_r and r<=1,

    r/|t|^12 <= r^(1-12q)=r^(1/2) -> 0.                (A13)

Combining (A11)-(A13) and shrinking r_* to absorb the constants gives

    Sigma_r=Cov_Qr(J_r)>=c1*|t|^12 I,
    ||Sigma_r^-1||<=C*|t|^-12, det(Sigma_r)^-1/2<=C*|t|^-18.  (A14)

These estimates also hold away from a fixed small axis neighborhood by compactness. There is no substitution of a shrinking eta into unspecified PR16 constants. The cutoff is selected AFTER deriving the explicit eigenfloor and O(r) perturbation. A q with 12q>=1 would not justify this perturbation step.

## 5. The joint height density keeps the rare-gradient exponential

Choose A1=(A0+1)/2>1. Fix epsilon>0 sufficiently small that epsilon<1, epsilon^2<A0^2-A1^2, and the mean estimate below holds. On K with |t|<=epsilon, necessarily |u|>=A1. Write d0=6k_-(A1^2-1/4)>0. From (A10) and bounded jet means,

    E_Qr J_r1 >= d0-C(|t|+r) >= d0/2,
    Var_Qr J_r1 <= C*(t^2+r^2).                         (A15)

The variance estimate uses the O_L2(r) remainder and the t factors in the first random contact row. On h_r<=|t|<=epsilon, r<=|t|, so Var J_r1<=C*t^2.

At witness gradient zero and height f(X)=b+r^3*tau, the J_r target is y=(0,0,tau), tau in [-k,0], a fixed compact range. For any positive covariance Sigma and vector z,

    z^T Sigma^-1 z >= z_1^2/Sigma_11                    (A16)

by Cauchy-Schwarz applied to Sigma^(1/2)e1 and Sigma^(-1/2)z. Thus (A15) supplies exp(-c/t^2) in the FULL three-dimensional Gaussian density at y, regardless of correlations with height. Together with (A14) and the exact determinant r^-6 of the affine map (f_x,f_z,f)->J_r,

    p_(grad,height)|pins(0,b+r^3*tau)
        <= C*r^-6*|t|^-18*exp(-c/t^2)                  (A17)

on the outer part of the small axis neighborhood. Away from |t|<=epsilon, the same argument without the exponential gives p<=C*r^-6 with constants depending on epsilon. We never infer a joint density bound from a marginal alone: the joint determinant is bounded separately by (A14).

## 6. Account for all three Hessians after gradient AND height pins

Let M3 be the supremum norm of third derivatives on a local convex ball containing M,S,X. Under the three zero-gradient constraints, deterministic Taylor identities give

    ||H_M e1||<=C*r*M3,
    ||H_M (u+1/2,t)^T||<=C*r*M3.

The two-direction matrix has determinant t. For t!=0,

    ||H_M||+||H_S||+||H_X|| <= C*r*|t|^-1*M3,           (A18)

using Hessian Lipschitz control between these points. The constant is uniform for bounded u,t. Therefore the typed determinant product is bounded, pointwise under the constrained law, by

    W_r F_j(H_X) <= C*r^6*|t|^-6*M3^6.                  (A19)

Smallness of an endpoint-only Hessian is NOT simply reused after extra conditioning.

To control M3 after the additional pins, start with the Q_r field in (A5). Its C3 moments and J_r moments are uniformly bounded. For every derivative through order three, the supremum of its covariance with J_r is bounded by Cauchy-Schwarz. Gaussian regression at J_r=y represents the additional conditional field as

    f^*=f^(r)+Cov_Qr(f^(r),J_r) Sigma_r^-1 (y-J_r).

This coupling retains all original pins and imposes the actual witness pins. By (A14), Minkowski, and bounded sixth moments of y-J_r,

    E[M3^6 | original pins, grad=0, height=b+r^3*tau]
        <= C*|t|^-72                                  (A20)

for h_r<=|t|<=epsilon. This is intentionally crude and includes possible large conditional means. Combining (A19)-(A20) gives a single, correlated triple-determinant expectation <=C*r^6*|t|^-78. No independence or factorization of endpoint and witness Hessians is assumed. On |t|>=epsilon the same reasoning gives <=C*r^6.

## 7. Weighted Kac-Rice yields a bounded angular envelope

At each fixed r>0 the annulus stays away from the pinned sites. Distinct-site derivative distributions are independent for this periodic field: if a finite distribution annihilates every Fourier mode, it is zero, and local test functions isolate its coefficients. This proves the needed positive finite-r gradient/value/Hessian covariance off the pins. Smooth Gaussian regression provides continuous conditional laws. The joint gradient and symmetric-Hessian jet is the relevant subspace, not an unconstrained vector derivative space.

Apply the weighted Kac-Rice formula to the gradient under Q_r with nonnegative weight W_r times the nonsingular index indicator and the open height-window indicator. Equivalently augment the auxiliary Gaussian field with f, its Hessians, and the endpoint Hessians. These weights are lower semicontinuous on the nonsingular index sets and can be truncated and approximated from below; continuous Gaussian regression checks the conditional-law requirement. Disintegrating the witness height then gives

    rho_win,j^W(X)=Z_r^-1 int_(b-k*r^3)^b
       p_(grad,height)|pins(0,s)
       E_Qr[W_r F_j(H_X) | grad=0,height=s] ds.          (A21)

One may use Armentano--Azais--Leon, arXiv:2304.07424v3, Theorem 7.1 with Theorem 2.2, and the critical-point example in Section 8.1. The paper supplies the integration framework; the actual r,t-uniform bounds are proved here. No formula for almost every gradient level is substituted for the required zero-gradient level.

For h_r<=|t|<=epsilon, (A8),(A17),(A19)-(A20), and the ACTUAL height interval of length k*r^3 give

    rho_win,j^W(ru,rt)
       <= C*k*r * |t|^-96 * exp(-c/t^2).                (A22)

The angular exponent is 18+72+6=96. Although this polynomial cost is large, its product with the exponential is uniformly bounded for 0<|t|<=epsilon. For example, e^y>=y^48/48! gives

    |t|^-96 exp(-c/t^2) <= 48!*c^-48.                  (A23)

Consequently rho_win,j^W<=C*k*r on this entire shrinking-cutoff outer region. For |t|>=epsilon, fixed-angle covariance and (A18)-(A20) yield the same bound directly. The r ledger is density -6, triple determinants +6, height +3, original normalizer -2: physical-area intensity +1.

## 8. Inner strip: all-height suppression is enough

It remains to handle |t|<h_r, where |u|>=A1 for small r. This is exactly the two-scale regime; the following summarizes its proved-in-package candidate ingredients without assuming outside review acceptance.

Put delta=sqrt(r^2+t^2), alpha=r/delta, beta=t/delta. Before unconditional covariance analysis subtract H'_r(ru)+r*t*L'_r(0) from f_x and L_r(ru) from f_z, then divide by (r^2*delta,r*delta). The joint contact rows on the genuinely random jets (f_xxxx,f_xxz,f_zz) are

    [u(u^2-1/4)*alpha/6, u*beta, 0],
    [0,(u^2-1/4)*alpha/2,beta].                         (A24)

Taylor remainders are O_L2(delta) uniformly in the aspect ratio: the first component's error is bounded by C(r^2+r|t|+t^2)/delta, the second by C(r^2+r|t|+r*t^2)/delta. The three minors, with a=u(u^2-1/4)/6 and b=(u^2-1/4)/2, are ab*alpha^2,a*alpha*beta,u*beta^2. Their squared sum is bounded below by c(alpha^4+beta^4)>=c/2. Thus the conditional covariance is uniformly positive, while its Gaussian target has size at least c/delta, after allowing a bounded mean. The raw gradient density is <=C*r^-3*delta^-2*exp(-c/delta^2).

For all three Hessians, conditional regression under this rare gradient target gives O(delta^-1) means and bounded covariance, so the single determinant-product expectation is <=C*delta^-6. With the original normalizer (A8), the ALL-HEIGHT physical-area intensity satisfies

    rho_all,j^W(ru,rt)<=C*r^-5*delta^-8*exp(-c/delta^2). (A25)

Detailed derivations are in unchanged TWO_SCALE_ADDENDUM.md Sections 2--7. In this inner strip, r<=delta and delta^2<=2r^(1/12), hence

    rho_all,j^W<=C*r^-13*exp[-c/(2*r^(1/12))]<=C*r.     (A26)

The last step follows explicitly from e^y>=y^168/168!, producing a factor C*r^14. Constants may be huge but finite; no usable numerical radius is claimed. A height restriction only decreases this intensity. Since k>=k_->0, C*r can be written C'*k*r with C'=C/k_-. This does NOT give uniformity as k tends to zero.

## 9. Conclusion, remaining review, and non-extrapolation

Equations (A22)-(A26) cover every point in the fixed scaled annulus. Integrating rho_win,j^W<=C*k*r over r E0, whose physical area is r^2 area(E0), proves the candidate (A2). Boundary lines can be assigned to either region; the count measure is absolutely continuous by Kac-Rice, so fixed area-null boundaries have zero expected count.

The proof uses TWO different strengths: a height-restricted, three-Hessian cancellation outside h_r, and an all-height exponential bound inside h_r. It does not give all-height O(r^3) over the full annulus, a positive leading coefficient, or a matching probability lower bound.

Scope covered at author-candidate level: one witness in each fixed scaled annulus A0>1,B0<infinity, d=2, full periodic covariance at fixed L, compact positive marks. Scope still excluded: pin neighborhoods and scaled radii near the pinned sites, a limit B0->infinity, r<<distance<<rho, simultaneous witness collisions, d>=3, k_->0, uniform torus-size/field-class limits, numerical24-jet constants, or persistence elder pairing. This is downstream D5 candidate work, not activation of main#94.

Independent review must check: R1 full six-pin regression and original normalizer; R2 conditioned-law expansion (A10) and O(r) covariance perturbation; R3 contact eigenfloor and cutoff constants; R4 full joint density with retained exponential; R5 deterministic three-gradient Hessian lemma and extra-conditioned C3 moments; R6 legitimate weighted/height-disintegrated Kac-Rice; R7 pointwise inner-strip estimate and union of regions; R8 exact scope. Test success, same-provider agreement, and publication provide no independent acceptance of these interfaces.

Source context inspected: PR22 head b2e1652f1374c3b45759324a1ad1fd4177458500; PR16 head32b80ee085dc6a40113d1e46e333cda50d57ba21, TRANSVERSE_BOUND_CANDIDATE.md blob024d927779f79fadaf932541ea6474f2995f8c50; current Math#24 review assignment. This addendum quantifies the fixed-transverse constants and supplies a new stitching argument; it neither edits nor claims independent review of those same-provider sources. The exact algebra tests verify finite pin completions, the contact matrix/minor, and exponent budgets only, not this analytic proof.
