# D5 two-scale addendum: a uniform near-axis crossover estimate

**Object:** D5-TWO-SCALE-20260925-v1. **Author:** OpenAI / ChatGPT.
**Disposition:** AUTHOR-SIDE ANALYTIC CANDIDATE; no independent acceptance.
**Scientific effect:** NONE. Original `PROOF.md` remains unchanged.

## 1. Purpose, exact model, and quantifiers

This addendum addresses the width gap between a physical O(r^2) axial tube and a fixed-width scaled transverse chart. It does NOT substitute a diverging K into the previous fixed-K theorem. Instead it derives a new covariance normalization uniform across the aspect ratio t/r, including the limiting regimes zero and infinity.

Fix L>0 and the centered variance-one Gaussian field on R^2/(L Z^2) with the FULL covariance

    K_L(h) = [sum_{n in Z^2} exp(-|h+Ln|^2/2)]
             / [sum_{n in Z^2} exp(-|Ln|^2/2)].

Choose any orthonormal frame E and write f(x,z)=F(x e1+z e2). Let b range over a fixed compact interval, 0<k_-<=k<=k_+<infinity, and 1<A<B<infinity. Condition on exactly

    f(-r/2,0)=b, f(r/2,0)=b-k*r^3,
    f_x(-r/2,0)=f_x(r/2,0)=f_z(-r/2,0)=f_z(r/2,0)=0.       (S1)

Call this continuous Gaussian regression law Q_r. With M,S the two endpoints, define

    F_j(H)=|det H|*1{negative index(H)=j},
    W_r=F_2(H_M)*F_1(H_S), Z_r=E_Qr W_r, dQ_r^W=(W_r/Z_r)dQ_r.

Singular matrices contribute zero. There is no elder or adjacency conditioning. At the witness (x,z)=(r*u,r*t), put

    delta=sqrt(r^2+t^2), alpha=r/delta, beta=t/delta.        (S2)

Actual parameters have alpha>0 and alpha^2+beta^2=1. The closed semicircle alpha>=0 is used ONLY to prove uniform bounds by compactification. It is not an additional physical point at r=0.

**Candidate conclusions.** Constants C,c,delta_0>0 exist, depending on the preceding fixed parameters, such that for A<=|u|<=B, r>0 and delta<=delta_0, uniformly in the frame and marks,

    p_{grad f(ru,rt)|S1}(0) <= C*r^-3*delta^-2*exp(-c/delta^2),   (S3)

and the endpoint-weighted index-j critical-point intensity per PHYSICAL area satisfies

    rho_j^W(ru,rt) <= C*r^-5*delta^-8*exp(-c/delta^2).            (S4)

Here rho_j^W includes the original endpoint-only normalizer once. Choose delta_0 also small enough that the local coordinate chart is embedded. Constants are existence bounds, not numerical enclosures.

For T_r(epsilon)={(r*u,r*t): A<=|u|<=B, |t|<=epsilon}, if sqrt(r^2+epsilon^2)<=delta_0, this gives the explicit all-height count estimate

    E_Qr^W N_j(T_r(epsilon))
      <= C*r^-3 * int_{-epsilon}^{epsilon}
                  (r^2+t^2)^-4 * exp[-c/(r^2+t^2)] dt.         (S5)

The sum over the three possible indices satisfies the same bound after changing C. Restricting witness heights only decreases this count.

## 2. Exact finite-r subtraction and the PR21/PR22 crosswalk

Let g(x)=f(x,0), h(x)=f_z(x,0), and a=r/2. Let H_r be the unique cubic matching g and g' at -a,+a, and L_r the unique affine function matching h at those endpoints. For arbitrary unconditioned data define

    S0=(g(-a)+g(a))/2, D0=(g(a)-g(-a))/(2a),
    S1=(g'(-a)+g'(a))/2, D1=(g'(a)-g'(-a))/(2a).

The vector U_r of midpoint derivative coefficients of H_r followed by those of L_r is

    U_r=(S0-a^2*D1/2, (3*D0-S1)/2, D1,
         3*(S1-D0)/a^2, L_r(0), L'_r(0)).                    (S6)

This is invertible: these six coefficients recover the original observations by evaluation. In particular the pinned target is

    v_r=(b-k*r^3/2, -(3/2)*k*r^2, 0, 12k, 0, 0),
    H_r(x)=b-k*r^3/2-(3/2)*k*r^2*x+2k*x^3, L_r(x)=0.        (S7)

The notation S1 in (S6) means the average slope, not the label of equation (S1).

The previous thin-tube proof used V_r=(S0,D0,D1,3*(S1-D0)/a^2,L_r(0),L'_r(0)). Our first two coordinates equal V_r[0]-r^2*V_r[2]/8 and V_r[1]-r^2*V_r[3]/24; the other coordinates are unchanged. This shear has determinant one. Thus the targets -k*r^2 and -(3/2)*k*r^2 in the two coordinate systems are consistent; neither author needs to invalidate the original centered pin transform. The invalid operation is replacing a finite-r target by its limit BEFORE singular rescaling.

Define two UNCONDITIONAL linear Gaussian observations

    Y1=[f_x(ru,rt)-H'_r(ru)-r*t*L'_r(0)]/(r^2*delta),
    Y2=[f_z(ru,rt)-L_r(ru)]/(r*delta).                       (S8)

Subtracting r*t*L'_r is essential even though L'_r vanishes at the prescribed pin values. Omitting it from the unconditioned observation would leave a divergent f_xz contribution in the covariance calculation. Under the pins, the affine identity is EXACT:

    grad f(ru,rt)=diag(r^2*delta,r*delta)
                   [Y + (6k*(u^2-1/4)/delta,0)].            (S9)

Consequently the gradient-zero condition is evaluated at a known target for Y, not imposed on a rank-deficient limiting deterministic row.

## 3. Uniform remainders across the aspect ratio

The periodic Fourier coefficients are strictly positive and decay as a Gaussian in lattice frequency. The sum of their square roots times any fixed polynomial in frequency is finite. Applying Minkowski to the Fourier series gives finite L^p norms of every fixed derivative supremum on the torus, uniformly in E, for every fixed finite p. The Taylor remainders below therefore have uniform L^p bounds, not merely pathwise bounds with untracked random constants.

Write Q=f_xxxx(0), T=f_xxz(0), S=f_zz(0), and

    a(u)=u*(u^2-1/4)/6, b(u)=(u^2-1/4)/2.

Cubic Hermite and affine interpolation, followed by Taylor expansion, give uniformly for bounded u,

    g'(ru)-H'_r(ru)=r^3*a(u)*Q+O_Lp(r^4),
    h'(ru)-L'_r(0)=r*u*T+O_Lp(r^2),
    h(ru)-L_r(ru)=r^2*b(u)*T+O_Lp(r^3).                    (S10)

For the first identity the exact quartic interpolation error is Q*(x^2-r^2/4)^2/24. The Taylor remainder through degree four and its first derivative give the displayed O(r^4) derivative bound: interpolation coefficients multiplying endpoint value errors are O(1/r), and those multiplying endpoint derivative errors are O(1). The bounded scaled evaluation point is allowed outside the interpolation interval. The last two identities follow from affine interpolation of h and its quadratic Taylor term.

Expand in the physical transverse coordinate z=r*t. Bounded derivative moments imply

    f_x(ru,rt)=g'(ru)+r*t*h'(ru)+O_Lp(r^2*t^2),
    f_z(ru,rt)=h(ru)+r*t*S+O_Lp(r^2*|t|+r^2*t^2).         (S11)

After (S8), the error in Y1 is bounded by

    C*(r^2+r*|t|+t^2)/delta <= 3C*delta,

and the error in Y2 by

    C*(r^2+r*|t|+r*t^2)/delta <= 3C*delta                  (r<=1).

There is no hidden factor 1/alpha or 1/beta in these bounds. Also U_r=U_0+O_Lp(r^2), where U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)(0). Thus, uniformly as delta->0 across ALL aspect ratios,

    Y=M(u,alpha,beta)*(Q,T,S)^T+O_L2(delta),
    M = [ a(u)*alpha   u*beta       0 ]
        [      0      b(u)*alpha  beta ].                  (S12)

These are unconditioned joint Gaussian observations. Their covariance and cross-covariance converge uniformly by Cauchy-Schwarz. No claim about convergence of densities at divergent targets is used.

## 4. Rank at both ends and in between

The nine distinct midpoint derivatives (U_0,Q,T,S) have positive-definite covariance for every frame. A hypothetical zero-variance combination defines a polynomial differential symbol vanishing at every rotated lattice frequency, because each Fourier variance is positive. Undo the rotation and fix one integer coordinate at a time: a polynomial vanishing on Z^2 is identically zero. Distinct derivative monomials then force the combination to be trivial. Compactness of O(2) gives uniform positive/finite eigenvalue bounds for this nine-jet covariance.

The three 2-by-2 minors of M are

    a*b*alpha^2,   a*alpha*beta,   u*beta^2.                (S13)

The first one alone vanishes when alpha->0; it cannot establish uniformity outside the original tube. By Cauchy-Binet,

    det(M M^T)=a^2*b^2*alpha^4+a^2*alpha^2*beta^2+u^2*beta^4.

Let m0=min_{A<=|u|<=B}{a(u)^2*b(u)^2,u^2}>0. Then

    det(M M^T)>=m0*(alpha^4+beta^4)>=m0/2,                (S14)

using alpha^2+beta^2=1. Its trace is uniformly bounded on the compact parameter set, so its least eigenvalue is uniformly positive. This proves rank both at beta=0 (axis) and alpha=0 (outer aspect-ratio limit), and throughout the transition. The fixed mark k is not used as a random covariance coordinate.

Adjoining U_0 to Y_0=M(Q,T,S) therefore gives uniform positive/finite joint covariance bounds. By Section 3 the same holds for (U_r,Y) at sufficiently small delta. Its Schur complement obeys

    lambda I <= Sigma=Cov(Y|U_r) <= Lambda I.             (S15)

The conditional mean m_Y=Cov(Y,U_r)*Cov(U_r)^-1*v_r is uniformly bounded, since the cross-covariance, inverse, and target are bounded.

## 5. Joint density at the compensating value

Let d(u)=6k*(u^2-1/4), so d(u)>=d0=6k_-*(A^2-1/4)>0. At the target tau=(-d(u)/delta,0) from (S9), boundedness of m_Y permits delta_0 to be chosen so that

    |tau-m_Y|>=d0/(2delta),   |tau-m_Y|<=C/delta.

The exact Gaussian density with (S15) gives p_Y(tau|U_r=v_r)<=C exp(-c/delta^2). The raw-gradient Jacobian is

    det diag(r^2*delta,r*delta)=r^3*delta^2.

This proves (S3). It is a joint two-component density estimate, not an inference from one marginal density.

## 6. Extra conditioning and the ORIGINAL normalizer

Stack the independent entries of H_M,H_S,H_w into a nine-dimensional vector H. Under Q_r its mean and covariance are uniformly bounded, by bounded derivatives and the stable U_r regression. Cauchy-Schwarz and (S15) bound Cov_Qr(H,Y). Conditioning further on Y=tau changes its mean by Cov(H,Y)*Sigma^-1*(tau-m_Y)=O(delta^-1), and decreases its covariance. A product of three 2-by-2 determinant absolute values is bounded by C*|H|^6, so

    E_Qr[W_r*F_j(H_w) | grad f(w)=0] <= C*delta^-6.        (S16)

This keeps all three correlated Hessians in one expectation. An endpoint-only O(r^2) estimate is not incorrectly retained after imposing the witness condition.

Separately, Z_r uses ONLY the original six pins. The finite-r identities and bounded conditional derivative moments give, with A_r=f_zz(0),

    det H_M/r=-6k*A_r+O_Lp(r), det H_S/r=6k*A_r+O_Lp(r),
    f_xx(M)/r=-6k+O_Lp(r).                                 (S17)

For example f_xx(0)=-(r^2/24)f_xxxx(0)+O_Lp(r^4) and f_xxx(0)=12k+O_Lp(r^2), obtained by subtracting the symmetric slope and height equations, give the first diagonal estimates. Transverse slope pins give f_xz(M),f_xz(S)=O_Lp(r). These establish (S17) by multiplying the entries with Holder's inequality.

The conditional variance of A_r is uniformly bounded above and below by the one-site jet rank and covariance convergence, and its mean is bounded. Thus P_Qr(-2<=A_r<=-1)>=p0>0. Intersect with determinant errors below 3k_- and longitudinal-diagonal error below 3k_-. Markov and the uniform L^p bounds show that the removed probability is O(r^p). For small r the intersection still has probability at least p0/2; there H_M is negative definite, det H_S<0, and both |det|>=3k_-*r. Therefore

    Z_r>=c_Z*r^2.                                         (S18)

No remote-conditioned normalizer replaces Z_r, and no numerical historical eigenfloor/radius is imported.

## 7. Weighted counting and the two width corollaries

For every fixed r>0 the witness domain stays away from the two endpoints. Distinct-site jet functionals are nondegenerate for the full periodic model: if a finite combination of point derivatives annihilates all Fourier modes, the corresponding distribution on the torus is zero; disjoint local test functions then isolate its coefficients. Gaussian regression preserves the required full gradient/symmetric-Hessian jet rank away from the pins. Smoothness, Gaussian jet transversality and continuous conditional regression verify the local Kac-Rice hypotheses at fixed r.

Use the weighted Kac-Rice formula (Stecconi, arXiv:2103.10853v1, Theorem 29, printed p.20; nonnegative-weight argument in Section 8.4, p.44). W_r is a nonnegative measurable functional of the gradient field on the torus with M,S removed: endpoint derivatives are recoverable as limits. Truncation handles its unbounded size. We have not assumed the tilted measure is Gaussian. Consequently

    rho_j^W(w)=Z_r^-1*p_grad(0|S1)
                   *E_Qr[W_r*F_j(H_w)|grad f(w)=0].       (S19)

Combining (S3),(S16),(S18) proves (S4). Physical area is r^2 du dt, proving (S5). There is no additional height-window factor; all heights are already counted.

**Power-width corollary.** Fix 0<gamma<=1 and K>0. For epsilon(r)=K*r^gamma, shrinking delta_0 if necessary so s^-8 exp(-c/s^2) increases for 0<s<=delta_0, (S5) implies

    E_Qr^W N_j(T_r(K*r^gamma))
       <= C_gamma*r^(-3-7gamma)*exp(-c_gamma/r^(2gamma)).  (S20)

Indeed s_max=sqrt(r^2+K^2*r^(2gamma)) lies between K*r^gamma and sqrt(1+K^2)*r^gamma for r<=1. The integration interval has length 2K*r^gamma. This proves the claimed power and exponential without substituting a diverging constant into the original theorem. For gamma=1 it recovers the fixed O(r^2) physical width; gamma=1/2 gives physical width O(r^(3/2)) and bound C*r^(-13/2)*exp(-c/r). Every fixed positive gamma yields O(r^N) for every fixed N.

**Logarithmic-width corollary (target-dependent width).** For each fixed N>0 there exists D_N>0 such that

    E_Qr^W N_j(T_r(D_N/sqrt(log(1/r)))) = O(r^N).           (S21)

For small r, s_max^2<=2D_N^2/log(1/r). Then (S5) is bounded by

    C*D_N^-7*r^(c/(2D_N^2)-3)*(log(1/r))^(7/2).

Choose D_N^2<=c/[2(N+4)] and use r*(log(1/r))^(7/2)->0. This D_N depends on N and on an existence constant c; it is not numerically certified. It does NOT assert that one fixed positive logarithmic-width coefficient gives every power N. Markov yields only upper bounds on the corresponding witness-existence probability.

## 8. Limits and source/review crosswalk

The old fixed-K proof is preserved byte-for-byte. PR21's density argument at b420099b2440da3a8ba62f7000feacc9fc88069b was read and supplies the useful unconditioned Hermite subtraction convention; it and PR22's original 35eddbb5ab7dee33a659b64d8e9b0efeec16b70d are exposed SAME-PROVIDER candidates, not independent acceptance. The relevant identities and count ingredients are rederived here, rather than accepted by citation.

This addendum narrows the unresolved intermediate-width region but does not establish a uniform O(r^3) bound on a fixed scaled-width annulus. For fixed positive epsilon, the crude bound (S5) retains r^-3; simply integrating it does not close that target. One still needs a complementary count estimate uniform down to the shrinking cutoff, or a sharper determinant/height argument. No result is asserted at the pin sites, near u=0, at intermediate physical distances r<<|x|<<rho, for witness-witness collisions, in d>=3, at k_-=0, or uniformly over L. No elder-selection or generalized lifetime theorem is promoted.

Independent review questions: S6--S9 exact pins and subtraction; S10--S12 aspect-uniform remainders; S13--S15 all-minors rank argument; large-target Gaussian evaluation; S16 rare-conditioning determinant cost; S17--S19 normalizer and weighted Kac-Rice; S20--S21 quantifier order; explicit complement. Exact polynomial tests and deliberately broken implementations check finite algebra only. They do not establish analytic validity or reviewer independence.

### Reconnaissance delta

On 25 September 2026 I searched for the weighted Kac-Rice and Gaussian critical-point interpolation references, opened arXiv:2103.10853v1, and inspected page images for printed pp.20 and44. Those pages support the weighted-count theorem and its nonnegative-weight extension, not the new two-scale estimates. Gass--Stecconi, PTRF190 (2024), DOI10.1007/s00440-024-01273-5, was located as interpolation/nondegeneracy context only; its finite-moment theorem is not used to claim any rate here. No comprehensive novelty or priority claim is made.
