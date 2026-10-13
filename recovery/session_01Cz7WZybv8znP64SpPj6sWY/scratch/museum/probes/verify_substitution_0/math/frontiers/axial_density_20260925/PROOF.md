# Inner axial belt: an exponential conditional-gradient density bound

Object: **RN-INNER-AXIAL-DENSITY-20260925-v1**. Author: OpenAI / ChatGPT.
Disposition: **AUTHOR-SIDE ANALYTIC CANDIDATE; distinct-lane review required**.
Scientific/register effect: **NONE**. This is a density sublemma, not a weighted critical-point count or RN_UNIF closure.

## 1. Model, quantifiers, and conclusion

Fix L>0. Let f be the centered variance-one Gaussian field on the two-dimensional torus R^2/(L Z^2) with covariance

    K_L(t) = sum_{n in Z^2} exp(-|t+Ln|^2/2)
             / sum_{n in Z^2} exp(-|Ln|^2/2).

Fix B0<infinity, 0<k0<=k1<infinity, 1<A<B<infinity, and 0<=W<infinity. For any orthonormal frame R, use local coordinates (x,z) along its columns. With M=R(-r/2,0), S=R(r/2,0), condition on the ORIGINAL six observations

    f(M)=b, f(S)=b-k*r^3, grad f(M)=grad f(S)=0,
    |b|<=B0, k0<=k<=k1.

Denote their continuous Gaussian regression law by Q_r. Gradients below are in the frame R; the orthogonal coordinate change preserves density at zero. At the witness X=R(r*u,r^2*w), assume A<=|u|<=B and |w|<=W.

**Candidate theorem.** There are C,c,r_*>0 depending only on L,B0,k0,k1,A,B,W such that, uniformly in the stated parameters and frame, for 0<r<=r_*,

    p_{grad f(X) under Q_r}(0) <= C*r^(-5)*exp(-c*k^2/r^2).       (1)

The covariance at the witness is nonsingular for these r. The density is with respect to two-dimensional Lebesgue measure in GRADIENT space. It contains neither a pin-density factor nor a spatial Jacobian nor an endpoint Palm weight.

This covers the physical belt |z|<=W*r^2, equivalently |v|<=W*r when X=R(r*u,r*v). It is not a bound for a fixed-width scaled belt |v|<=epsilon. The omitted region W*r<|v|<eta is substantial and remains open here. Taking W=0 gives the axial line, but W>0 also covers positive area.

## 2. Exact Hermite observations before the limit

Put a=r/2, g(x)=f(R(x,0)), h(x)=partial_z f(R(x,0)). Let H_r be the unique cubic matching g and g' at both -a and a. Let L_r be the unique linear function matching h at those two points. Define

    U_r=(H_r(0),H'_r(0),H''_r(0),H'''_r(0),L_r(0),L'_r(0)).

For arbitrary, UNCONDITIONED input observations set

    S0=(g(-a)+g(a))/2, D0=(g(a)-g(-a))/(2a),
    S1=(g'(-a)+g'(a))/2, D1=(g'(a)-g'(-a))/(2a).

Then the first four coordinates of U_r are

    S0-a^2*D1/2, (3*D0-S1)/2, D1, 3*(S1-D0)/a^2,              (2)

and the last two are (h(-a)+h(a))/2 and (h(a)-h(-a))/(2a). These identities follow by substituting a cubic with derivative coefficients into all four interpolation equations. They also prove invertibility for every r>0: the coefficient vector recovers every original observation by evaluation.

Under the six specified pins, exactly, not asymptotically,

    H_r(x)=b-k*r^3/2-(3/2)*k*r^2*x+2*k*x^3, L_r(x)=0,
    U_r=v_r=(b-k*r^3/2,-(3/2)*k*r^2,0,12*k,0,0).             (3)

Thus H'_r(r*u)=6*k*r^2*(u^2-1/4). The offset does not disappear after division by r^2.

**Crosswalk to the fixed-remote source.** Its exact centered vector is

    V_r=(S0,D0,D1,3*(S1-D0)/a^2,L_r(0),L'_r(0)).

Our first two coordinates are V_0-r^2*V_2/8 and V_1-r^2*V_3/24, with the other four unchanged (indices in this sentence are zero-based). This change has determinant one. The old finite-r vector is therefore a VALID normalization of the same pins. The error in the unshifted chart is replacing finite-r data by their contact limit before rescaling, not an invalidity of every use of the centered vector. No parent normalizer or theorem is refuted by changing coordinates here.

## 3. The normalized UNCONDITIONAL Gaussian witness

Define the two linear Gaussian observations before any conditioning:

    Y_r1 = [f_x(r*u,r^2*w)-H'_r(r*u)-r^2*w*L'_r(0)]/r^3,
    Y_r2 = [f_z(r*u,r^2*w)-L_r(r*u)]/r^2.                     (4)

The L' subtraction in Y_r1 is essential. Under the pins L'=0, but without the subtraction the unconditioned covariance would contain the divergent term w*f_xz(0)/r. A formula that works only after substituting zero pin values is not automatically a valid covariance normalization.

Write the random midpoint jets Q=f_xxxx(0), T=f_xxz(0), S=f_zz(0), and

    alpha(u)=u*(u^2-1/4)/6, beta(u)=(u^2-1/4)/2.

Uniformly on the declared compact parameter set,

    U_r -> U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)(0),
    Y_r -> Y_0=(alpha(u)*Q+u*w*T, beta(u)*T+w*S)              (5)

in L^2, with errors O(r^2) and O(r), respectively.

Here are explicit remainder reasons. Positive Fourier coefficients, proportional to exp(-2*pi^2*|n|^2/L^2), give summability of the coefficient square roots times every polynomial in |n|. The Fourier series consequently has finite L^p moments of every fixed derivative supremum, uniformly in frame. Taylor estimates below therefore hold in L^2 uniformly, not merely pathwise with uncontrolled random constants.

For g, cubic Hermite interpolation reproduces degrees <=3. Its quartic error is

    (Q/24)*(x^2-a^2)^2.

Differentiating and evaluating x=r*u gives r^3*alpha(u)*Q. The fifth-order value remainder and fourth-order derivative remainder give O_L2(r^4) after interpolation and differentiation at r*u: the value-data coefficients grow at most as 1/r and the derivative-data coefficients stay bounded. For h, linear interpolation gives

    h(r*u)-L_r(r*u)=r^2*beta(u)*T+O_L2(r^3),
    h'(r*u)-L'_r(0)=r*u*T+O_L2(r^2).

Finally Taylor expansion in z=r^2*w gives

    f_x(r*u,r^2*w)=g'(r*u)+r^2*w*h'(r*u)+O_L2(r^4),
    f_z(r*u,r^2*w)=h(r*u)+r^2*w*S+O_L2(r^3).

Substitution in (4) proves (5). These are statements about the unconditioned jointly Gaussian observations; covariance and cross-covariance convergence follow by Cauchy-Schwarz. No convergence at a target tending to infinity is being assumed.

## 4. Uniform conditional covariance and bounded regression mean

The nine distinct jet functionals

    J=(f,f_x,f_xx,f_xxx,f_z,f_xz,Q,T,S)(0)

have positive-definite covariance for every frame. Indeed a zero-variance linear combination would have zero action on every Fourier mode, since all mode variances are strictly positive. Its derivative symbol is a polynomial in the rotated lattice frequency. A polynomial vanishing on all of Z^2 is zero: first fix either integer coordinate and use the one-variable polynomial identity, then apply it to each coefficient in the other coordinate. Rotation is invertible, so the original polynomial would also be zero. Distinct derivative monomials are linearly independent. This contradicts a nontrivial combination.

After adjoining the six U_0 coordinates, the two Y_0 rows are independent: on the unpinned random columns (Q,T) their 2-by-2 minor is

    alpha(u)*beta(u)=u*(u^2-1/4)^2/12 != 0.                  (6)

The mark k is fixed and is NOT a random covariance direction. The minor stays away from zero on A<=|u|<=B. Continuity and compactness of O(2) and the (u,w) set give a uniform positive minimum eigenvalue for Cov(U_0,Y_0), and a finite maximum. This uses compactness of frames, not rotational invariance of the torus covariance.

By (5), the same bounds, with possibly changed constants, hold for Cov(U_r,Y_r) at all sufficiently small r. Its Schur complement

    Sigma_r=Cov(Y_r)-Cov(Y_r,U_r)*Cov(U_r)^(-1)*Cov(U_r,Y_r)

is therefore uniformly positive and bounded: choose 0<lambda<=Lambda<infinity with

    lambda*I <= Sigma_r <= Lambda*I.                         (7)

For completeness the lower Schur bound follows by minimizing the quadratic form of the positive joint covariance over the U coordinates; the squared norm of the remaining Y coordinates is unchanged. The upper bound follows by taking the U coordinates zero.

Under U_r=v_r, Y_r is Gaussian with mean

    m_r=Cov(Y_r,U_r)*Cov(U_r)^(-1)*v_r.

The cross-covariances and inverses are uniformly bounded and (3) is a bounded vector for the fixed mark range. Thus |m_r|<=M0 uniformly. This does not declare the conditional mean zero.

## 5. Density at the required compensating value

Under the original pins, equation (4) becomes exactly

    grad f(X)=(6*k*r^2*(u^2-1/4)+r^3*Y_r1, r^2*Y_r2).

The transformation Y_r to this raw gradient has determinant r^5. Consequently

    p_{grad f(X)|U_r=v_r}(0)
      =r^(-5)*p_{Y_r|U_r=v_r}((-6*k*(u^2-1/4)/r,0)).        (8)

Let delta=6*(A^2-1/4)>0. The target t in (8) satisfies |t|>=delta*k/r. Shrink r_* so that r_*<=delta*k0/(2*M0) when M0>0, and ensure all local coordinates remain inside the torus injectivity scale. Then |t-m_r|>=delta*k/(2*r).

The exact two-dimensional Gaussian density and (7) imply

    p_Y(t) <= (2*pi*lambda)^(-1)
              *exp(-|t-m_r|^2/(2*Lambda))
           <= (2*pi*lambda)^(-1)
              *exp(-delta^2*k^2/(8*Lambda*r^2)).

Together with (8), this proves (1), with C=(2*pi*lambda)^(-1) and c=delta^2/(8*Lambda). These are existence constants, not numerical enclosures. The lower bound k0>0 is used essentially to dominate the bounded mean uniformly. No claim uniform down to k=0 follows.

## 6. Import boundary and review targets

This is an author-side proof of the declared density sublemma. It does NOT establish the height-conditioned density, conditional Hessian moments after forcing the large compensating value, the endpoint-weighted Palm count, its full normalizer, a bound across the remaining thin belt, pin/witness collisions, intermediate distances, d=3, numerical C_H, or historical 24-jet certification. In particular exponential smallness of a Gaussian gradient density must not be substituted for a weighted Kac-Rice integral without controlling its conditional determinant factors.

Distinct reviewer: independently check (2)-(4), the uniform Taylor remainders in (5), the random-jet minor (6), the Schur/mean bounds (7), and the affine density identity (8). Check all quantifiers and scope exclusions. Return ACCEPT/AMEND/COUNTEREXAMPLE per interface against the exact reviewed commit. Tests verify finite algebra, not these analytic arguments.

Source context, not unproved theorem imports:
- Math- main at baca69c394ab42130c61771bee74e808703f1ce7, frontiers/remote_window_20260924/PROOF.md, blob b383bfcc88ec4ad497dff01fb6640e429ba24a84: exact field and observation conventions.
- Math- PR19 at ee8629f37977f4132da4c5af7e1dd61e9d94aa1d, reviews/d5_finite_r_hermite_repair_20260925/REPAIR.md, blob ac13515dfb942fbc5468e5006547fc434d5c00ea: finite-pin repair context. Equations used here are rederived above.
- Math- PR9 at c393f6bd1b2a24bb4741baccc30f5c88969c474a, chart PROOF.md blob c96de6b907f23347964acc8b35c690e68e5c91f0: inspected erroneous unshifted rows; NOT imported.
- Gass and Stecconi, The number of critical points of a Gaussian field: finiteness of moments, PTRF 190 (2024), 1167-1197, DOI 10.1007/s00440-024-01273-5; preprint arXiv:2305.17586v2 (2023). Methodological prior art only; see RECONNAISSANCE.md. No novelty priority established by this search.
