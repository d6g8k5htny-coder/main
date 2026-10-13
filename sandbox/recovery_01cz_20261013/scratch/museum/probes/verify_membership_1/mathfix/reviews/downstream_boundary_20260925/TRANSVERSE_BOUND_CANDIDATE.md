# Corrected fixed-transverse chart: a cubic conditional count bound

**Author:** OpenAI / ChatGPT, 25 September 2026.  
**Disposition:** corrective downstream proof candidate; separate-agent review required.  
**Scope:** dimension two, a fixed compact scaled chart bounded away from the axis. No full annulus, collision, finite numerical constant, or scientific promotion is claimed.

This note is a repair proposal following the exact finite-pin counterexample to Math PR9. It does not amend Cursor's proof body. The algebraic obstruction is settled by the accompanying rational-polynomial oracle; the continuum argument below still needs independent analytic review.

## 1. Model and exact count

Fix the variance-one normalized periodized Gaussian covariance

    K_L(z) = sum_(n in Z^2) exp(-|z+Ln|^2/2)
             / sum_(n in Z^2) exp(-|Ln|^2/2)

on the fixed two-dimensional torus of side L>0. Fix compact birth and gap intervals, with 0<k_min<=k<=k_max. Constants below may depend on L, these intervals, and the declared chart.

In an arbitrary orthonormal local frame, prescribe M=(-r/2,0), S=(r/2,0), heights b and b-k*r^3, and both gradients zero. Let Q_r be the full Gaussian regression law at these six observations. Set

    W_r = |det H_M det H_S| 1{H_M negative definite, H_S index one},
    Z_r = E_Qr W_r,    dQ_r^W = (W_r/Z_r) dQ_r.

There is no adjacency condition or pin Jacobian in Z_r.

For fixed A>1, B>A and eta>0, let

    K = {(u,v): A<=sqrt(u^2+v^2)<=B, |v|>=eta}.

Let E be any Borel subset of K, and N_j(rE) count critical points of index j in rE whose heights lie strictly between b-k*r^3 and b.

**Candidate theorem.** There are C<infinity and r_*>0 such that, uniformly over the compact marks, all frames, E and j=0,1,2,

    E_(Qr^W) N_j(rE) <= C*k*r^3*area(E),    0<r<=r_*.

The constants are qualitative. They are NOT uniform as eta tends to zero, B tends to infinity or the gap interval approaches zero.

## 2. Endpoint regression and full normalizer

Use the exact centered six-observation transform

    U_r = ((fM+fS)/2, (fS-fM)/r, (fxS-fxM)/r,
           (6/r^2)*(fxM+fxS-2*(fS-fM)/r),
           (fzM+fzS)/2, (fzS-fzM)/r).

Its target is (b-k*r^3/2,-k*r^2,0,12k,0,0). Its contact limit is

    U_0=(f,fx,fxx,fxxx,fz,fxz)_0,
    target=(b,0,0,12k,0,0).

Positive Fourier variances at every lattice mode imply positive covariance for any independent finite list of derivative-evaluation functionals. A zero-variance linear combination would annihilate every Fourier mode and hence be the zero derivative distribution. The Gaussian Fourier coefficients decrease rapidly enough that all fixed derivative-supremum moments are finite.

Taylor integral formulas give uniform convergence of the transformed observation covariance and its derivative-field cross-covariances on the compact frame set. The limiting covariance is positive, so inverse covariances remain bounded. Gaussian regression therefore gives uniformly bounded C^m moments for every fixed m and finite moment order under Q_r, and convergence of all the needed midpoint jets to their U_0-conditioned counterparts. No rotational invariance of the periodic field is assumed.

Appending a=fzz(0) leaves the contact jet list independent. Under Q_r, the endpoint Hessian entries satisfy

    fxx(M)/r -> -6k, fxx(S)/r -> 6k,
    fxz(M),fxz(S)=O_Lp(r), fzz(M),fzz(S)->a.

Consequently, by type convergence away from a=0 and uniform integrability,

    Z_r/r^2 -> 36*k^2 E[a^2 1{a<0} | U_0=target] = z_0 > 0.

The contact a has a nondegenerate Gaussian law. The limiting expectation is continuous in the compact parameters. Thus Z_r>=z_* r^2 uniformly for sufficiently small r. A compact-subsequence argument supplies uniformity of the convergence, without assuming an unproved uniform pointwise type margin.

These are local Gaussian regression and full-normalizer interfaces also used in the existing fixed-remote note. They do not consume a global elder-selection conclusion or any historical numerical RN certificate.

## 3. Keep finite-r midpoint corrections before taking limits

Write a=fzz(0), q=fxxz(0), c=fxzz(0), d=fzzz(0), for the finite-r field. Taylor's formula and the original pins imply, with remainders bounded in every fixed Lp uniformly on K,

    fx(ru,rv)/r^2
       = 6k*(u^2-1/4)+q*u*v+(c/2)*v^2 + O_Lp(r),

    fz(ru,rv)
       = r*a*v+r^2*((q/2)*(u^2-1/4)+c*u*v+(d/2)*v^2)
         + O_Lp(r^3),

    f(ru,rv)-b
       = r^2*a*v^2/2+r^3*(k*(2u^3-3u/2-1/2)
         +(q/2)*(u^2-1/4)*v+(c/2)*u*v^2+(d/6)*v^3)
         + O_Lp(r^4).

The origin corrections are fx(0)=-(3/2)kr^2+O(r^3 ||f||C4), fz(0)=-qr^2/8+O(r^3 ||f||C4), and f(0)=b-kr^3/2+O(r^4 ||f||C4). They cannot be discarded before rescaling. The accompanying exact cubic family verifies every displayed coefficient without any remainder.

Define the ACTUAL normalized witness vector at X=(ru,rv):

    J_r = (fx(X)/r^2,
           fz(X)/r,
           (f(X)-b-(r*v/2)*fz(X))/r^3).

Its limiting random part is a linear map of (a,q,c,d), with mean shifts depending on k:

    J_1 -> 6k*(u^2-1/4)+q*u*v+(c/2)*v^2,
    J_2 -> a*v,
    J_3 -> k*(2u^3-3u/2-1/2)+(q/4)*(u^2-1/4)*v-(d/12)*v^3.

The coefficient minor on columns (a,c,d) is v^6/24. It is bounded away from zero on |v|>=eta. The four added midpoint jets are independent from U_0, so their conditional covariance is positive definite. Compactness gives uniform positive covariance for J_r under Q_r when r is small. This is the true three-variable gradient-and-height density; it does NOT treat k as a free Gaussian variable after k has been pinned.

The exact affine change from (fx,fz,f) to J_r has determinant r^-6. At gradient zero and f(X)=b+r^3*tau, the J_r target is (0,0,tau), with tau in [-k,0]. Hence

    p_(grad f(X),f(X)|pins)(0,b+r^3*tau) <= C*r^-6.

The height Jacobian has not disappeared. It is kept explicitly here and canceled only by the actual dt=r^3*d tau in the final integral.

## 4. Three critical gradients force all three Hessians to be small

This step is deterministic. Under the endpoint pins and the additional witness condition grad f(X)=0, let M_3 bound the third directional derivatives on the local convex ball containing M,S,X.

The two endpoint gradient conditions and Taylor's integral formula give

    |fxx(M)| + |fxz(M)| <= C*r*M_3.

Expanding fz(X) from M, and using fz(M)=fz(X)=0, gives

    0 = r*(u+1/2)*fxz(M) + r*v*fzz(M) + O(r^2*M_3).

Since |v|>=eta, it follows that |fzz(M)|<=C*r*M_3. Hessian Lipschitz control along segments then gives

    ||H_M||+||H_S||+||H_X|| <= C*r*M_3.

In dimension two, therefore,

    W_r * |det H_X| <= C*r^6*M_3^6.

This is where the two endpoint determinants AND the witness determinant acquire the needed extra powers. Using the raw unconditioned r^1 determinant power after conditioning would miss this mechanism.

The same estimate holds with any witness index indicator, since dropping it only increases the absolute bound.

## 5. Uniform derivative moments after the extra witness pins

Under Q_r, J_r has uniformly bounded mean and nonsingular covariance from Section3. Its cross-covariance with any fixed derivative field is uniformly bounded in the corresponding supremum norm: use the uniformly bounded Q_r derivative moments, the bounded J_r moments, and Cauchy-Schwarz.

The Gaussian regression coupling

    F_conditioned = F_Qr + Cov(F_Qr,J_r) Cov(J_r)^-1 (target-J_r)

retains all original endpoint pins and has exactly the extra-conditioned law. Because target=(0,0,tau) lies in a fixed compact set, its C^3 moments of every fixed finite order are uniformly bounded. In particular E[M_3^6 | pins,J_r=(0,0,tau)]<=C. No independence between the endpoint and witness determinants is asserted.

Thus the full three-determinant conditional expectation is at most C*r^6.

## 6. The complete count ledger

Apply the marked Kac-Rice formula under Q_r on the spatial chart, disintegrating the witness height and retaining the endpoint weight. At fixed r>0 the witness stays away from both endpoints; distinct-site finite-jet positivity supplies the required nondegenerate conditional gradient/value laws. Filtered absolute determinants are continuous through singular matrices and have polynomial growth. Truncation and the uniform moments justify the weight; height indicators can be obtained by nonnegative approximation.

The unnormalized numerator is bounded by

    integral_(rE) integral_(b-k*r^3)^b
         p_(grad,height|pins)(0,t)
         E[W_r*|det H_X|*1{index=j} | pins,grad=0,height=t] dt dX
      <= C * r^2*area(E) * k*r^3 * r^-6 * r^6
      = C*k*r^5*area(E).

Division by the FULL Z_r>=z_*r^2 proves the stated O(k*r^3) expected count. Markov then supplies the corresponding upper bound on the probability of at least one witness. No probability-to-expected-count reversal is used.

## 7. Exactly what is and is not repaired

This proposal supplies a route through the fixed two-dimensional transverse chart with |v|>=eta. It replaces the mistaken finite-pin contact rows and accounts for the full gradient/height Jacobian and all three determinant factors. It does not cover eta->0 (thin belt/axis), pin collisions, intermediate r<<distance<<rho, multiple witness collisions, dimension three, numerical constants, or the historical all-cell/24-jet certificate. It establishes no positive leading count coefficient and no matching probability lower bound.

The argument is an OpenAI-authored repair candidate requiring another agent's review. The independent polynomial oracle validates its local algebra, not the continuum Gaussian regression/Kac-Rice proof.

Primary framework: Armentano, Azais and Leon, *On a general Kac-Rice formula for the measure of a level set*, arXiv:2304.07424v3, Theorem7.1 and Section8.1. The paper supplies the expected-integral framework, not this project's conditioned estimate. The actual application and its bounds are given above.
