# Remote critical points in a shrinking height window

**Object:** RN-FIXED-REMOTE-WINDOW-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side proof candidate; nonauthor analytic review open.

[Research home](https://github.com/d6g8k5htny-coder/main) · [Math index](../../README.md) · [Earlier count interface](../three_fronts_20260924/RN_COUNT_INTERFACE.md)

## 1. The count and the new scope

The earlier RN count interface identified a three-determinant integral whose unnormalized numerator must be O(r^5), but did not bound it. This note supplies that bound for a precise part of the problem: **a fixed remote spatial region and the shrinking height window between the pinned maximum and saddle**. The spatial exclusion radius must not shrink with r.

Fix d>=2 and L>0. On X=R^d/(L Z^d), use the centered variance-one Gaussian field with the exact normalized periodic covariance

    K_L(z)=sum_(n in Z^d) exp(-|z+Ln|^2/2)
             / sum_(n in Z^d) exp(-|Ln|^2/2).

Fix compact B=[b_-,b_+] and K=[k_-,k_+] with 0<k_-<=k_+<infinity. Let R be any orthonormal frame, with axial direction u=Re_1. For small r>0 the pins are

    M=-(r/2)u, S=(r/2)u,
    f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0.

Q_r denotes continuous Gaussian regression on these 2(d+1) observations. For a symmetric matrix H define

    F_j(H)=|det H| 1{H has j negative eigenvalues},

with value zero on singular matrices. Define the actual endpoint weight and FULL normalizer

    W_r=F_d(H_M) F_(d-1)(H_S),
    Z_r=E_(Q_r) W_r,             dQ_r^W=(W_r/Z_r)dQ_r.

There is no adjacency condition, no additional pin-density factor, and no Jacobian inside Z_r.

Fix 0<rho<L/4 and D_rho={x:dist_X(x,0)>=rho}. For a Borel E contained in D_rho and j in {0,...,d}, let

    N_(r,j)(E)=#{x in E: grad f(x)=0, index H_x=j,
                            b-k r^3<f(x)<b}.             (1)

Thus the height window shrinks; the spatial exclusion does not. Boundary heights may equivalently be included: their expected critical-point counts are zero by the height-density formula below. The set E is arbitrary; the uniform measure estimate even permits choosing different Borel E_r inside the SAME D_rho.

**Theorem A (fixed-remote mean measure).** There are r_*>0 and C<infinity, depending on d,L,rho,B,K, such that for r<=r_* and all b,k,R,j and Borel E subset D_rho,

    | E_(Q_r^W) N_(r,j)(E)/(k r^3)
           - integral_E Lambda_j(x;b,k,u) dx |
        <= C r |E|.                                      (2)

The explicitly defined contact kernel Lambda_j in Section 5 is continuous, finite and strictly positive. On these compact parameter sets it is bounded above and away from zero. In particular

    E_(Q_r^W) N_(r,j)(E) <= C k r^3 |E|,
    P_(Q_r^W){N_(r,j)(E)>=1} <= C k r^3 |E|.              (3)

For fixed E of positive volume the expected count is asymptotic to a strictly positive constant times r^3. This is NOT a matching lower bound on the event probability. Sum over the finitely many indices to count all critical types.

The normalized mean measures converge in total variation on D_rho at rate O(r): a density version differs uniformly from Lambda_j by O(r). No numerical value of C or r_* is supplied. No constant is claimed uniform as rho tends to zero, marks become unbounded, or d,L vary.

## 2. Exact observations and uniform remote nondegeneracy

Order the original axial observations as f(M),f_x(M),f(S),f_x(S), followed by pairs f_yj(M),f_yj(S). Use the centered transform

    U_r=[(f(M)+f(S))/2,
         (f(S)-f(M))/r,
         (f_x(S)-f_x(M))/r,
         (6/r^2){f_x(M)+f_x(S)-2(f(S)-f(M))/r},
         (f_yj(M)+f_yj(S))/2, (f_yj(S)-f_yj(M))/r for j].

It has absolute determinant 12 r^(-(d+3)) and target

    v_r=(b-k r^3/2,-k r^2,0,12k,0,...,0).

Its contact limit and target are

    U_0=(f,f_x,f_xx,f_xxx,f_yj,f_xyj for j) at 0,
    v_0=(b,0,0,12k,0,...,0).                              (4)

These are the exact conventions of the parent matrix-cap source, not a new normalization. Since U_r is an invertible re-expression of the original pins, conditioning on U_r=v_r gives the SAME Q_r. Its determinant does not get multiplied into the conditional count formula again.

Poisson summation gives strictly positive Fourier variances proportional to exp(-2pi^2|n|^2/L^2). All sums of their square roots times any fixed polynomial in |n| converge. The real Fourier series therefore has smooth versions and finite moments of every fixed derivative supremum.

Any finite collection of distinct derivative-evaluation functionals at distinct torus sites has positive covariance. A zero-variance linear combination would annihilate every Fourier mode, hence would be the zero distribution. Derivatives of point masses at distinct sites are independent, as tested by smooth functions with individually prescribed finite jets. At one site the rotated derivative monomials are independent; only independent symmetric Hessian entries are listed. This proves the assertion without a finite-mode or sampled-eigenvalue substitution.

For x in D_rho write Y_x=(grad f(x),f(x)). The joint vector (U_0,Y_x) has no repeated jet functionals and positive covariance. Its covariance varies continuously on the compact set O(d) x D_rho. It consequently has a uniform positive smallest eigenvalue and finite largest eigenvalue. The same holds after adjoining the independent entries of A_0=D_y^2 f(0) and H_x. These extra entries are distinct from the observations.

Centered differences imply U_r-U_0=O(r^2) in every finite L^p norm. More precisely, modewise Taylor remainder bounds and the Fourier summability above give second-order convergence of the observation covariance and its cross-covariances with any fixed derivative field, in every fixed C^q norm. For example the fourth axial row is f_xxx(0)+(r^2/40)f_xxxxx(0)+O(r^4). The bounds are uniform in frame and remote position.

Choose r_*<=rho and below the torus injectivity scale. Uniform convergence shows that the covariances of (U_r,Y_x), their inverses, and the relevant Schur complements are uniformly bounded and positive for r<=r_*. Every x stays at least rho/2 from both endpoints. The appended endpoint-transverse and witness-Hessian conditional covariances are also positive by their contact limits. These arguments use compactness, not rotational invariance of the torus.

## 3. A common regression coupling, with the extra height pin retained

For t in [b-k r^3,b], let V_r=(U_r,Y_x) have target a_r=(v_r,0,t). At contact take V_0=(U_0,Y_x) with target a_0=(v_0,0,b). If F is one unconditioned field copy, define

    F_(r,x,t)(z)=F(z)+Cov(F(z),V_r) Cov(V_r)^(-1)(a_r-V_r),
    F_(0,x,b)(z)=F(z)+Cov(F(z),V_0) Cov(V_0)^(-1)(a_0-V_0).

Each marginal has precisely the indicated conditional Gaussian law. All targets are bounded on the stated compact sets, and |a_r-a_0|<=C(r^2+|t-b|). Section 2 gives, for every fixed finite p>=1 and derivative order q,

    ||F_(r,x,t)-F_(0,x,b)||_(L^p(C^q)) <= C_(p,q) r^2,
    E[(1+||F_(r,x,t)||_(C^q)+||F_(0,x,b)||_(C^q))^p]<=C. (5)

Here |t-b|<=k_+ r^3 is included. The same construction without Y_x applies to the endpoint-only laws. The fields, their Hessians and remote observations are NOT declared independent; this is a coupling, not a factorization.

The conditional density p_(Y_x|U_r=v_r)(0,t) differs uniformly by O(r^2) from p_(Y_x|U_0=v_0)(0,b). To see this, the conditional means and covariances are Schur expressions in the uniformly positive joint covariance, and differ by O(r^2); t-b=O(r^3). On the bounded target/positive covariance domain the Gaussian density is continuously differentiable with bounded derivatives. In particular all these densities are uniformly bounded. No actual mean value is replaced by a zero mean unless the regression supplies it.

## 4. The endpoint weights vanish to second order even after remote conditioning

All extra conditional fields retain both original gradient pins and the exact height gap. Put

    alpha_i=f_xx(i)/r, beta_i=grad_y f_x(i)/r, A_i=D_y^2 f(i),
    H_i=[[r alpha_i,r beta_i^T],[r beta_i,A_i]].

The gradient integral identities give |alpha_i|,||beta_i||<=M3/2. The Hermite height identity forces an averaged f_xxx equal to 12k, whence

    alpha_M=-6k+O(r M4), alpha_S=6k+O(r M4).              (6)

All derivative moments in these bounds remain uniform under the EXTRA remote value and gradient pins by (5). This extra-conditioning verification is essential: an estimate established only under Q_r cannot simply be presumed uniform after Y_x has been fixed.

Let A_0 be the transverse block of F_(0,x,b) at 0. Equation (5), displacement of endpoints by r/2, and (6) show, in every fixed finite L^p norm,

    A_i-A_0=O(r), alpha_M+6k=O(r), alpha_S-6k=O(r),
    beta_i=O(1), H_x(F_(r,x,t))-H_x(F_(0,x,b))=O(r^2).    (7)

For clarity two purely deterministic facts are reproved. For symmetric X,Y,

    |F_j(X)-F_j(Y)|
       <= n max(||X||,||Y||)^(n-1)||X-Y||.               (8)

If one filtered determinant contributes and the indices differ, the segment from X to Y meets a singular matrix. The determinant difference bound between that endpoint and the singular matrix proves (8). Equal-index cases follow by the same determinant bound; if neither contributes the difference is zero. Thus the index indicator itself is not asserted Lipschitz.

For K_s=[[alpha,sqrt(s) beta^T],[sqrt(s) beta,A]],

    det K_s=alpha det A-s beta^T adj(A)beta,
    |F_j(K_r)-F_j(K_0)|<=r |beta^T adj(A)beta|.            (9)

The determinant is affine in s, including singular A. If an index changes, the continuous matrix path has a singular intermediate point and its affine determinant controls the contributing endpoint. If the determinant is identically zero both filtered determinants vanish. No inverse Hessian or lower eigenvalue bound occurs in (9).

With D_r=diag(sqrt(r),I), H_i=D_r K_i D_r. Congruence preserves inertia and det(D_r)^2=r. Remove the sqrt(r) block by (9), then compare each endpoint to its own filtered block-diagonal determinant F_d(diag(-6k,A_0)) and F_(d-1)(diag(6k,A_0)) by (8) and (7). Since k>0, each reference factor is 6k |det A_0|1{A_0<0}. Uniform higher moments and Holder control the product error, giving

    W_r/r^2 = w_0+O_(L^p)(r),
    w_0=(6k)^2(det A_0)^2 1{A_0<0}.                      (10)

No equality of the separate finite-r type indicators, and no independence of endpoint weights, is asserted.

Under the endpoint-only contact law, z_0=E[w_0|U_0=v_0] is continuous and strictly positive: the conditional law of A_0 has full density on an open negative-definite matrix ball. Compactness, k_->0, and the endpoint-only version of (10) give

    Z_r/r^2=z_0+O(r),     inf z_0>0,
    (Z_r/r^2)^(-1)=z_0^(-1)+O(r).                       (11)

This is the FULL normalizer, not a restriction to a geometric good event. The remote conditioning changes the law inside (10), not the denominator in (11).

## 5. Kac-Rice yields the actual bound and the limiting density

Apply Kac-Rice under Q_r to grad f(x) on the compact remote domain. Its conditional gradient covariance is uniformly nonsingular there. Use the endpoint determinant product as a field mark, and F_j(H_x) to incorporate the witness Jacobian and index. The endpoint and witness filtered determinants are continuous functions of finite jets by (8). Truncation and uniform Gaussian moments handle their polynomial growth. Approximate the height-window indicator by continuous functions and then use height disintegration; location restrictions extend to Borel sets as equality of the finite measures. The standard Gaussian weighted Kac-Rice formula therefore gives

    E_(Q_r^W) N_(r,j)(E)
      = (r^2/Z_r) integral_E integral_(b-k r^3)^b
          p_(Y_x|U_r=v_r)(0,t)
          E[(W_r/r^2) F_j(H_x)|U_r=v_r,Y_x=(0,t)] dt dx. (12)

An underlying critical-point Jacobian is counted exactly once through F_j(H_x). No separate |det H_x| should be multiplied into F_j. Endpoint weight is counted exactly once. The height disintegration uses dt, not a radial/pin Jacobian. Equality of (12) also establishes zero expected count at either single boundary height.

The contact kernel is

    Lambda_j(x;b,k,u)
      = p_(Y_x|U_0=v_0)(0,b)/z_0
          * E[w_0 F_j(H_x)|U_0=v_0,Y_x=(0,b)].           (13)

The dependence between A_0 and H_x is retained INSIDE this expectation. The contact field has a forced degenerate critical jet at 0 and is not a globally Morse field; no such claim is needed. Only its separated remote jets and their Gaussian density are used.

By (5),(8),(10), Holder and uniform moments, the expectation in (12) differs from its contact counterpart by O(r), uniformly in x,b,k,R,t. Multiply by the density estimate from Section 3 and (11); the complete integrand after r^2/Z_r differs uniformly from Lambda_j by O(r). Integrating over a window of exact length k r^3 proves (2). The unnormalized version of (12) has the explicit bound

    integral_E integral_window p_(Y_x|U_r)(0,t)
          E[W_r F_j(H_x)|U_r,Y_x] dt dx
        <= C k r^5 |E|.                                 (14)

This is the required three-determinant numerator for THIS declared spatial/height region. It is not deduced from a cap-failure probability. Markov is used only in the valid direction after the expectation is bounded, giving (3).

Continuity of Lambda_j follows from continuous nondegenerate Gaussian conditional parameters and polynomial moment domination, since the filtered determinants are continuous. Strict positivity follows from full conditional support of (A_0,H_x): use an open ball around A_0=-I and a separate open ball around a nonsingular H_x of index j. Their joint density is positive; it need not factor. Compactness supplies common positive lower and finite upper bounds. Invariance under changing the transverse frame follows from determinant/inertia invariance, so Lambda_j is a function of u. No global continuous choice of transverse frame over the sphere is required.

## 6. Several mutually separated witnesses

Fix an integer m>=2 and eta>0. Restrict the ordered location tuples to

    D_(rho,eta,m)={ (x1,...,xm): xi in D_rho,
                                      dist_X(xi,xj)>=eta for i!=j }.

The domain may be empty; then all corresponding quantities are zero. Count ordered distinct m-tuples of index-j critical points in the height window on this domain. Use the SAME endpoint weight W_r once, not its m-th power, and normalize by Z_r once. Append all Y_xi to U_r. Distinct-site jet rank and compact separation give the same uniform covariance, coupling and moment argument. Thus, writing T_(r,j,m,eta) for this ordered count,

    E_(Q_r^W) T_(r,j,m,eta)
      = (k r^3)^m integral_(D_(rho,eta,m)) Lambda_(j,m)(x1,...,xm) dx
          + O((k r^3)^m r).                              (15)

The coefficient uses the JOINT conditional density of all remote gradients/heights and the conditional expectation of w_0 times the product of the m witness filtered determinants, divided by z_0. It is not a product of one-point kernels. If the tuple domain has positive measure, its integral is strictly positive.

There are m+2 Hessian determinant factors, one O(r^2) endpoint product and m height-window integrations. The radius exponent is 2+3m-2=3m. Since each unordered qualifying m-set contributes m! ordered tuples,

    P(exists m pairwise-eta-separated qualifying points)
       <= E T_(r,j,m,eta)/m! = O(r^(3m)).                 (16)

For m=2 this supplies an O(r^6) bound on the probability of a separated pair. It does not control two nearly coincident witnesses. There is no uniformity as eta decreases to zero, no full factorial-moment bound, no matching probability lower bound, and no Poisson-limit claim.

## 7. RN crosswalk and remaining work

The earlier RN_COUNT_INTERFACE.md, SHA256 aa993f5217bd70e6d1060402d121ab659f6582a098e29042c67585a345a8e3ab, required an actual O(r^5) count numerator before division by Z~r^2. Equation (14) now supplies it for every fixed remote region and the entire BETWEEN-PIN HEIGHT WINDOW. Any additional 0/1 witness selector is bounded by this count when its type/location/value conditions imply (1). Arbitrary global selectors are not asserted to share the explicit positive leading kernel (13).

What remains outside this result:

- Spatial regions with distance from the pins tending to zero, including the intermediate annulus between a multiple of r and fixed rho. Compactness constants here may diverge there.
- Witness tuples whose mutual separation tends to zero; (15) does not control that collision.
- All remote critical points with NO shrinking height restriction. The r^3 window factor is absent for that different count.
- A numerical value of C or r_*, the original RN all-cell/24-jet enclosures, and a proof that a particular legacy witness definition lies in this note's count region.
- Independent analytic review of this note or any parent persistence theorem. A source checksum or green test cannot supply that review.

The Gaussian model and exact observation conventions are those of UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1, SHA256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7. The weight stability algebra is reused, and rederived above, from LIFETIME-BOUNDED-REMAINDER-20260924-v1, SHA256 380b7d0abdb0fe2de5a6564565af9560d3f1b1ce22fd5538927db0c52f4f3a4a. This proof does NOT consume their global elder-selection conclusion or the deterministic cap theorem. It treats the actual pinned law and remote critical-point count directly.

## 8. Attribution and test scope

Armentano, Azais and Leon, *On a general Kac-Rice formula for the measure of a level set*, arXiv:2304.07424v3, Theorem 7.1 and Section 8.1 supply the established Gaussian expected-integral/critical-point framework. The application to the stated pinned law, extra height disintegration, uniformity and endpoint scaling is argued here. This note claims no historical priority for Kac-Rice, Gaussian regression, rare-window scaling or factorial moment methods.

The standard-library exact tests check observation transforms, singular typed-determinant cases, conditional covariance calculations, dependence controls, height-window integration and ordered-count radius factors. They do not compute Lambda_j for the actual field, certify continuum bounds, or substitute for a mathematical review.
