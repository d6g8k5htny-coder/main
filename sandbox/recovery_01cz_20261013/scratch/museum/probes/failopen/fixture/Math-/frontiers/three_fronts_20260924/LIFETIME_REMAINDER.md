# A bounded unrestricted short-lifetime remainder

**Object:** LIFETIME-BOUNDED-REMAINDER-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side derivation; nonauthor analytic review open.

## 1. Exact dependency and new conclusion

This is an additive successor to `UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1`, main issue 63, Drive `1foDgiDi4XIKOfbrZEE8dWKV_BkZU8LIb`, SHA256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`. The unchanged deterministic import is `MARKED-CYLINDER-CAP-20260924-v1`, SHA256 `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc`.

Use the parent's exact centered variance-one Gaussian field on the fixed torus X=R^d/(L Z^d), for each fixed d>=2 and L>0:

    K_L(z)=sum_n exp(-|z+Ln|^2/2) / sum_n exp(-|Ln|^2/2).

Every constant below may depend on d,L. It need not be uniform as L or d varies. No infinite-volume theorem is asserted. Pins are M=-ru/2, S=ru/2, heights b and b-k r^3, and zero gradients at both points, with b in R, k>0 and u a unit vector. Q is the continuous Gaussian regression law, W=|det H_M det H_S| times the maximum/index-(d-1)-saddle indicator, and Z=E_Q W is the FULL normalizer. Let p denote the Q^W=(W/Z)Q probability of global ordinary elder pairing.

The marked Kac-Rice identity, pinned genericity, measurable elder convention and separating-cylinder implication are used exactly as in the parent. This note supplies quantitative estimates missing there; it is not independent acceptance of those interfaces. The coefficient c=c_{d,L} is exactly the parent's equation (15.2).

**Theorem R (candidate).** There are ell_*>0 and C<infinity such that versions of the unrestricted expected per-unit-volume densities satisfy, for 0<ell<=ell_*,

    |nu_cand(ell)-c ell^(-1/3)| <= C,
    0 <= nu_cand(ell)-nu_eld(ell) <= C,
    |nu_eld(ell)-c ell^(-1/3)| <= C.                       (R1)

Here candidates are ALL ordered maximum/index-(d-1)-saddle pairs with positive height gap ell, and selected pairs are all finite ordinary superlevel H0 bars. All birth heights, all positive gap marks, all orientations and all separations on the fixed torus are included. The essential global-maximum class is excluded.

Thus the relative error is O(ell^(1/3)). We do not prove convergence of the bounded remainder, identify a second coefficient, supply a numerical C or ell_*, or claim that the compact-window O(ell^(2/3)) selection difference holds without cutoffs. A bounded off-diagonal contribution need not vanish.

The mechanism is a quadratic determinant cancellation for a sqrt(r) off-diagonal Hessian block, combined with explicit k-dependence in an UNNORMALIZED cap-loss estimate. Neither a globally uniform lower bound for Z/r^2 over b,k nor a globally uniform Cr^3 pairing-probability constant is needed.

## 2. Symmetric contact observations give a second-order coupling

Use the exact parent observation transform U_r=T_r O_r. For a=-r/2,c=r/2 its axial rows are

    (f(a)+f(c))/2,
    (f(c)-f(a))/r,
    (f_x(c)-f_x(a))/r,
    (6/r^2)[f_x(a)+f_x(c)-2(f(c)-f(a))/r],

with transverse rows (f_yj(a)+f_yj(c))/2 and (f_yj(c)-f_yj(a))/r. Its absolute determinant is 12 r^(-(d+3)), and its target is

    v_r=(b-k r^3/2,-k r^2,0,12k,0,...,0).

At r=0 let U_0=(f,f_x,f_xx,f_xxx,f_y1,f_xy1,...), with target v_0=(b,0,0,12k,0,...). These are exact source conventions, including the factor 12.

All difference rules are centered. Taylor expansion with integral remainder gives U_r-U_0=O(r^2) in every finite L^p norm, uniformly over orthonormal frames. In particular the fourth axial row is f_xxx(0)+(r^2/40)f_xxxxx(0)+O(r^4), not an O(r) error or an unscaled residual. Cross-covariances with any fixed derivative order have the same second-order estimate.

A direct justification with no random-grid approximation uses the full positive Fourier expansion. Its coefficients satisfy sum_n sqrt(a_n)(1+|n|)^q<infinity for every fixed q. For each mode, the centered-rule error is bounded by C r^2 times a fixed higher power of |n|. Minkowski and this summability control the random series, derivative suprema and their moments. The same argument, with products of coefficients, controls deterministic covariance derivatives uniformly in position and frame.

Write Sigma_r=Cov(U_r), C_r(z)=Cov(F(z),U_r), where F is an unconditioned copy. The distinct contact functionals have positive covariance. Compactness of the frame space and convergence give, on one band 0<=r<=r_0, uniform positive lower/upper eigenvalue bounds and

    Sigma_r-Sigma_0=O(r^2),
    Sigma_r^(-1)-Sigma_0^(-1)=O(r^2),
    ||C_r-C_0||_{C^q}=O(r^2).                            (R2)

These covariance facts do not depend on b,k. Choose r_0<=1 inside the torus injectivity scale. On a common probability space set

    F_r=F+C_r Sigma_r^(-1)(v_r-U_r),
    F_0=F+C_0 Sigma_0^(-1)(v_0-U_0).                     (R3)

Gaussian regression shows that each F_r has the required law. Since |v_r-v_0|<=C r^2 k and |v_r|+|v_0|<=C(|b|+k), equations (R2)-(R3) imply, for finite p>=1 and each finite derivative order q,

    ||F_r-F_0||_{L^p(C^q)} <= C_{p,q} r^2 P,
    ||1+||F_r||_{C^q}+||F_0||_{C^q}||_p <= C_{p,q} P,
    P=1+|b|+k.                                          (R4)

This is a coupling of laws, not an assertion of equality of two conditioned fields or independence of endpoint variables.

Let pi_r be the density of U_r. For all b,k and r<=r_0,

    pi_r(v_r) <= C exp[-c(b^2+k^2)],
    |pi_r(v_r)-pi_0(v_0)|
       <= C r^2 P^2 exp[-c(b^2+k^2)].                    (R5)

To prove the difference estimate, interpolate the covariance and target linearly between r and 0. The covariance stays uniformly positive. The target keeps its 12k coordinate; its other nonzero height coordinate is b-theta*k*r^3/2, so its squared norm is uniformly coercive in b^2+k^2. Differentiating log Gaussian density gives a trace term O(r^2), a quadratic term O(r^2 P^2), and a target term O(r^2 kP). Integrating the derivative yields (R5), after reducing c. This explicitly handles arbitrarily large targets.

## 3. An index-filtered determinant is more stable than its inertia indicator

For a real symmetric n by n matrix H define

    F_j(H)=|det H| 1{H has exactly j negative eigenvalues},

and give it value zero on singular matrices. Two elementary lemmas suffice.

**Lemma R3.1.** For symmetric X,Y,

    |F_j(X)-F_j(Y)| <= n max(||X||,||Y||)^(n-1)||X-Y||.  (R6)

If their indices agree, this follows from the determinant difference bound (or both terms are zero). If their indices differ and one term is nonzero, the straight segment reaches a singular matrix Z. The same determinant bound from that endpoint to Z applies, with norm and distance bounded by those of the full segment. When an endpoint is singular the same reasoning applies directly. The determinant bound follows by telescoping columns and Hadamard's inequality. In particular the discontinuous inertia indicator itself was not assumed Lipschitz.

**Lemma R3.2 (quadratic off-diagonal cancellation).** For a symmetric (n-1) by (n-1) matrix A, scalar alpha, vector beta and t>=0, set

    K_t=[[alpha,sqrt(t) beta^T],[sqrt(t) beta,A]].

Then for every j and every r>=0,

    |F_j(K_r)-F_j(K_0)|
       <= r |beta^T adj(A) beta|.                        (R7)

Indeed det K_t=alpha det A-t beta^T adj(A) beta, INCLUDING singular A. If the endpoint indices agree, use the difference of absolute determinants. If only one endpoint contributes, a singular K_t lies in between by continuity of eigenvalues, and the affine determinant bounds the contributing endpoint by r times its slope. If the determinant is identically zero, both sides' filtered determinants are zero. The proof requires no inverse matrix and no lower eigenvalue margin.

A naive application of (R6) to the sqrt(r) entries would lose a square root. Equation (R7) is the stronger fact that prevents this loss, also as k tends to zero.

## 4. A refined all-mark intensity and contact error

Under (R3), put A_0=D_y^2 F_0(0), A_i=D_y^2 F_r(i),
alpha_i=F_r,xx(i)/r and beta_i=grad_y F_r,x(i)/r. Let D_r=diag(sqrt(r),I). Then

    K_i=D_r^(-1) H_i D_r^(-1)
       =[[alpha_i,sqrt(r) beta_i^T],[sqrt(r) beta_i,A_i]],
    |det H_i|/r=|det K_i|.

The exact gradient/height pin identities give

    |alpha_M+6k|+|alpha_S-6k| <= C r M4,
    ||beta_i|| <= M3/2.                                  (R8)

Equation (R4) and the endpoint displacement give a nonnegative random T>=1+k with all finite moments bounded by C_p P^p such that

    ||A_i-A_0||<=r T,
    ||A_i||+||A_0||+||beta_i||+M3+M4<=C T,
    |alpha_M+6k|+|alpha_S-6k|<=C r T.                     (R9)

For example T can include the C^4 norms in (R4) and r^(-2)||F_r-F_0||_{C^2}; the latter has the asserted uniform moments by (R4). No independence of T and A_0 is required here.

Use j=d at M and j=d-1 at S. First remove the off-diagonal block by (R7); then compare diag(alpha_i,A_i) with diag(+-6k,A_0) by (R6). Equations (R8)-(R9) yield an O(r T^d) filtered determinant error at each endpoint. The exact block determinant also gives

    F_j(K_i) <= C(k+r) T^d,
    F_d(diag(-6k,A_0)) <= C k T^(d-1),
    F_(d-1)(diag(6k,A_0)) <= C k T^(d-1).

The reference product is (6k)^2 det(A_0)^2 1{A_0<0}; no other transverse inertia can give both specified full indices. Multiplying the two estimates and taking expectations proves

    Z_r/r^2 <= C(k+r)^2 P^N,
    |Z_r/r^2-z_0(b,k,u)| <= C r(k+r) P^N,                (R10)

for a fixed finite integer N depending only on d. Thus, defining exactly as in the parent

    A_r(b,k,u)=12 pi_r(v_r) Z_r/r^2,
    A_0=12 pi_0(v_0) z_0,

there is a majorant H(b,k)=C P^N exp[-c(b^2+k^2)] (increasing N when necessary) with

    0<=A_r<=(k+r)^2 H,
    0<=A_0<=k^2 H,
    |A_r-A_0|<=r(k+r) H.                                (R11)

For the density difference term use (R5) and z_0<=C k^2 P^N; its r^2 k^2 factor is absorbed by r(k+r) times a further polynomial in P. These bounds hold for ALL b in R,k>0 and 0<r<=r_0. There is no division by k or Z in their proof.

## 5. The unnormalized cap-loss bound with its actual k dependence

Let G_r be the deterministic sufficient event

    lambda_min(-A_M) > (4/(3k)) r M3^2,
    r M4 <= 3k/10.

On typed support the cap theorem gives elder pairing on G_r. Define

    B_r=12 pi_r(v_r) E_Q[(W/r^2) 1{G_r^c}].

Always A_r(1-p_r)<=B_r<=A_r. We prove that with delta=r/k,

    B_r <= delta^3 H(b,k),       whenever delta<=1.       (R12)

The constants in (R12) are independent of b,k; their target growth is inside H. This is NOT a globally uniform normalized probability estimate.

Append the independent symmetric entries of A_M to U_r. The joint covariance is uniformly positive at contact and for r<=r_0, by distinct finite jets and compactness. Consequently its JOINT density obeys

    pi_r(v_r) density(A_M=A | U_r=v_r)
       <= C exp[-c(b^2+k^2+||A||_F^2)].                  (R13)

Regress the whole field on this appended vector. Its centered residual g is independent of that entire vector and has uniform C^4 moments of every finite order. Its mean has C^4 norm bounded by C(P+||A||). Set J=1+||g||_{C^4}, T=P+J. On A_M<0 write B=-A_M with eigenvalues 0<lambda_1<=...<=lambda_m=Lambda, m=d-1. Then h=M3 and M4 are at most K(T+Lambda), absorbing sqrt(m) into K. J is independent of B, though eigenvalues within B are not independent.

The exact parent determinant envelope is

    W/r^2 <= (h^2/4) lambda_1(lambda_1+3rh/2)
                      product_(j=2)^m lambda_j(lambda_j+rh). (R14)

For m>=2, failure of the depth condition implies

    0<lambda_1<=D delta(T+Lambda)^2,     D=4K^2/3.

Use symmetric-matrix Lebesgue eigenvalue coordinates. Their Vandermonde factor is at most Lambda^[m(m-1)/2] on the positive ordered domain, and (R13) supplies a Gaussian bound in all eigenvalues, uniformly in their eigenvectors. This is a Lebesgue change of variables, not a GOE claim. Put U=T+Lambda. All factors in (R14) except the first two soft factors are at most C U^(2m), for r<=1. After dropping the lambda_1 Gaussian factor, its integral is bounded by

    integral_0^(D delta U^2) lambda(lambda+E rU) d lambda
      = (D^3/3)delta^3 U^6+(ED^2/2)r delta^2 U^5.

Since r delta^2=k delta^3, both terms are bounded by delta^3 times a polynomial in P,J,Lambda. The Gaussian integration in lambda_2,...,lambda_m and finite moments of J give delta^3 H. The largest eigenvalue is integrated, not frozen at a chosen value. No lower cutoff on lambda_2 is used: every higher-corank intersection is included.

For m=1, lambda_1=Lambda=lambda. The scalar quadratic inequality is instead

    lambda<=D delta(T+lambda)^2
      => [lambda<=4D delta T^2] OR [lambda>1/(4D delta)]. (R15)

On the near branch delta<=1 gives h<=C T^2. Direct scalar integration of lambda(lambda+CrT^2), with width 4D delta T^2 and (R13), again gives delta^3 H, since r delta^2=k delta^3. For the FAR branch, fourth-moment Markov gives

    pi_r E_Q[(W/r^2) 1_far]
       <=(4D delta)^4 pi_r E_Q[(W/r^2)lambda^4]
       <=delta^4 H<=delta^3 H.

Joint moments here are bounded by polynomial Gaussian integrals from (R13), or directly by the target-growth regression bound. The scalar far branch was not discarded.

Finally the fourth-derivative exception M4>3/(10delta) gives

    pi_r E_Q[(W/r^2)1_exception]
       <=(10delta/3)^4 pi_r E_Q[(W/r^2)M4^4]
       <=delta^4 H<=delta^3 H.

These are weighted pointwise Markov inequalities, not Cauchy-Schwarz without a square root. Increasing the common polynomial degree and constants proves (R12).

## 6. Integrate candidate error over the entire mark domain

The exact parent polar/height/pin/determinant accounting gives r A_r dr db dk d sigma, with ordinary sphere area and no role-order factor 1/2. At lifetime ell set r=(ell/k)^(1/3). The near restriction r<=r_0 is k>=a, a=ell/r_0^3, and

    ell^(1/3) nu_cand^near(ell)
       = integral_(k>=a) A_(ell/k)^(1/3)/(3 k^(2/3)) db dk d sigma.

The full contact coefficient is integral A_0/(3 k^(2/3)) over all b,k>0,u. From (R11), the missing k<a contact mass is O(a^(7/3)). For the remaining difference the two monomials are exactly

    k^(-2/3) r k = ell^(1/3),
    k^(-2/3) r^2 = ell^(2/3) k^(-4/3).

Integrating H in b,k controls the first by C ell^(1/3). For the second, the integral from a to 1 is at most C a^(-1/3), and its tail is finite. Hence it is at most C ell^(2/3) a^(-1/3)=C r_0 ell^(1/3), up to another O(ell^(2/3)) tail. The a^(7/3) missing mass is smaller for small ell. Thus

    |ell^(1/3) nu_cand^near(ell)-c|<=C ell^(1/3).          (R16)

This is a rate proved from (R11), not an assigned rate for the parent's dominated convergence.

## 7. Integrate the actual pairing loss

Take eta=ell^(1/6), and shrink ell_* so that a<eta<1. For a<=k<=eta use B_r<=A_r and the first line of (R11), with (k+r)^2<=2k^2+2r^2. Its scaled contribution is

    C[eta^(7/3)+ell^(2/3) a^(-1/3)]
       <=C[ell^(7/18)+ell^(1/3)].                       (R17)

For k>=eta, delta=ell^(1/3)k^(-4/3)<=ell^(1/9)<=1. The loss estimate (R12) is therefore applicable, including large b,k. The scaled integrand is bounded by

    C ell k^(-14/3) H(b,k).

Its integral is at most C ell eta^(-11/3)=C ell^(7/18), plus a smaller Gaussian-tail term. Combining with (R17), since 7/18>1/3, gives

    0<=ell^(1/3)(nu_cand^near-nu_eld^near)<=C ell^(1/3).   (R18)

No normalized lower floor in the small-k region was used. The lower cutoff a must be retained: replacing it by zero in the r^2 k^(-2/3) term would create a divergent integral and lose the valid bound.

On torus separations at least r_0 the original full-pin covariance is uniformly positive, as in parent Section 14. The direct unconditioned height-density estimate there gives 0<=nu_eld^far<=nu_cand^far<=C for 0<ell<=1. Adding these terms to (R16)-(R18) proves (R1).

## 8. Quantitative consequences and exact limits

Per unit volume, integration of (R1) yields

    E N_eld(0,t] = (3/2)c t^(2/3)+O(t),
    0<=E N_cand-but-not-eld(0,t]<=C t.                   (R19)

For q>-2/3, the individual candidate and finite-bar q-moments obey

    E sum_(ell_i<=t) ell_i^q / Vol(X)
       = c t^(q+2/3)/(q+2/3)+O(t^(q+1)).                (R20)

For q>-1 the NONSELECTED counting measure alone has weighted mass at most C_q t^(q+1). This remains meaningful for -1<q<=-2/3 even though the two individual expectations diverge; it is not subtraction of infinities. It proves a sufficient p<1 inverse-lifetime integrability range for nonselected candidates, not an exact threshold.

The full individual inverse-lifetime threshold from the leading positive density remains p<2/3. No new fixed-r inverse-Hessian moment is inferred. This note supplies no RN expected-critical-count estimate and no original 24-jet certificate.

## 9. Review and validation interface

Critical review points are the second-order observation/coupling estimate (R4), the inertia-filtered affine determinant lemma (R7), target-uniform joint density (R13), both soft eigenvalue factors in (R14), the scalar far branch (R15), and the lower/upper k cutoffs in (R16)-(R18). Each is argued above, not delegated to numerical tests. The supplied finite exact tests check representative inertia transitions, covariance-free determinant identities, rational exponents and integration models. They do not verify continuum Gaussian estimates or independently accept the parent.

External background: Armentano, Azais and Leon, arXiv:2304.07424v3, Theorem 7.1 and Section 8.1, provide established Gaussian marked Kac-Rice context. The parent supplies its stated Borel-mark extension and event convention. The new determinant cancellation and rate argument are derived here; this bounded reconnaissance establishes no novelty or priority claim.
