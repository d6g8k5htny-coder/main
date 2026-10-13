# Uniform matrix-cap selection and unrestricted short-lifetime asymptotics

**Object:** UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1.  
**Author:** OpenAI / ChatGPT, 24 September 2026.  
**Disposition:** author-side mathematical candidate; no nonauthor acceptance, formal verification, numerical finite-band certificate, or program-level closure is asserted.

This is an additive continuation of MARKED-CYLINDER-CAP-20260924-v1, SHA256 `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc`, reproduced unchanged in `sources/`. It addresses the matrix-probability, angular-uniformity, compact-mark, and lifetime-composition interfaces of campaign #61. It does **not** extend the predecessor's numerical constants to new parameters. Instead it derives new, non-numerically-evaluated constants by a uniform Gaussian argument.

## 1. Model, convention, and the three theorems

Fix a dimension d >= 2 and a side length L > 0. On the flat torus X = R^d/(L Z^d), let f be the centered stationary Gaussian field of variance one with covariance

    K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)
             / sum_{n in Z^d} exp(-|Ln|^2/2).

In particular d=3, L=24 is an actual three-dimensional SIDE24 application. No replacement by a nonperiodic covariance is made. Fix compact intervals B=[b_-,b_+] and K=[k_-,k_+] with 0<k_-<=k_+<infinity. Unless stated otherwise, constants may depend on d,L,B,K. They do not depend on radius, birth within B, gap mark within K, or orientation.

Let R be an orthonormal frame, with axial unit vector u=R e_1. For sufficiently small r>0, use the embedded local cylinder centered at the origin with pins

    M=-(r/2)u,     S=(r/2)u,
    f(M)=b,        f(S)=s=b-k r^3,
    grad f(M)=grad f(S)=0.

Q_{r,b,k,R} is the continuous Gaussian regression law at these 2(d+1) observations, not a conditioning on a positive-probability event. A maximum has d negative Hessian eigenvalues; the relevant saddle has d-1 negative and one positive eigenvalue. Set

    W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1},
    Z_r = E_Q W_r,       dQ^W = (W_r/Z_r)dQ.

The index counts negative eigenvalues. The FULL normalizer Z_r has no extra pin-density, coordinate Jacobian, or adjacency condition. Let p_r(b,k,R) be the Q^W probability that the **global ordinary superlevel elder death partner** of M is S.

**Theorem A (uniform selection candidate).** There exist r_*>0 and C<infinity such that, for all 0<r<=r_*, b in B, k in K, and all orthonormal frames,

    0 <= 1-p_r(b,k,R) <= C r^3.                         (1.1)

The proof includes a positive full-normalizer lower bound Z_r>=z_* r^2 and the full transverse-matrix boundary layer, including simultaneous small transverse eigenvalues. It does not presume isotropy under arbitrary rotations of the torus.

**Theorem B (compact-mark density candidate).** Suppose B and K have positive lengths. Count ordered maximum/saddle pairs with 0<distance<=r_*, birth in B, and scaled gap (f(M)-f(S))/distance^3 in K. Let nu_cand(ell) be their expected lifetime density per unit volume; let nu_eld(ell) count only pairs which are actual ordinary elder partners. These densities admit versions with

    nu_cand(ell) ~ c_{B,K} ell^(-1/3),
    nu_eld (ell) ~ c_{B,K} ell^(-1/3),       c_{B,K}>0,
    0 <= nu_cand(ell)-nu_eld(ell) <= C_{B,K} ell^(2/3),    (1.2)

as ell decreases to zero. An exact finite-dimensional Gaussian integral for c_{B,K} appears in Section 11.

Theorem B is a COMPACT MARK WINDOW result, especially its quantitative difference estimate. Theorem C below separately removes the mark and distance restrictions for the LEADING asymptotic. There is no stated numerical value of C, r_*, z_*, or c_{B,K}, and no claim that r_*=1/20. The new asymptotic normalizer proof does not replace or re-certify the historical H3 finite-band floor.

**Theorem C (unrestricted finite-lifetime leading density candidate).** For the same fixed torus and dimension, let nu_cand^all(ell) count ALL ordered maximum/index-(d-1)-saddle pairs with height difference ell>0, and let nu_eld^all(ell) count all finite ordinary superlevel H0 persistence bars, both in expectation per unit volume. There is a positive finite constant c_{d,L} such that

    nu_cand^all(ell) ~ c_{d,L} ell^(-1/3),
    nu_eld^all(ell)  ~ c_{d,L} ell^(-1/3).                (1.3)

All birth heights, all positive scaled-gap marks and all spatial separations on the torus are included in (1.3). The essential global-maximum class is excluded from the finite-lifetime count. The constant is an explicitly identified finite Gaussian/angular integral, with the birth and gap integrals analytically eliminated in Section 15. No elementary closed form or numerical enclosure is claimed. Crucially, the compact-window O(ell^(2/3)) DIFFERENCE estimate in (1.2) is not asserted for the unrestricted densities. Theorem C has a separate domination and off-diagonal proof, not a change of label on Theorem B.

## 2. Positive Fourier spectrum and finite-jet rank

Poisson summation gives

    K_L(z)=sum_{n in Z^d} a_n exp(2 pi i n.z/L),
    a_n=exp(-2 pi^2 |n|^2/L^2) / sum_j exp(-2 pi^2 |j|^2/L^2)>0.

Thus every Fourier mode has positive variance, and sum_n sqrt(a_n)(1+|n|)^q is finite for every q. A real sine/cosine expansion therefore converges in every C^q norm in every finite L^p norm. In particular the field has a smooth version and all finite moments of its global derivative suprema exist.

Any finite list of distinct derivative evaluation functionals at distinct sites is linearly independent. Indeed a vanishing-variance linear combination must annihilate every Fourier mode because every a_n is positive. As a distribution it is a finite sum of derivatives of point masses with every Fourier coefficient zero, so it is zero. Smooth test functions supported at one site and with arbitrarily prescribed finite jets show that every coefficient is zero. At one site in a rotated frame the same assertion also follows from a polynomial P(R^T n) vanishing on all n in Z^d: the invertible rotation reduces this to a polynomial vanishing on the full integer lattice, hence the polynomial is zero.

Only lists without repeated derivative functionals are called distinct. A Hessian is parametrized by its independent symmetric entries, not by d^2 allegedly independent entries.

These arguments justify positive covariance for the original pins at each pair of distinct points and for every residual jet list modulo the pins used below. They also explain why finitely many observed modes or a numerically positive sampled covariance would not suffice for this proof.

## 3. An exact nonsingular contact frame for all orientations

Write x for the axial coordinate and y_1,...,y_m for the transverse coordinates, m=d-1. On the line y=0 set a=-r/2, c=r/2. Order the original observations as

    O_r=(f(a), f_x(a), f(c), f_x(c),
         f_y1(a),f_y1(c), ..., f_ym(a),f_ym(c)).

This is a permutation and orthogonal re-expression of the specified physical pins. Define U_r=T_r O_r by four axial rows

    U0 = (f(a)+f(c))/2,
    U1 = (f(c)-f(a))/r,
    U2 = (f_x(c)-f_x(a))/r,
    U3 = (6/r^2)[f_x(a)+f_x(c)-2(f(c)-f(a))/r],           (3.1)

and, for every transverse direction j, two rows

    Vj0=(f_yj(a)+f_yj(c))/2,
    Vj1=(f_yj(c)-f_yj(a))/r.

The transformation is invertible for r>0 and has exact absolute determinant

    |det T_r| = (12/r^4)(1/r)^m = 12 r^(-(d+3)).          (3.2)

The prescribed target becomes exactly

    v_r=(b-k r^3/2, -k r^2, 0, 12k, 0,...,0).            (3.3)

This is not an approximate transformation, and the factor 12 in (3.2) is retained in the lifetime ledger.

Taylor formulas with integral remainders give mean-square convergence, uniformly over frames, of U_r to the contact jet

    U0*=(f, f_x, f_xx, f_xxx,
          f_y1,f_xy1,...,f_ym,f_xym) at 0.                (3.4)

The potentially dangerous U3 cancellation is an averaged third derivative, not a subtraction of uncontrolled random values. Uniform convergence of its covariances, and of cross-covariances with any fixed finite derivative order, follows either from those integral formulas or from the rapidly summable Fourier series. The target v_r converges uniformly on B x K to v_0=(b,0,0,12k,0,...,0).

The functionals in (3.4) are distinct. Hence their covariance Sigma_0(R) is positive definite by Section 2. Its entries are continuous in R. The orthogonal group is compact, so its least eigenvalue has a positive uniform minimum. Consequently Sigma_r(R) remains uniformly positive definite for all sufficiently small r; Sigma_r and Sigma_r^{-1} converge uniformly.

This compactness argument establishes existence, not an evaluated interval enclosure. It never equates Sigma_r at different orientations. Full torus rotational invariance is neither assumed nor true in general.

Let A_r=D_y^2 f(M), represented by its m(m+1)/2 independent entries. Append these entries to U_r. At r=0 the extra functionals are f_yi_yj, i<=j; none duplicates a functional in (3.4). The enlarged covariance is therefore uniformly positive definite at contact and for small r. Its Schur complement shows that the Q-law of A_r has covariance uniformly bounded above and uniformly bounded away from zero. Its mean is uniformly bounded. In symmetric-matrix Lebesgue coordinates it has a density satisfying

    density_Q(A) <= C_0 exp(-c_0 ||A||_F^2),              (3.5)

for fixed positive C_0,c_0. This remains valid after any change of eigenvector orientation. Orthogonal invariance of the density is not required.

## 4. Uniform conditional derivative moments and the independent residual

Because Sigma_r^{-1} and v_r are uniformly bounded, the regression energy v_r^T Sigma_r^{-1}v_r is uniformly bounded. For each unconditioned standardized Fourier coefficient xi, covariance Cauchy-Schwarz bounds its conditional mean by the square root of this energy, and its conditional centered variance by one. Minkowski's inequality and the summable Fourier coefficients then give, for every finite p and derivative order q,

    sup_{r,b,k,R} || ||f||_{C^q(X)} ||_{L^p(Q)} < infinity. (4.1)

The same proof applies to a centered residual after further conditioning on A_r. Write, in one full-field Gaussian coupling,

    f(z)=mu_r(z)+B_r(z).(A_r-E_Q A_r)+g_r(z),             (4.2)

where g_r is jointly independent of the entire symmetric matrix A_r. The regression coefficient B_r and its derivatives through order four are uniformly bounded, by the uniform inverse covariance from Section 3 and bounded derivative cross-covariances. The residual has uniformly bounded C^4 moments of every finite order.

Independence in (4.2) is not asserted separately for Hessian entries, endpoint variables, or sampled derivative maxima. All residual evaluations are orthogonal to A_r; Gaussian independence on a countable dense collection and continuity give independence of the random smooth field. The residual's internal correlations are retained.

Let J_r=1+||g_r||_{C^4(X)}, with a fixed coordinate norm. The partial-block operator norms of the cap theorem differ from this coordinate norm by finite constants depending only on d. Hence, for M3,M4 on the local cylinder,

    M3 <= K_0(J_r+||A_r||_F),
    M4 <= K_0(J_r+||A_r||_F),
    J_r >= 1,  J_r independent of A_r,
    sup ||J_r||_p < infinity for every finite p.           (4.3)

No inverse Hessian moment is used. Gaussian pin covariance and random field Hessians are different objects.

## 5. Endpoint identities and the actual full normalizer

For either pin i in {M,S}, put

    alpha_i=f_xx(i)/r,    beta_i=grad_y f_x(i)/r,
    A_i=D_y^2 f(i),
    H_i=[[r alpha_i, r beta_i^T], [r beta_i,A_i]].

The two gradient pins imply the vector and scalar average bounds

    |alpha_i| <= M3/2,         ||beta_i|| <= M3/2,
    ||A_S-A_M||_op <= r M3.                               (5.1)

For example integral_a^c f_xx=0; subtract this average at an endpoint and integrate the M3-Lipschitz bound. For beta use the average of grad_y f_x and its vector derivative. This does not invoke a simultaneous vector Rolle zero.

The exact height identity gives a weighted mean of f_xxx equal to 12k on [a,c]. Variation of f_xxx over that interval is at most rM4. The same average calculation yields

    |alpha_M+6k| <= r M4/2,
    |alpha_S-6k| <= r M4/2.                               (5.2)

The block determinant identity, valid even when alpha_i or A_i is zero, is

    det H_i/r = alpha_i det A_i
                -r beta_i^T adj(A_i) beta_i.              (5.3)

By (4.1),(5.1), det H_i/r and W_r/r^2 have uniformly bounded moments of every finite order.

Let A_0 be D_y^2 f(0) under regression on U0*=v_0. Its law is a full-support nondegenerate Gaussian on symmetric matrices. Finite-dimensional Gaussian regression gives uniform convergence of the means and covariances of A_M to those of A_0. Equations (5.1),(5.2) imply that the congruence-scaled endpoint Hessians converge, in probability uniformly over parameters, to

    diag(-6k,A_0) at M,        diag(6k,A_0) at S.

Here the congruence uses diag(sqrt(r),I), so its off-diagonal entries are sqrt(r) beta_i. The limiting matrices are nonsingular almost surely. This follows from the full matrix density and k>=k_->0. Thus the type indicator converges to 1{A_0<0}. Uniform integrability from (5.3) gives

    Z_r/r^2 -> z_0(b,k,R)
       := (6k)^2 E[det(A_0)^2 1{A_0<0}],                 (5.4)

uniformly on the compact parameter set.

For completeness, uniformity can be checked by a subsequence argument: any allegedly bad sequence of parameters has a convergent subsequence; continuous Gaussian means/covariances, (5.2), and uniform integrability give the stated limit on that subsequence. The discontinuity set of the type indicator has Gaussian probability zero. Equation (3.5) supplies a common integrable majorant.

The right side of (5.4) is continuous and strictly positive: the Gaussian law has positive density on an open negative-definite matrix ball. Compactness therefore yields

    0<z_*<=Z_r/r^2<=z^*<infinity                         (5.5)

after reducing r_* if needed. This is a proof of a full asymptotic normalizer floor, not the historical numerical floor 17/10, and it involves neither an added pin density nor a restricted good-event normalization.

## 6. The double soft-eigenvalue factor in the typed weight

Write m=d-1. On the maximum support let B=-A_M>0, with eigenvalues

    0<lambda_1<=...<=lambda_m=Lambda.

Let h=M3. Since H_M<0, alpha_M<0 and the negative transverse Schur complement is

    B - (r/(-alpha_M)) beta_M beta_M^T >0.

Determinant monotonicity on positive-definite matrices therefore gives

    |det H_M| <= r (h/2) det B.                           (6.1)

This is a canceled-pivot bound: no estimate of 1/alpha_M is taken.

From ||A_S+B||<=rh, the ordered singular values of A_S are at most lambda_j+rh. The adjugate norm is at most the product of the largest m-1 such bounds. With the empty product equal to one when m=1, (5.3) gives

    |det H_S| <= r(h/2) [lambda_1+(3/2)rh]
                         product_{j=2}^m(lambda_j+rh).

Hence the actual typed weight satisfies

    W_r <= (r^2 h^2/4) lambda_1[lambda_1+(3/2)rh]
                    product_{j=2}^m lambda_j(lambda_j+rh). (6.2)

The second soft factor includes the saddle mixed-square term. Omitting it, or replacing it by an unrelated order-one factor, changes the radius power. No Gaussian-entry independence is needed for (6.2).

## 7. Full matrix boundary-layer integration, including corank intersections

By the predecessor's deterministic marked-cylinder theorem, pairing and the one-branch conclusion hold on

    G_r={lambda_min(-A_M)>[4/(3k)]rM3^2,
                                      rM4<=3k/10}.

On typed support A_M<0 automatically. It suffices to control G_r^c directly, not infer a converse from a local path-event bound.

First suppose m>=2. In (4.3) absorb sqrt(m) into K and set U=J_r+Lambda>=1. Then h<=KU. Failure of the depth condition is contained in

    0<lambda_1<=D r U^2,       D=4K^2/(3k_-).             (7.1)

Unlike a scalar strip argument, this width may depend on the largest eigenvalue. That dependence is retained in the integral. J_r, but not the eigenvalues within B, is independent of B.

Symmetric-matrix eigenvalue coordinates have volume factor equal to a fixed constant times product_{i<j}|lambda_j-lambda_i| times angular volume. One way to see the factor is to differentiate B=O diag(lambda) O^T: an infinitesimal rotation in the ij plane changes the off-diagonal coordinate by (lambda_j-lambda_i) times that angular increment. The angular space is compact; repeated-eigenvalue strata are Lebesgue null. Its finite multiplicity is absorbed into the constant. This is a change of variables for Lebesgue measure, **not a claim that the conditional matrix is GOE**.

Using (3.5), angular integration, and 0<=lambda_i<=Lambda, the resulting eigenvalue measure is bounded by

    C exp(-c sum lambda_i^2) Lambda^[m(m-1)/2]
                     d lambda_1 ... d lambda_m.          (7.2)

In (6.2), with r<=1, all factors other than lambda_1[lambda_1+(3/2)rh] are bounded by C r^2 U^(2m). Condition on J_r and lambda_2,...,lambda_m. Extend the positive majorant's lambda_1 integration from the actual interval to [0,DrU^2]. Then

    integral lambda_1(lambda_1+E rU) d lambda_1
       = r^3[(D^3/3)U^6+(ED^2/2)U^5]
       <= C r^3 U^6.                                    (7.3)

The Gaussian factor involving lambda_1 can be dropped for this upper bound. Equations (6.2),(7.2),(7.3) give

    E_Q[W_r 1{depth failure}] <= C r^5 E integral
       (J_r+Lambda)^(2m+6) Lambda^[m(m-1)/2]
       exp(-c sum_{j=2}^m lambda_j^2)
       d lambda_2 ... d lambda_m <= C' r^5.              (7.4)

The last integral is finite by Gaussian polynomial integrability and the uniform moment of J_r of order 2m+6. All lambda_2,...,lambda_m are integrated over their ordered positive domain. In particular **lambda_2 is not bounded away from zero**. The possible corank-two and higher intersections are included, with only nonnegative polynomial powers. There is no angular singularity or inverse-eigenvalue moment left to certify.

For m=1 the largest and smallest eigenvalues coincide; using (7.3) with Lambda held fixed would be invalid. Write lambda=-A_M>0 and use h<=K(J_r+lambda). Depth failure implies

    lambda<=Dr(J_r+lambda)^2
            <=2DrJ_r^2+2Dr lambda^2.

It is contained in the union

    0<lambda<=4DrJ_r^2,     or     lambda>1/(4Dr).          (7.5)

The large-lambda branch is retained. On the near branch, h<=K(1+4D)J_r^2 for r<=1, J_r>=1. The bounded scalar density, (6.2), and direct integration in lambda give a numerator <=C r^5 E J_r^10. On the far branch, direct fourth-moment Markov with the actual weight gives

    E_Q[(W_r/r^2)1_far]
       <=(4Dr)^4 E_Q[(W_r/r^2)lambda^4] <=C r^4.          (7.6)

Thus the scalar numerator is <=C r^5+C' r^6. This separate argument is why the all-dimension statement does not silently drop the scalar far branch.

Finally, for every dimension, the fourth-derivative exception is bounded directly by

    E_Q[(W_r/r^2)1{rM4>3k_-/10}]
       <=[10r/(3k_-)]^4 E_Q[(W_r/r^2)M4^4] <=C r^4.      (7.7)

This is a pointwise Markov bound followed by a joint moment estimate. It is not Cauchy-Schwarz with a missing square root. All joint moments in (7.6),(7.7) are finite by Sections 4-5.

Divide the numerators by the FULL lower bound (5.5). The result is

    Q^W(G_r^c)<=C_3 r^3+C_4 r^4<=C r^3.                 (7.8)

## 8. Pinned genericity and global event measurability

The following argument applies at every fixed positive r and parameter, not on an asserted common probability-one set for uncountably many laws.

Away from the pins, consider (grad f(x),Hf(x)v), with ||v||=1. Its range has dimension 2d; its parameter manifold has dimension 2d-1. Modulo the pin observations, these functionals have full covariance rank: gradient and Hessian orders are distinct, and the map H -> Hv from symmetric matrices onto R^d is surjective. For two distinct unpinned points, the map

    (grad f(x),grad f(y),f(x)-f(y))

has range dimension 2d+1 and parameter dimension 2d. For a tie to a pinned height use (grad f(x),f(x)-b), or the analogous s map, with dimensions d+1 and d.

Exhaust the relevant domains by compact sets separated from pins and from coincident points; use finitely many sphere charts when v occurs. Covariance positivity from Section 2 and compactness give bounded Gaussian densities on each such set. An elementary overdetermined-zero lemma applies: on a mesh of size epsilon, a random C^1 map from dimension q to dimension p>q, with derivative norm bounded by N, could have a zero only if one of O(epsilon^-q) mesh values has norm O(N epsilon). The density bound makes the probability O(N^p epsilon^(p-q)), tending to zero. First fix N, then let N increase; the random derivative norm is almost surely finite. This proves absence of all these zeros.

Pinned Hessians have a full density on their symmetric entries for fixed distinct pins, so their determinants are nonzero almost surely. The pinned heights differ because k,r>0. Countable exhaustion now proves that Q is almost surely Morse with distinct critical values. The transfer to Q^W follows by absolute continuity and 0<Z_r<infinity.

The ordinary elder event is Borel. For fixed pins define the maximin connection level

    d_f(M)=sup_{gamma(0)=M, f(gamma(1))>f(M)} min_t f(gamma(t)),

with supremum of an empty set equal to -infinity. The event d_f(M)>h can be represented by a countable union of polygonal-path tests, with rational interior vertices and endpoints, all staying strictly above h and with endpoint value above f(M). Strict clearances allow approximation by such paths. For variable M, use countably many coordinate charts and an initial segment depending continuously on M. These are Borel tests in (M,f). Thus equality d_f(M)=f(S) is Borel. On the Morse distinct-value locus it expresses ordinary elder death at S; an essential maximum has no older endpoint.

On G_r the unchanged deterministic cap theorem controls every boundary exit and supplies a connection through S to a point higher than b. Therefore the GLOBAL ordinary elder partner is S, not merely a local adjacency candidate. Equations (7.8) and the full-probability locus prove Theorem A. The cap theorem also gives exactly one actual ascending branch to M. It does not identify that branch with the transverse ridge.

## 9. A source-bound weighted Kac-Rice interface

For clarity, this section specifies why a global elder mark is admissible; a formula for local Hessian index alone would not suffice.

Apply Kac-Rice first on compact spatial pair domains separated from the diagonal, to the Gaussian vector field (grad f(x),grad f(y)). It is C^1 and has positive definite point covariance for x!=y by Section 2. Its derivative is block diagonal with Jacobian |det H_x det H_y|. Conditional laws of the whole smooth field depend continuously on the imposed gradient values by finite-dimensional Gaussian regression. Derivative moments are locally bounded.

The expected-integral extension in Armentano-Azais-Leon, *On a general Kac-Rice formula for the measure of a level set*, arXiv:2304.07424v3, Theorem 7.1, gives the identity for bounded nonnegative continuous weights in the location and an additional Gaussian field mark. It is enough first to use cylinder weights depending on finitely many fixed evaluations of f and its derivatives through order two. Their conditional distributions depend continuously on the imposed gradients by Gaussian regression, so the paper's continuity requirements hold. The field is a random element of the separable C^2 space on the torus. On each compact pair domain the two measures on location x C^2 are finite, by the unweighted formula. Cylinder evaluations on a countable dense set, together with location coordinates, generate this Borel sigma-algebra. Equality on their bounded continuous test-function algebra gives equality of the Borel measures by the monotone-class theorem. This extends the identity to bounded Borel global marks, rather than pretending the elder indicator is continuous or lower semicontinuous. No independence between the mark and gradient field is assumed.

Use the Borel elder mark from Section 8 and the type indicators. Disintegrate the two heights using their nondegenerate joint Gaussian density conditional on the gradients. This gives the full-pin density times E_Q[W] for candidates, and the same quantity times p_r for selected pairs. The continuous Gaussian regression kernel defines the canonical density version; the underlying measure identity is unaffected by height-null-set choices.

Exhaust the off-diagonal spatial domain toward r=0. The r dr bound proved in the next section makes the expectations locally integrable, and nonnegative monotone convergence completes the near-diagonal formula. No finite-grid approximation of the continuum field is substituted.

## 10. Exact radial ledger: the dimension cancels

Let pi_r(R;v_r) denote the density of U_r at (3.3). Since U_r=T_r O_r,

    density_{O_r}(specified pins)
           =12 r^(-(d+3)) pi_r(R;v_r).                   (10.1)

Use midpoint z, directed separation h=r u, birth b, and gap mark k. The change from the ordered pair (M,S) to (z,h) has absolute determinant one. Polar separation contributes r^(d-1) dr d sigma(u), where sigma is ordinary, NOT probability-normalized, surface area on S^(d-1). The height change (f(M),f(S))=(b,b-k r^3), at fixed r, contributes r^3 db dk. There is no factor 1/2: maximum/saddle pairs are ordered by their different roles.

Consequently the candidate intensity per unit midpoint volume is

    r A_r(b,k,u) dr db dk d sigma(u),
    A_r=12 pi_r(R;v_r) (Z_r/r^2).                        (10.2)

The powers are

    spatial (d-1) + height-mark 3 + pin -(d+3)
                         + determinant-normalizer 2 = 1.

By Sections 3 and 5,

    A_r -> A_0=12 pi_0(R;v_0) z_0(b,k,R)>0               (10.3)

uniformly and with a uniform finite upper bound. Although frames were used in the proof, (10.2) is invariant under a rotation of the transverse frame: the original zero-gradient law and determinant weight are unchanged by that orthogonal coordinate change. Thus it is a function of u alone. Uniformity was proved on the compact orthogonal group and descends to the sphere; no nonexistent global continuous transverse frame on the sphere is assumed.

The selected intensity is r A_r p_r with the same measures. This supplies the domination needed below, rather than assuming a fixed-axis statement can be integrated.

## 11. Lifetime pushforward and the exact compact coefficient

For ell<k_- r_*^3, every k in K gives r=(ell/k)^(1/3)<r_*. At fixed k,

    r (dr/dell) = (1/3) k^(-2/3) ell^(-1/3).             (11.1)

Hence a version of the candidate density is

    nu_cand(ell)=ell^(-1/3) integral_{B x K x S^(d-1)}
        A_{(ell/k)^(1/3)}(b,k,u)/(3 k^(2/3))
                                     db dk d sigma(u),  (11.2)

and nu_eld has an additional factor p_{(ell/k)^(1/3)}. Uniform convergence and compact domination give the common positive leading coefficient

    c_{B,K}=4 integral_{B x K x S^(d-1)}
           k^(-2/3) pi_0(R;v_0) z_0(b,k,R)
                                     db dk d sigma(u).  (11.3)

In (11.3), z_0=(6k)^2 E[det(A_0)^2 1{A_0<0}], with A_0's actual conditional covariance. Formula (11.3) is an identified Gaussian integral, not an evaluated closed form. There is no isotropic GOE substitution and no unexamined image-tail truncation.

The pointwise selection estimate supplies the stronger difference statement

    0<=nu_cand(ell)-nu_eld(ell)
      <= C ell^(2/3) integral_{B x K x S^(d-1)}
                           A^*/(3 k^(5/3)) db dk d sigma,

which is finite because k>=k_->0. This proves (1.2). It is an O(ell^(2/3)) difference, not a claim that either density separately has a second-order expansion with that remainder. Convergence of A_r to A_0 was not assigned a numerical rate.

## 12. Short bars, moments, and the exact remaining boundary

Integrating the density versions gives the following expected per-unit-volume results for the same spatial/mark-restricted populations:

    E N_eld(0,t] ~ (3/2)c_{B,K} t^(2/3),
    0<=E[N_cand(0,t]-N_eld(0,t]]<=C t^(5/3).             (12.1)

For any real q>-2/3,

    E sum_{selected, ell<=t} ell^q
           ~ c_{B,K} t^(q+2/3)/(q+2/3).                 (12.2)

The same leading expression holds for candidates. For q>-5/3 the nonselected difference alone satisfies

    E sum_{candidate but not selected, ell<=t} ell^q
                              <= C_q t^(q+5/3).         (12.3)

For q<=-2/3 the two individual expectations in (12.2) are infinite for every sufficiently small positive t, because their densities have a positive ell^(-1/3) leading term. Equation (12.3) concerns the nonselected counting measure directly, not subtraction of two infinities. In particular the compact-window inverse-lifetime threshold is p<2/3 for integrability of ell^(-p), with logarithmic divergence at p=2/3. This is unrelated to the inverse-Hessian/Schur thresholds in issue #56.

The corresponding unrestricted cumulative and individual-moment corollaries follow from Theorem C with c_{d,L} in place of c_{B,K}, by the same integrable-power argument. The compact-window nonselected difference rates (12.1),(12.3) are not automatically unrestricted rates.

## 13. Removing all birth and gap cutoffs by a direct intensity majorant

The covariance Sigma_r of U_r depends on radius and frame, NOT on the imposed target. Section 3 therefore supplies a fixed r_0>0, independent of b and k, on which its eigenvalues are bounded above and away from zero. Reduce r_0 below the torus injectivity scale and below one. Do NOT require a globally uniform lower bound for Z_r or a globally uniform constant in (1.1).

For every b in R, k>0 and 0<r<=r_0, the exact target (3.3) satisfies

    |v_r| <= C(|b|+k),
    |v_r|^2 >= c(b^2+k^2),                               (13.1)

with fixed c,C>0. For the second inequality it suffices to use its two coordinates b-k r^3/2 and 12k: if r<=1, then b^2<=2(b-k r^3/2)^2+k^2/2. Thus the invertible covariance bound yields

    pi_r(R;v_r) <= C exp[-c(b^2+k^2)].                    (13.2)

Conditional Fourier-coefficient means are bounded by C|v_r| and their centered variances by one. The same summability/Minkowski proof as Section 4 now gives the TARGET-GROWTH bound

    E_Q (1+||f||_{C^3})^p <= C_p(1+|b|+k)^p              (13.3)

for every fixed finite p, uniformly over r and frames on this r_0 band. This is not the compact-target bound (4.1) applied outside its scope; it is its explicit polynomial-growth version.

By (5.1),(5.3), for r<=1,

    |det H_i|/r <= C (1+||f||_{C^3})^d,
    0<=W_r/r^2<=C(1+||f||_{C^3})^(2d).

These inequalities remain true when k is arbitrarily small; no division by k or by Z_r has occurred. Combining with (13.2),(13.3) gives the full unnormalized-intensity majorant

    0<=A_r(b,k,u)=12 pi_r(R;v_r) Z_r/r^2
      <= C(1+|b|+k)^(2d) exp[-c(b^2+k^2)] =: H(b,k).    (13.4)

For every fixed positive r and every b,k>0, Z_r is positive and finite: append BOTH endpoint Hessians to the distinct-site pins; the full residual Gaussian density is positive on a product of an open maximum cone and an open index-(d-1) cone. Thus p_r is defined, even though no globally uniform lower floor has been used. For every fixed b,k>0, Theorem A on a compact neighborhood implies p_r->1 as r->0.

At a fixed lifetime ell, the near-pair restriction r<=r_0 is exactly k>=ell/r_0^3. After the exact change (11.1),

    ell^(1/3) nu_cand^near(ell)
       = integral_{R x (0,infinity) x S^(d-1)}
           1{k>=ell/r_0^3} A_{(ell/k)^(1/3)}(b,k,u)
                       /(3 k^(2/3)) db dk d sigma,       (13.5)

with the integrand defined to be zero where the indicator is zero. The selected near density has the additional factor p_r. Both integrands are dominated by H(b,k)/(3 k^(2/3)). This majorant is integrable: the k exponent -2/3 is strictly greater than -1 at zero, and Gaussian tails absorb the polynomial at infinity and in b. This explicitly handles the shrinking-k boundary where a compact-window selection constant can blow up.

For each fixed b,k>0,u, the indicator tends to one, A_r->A_0 and p_r->1. Dominated convergence therefore gives the same limit for the two near densities:

    c_{d,L}=4 integral_{R x (0,infinity) x S^(d-1)}
           k^(-2/3) pi_0(R;v_0) z_0(b,k,R)
                            db dk d sigma(u),           (13.6)
    0<c_{d,L}<infinity.

In particular c_{B,K} increases to c_{d,L} along an exhausting sequence of mark rectangles. This step uses the unnormalized product pi_r Z_r, not an unjustified globally uniform Q^W probability bound. It proves an unrestricted leading limit; it does not invent a quantitative unrestricted remainder from dominated convergence.

## 14. The off-diagonal contribution is bounded, not silently discarded

On the compact set of spatial pairs with torus distance at least r_0, the ORIGINAL full-pin covariance is uniformly positive definite. Every distinct-site jet list has full rank by Section 2, and compactness applies without a diagonal singularity. Its inverse is uniformly bounded. For 0<ell<=1, the original target is (b,0,...,0,b-ell,0,...,0), whose squared norm is at least b^2. Its Gaussian density is therefore at most C exp(-c b^2).

Conditional endpoint determinant-product moments on this domain are at most C(1+|b|)^(2d). The Kac-Rice height transformation (f(M),f(S))=(b,b-ell) has absolute determinant ONE here. Integrating over b and the compact relative-position domain gives

    0<=nu_eld^far(ell)<=nu_cand^far(ell)<=C,  0<ell<=1.   (14.1)

No remote-critical count bound conditional on a rare cap failure is inferred. Equation (14.1) is an unconditional off-diagonal height-density bound proved directly from a NONSINGULAR covariance on a fixed compact set. It is not the old shrinking-annulus RN enclosure and does not discharge that numerical obligation.

The near and far populations cover all distinct pairs (a distance boundary may be assigned to either; it is spatially null). Stationarity converts the spatial integral into a per-unit-volume density. Combining (13.5)-(13.6) with ell^(1/3) times (14.1) proves Theorem C. On the almost-sure Morse distinct-critical-value locus, each finite superlevel H0 bar has exactly one local-maximum birth and one index-(d-1) merging-saddle death. Thus the selected full-pair count is precisely the finite-bar count, not just a local proxy. The essential global-maximum class is not counted.

## 15. Analytically eliminate birth and gap from the leading coefficient

This simplification uses inversion symmetry/parity of a stationary real covariance, NOT rotational isotropy. At the contact point write G=grad f and t_u=partial_u^3 f. Let V_u=H_f u, expressed in the frame R; it has d coordinates (f_xx,f_xy1,...,f_xym). Let A_u be the transverse Hessian block on u-perp.

All odd derivatives (G,t_u) are jointly independent of the even derivatives (f,V_u,A_u), since every odd derivative of the even covariance at zero vanishes. Put

    tau_u^2=Var(t_u | G=0)>0,
    D_u=E[(det A_u)^2 1{A_u<0} | V_u=0].                 (15.1)

Here A_u conditional on V_u=0 is an actual centered nondegenerate Gaussian symmetric matrix, not automatically GOE. Finite-jet rank proves tau_u>0, positive densities p_G(0),p_Vu(0), and D_u>0. All are continuous in the frame and invariant under transverse-frame changes.

The contact density and cone moment factor as

    pi_0(R;v_0)=p_(f,V_u)(b,0) p_G(0) phi_tau_u(12k),
    z_0=(6k)^2 E[(det A_u)^2 1{A_u<0} | f=b,V_u=0],

where phi_tau is the centered scalar Gaussian density of standard deviation tau. Integrating b by ordinary Gaussian disintegration gives p_Vu(0) D_u. Thus (13.6) becomes

    c_{d,L}=144 integral_{S^(d-1)} p_G(0) p_Vu(0) D_u
                        [integral_0^infinity k^(4/3)
                                   phi_tau_u(12k) dk] d sigma.

The scalar integral can be evaluated exactly. With t=12k and the elementary gamma integral,

    144 integral_0^infinity k^(4/3) phi_tau(12k) dk
      = Gamma(7/6) tau^(4/3) / [24^(1/3) sqrt(pi)].

Consequently the full leading coefficient is

    c_{d,L}= Gamma(7/6) / [24^(1/3) sqrt(pi)]
       * integral_{S^(d-1)} p_G(0) p_Vu(0) tau_u^(4/3)
                              D_u d sigma(u).            (15.2)

This eliminates both unbounded mark integrals. The remaining angular and negative-definite-cone expectation are finite, strictly positive and specified by derivatives of the EXACT periodized covariance. Formula (15.2) is an explicit finite-dimensional Gaussian representation, not a reported elementary closed form or numerical enclosure. The factor24 is a numerical Jacobian/gamma factor, NOT the side length L; it is the same factor for every L. Changing the field amplitude by a>0 scales the coefficient by a^(-2/3), consistent with nu_(af)(ell)=a^(-1)nu_f(ell/a); this is an additional normalization check.

## 16. Supplied interfaces, retained limits, and review

**Interfaces supplied by this candidate:** the actual transverse matrix boundary argument (including d=3); a uniform compact-mark asymptotic full normalizer; all orientations without false isotropy; the global Borel marked Kac-Rice interface; exact pin/radial/lifetime factors; compact-window leading density and quantitative selection difference; a separate integrable unnormalized Gaussian majorant removing ALL birth/gap cutoffs; an off-diagonal bound completing ALL spatial separations; the unrestricted finite-bar leading density and its cumulative/moment consequences; and analytic elimination of birth and gap in its coefficient.

**Not supplied:** a numerical r_* or C on a prescribed band; a globally uniform O(r^3) probability constant over unbounded b,k; the compact O(ell^(2/3)) difference as an unrestricted remainder; an elementary closed form or numerical enclosure of (15.2); a rigorously evaluated lower coefficient for pairing failure; the original numerical H3/LPW replay; a shrinking-annulus RN certificate; a24-jet displacement enclosure; formal proof verification; nonauthor mathematical acceptance; or unrestricted P15 prize closure. No historical scientific status is changed by this author-side proof.

The exact-arithmetic program checks pin transforms/targets, block determinants, finite typed-matrix envelopes, spectral Jacobians, radius/lifetime exponents, explicit small-eigenvalue integral models, global target coercivity and coefficient scaling. Semantic mutations challenge the load-bearing factors. These are finite algebra/implementation controls, NOT Gaussian simulation or proof of continuum estimates.

Priority nonauthor review: (i) desingularized full residual covariance; (ii) uniform all-field moments and independent-matrix regression; (iii) both soft determinant factors; (iv) eigenvalue integration retaining the random largest-eigenvalue width and all corank strata; (v) scalar far branch; (vi) full-normalizer type convergence; (vii) continuous-to-Borel marked Kac-Rice extension; (viii) every factor in the pin/lifetime ledger; (ix) target-growth bound (13.3) and integrability at k=0 in (13.5); (x) distinct-site compact covariance in the far domain; and (xi) parity, b-disintegration and the gamma factor in (15.2).

### Primary-source attribution

Armentano, Azais and Leon, arXiv:2304.07424v3, Sections 2 and 7, supply established Gaussian Kac-Rice/expected-integral background. The precise full-field/Borel and pair application is explained in Section 9, not attributed wholesale to that paper. Curry, *The Fiber of the Persistence Map for Functions on the Interval*, arXiv:1706.06059v2, Section 3, supplies elder-rule background; the all-dimensional separating cap is the explicitly identified in-project predecessor. Beliaev, Cammarota and Wigman, *No repulsion between critical points for planar Gaussian random fields*, arXiv:1911.03455, is neighboring two-point/index-asymptotic work, not a source of our elder-selection conclusion. No priority or novelty claim follows from this bounded reconnaissance.
