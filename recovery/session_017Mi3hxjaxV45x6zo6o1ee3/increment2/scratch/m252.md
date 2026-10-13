# Independent adversarial audit — central lifetime chain, first pass
Date: 2026-10-03. Audit ID: IBA-20261003-P1.

**Dylan Roy — delegated AI review. Actual performer: OpenAI Codex, this isolated session.** Dylan's personal reading: PENDING. Scientific/status effect: NONE.

## Executive conclusion

**The scoped central theorem is apparently supported by the supplied ordinary mathematical argument**, with the explicit congruence correction described below. I reconstructed the cap, weighted boundary layer, normalization, leading density and bounded-remainder argument and did not find a concrete counterexample or an unresolved load-bearing gap in that chain.

This is NOT a completed audit of every theorem in the research program. In particular, downstream higher-order expansions, cluster-limit theorems, numerical finite-band certificates, formal statement alignment, and all historical proof packets have not been exhaustively audited. The entire-program request remains incomplete in those respects. No status label, prior review, CI result or formal-verification label is evidence for this conclusion.

The supported scope examined here is: for each **fixed** integer d>=2 and fixed L>0, the exact normalized periodized Gaussian field, expected ordinary superlevel H0 finite-bar density per unit volume, and a suitable density version,
    
    nu_eld(ell) = c_{d,L} ell^(-1/3) + O_{d,L}(1), ell -> 0+,
    0 < c_{d,L} < infinity.

There is no demonstrated constant uniform in L or d, no samplewise density assertion, and no numerical small-radius threshold from this audit.

## Independence and exposure record

No other agent was contacted, delegated work, consulted, or asked to validate this analysis. No PR discussions, issue discussions, or files under reviews/ were opened. The repository tree was used only to locate source paths; seeing report filenames is not reading their arguments.

Strict blindness from session inception was unavailable: the preloaded conversation context already contained reviewer conclusions. These are contaminated and were excluded as evidence. A source amendment BOREL.md also embedded commentary about an earlier review in its introductory section; that commentary is contaminated and excluded. Its actual mathematical replacement was assessed as research source, with the underlying Kac-Rice theorem checked directly. Operational governance files contained historical status metadata, likewise excluded. This record must not be represented as a pristine never-exposed blind experiment or as organizationally independent review: source authors and this executor share a provider.

The current user instruction requires isolation, overriding repository instructions to consult peer findings. All publication is one-way. This report is a new append-only record: do not edit its body; corrections belong in a new separately dated report.

## Frozen source set and exact locations

Repository: d6g8k5htny-coder/Math-. All following files were read at commit **c2836c2ae5df1dfdf72549fd9480b83b4621113e**. Blob IDs below are Git blob SHA-1 identifiers, not SHA256 digests.

| Key | Source path | Git blob |
|---|---|---|
| P | imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md | dfed3b8d318a3ab1950957f393307733a4bef3f2 |
| C | imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md | 0633aca3c2a2882b0de4399da0a75d64c2e6b2e1 |
| E | imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md | 213594d6ca6a86fb938110f4d166d9ce275a02d0 |
| R | frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md | 247b3ecf80bfbe896948d5d489b2d5842a81c481 |
| B | frontiers/lifetime_moment_boundary_20260930/sources/BOREL.md, mathematical replacement section 2 | fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a |

Sources can be retrieved as https://github.com/d6g8k5htny-coder/Math-/blob/c2836c2ae5df1dfdf72549fd9480b83b4621113e/ followed by the listed path.

External primary source inspected: Armentano–Azaïs–León, arXiv:2304.07424v3, Theorem 7.1 and its assumptions, https://arxiv.org/html/2304.07424v3 . It supplies the continuous-weight starting point; it is not a theorem about this persistence pairing.

## Attack ledger

D means the stated component survives the attack within the frozen scope. OPEN is a coverage limitation, not an accusation of missing proof.

| ID | Claim / source | Attack and result | Level / severity | Required repair or follow-up |
|---|---|---|---|---|
| A-001 | Endpoint rescaling, P §5 | Literal multiplying congruence diag(sqrt(r),I) gives the wrong axial normalization; inverse factors are required. E and R §4 give the correct formula. | A in original displayed wording; repaired, low residual severity | Consume E with P; never quote the defective factor alone. |
| A-002 | Smoothness and jet rank, P §§2–3 | Positive full Fourier spectrum prevents any finite distinct-jet linear relation; Gaussian spectral decay controls all derivative moments. Survives. | D | Retain fixed L>0. |
| A-003 | Rotation, P §§3,15 | Torus is not O(d)-isotropic. Proof retains frame covariance and uses compactness, not equality of laws. Survives. | D | Do not substitute GOE at finite L. |
| A-004 | Pin transform, P §3 | Recomputed determinant 12 r^(-(d+3)) and target 12k. Survives. | D | None. |
| A-005 | Full normalizer, P §5 | Correctly rescaled typed determinants converge; transverse Gaussian density and uniform integrability give positive compact-mark floor. | D | Floor is compact in marks, not global. |
| A-006 | Deterministic cap, C §§2–5 | Tested every face and older endpoint, including arbitrary remote exits. Survives. | D | Keep both depth and fourth-derivative hypotheses. |
| A-007 | Ridge versus gradient flow, C §5 | Ridge is only a connecting path. Actual branch confinement follows separately from boundary values and gradient monotonicity. | D | Do not identify the two curves. |
| A-008 | Weighted bad-event mass, P §§6–7 | Double soft factor yields numerator O(r^5), not merely unweighted probability. Survives. | D | Keep actual W in all estimates. |
| A-009 | d=2 scalar layer, P §7 | Quadratic inequality has a large-lambda branch; source retains and controls it by weighted Markov. | D | Never freeze lambda as an independent largest eigenvalue. |
| A-010 | d=3,4 and higher corank, P §7 | Vandermonde is a Lebesgue coordinate Jacobian; all remaining eigenvalues integrated, no inverse powers. | D | Constants depend on dimension. |
| A-011 | Pinned Morse/ties, P §8 | Overdetermined mesh count on compact exhaustions has positive exponent; endpoint Hessians nondegenerate at fixed r. | D | No simultaneous uncountable-parameter probability-one claim. |
| A-012 | Borel elder Kac-Rice, P §§8–9, B §2 | Explicit finite-measure determining-class argument permits global mark; continuity of elder indicator not assumed. | D | Use whole-field Gaussian kernel, not just a formal conditional symbol. |
| A-013 | Exponent and coefficient, P §§10–15 | Independently recomputed all radial and gamma factors. | D | Ordinary sphere area and role ordering must remain explicit. |
| A-014 | Small k leading limit, P §13 | Integrable k^(-2/3) majorant controls joint r,k layer. | D | Dominated convergence alone gives no O(1) remainder. |
| A-015 | Far pairs/global short bars, P §14 | Direct nonsingular all-pair density is O(1); no theorem that every short bar is local is necessary. | D | Do not claim deterministic spatial localization. |
| A-016 | Higher-order mechanisms | All-mark envelope gives quantitative shrinking-band bounds; no codimension-only exclusion used. | D for leading-order exclusion | Separate sharp mechanism coefficients remain uncomputed. |
| A-017 | Multi-point clusters | Compact-mark local extra critical point contradicts cap uniqueness; its weighted pair contribution is in O(r^3) failure mass. | D for this first-intensity application | Does not establish cluster probabilities or higher factorial moments. |
| A-018 | Remainder determinant lemmas, R §3 | Filtered determinants remain controlled across index changes because crossings have determinant zero; off-block determinant is affine in r. | D | None found. |
| A-019 | Remainder target uniformity, R §§4–5 | Joint pin/matrix Gaussian density retains b,k decay; independent residual only, not independent eigenvalues. | D | Preserve polynomial target growth. |
| A-020 | eta split, R §§6–7 | Exact integration gives scaled O(ell^(1/3)); eta=ell^(1/6) is sufficient, not unique. | D | Retain k>=ell/r0^3. |
| A-021 | Scope beyond fixed d,L / expectation | Audited argument does not establish uniform L,d or samplewise laws. Sources themselves state fixed-parameter expectation scope. | C if broadened; no such source defect found | Prevent downstream scope inflation. |
| A-022 | Published finite-L numerical coefficient | No independent finite-L directional/cone quadrature completed; only a nonperiodic reference computed. | OPEN, audit limitation | Independent exact-periodic implementation and error enclosure. |
| A-023 | Entire research program / formal alignment | Not exhaustively inspected. | OPEN, substantial coverage limitation | Separate source-only passes over remaining theorems and formal statement alignment. |

## Independent reconstruction

### 1. Field, symmetry and nondegeneracy

Poisson summation converts the normalized covariance to a Fourier series with
    
    a_n = exp(-2 pi^2 |n|^2/L^2) / sum_j exp(-2 pi^2 |j|^2/L^2) > 0.

For each fixed L>0, sum sqrt(a_n)(1+|n|)^q is finite for every q. Minkowski therefore bounds C^q norms in every finite L^p; almost sure smooth versions follow using a countable sequence of derivative orders. If a finite linear combination of distinct derivative evaluations had variance zero, positivity of every a_n would force the corresponding finite-support distribution to annihilate all Fourier modes. Fourier uniqueness and local jet test functions then force every coefficient to vanish.

This proves finite distinct-jet rank, not just sampled numerical rank. Degenerating separation still needs desingularization; raw two-point covariance is not uniformly nonsingular near collision. For fixed positive L there is no exceptional value at which the Fourier support collapses. Uniformity as L tends to zero or infinity is not supplied.

Translations, coordinate reflections and coordinate permutations are symmetries. Arbitrary rotations generally are not. The source's use of an orthonormal coordinate frame is legitimate because covariance tensors are transformed with it and bounds are uniform on the compact frame group. It never needs the conditional matrix distribution to be rotationally invariant.

### 2. Pinning and endpoint geometry

For axial observations (f(a),f_x(a),f(c),f_x(c)), a=-r/2,c=r/2, the four transformed rows are the average height, secant, derivative difference/r, and
    
    6/r^2 [f_x(a)+f_x(c)-2(f(c)-f(a))/r].

Elimination gives absolute determinant 12/r^4. Each of d-1 transverse two-row blocks contributes 1/r. Target is (b-kr^3/2,-kr^2,0,12k,0,...). The limiting jet list contains distinct derivatives f, grad f, H u and partial_u^3 f. Its covariance has a uniform positive minimum over frames by continuity and compactness.

The Hermite identity independently forces the averaged third derivative to equal 12k. Thus f_xx(M)/r -> -6k and f_xx(S)/r -> 6k, whereas transverse Hessian A has a nondegenerate conditional Gaussian law. Writing H_i=[[r alpha_i,r beta_i^T],[r beta_i,A_i]], the correct rescaling is
    
    diag(r^(-1/2),I) H_i diag(r^(-1/2),I)
       = [[alpha_i,sqrt(r) beta_i^T],[sqrt(r) beta_i,A_i]].

This verifies the literal original wording issue in A-001 and verifies its existing repair independently. It does not invalidate the determinant identity
    
    det H_i/r = alpha_i det A_i - r beta_i^T adj(A_i) beta_i.

For fixed k>0, the limiting typed product is
36 k^2 det(A)^2 1{A<0}. Its expectation is strictly positive because a full Gaussian matrix density assigns positive mass to an open negative-definite ball. Polynomial determinant bounds and Gaussian derivative moments provide uniform integrability. This is why indicator discontinuities at singular Hessians do not spoil the limit.

### 3. Deterministic global cap attack

Normalize k=1/6 by dividing f by 6k. Write m=M3,n=M4. The hypotheses become lambda>8rm^2 and rn<=1/20. The pin identity implies m>=2. On the cylinder, transverse concavity is at least delta=lambda-5rm and f_xxx>=7/4.

The transverse gradient at y=0 has two zeros and second derivative bounded by m. Projection onto an arbitrary unit vector gives the interpolation estimate ||w(x)||<2mr^2, including outside the pins. Strong concavity gives a unique interior transverse maximizer h(x), with
||h||<2r/(8m-5). The implicit derivative estimate gives ||h'||<2/q+2/q^2, q=8m-5>=11. Differentiating g(x)=f(x,h(x)) eliminates the h'' term because the transverse component of H(1,h') is zero. The exact rational lower bound on g''' is
    
    7/4 - 2554128/1771561 = 2184415/7086244 > 1/4.

Thus F=g' is strictly convex with its prescribed two zeros. There are exactly two critical points in the whole cylinder. Comparison with (x-a)(x-c)/8 and integration gives g(-2r)<=b-9r^3/32 and g(2r)>=s+9r^3/32. The surplus over the gap is 11r^3/96.

For C=[-2r,r/2] times the closed transverse ball:
* the left face is strictly below s;
* the curved side is strictly below s by strong concavity and the ridge displacement bound;
* the right face is at most s with equality only at S;
* the interior has maximum b;
* the ridge path from M through S reaches a point with value greater than b.

Every path from M to an older point must cross this boundary, including paths that later leave the local chart, circle the torus or reenter. Hence its bottleneck is at most s. The exhibited path has bottleneck exactly s. For a compact Morse function with distinct critical values, this maximin value is the ordinary elder death level, and the unique critical point at that value is S. Attempts to install a remote competing saddle cannot change this conclusion without violating the boundary barrier or older-endpoint premise.

The ridge need not be a gradient trajectory. For any smooth positive metric, an unstable eigenvector at S has positive Hessian quadratic form and therefore is not wholly transverse. One half-branch enters C above s and cannot exit; compactness and finiteness of critical points force convergence to M. This separately establishes the branch assertion.

### 4. Weighted eigenvalue layer, including dimensions two through four

On typed maximum support B=-A_M>0, Schur complement monotonicity gives |det H_M|<=rh det B/2, h=M3. At the saddle, endpoint block displacement and the adjugate identity give
    
    W/r^2 <= (h^2/4) lambda1(lambda1+3rh/2)
                    product_{j=2}^m lambda_j(lambda_j+rh), m=d-1.

This is the crucial two-soft-factor bound. An unweighted probability estimate would not reproduce the required suppression.

Condition the full field further on A_M. Only the residual field J is independent of A_M; no independence between eigenvalues is used. Its derivative moments are finite and h<=K(J+Lambda). For m>=2, depth failure restricts lambda1 to [0,Dr(J+Lambda)^2]. Holding the other eigenvalues fixed is legitimate because Lambda=lambda_m is among them. Integrating lambda1(lambda1+Er(J+Lambda)) gives O(r^3) times a polynomial. The Gaussian matrix density times the Lebesgue spectral Jacobian supplies Gaussian integrability of all other factors.

For d=3, m=2, the Vandermonde is lambda2-lambda1, bounded by Lambda. For d=4, m=3, its degree is 3. For fixed d>=5 the degree is m(m-1)/2. In none of these cases is lambda2 bounded away from zero; simultaneous soft eigenvalues are included.

For d=2, m=1, freezing Lambda would be wrong. Instead lambda<=Dr(J+lambda)^2 implies lambda<=4DrJ^2 OR lambda>1/(4Dr). The first branch integrates to O(r^3) for W/r^2. On the second,
    
    E[(W/r^2) 1_far] <= (4Dr)^4 E[(W/r^2)lambda^4] = O(r^4).

The fourth-derivative exception has the same weighted Markov treatment. Dividing by the compact-mark full normalizer of order r^2 produces failure O(r^3), uniformly in compact b,k and all directions.

### 5. Measurability and Kac-Rice

At fixed positive separation, the overdetermined maps excluding degenerate critical points and value ties have target dimensions exceeding their parameter dimensions by one. On each compact domain away from pins and coincidences, covariance rank gives bounded densities. An epsilon mesh with a derivative bound N has O(epsilon^(-q)) sites and zero-event bound O(N^p epsilon^p); p>q makes the union bound vanish. Exhaustion and then N->infinity give the required pinned genericity. Hessians at the pins themselves have full residual density.

The elder maximin event is Borel: strict path clearances allow countably many rational polygonal tests in charts. Equality of the resulting connection value with f(S) remains Borel. No continuity of the elder indicator is asserted.

For a compact separated pair domain, define the expected marked zero-count measure on location times C^2(field), and the measure obtained by integrating the Gaussian regression kernel times the Hessian Jacobian. Both are finite. Gaussian regression provides a continuous full-field conditional kernel for continuous cylinder weights. The continuous-weight Kac-Rice identity equates their integrals on a bounded cylinder algebra generating the Borel sigma-algebra. Finite-measure monotone class then gives equality for every bounded Borel mark. This independently validates the extension rather than applying a lower-semicontinuous-weight theorem directly to a discontinuous elder equality event.

Height disintegration introduces the full-pin density once and the Hessian product once. Exhaustion toward the diagonal is justified by the integrable radial intensity below.

### 6. Every normalization and the leading exponent

For ordered maximum/saddle locations, (M,S) -> (midpoint,h) has absolute determinant one locally. Translation invariance gives per-unit-volume normalization without a second factor L^d. Polar h=ru gives r^(d-1) dr d sigma, with ordinary sphere area. Heights (b,b-kr^3) give r^3 db dk. Pins give 12 r^(-(d+3)); Hessian expectation is r^2 z_r. Thus
    
    d intensity = r A_r db dk dr d sigma, A_r=12 pi_r z_r.
    r dr = (1/3) ell^(-1/3) k^(-2/3) d ell.

No 1/2 is present because the roles of maximum and saddle distinguish ordering. This reconstructs the exponent from the full critical-pair law, not from a postulated cubic manifold.

The source's all-mark bound is A_r<=C(1+|b|+k)^N exp[-c(b^2+k^2)]. Therefore k^(-2/3)A_r has an integrable envelope. At lifetime ell the near domain is k>=ell/r0^3. Its indicator converges pointwise to one; fixed k positive gives the cap selection limit. Dominated convergence proves the unrestricted leading coefficient, even where r/k becomes large. Off-diagonal original covariance is uniformly positive on the compact separated domain; integrating heights gives an O(1) far-pair density.

Each finite ordinary H0 bar is among these typed ordered pairs. Consequently no missing deterministic theorem saying every short bar has nearby endpoints is needed. Far short bars may exist and are explicitly included.

### 7. Alternative mechanisms and clusters

A height law ell=a r^m alone never determines a density exponent. If the relevant radial weighted intensity is r^beta dr with coefficient a fixed positive, the pushforward is
    
    (1/m) a^(-(beta+1)/m) ell^((beta+1)/m-1) d ell.

Any shrinking coefficient-region measure must also be included. Applying beta=1 blindly to quartic events would be unjustified.

Here the exact endpoint pins already imply a cubic gap bound under bounded third derivatives. Quadratic gaps at collision demand derivative growth of order at least r^(-1); such events are in the target/derivative tails. For higher-order bands use the stronger bound from R,
    
    A_r <= (k+r)^2 H(b,k).

In a band k comparable to ell^alpha, 0<alpha<1, integrating its width comparable to k gives the actual density upper bound
    
    O(ell^(-1/3+7alpha/3) + ell^(1/3-alpha/3)).

Both terms are o(ell^(-1/3)). A mechanism k comparable to r^(m-3) has alpha=(m-3)/m. For quartic scaling m=4 this gives O(ell^(1/4)+ell^(1/4)) = O(ell^(1/4)); for higher m the bounds remain strictly lower order for every fixed finite m. These are upper bounds, not exact mechanism asymptotics. The collective small-k layer is controlled separately by the integrable envelope and remainder estimates; infinitely many informal mechanism labels are not summed.

If an additional critical point lies inside the cap cylinder, the cap good event is impossible because the ridge analysis gives exactly two critical points there. For compact b,k the Kac-Rice-weighted pair intensity associated with that event is therefore suppressed by O(r^3), giving O(ell^(2/3)) in the compact-mark density. This bounds pair contributions, not the probability or factorial moments of an arbitrary cluster population. Remote extra critical points can coexist with the good event and do not interfere with the cap barrier. No unjustified assertion that all clusters are rare is needed for the first-intensity theorem.

### 8. Bounded remainder rederived

For F_j(H)=|det H| times a fixed-index indicator, changing index requires crossing determinant zero. Hence the polynomial determinant Lipschitz estimate extends across index changes. For
    
    K_t=[[alpha,sqrt(t) beta^T],[sqrt(t) beta,A]],

det K_t=alpha det A-t beta^T adj(A)beta is affine in t, giving the stronger filtered difference bound r|beta^T adj(A)beta|. This avoids a spurious sqrt(r) error and is valid at singular A.

Centered observation rules and Gaussian regression coupling give second-order law error with polynomial target growth. Endpoint displacement is first order. Combining filtered endpoint estimates produces
    
    A_r <= (k+r)^2 H,
    A_0 <= k^2 H,
    |A_r-A_0| <= r(k+r)H.

Let a=ell/r0^3. After multiplying the density by ell^(1/3), candidate error monomials are ell^(1/3)H and ell^(2/3) k^(-4/3)H. Integration k>=a gives O(ell^(1/3)); missing contact mass k<a is O(a^(7/3)).

For selection loss, joint pin-plus-transverse-matrix density retains exp[-c(b^2+k^2+||A||^2)]. The spectral layer width is now proportional to delta=r/k; weighted integration yields B_r<=delta^3 H for delta<=1. This is unnormalized; no globally positive floor for Z is assumed.

Splitting at eta=ell^(1/6), small-k scaled mass is bounded by
    
    C eta^(7/3) + C ell^(2/3) a^(-1/3)
       = O(ell^(7/18)) + O(ell^(1/3)).

For k>=eta, delta<=ell^(1/9) and the scaled loss is bounded by
    
    C ell integral_eta^infinity k^(-14/3)H dk
       = O(ell eta^(-11/3)) = O(ell^(7/18)).

Since 7/18>1/3, division by ell^(1/3) gives O(1). More generally eta=ell^alpha works for 1/7<=alpha<=2/11; 1/6 lies inside. The split is sufficient, not uniquely necessary. Far pairs contribute another O(1). This proves the bounded remainder at the tested dependency chain's scope, not merely a little-o leading result.

### 9. Coefficient and numerical falsification check

Odd and even contact derivatives are independent by inversion symmetry. Put G=grad f, V=H u, t=partial_u^3 f, tau_u^2=Var(t|G=0), and
D_u=E[det(A)^2 1{A<0}|V=0]. Integrating b disintegrates the even block. The remaining scalar integral is
    
    144 integral_0^infinity k^(4/3) phi_tau(12k) dk
       = Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt(pi)).

Thus
    
    c = Gamma(7/6)/(24^(1/3)sqrt(pi))
          integral_sphere p_G(0)p_V(0) tau_u^(4/3) D_u d sigma(u).

Finite-jet rank and compact directions give positivity and finiteness. This keeps directional dependence at finite L. At k=0 the actual contact integrand behaves as k^(4/3), while tails in k and b are Gaussian. Field amplitude multiplication by a>0 changes c by a^(-2/3), consistent with density rescaling.

Independent numerical reference only: for the NONPERIODIC d=2 covariance exp(-|z|^2/2), p_G(0)=1/(2pi), p_V(0)=1/(2pi sqrt(3)), tau^2=6, and Var(A|V=0)=8/3, so D=4/3. This gives c=0.0734069193060343. This is not a claimed finite-L numerical enclosure or a comparison with an inspected published numerical value.

Executed fresh Python standard-library checks:
* rational Gaussian elimination of the axial pin transform at r=1,1/7,2/9: all match 12/r^4;
* exact rational cap constants and 7/18 remainder exponent: pass;
* independent midpoint quadrature of the scalar gamma integral (200000 intervals, t in [0,12], tau=1): relative difference about 3.27e-12 from the formula.

These finite computations falsify some normalization mistakes; they do not certify the continuum proof.

## Remaining work and falsification experiments

1. Complete source-only coverage of downstream higher-order remainder expansions and cluster distribution/factorial-moment claims. This report supplies no verdict on them.
2. Independently implement exact-periodic derivative tensors and conditional covariance Schur complements for several directions in d=2,3,4; check positivity and explicitly measure anisotropy. Use image sums or spectral sums with controlled tails, not an isotropic surrogate.
3. Evaluate the finite-L coefficient with directional and cone integration plus numerical error control. The nonperiodic benchmark above is useful for detecting factors 2, 3, 12 and sphere normalization.
4. Sample the full conditional residual field with determinant/index importance weights; estimate E[W 1_bad], E[W], and cap failure separately. Examine d=2 far-lambda events as well as the near layer. Unweighted failure frequencies are not a test of the theorem.
5. Build deterministic polynomial-plus-transverse-quadratic cap examples, perturb the exterior to add competing maxima and saddles, and verify maximin connectivity. Violating a cap premise is not a counterexample; record premise checks for every attempted falsifier.
6. Compare joint scaling bands k~ell^alpha and refine spatial resolution faster than r. Discrete persistence agreement without continuum discretization control is empirical only.
7. Audit formal theorem statements and axioms separately against these analytic claims. No Lean execution or alignment was performed here.

Smallest repair to the examined written chain: always bind the already-present congruence erratum to P §5. I found no need to change the scoped central theorem. Remaining items above are audit coverage or stronger-result obligations, not manufactured proof gaps in the scoped theorem.

## Dependency map

    positive Fourier spectrum
       -> smooth moments + finite-jet rank
       -> contact regression + full normalizer + matrix density envelope
       -> weighted cap failure control
    deterministic cap -> global elder selection
    pinned genericity + Borel kernel/marked Kac-Rice
       -> exact all-pair intensity
    exact intensity + contact limit + all-mark domination + far bound
       -> positive leading lifetime coefficient
    filtered determinant cancellation + joint-density loss bound
       + retained lower k cutoff + eta split + far bound
       -> O(1) remainder

The exponent, coefficient and remainder have different dependencies. A failure in the cap would threaten equality of elder and candidate leading coefficients but would not by itself refute the candidate normalization calculation. A failure of quantitative loss control could invalidate the O(1) claim while leaving the leading limit intact.

## Continuation protocol

This record closes the first pass over the central chain only. Entire-program coverage is explicitly incomplete. Preserve this report unchanged. A delayed fresh pass should start from canonical proof sources, not this report or others' reports; prioritize counterexample construction, exact-periodic coefficient computation and the unexamined downstream claims. Do not claim that a waiting period itself creates independence or that the scheduled future work has already run.

