# D5 all-height fixed-annulus bridge candidate

**Object:** D5-ALL-HEIGHT-ANNULUS-BRIDGE-20260925-v1.  
**Author:** OpenAI / ChatGPT, session OA-D5-ANNULUS-BRIDGE-20260925.  
**Disposition:** AUTHOR-SIDE ANALYTIC CANDIDATE; independent review pending.  
**Scientific effect:** NONE. No theorem/claim-status register is changed.

## 1. Statement and exact scope

Fix L>0 and the centered variance-one Gaussian field on the two-dimensional torus with covariance

    K_L(h) = sum_(n in Z^2) exp(-|h+Ln|^2/2)
             / sum_(n in Z^2) exp(-|Ln|^2/2).

Fix a compact birth interval, 0<k_-<=k<=k_+<infinity, and 1<A<B<infinity. For every orthonormal frame write f(x,z)=F(x e1+z e2), with midpoint zero. Impose exactly the six finite-r observations

    f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0,
    M=(-r/2,0), S=(r/2,0).                                  (1)

Q_r denotes their continuous Gaussian regression law, not an adjacency law. For a symmetric 2x2 matrix put F_j(H)=|det H| times the indicator of negative index j, zero on singular matrices. Define

    W_r=F_2(H_M) F_1(H_S), Z_r=E_Qr W_r, dQ_r^W=(W_r/Z_r)dQ_r.

For a Borel set E in K_AB={(u,v): A<=sqrt(u^2+v^2)<=B}, let N_j(rE) count index-j critical points in rE at **all heights**.

**Candidate theorem.** There are C,r_*>0, uniform in the stated marks, frames, Borel E and j=0,1,2, such that

    E_(Q_r^W) N_j(rE) <= C r^3 area(E), 0<r<=r_*.           (2)

Equivalently, the physical-area weighted critical-point intensity satisfies rho_j^W(ru,rv)<=Cr throughout this fixed scaled annulus. Summing over indices only changes C. Markov gives an upper bound on witness existence, not a lower bound. The between-pin-height count is a subset and satisfies the same bound. Since k>=k_->0, (2) can also be written <=C' k r^3 area(E), but that rewriting uses k_-: **there is no height-window factor in the proof and no assertion uniform as k approaches zero**.

This addresses the fixed scaled-annulus region in d=2. It does not include pin neighborhoods, A approaching 1/2, B growing with r, intermediate physical distances r<<distance<<rho, witness-witness collisions, d>=3, uniform-L estimates, numerical all-cell certificates, elder selection or unrestricted lifetime acceptance. Choose r_* small enough that all local segments used below embed in the torus.

## 2. What was missing, and what is added

PR22's two-scale estimate controlled the conditional product of three Hessians by C delta^-6, leading to the physical intensity C r^-5 delta^-8 exp(-c/delta^2). That estimate alone retains an adverse r power on a strip of fixed positive scaled width.

The additional fact is deterministic: **three noncollinear zero gradients force all three Hessians to be small**. The Gaussian input must control the FULL conditional C3 norm after the witness gradient condition, not only endpoint-only moments. Combining the two facts supplies the factor

    min(1,(r/|v|)^6),                                      (3)

with value 1 at v=0. Splitting at |v|=r, not sending a fixed-chart constant to infinity, absorbs the remaining inverse powers into the Gaussian penalty. No new height conditioning or shrinking polynomial/logarithmic cutoff is needed.

All precursor sources remain unreviewed same-provider candidates. Sections 3-8 give the required arguments explicitly rather than treating those sources as accepted premises. The separate reviewer must verify these arguments, not infer their validity from matching implementations.

## 3. Uniform endpoint regression, moments and original Z

Let a=r/2, g(x)=f(x,0), h(x)=f_z(x,0). Let H_r be the cubic interpolating g and g' at +/-a, and L_r the affine interpolant of h at +/-a. Their midpoint coefficients form an invertible observation vector

    U_r=(H_r(0),H'_r(0),H''_r(0),H'''_r(0),L_r(0),L'_r(0)).

For arbitrary data define s0=(g(-a)+g(a))/2, d0=(g(a)-g(-a))/(2a), s1=(g'(-a)+g'(a))/2 and d1=(g'(a)-g'(-a))/(2a). Then

    U_r=(s0-a^2 d1/2,(3d0-s1)/2,d1,3(s1-d0)/a^2,L_r(0),L'_r(0)).

The exact target under (1) is

    v_r=(b-k r^3/2,-3k r^2/2,0,12k,0,0),
    H_r(x)=b-k r^3/2-3k r^2 x/2+2k x^3, L_r=0.            (4)

The finite-r targets are retained BEFORE rescaling. U_r converges in every fixed Lp at rate O(r^2) to U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)(0).

The covariance has strictly positive, Gaussian-decaying Fourier variances at every lattice mode. The sum of their square roots times any polynomial frequency weight is finite. Minkowski applied to the Fourier series therefore bounds every fixed C^m supremum in each finite Lp, uniformly over frames. For any distinct one-site derivative monomials, a zero-variance combination gives a polynomial symbol zero on the rotated lattice. Undo rotation and successively fix integer coordinates: that polynomial is identically zero. Thus the jet covariance is positive definite; compactness of O(2) gives uniform positive/finite bounds. At distinct sites the analogous statement follows from uniqueness of Fourier coefficients of distributions and isolation of the supports by test functions.

Taylor formulas justify the U_r convergence, including mixed covariances. Its inverse covariance and its C^m field cross-covariances are bounded. The Gaussian coupling

    F_Q = F + Cov(F,U_r) Cov(U_r)^-1 (v_r-U_r)

shows that all fixed C^m/Lp moments under Q_r remain uniformly bounded. This uses the whole local Taylor domain, including pins, midpoint and segments, rather than an annulus-only derivative bound.

For completeness the full endpoint-only normalizer has a uniform lower bound

    Z_r >= c_Z r^2.                                        (5)

Indeed, with A_r=f_zz(0), the exact finite-pin Taylor relations give, under Q_r,

    det H_M/r = -6k A_r+O_Lp(r),
    det H_S/r =  6k A_r+O_Lp(r),
    f_xx(M)/r = -6k+O_Lp(r).

For example f_xx(0)=-(r^2/24)f_xxxx(0)+O_Lp(r^4), f_xxx(0)=12k+O_Lp(r^2), and f_xz at the endpoints is O_Lp(r). Products of remainders are controlled with Holder. Appending f_zz to U_0 leaves distinct derivative monomials. Thus A_r has conditional variance bounded above and below and bounded mean, so Q_r(-2<=A_r<=-1)>=p0>0. Remove the events where any displayed error exceeds 3k_-; their union has probability O(r^p). On the remainder H_M is negative definite, H_S is indefinite, and both absolute determinants are >=3k_-r. This proves (5). No witness-conditioned quantity replaces Z_r.

## 4. Deterministic three-Hessian lemma

Assume grad f(M)=grad f(S)=grad f(X)=0, X=(ru,rv), |u|,|v|<=B and v!=0. Let M_3 bound the trilinear operator norm of D^3 f on a convex neighborhood of M,S,X.

Taylor's integral formula from M to S gives

    ||H_M e1|| <= r M_3/2.

Put d=(u+1/2,v). The formula from M to X gives

    ||H_M d|| <= r |d|^2 M_3/2.

Consequently

    |v| ||H_M e2|| <= r M_3 (|u+1/2|+|d|^2)/2.

Hessian Lipschitz control along the remaining segments yields

    max(||H_M||,||H_S||,||H_X||) <= C_B r M_3/|v|,          (6)

where, for example, a sufficiently generous choice is C_B=16(B+1)^2. Only a finite constant depending on B is used. Since |det H|<=||H||^2 in d=2,

    W_r F_j(H_X) <= C_B^6 r^6 |v|^-6 M_3^6.               (7)

Dropping types only increases the bound. Also, without noncollinearity,

    W_r F_j(H_X) <= M_2^6,                                 (8)

if M_2 bounds the Hessian norm everywhere on the local domain. Estimate (8) is the valid fallback on the axis. No division by v is made there. The lemma constrains all three Hessians; an endpoint-only smallness estimate would not suffice.

## 5. The stable two-scale Gaussian block near the axis

Fix 1<a0<B and use a0<=|u|<=B. At X=(ru,rv), set

    delta=sqrt(r^2+v^2), alpha=r/delta, beta=v/delta.

Define UNCONDITIONAL linear Gaussian observations

    Y1=[f_x(X)-H'_r(ru)-r v L'_r(0)]/(r^2 delta),
    Y2=[f_z(X)-L_r(ru)]/(r delta).                          (9)

The L'_r subtraction must be made before covariance calculations even though it vanishes at the prescribed pins. Under (1),

    grad f(X)=diag(r^2 delta,r delta)
                [Y+(6k(u^2-1/4)/delta,0)].                (10)

Write Q=f_xxxx(0), T=f_xxz(0), S=f_zz(0), A(u)=u(u^2-1/4)/6 and B(u)=(u^2-1/4)/2. The reuse of B(u) here is a polynomial, not the fixed outer annulus radius. To avoid ambiguity in calculations denote these polynomials A_u,B_u.

Exact quartic Hermite error and affine interpolation give

    g'(ru)-H'_r(ru)=r^3 A_u Q+O_Lp(r^4),
    h'(ru)-L'_r(0)=r u T+O_Lp(r^2),
    h(ru)-L_r(ru)=r^2 B_u T+O_Lp(r^3).

Expand in the physical transverse coordinate rv. The Y1 error is bounded by C(r^2+r|v|+v^2)/delta, and the Y2 error by C(r^2+r|v|+r v^2)/delta. Both are O_Lp(delta), uniformly in the aspect ratio. Thus

    Y = M(u,alpha,beta) (Q,T,S)^T + O_L2(delta),
    M = [A_u alpha, u beta, 0; 0, B_u alpha, beta].        (11)

Its three two-column minors are A_u B_u alpha^2, A_u alpha beta and u beta^2. Because alpha^2+beta^2=1, alpha>=0,

    det(M M^T) >= min(A_u^2 B_u^2,u^2)/2 > 0              (12)

uniformly on the compact longitudinal chart. The trace is bounded. The nine jets (U_0,Q,T,S) have uniformly positive covariance by Section 3. Hence the joint covariance of (U_r,Y), and the Schur complement Sigma=Cov_Qr(Y), have positive/finite eigenvalue bounds for delta<=delta0. The conditional mean m_Y is bounded. This proves uniformity also as alpha->0; using only the axial minor would not.

Let d(u)=6k(u^2-1/4)>=d_*>0. At zero gradient the Y target is tau=(-d(u)/delta,0). For small delta,

    |tau-m_Y|>=d_*/(2delta), |tau-m_Y|<=C/delta.

The exact Gaussian density and det diag(r^2 delta,r delta)=r^3 delta^2 give

    p_(grad f(X)|pins)(0) <= C r^-3 delta^-2 exp(-c/delta^2). (13)

This is a JOINT two-dimensional density, not a marginal small-ball probability.

## 6. The added conditional-field moment interface

The essential extension beyond a finite Hessian-vector estimate is

    E_Qr[M_2^6+M_3^6 | grad f(X)=0] <= C delta^-6.          (14)

Here M_2,M_3 may be bounded by a fixed multiple of ||F||_C3 on the whole torus. Under Q_r these supremum moments are uniformly finite. The conditional Y moments are uniformly bounded by (11) and its stable regression on U_r. For C(x)=Cov_Qr(F(x),Y), Cauchy-Schwarz bounds each derivative through degree three uniformly in x, so ||C||_C3<=C0.

Under Q_r define the Gaussian field residual

    R=F_Q-E_QF_Q-C Sigma^-1 (Y-m_Y).

It is independent of Y and has uniformly bounded C3/L6 norm: apply the triangle inequality to this explicit expression, bounded ||C Sigma^-1||_C3, and the uniform moments of F_Q and Y. Under the additional condition Y=tau the field has law

    E_QF_Q+C Sigma^-1(tau-m_Y)+R.

Its C3 sixth moment is <=C(1+|tau-m_Y|^6)<=C delta^-6. This proves (14) after *all* gradient pins. It does not assume independence between endpoint and witness Hessians. The regression construction retains the original pins and gives the exact additional zero gradient, so (6)-(8) hold under this conditional law.

Combining (7),(8),(14) gives

    E_Qr[W_r F_j(H_X) | grad f(X)=0]
      <= C delta^-6 min(1,(r/|v|)^6),                      (15)

where the minimum is defined to be 1 at v=0. Different constants in the two bounds are absorbed into C. This is the missing factor, not a new power inferred from unconditioned Hessians.

## 7. Weighted intensity, crossover and Gaussian absorption

For each fixed r>0 the annulus excludes M,S. Distinct-site derivative nondegeneracy and smooth Gaussian regression justify weighted Kac-Rice there. In detail, use the gradient as the zero-counted field and take the nonnegative mark W_r times the indicator of a nonsingular index-j witness Hessian. The endpoint filtered determinants are continuous through singular matrices (their determinant factors vanish); the witness type set is open. This mark is lower semicontinuous, and joint Gaussian regression supplies the requisite continuity. Truncation removes polynomial-growth restrictions. Height is not a mark in this application.

The exact intensity per physical area is

    rho_j^W(X)=Z_r^-1 p_(grad f(X)|pins)(0)
                  E_Qr[W_r F_j(H_X) | grad f(X)=0].       (16)

This formula is under the underlying Gaussian Q_r; the tilted Q_r^W is not declared Gaussian. Applying (5),(13),(15) proves

    rho_j^W(ru,rv)
      <= C r^-5 delta^-8 exp(-c/delta^2)
                         min(1,(r/|v|)^6).               (17)

Now split at the actual crossover |v|=r.

* If |v|<=r then r<=delta<=sqrt(2)r, so (17) is <=C r^-13 exp(-c/(2r^2)).
* If r<=|v| and delta<=delta0 then |v|<=delta<=sqrt(2)|v|, and (17) is <=C r |v|^-14 exp(-c/(2v^2)).

For every s,c0>0, the seventh positive term in exp(c0/s^2) gives

    exp(-c0/s^2) <= 7! c0^-7 s^14.                         (18)

Use (18) with s=r in the first case and s=|v| in the second. Both bounds become <=Cr. Thus there is a FIXED near-axis strip, not merely a shrinking-width region, on which

    rho_j^W(ru,rv) <= Cr                                  (19)

uniformly whenever a0<=|u|<=B and delta<=delta0. The constants are qualitative and may be large; no optimal power prefactor or numerical bound is claimed.

## 8. Off-axis chart and complete fixed-annulus cover

Choose a0=(A+1)/2>1 and epsilon>0 so small that sqrt(A^2-epsilon^2)>a0 and epsilon<=delta0/2. Then choose r_* small enough that r_*<=delta0/2. Every annulus point with |v|<=epsilon satisfies a0<|u|<=B and delta<delta0, so Section 7 applies.

On the complementary compact chart |v|>=epsilon, retain the finite-r pin corrections and use only

    J=(f_x(ru,rv)/r^2, f_z(ru,rv)/r).

With a=f_zz(0), q=f_xxz(0), c1=f_xzz(0), the expansion under Q_r is

    J = (6k(u^2-1/4)+q u v+c1 v^2/2, a v)+O_Lp(r).        (20)

The coefficient matrix on genuinely random jets (a,q,c1) is

    [0,u v,v^2/2; v,0,0].

Its minor in the a,c1 columns is -v^3/2, uniformly separated from zero on this chart. Appending these jets to U_0 preserves covariance positivity. Therefore J has uniformly bounded mean, positive covariance and bounded C3 field cross-covariance; conditioning J=0 yields a uniformly bounded C3 sixth moment. The raw gradient Jacobian is r^3, hence its density is <=Cr^-3. Equation (7) gives conditional triple weight <=Cr^6 since |v|>=epsilon. Divide by (5) to obtain rho_j^W<=Cr on the off-axis chart as well.

The two charts cover K_AB without a gap. Physical area is r^2 du dv, so (16),(19),(20) give (2) for every Borel E. Null boundaries can be assigned to either chart; the fixed-r intensity is absolutely continuous. There is exactly one endpoint normalizer and one witness determinant. No height-window Jacobian is multiplied into this all-height count.

## 9. Scope and independent-review requirements

This is a stronger local count candidate than the between-pin-height statement in PR16, but it has the SAME fixed d2/compact-mark/fixed-annulus restrictions. The estimate does not give a matching lower bound, a numerical constant/radius, a fixed-remote all-height bound, an all-scales RN theorem, a persistence partner statement or a 3D result. Its k_->0 dependence is essential to the Gaussian penalty.

The decisive review interfaces are: R1 exact unconditional subtraction and finite-r remainders in (9)-(11); R2 full-rank compactification (12); R3 conditional full-field moments after the large target (14); R4 deterministic three-Hessian bound (6)-(8); R5 ORIGINAL Z (5), not a witness-conditioned replacement; R6 weighted Kac-Rice and the absence of a height factor; R7 both crossover powers, absorption and complete annulus cover. Any unsupported interface prevents acceptance of (2).

## 10. Source relationships and external framework

Read source PR22 at b2e1652f1374c3b45759324a1ad1fd4177458500, `frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md` (SHA256079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d); PR16 at32b80ee085dc6a40113d1e46e333cda50d57ba21, `reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md` (blob024d927779f79fadaf932541ea6474f2995f8c50); and PR19 at e93eade078cac2e8b14aa39d402a3043b7cdb1c7. These are exposed same-provider author sources, not outside confirmation. Their files and review dispositions are unchanged by this package.

External framework inspected: Armentano, Azais and Leon, arXiv:2304.07424v3, Theorems 2.2 and 7.1, Remark 8 and section 8.1, https://arxiv.org/html/2304.07424v3. These supply the Gaussian/weighted expected-count framework and its hypotheses, not estimates (13)-(20). NIST DLMF 3.3, https://dlmf.nist.gov/3.3, is interpolation context only. No comprehensive novelty or priority claim is made.

The finite program verifies exact row/minor identities, raw six-pin interpolation, polynomial triple-gradient fixtures and exponent accounting, including deliberate omission controls. It is not a proof checker for the continuum argument, nor independent organizational review. A separate analyst must review the exact candidate source before it is consumed as a closed prerequisite.
