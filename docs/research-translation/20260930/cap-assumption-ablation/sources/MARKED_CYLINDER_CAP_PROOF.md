# A weaker separating-cylinder criterion, dimension-independent geometry, and an improved SIDE24 cubic bound

Object: **MARKED-CYLINDER-CAP-20260924-v1**. Author: OpenAI / ChatGPT, 24 September 2026.
Disposition: author-side proof candidate. Ordinary mathematical argument below; exact finite controls accompany it. No independent acceptance or formal verification is claimed.
Tasks: CAP-01 and CAP-02, GitHub campaign #61. This is an additive successor to #57/#58, not a change to their historical statements.

## 1. Results and exact division of scope

There are two results, not one unqualified higher-dimensional Gaussian theorem.

**Deterministic theorem.** Let d>=2 and use coordinates (x,y) in R x R^(d-1) on an embedded product cylinder

    D=[-2r,2r] x closed_ball(0,2r),  r>0.

Suppose f is C4 on a neighborhood of D, has critical pins M=(-r/2,0), S=(r/2,0), with f(M)=b and f(S)=b-kappa*r^3, kappa>0. For j=3,4 define the partial-block norm

    M_j=max_{a+c=j} sup_D || partial_x^a D_y^c f ||_op.

For c=0 this is absolute value. For c>=1 the operator norm is the supremum on c transverse Euclidean unit vectors, independently chosen. In dimension two this is exactly the maximum of the coordinate partial derivatives used by #57, not the larger full-tensor operator norm. Put

    lambda=lambda_min(-D_y^2 f(M)).

The explicit sufficient conditions are

    lambda > (4/(3*kappa))*r*M_3^2,
    r*M_4 <= 3*kappa/10.                                  (1)

They give a unique transverse-critical ridge on D and a closed cap through S separating M from every older point. On a compact manifold, for a global Morse extension with distinct critical values, the ordinary superlevel elder death partner of M is S. Exactly one ascending unstable branch of S ends at M, for a smooth Riemannian gradient. No Morse-Smale assumption and no identification of the ridge with a flow trajectory are used.

**Probabilistic corollary, narrower scope.** Retain EXACTLY the fixed-x-axis 2D normalized SIDE24 six-pin law of #57/#58: b=6/5, kappa=1/6, 0<r<=1/20, Q_r the continuous Gaussian regression law and

    W=|det H_M det H_S| 1{H_M<0, det H_S<0},
    Z=E_Q W,                 dQ^W=(W/Z)dQ.

There is no adjacency conditioning and no additional pin-density/Jacobian factor in Z. Consuming the explicitly stated analytic imports in Section 6, the new theorem gives

    1-p_elder(r) <= min(1, C3_new*r^3+C4_new*r^4)
                <= min(1, 2.4*10^23*r^3),                 (2)

where

    C3_new=(10/51)[(32^3/3)*320^8+(3*32^2/2)*320^7],
    C4_new=(23/5)[10240^4+6800^4].                          (3)

The predecessor's exact absorbed constant exceeds this one's by more than 500. This is an improvement of a conservative proof constant, not a claim of sharpness or improved actual field probability. At r=1/20 the bound is still vacuous. At r=10^-8 it is strictly below 0.24. Both facts are stated rather than hiding the remaining scale problem.

The deterministic theorem covers arbitrary birth, positive gap mark, coordinate frame and dimension with its stated norms. The probability estimate does NOT thereby acquire those extensions. No 3D eight-pin matrix-boundary estimate, all-angle covariance bound, all-mark normalizer or lifetime integration is proved here.

## 2. Normalize the mark and exploit the forced third derivative

First prove the theorem with kappa=1/6. Write m=M_3, n=M_4 and assume

    lambda>8*r*m^2,            r*n<=1/20.                  (4)

The exact critical pins give the Hermite identity

    f(S)-f(M)=-(1/2) integral_{-r/2}^{r/2}
                  (x+r/2)(r/2-x) f_xxx(x,0) dx.

The positive kernel has integral r^3/6, so its weighted average of f_xxx is 2. In particular m>=2 and some x0 in the pin interval satisfies f_xxx(x0,0)=2. This forced lower bound on m is useful; replacing m by the larger 1+m is unnecessary.

The partial-block norms control increments by |Delta x|+||Delta y||, with no factor depending on dimension. Every point of D is within this product distance <5r of (x0,0), and likewise <5r from M. Thus

    f_xxx >= 2-5rn >= 7/4,
    D_y^2 f <= -(lambda-5rm) I = -delta I,                 (5)
    delta>rm(8m-5),                 8m-5>=11.

The Hessian inequality is in the quadratic-form order throughout D. A scalar lower bound on a diagonal entry cannot substitute for this matrix inequality when d>2.

## 3. A vector ridge exists globally across the cylinder

Let w(x)=grad_y f(x,0). It vanishes at the two pins and ||w''||<=m. Applying the scalar two-node interpolation remainder to every unit-vector projection of w gives, including outside the pin interval,

    ||w(x)|| <= (m/2)|x^2-r^2/4| <= (15/8)mr^2 <2mr^2.    (6)

The bound outside the nodes follows from the interpolation remainder on the convex hull of the three points, not from an unjustified inside-only estimate.

Also integral_{-r/2}^{r/2} w'(t)dt=0. Subtract this average from w'(x), use the Lipschitz constant m, and integrate |x-t| to obtain

    ||w'(x)|| <= (m/r) integral_{-r/2}^{r/2}|x-t|dt
               <=2mr,       |x|<=2r.                     (7)

This vector average argument does not posit a common point at which every component of w' vanishes.

For fixed x, f(x,.) is delta-strongly concave on the transverse ball. On its boundary ||y||=2r,

    grad_y f(x,y).y <= ||w(x)|| ||y|| -delta||y||^2 <0.

The maximum on the compact ball is therefore interior (moving inward increases the value at a boundary point). Strict concavity makes it unique. Call it h(x). It solves grad_y f(x,h(x))=0, has h(+-r/2)=0 and is C3 by the implicit function theorem. Strong monotonicity gives

    ||h(x)|| <= ||w(x)||/delta <2r/k,        k=8m-5>=11.    (8)

Its derivative satisfies

    h'=-(D_y^2 f)^(-1) partial_x grad_y f.

Equations (7),(8) and the mixed derivative bound give

    ||h'|| < (2rm+m||h||)/delta
            <2/k+2/k^2=:u,                 u<=24/121.    (9)

Since m=(k+5)/8,

    m*u=(1+6/k+5/k^2)/4 <=48/121.                         (10)

All these inequalities use transverse operator norms; there is no hidden dimension-dependent coordinate sum.

## 4. The correct ridge derivative and all cap boundary faces

Set g(x)=f(x,h(x)), F=g'=f_x(x,h(x)). With v=(1,h'), differentiating the transverse critical equation gives (H_f v)_y=0. In the third derivative of g, every term involving h'' is therefore zero. The exact identity is

    F''=f_xxx +3 f_xxy[h'] +3 f_xyy[h',h']
                  + f_yyy[h',h',h'].                     (11)

This is valid for a vector h. It does not assert that the ridge is an integral curve of the gradient.

Using (9),(10), the adverse part is at most

    (48/121)[3+3*(24/121)+(24/121)^2]
       =2554128/1771561.

Consequently

    F'' > 7/4-2554128/1771561
         =2184415/7086244 >1/4.                           (12)

The bound is weaker than #58's F''>1, but it is already sufficient for an older connecting endpoint. This is the second source of the improved depth condition.

F has the prescribed zeros a=-r/2,c=r/2. Strict convexity makes them its only zeros, with F negative between and positive outside. Every critical point in D lies on the ridge, hence D contains precisely M and S. The Schur complement of D_y^2 f at these points is F'(a)<0 and F'(c)>0, proving that M is a maximum and S has exactly one positive Hessian direction.

Define

    C=[-2r,r/2] x closed_ball(0,2r).

Strong concavity gives f(x,y)<=g(x) and g(x)<=b on C. To control the longitudinal faces, subtract (x-a)(x-c)/8 from F. The difference is convex with zero values at a,c, hence nonnegative outside [a,c]. Integrating on either exterior interval gives

    g(-2r) <= b-(9/32)r^3 = s-(11/96)r^3,
    g( 2r) >= s+(9/32)r^3 = b+(11/96)r^3,                (13)

where s=b-r^3/6. These are lower/upper inequalities at the actual ridge endpoints, not Taylor fits.

For a transverse boundary point ||y||=2r, (5),(8) imply

    f(x,y) <=g(x)-(delta/2)||y-h(x)||^2
            <b-(4400/121)r^3 <s.                         (14)

Indeed delta>rm(8m-5)>=22r and ||y-h||>20r/11. This covers the entire curved side of the cylinder, including its intersections with the end faces; in dimension two it covers both horizontal edges and their corners.

On the right cap face x=c, h(c)=0, so

    f(c,y) <= s-(delta/2)||y||^2,

with equality at S only. On the left face use f<=g and (13). Thus every boundary exit from C has value <=s.

Finally, the ridge segment a<=x<=2r starts at M, passes through S and ends at z=(2r,h(2r)), where f(z)>b by (13). Its minimum is exactly s. A path through S at level s is not a preemption path staying strictly above s.

## 5. Global elder pairing, actual branches, and mark scaling

Every point with value >b lies outside C. Every continuous path from M to such a point must first intersect the boundary of C, so its minimum is <=s regardless of later remote excursions or reentry. The preceding ridge path has minimum exactly s and endpoint >b. Hence the maximin connection level to an older point is s.

For a compact Morse function with distinct critical values, ordinary superlevel-component persistence identifies this maximin value with M's elder death level; the older endpoint excludes the essential/global-maximum case. The unique critical value s identifies S. This is the same separating-cap criterion established in #58, now with all its boundary premises re-proved under the weaker event.

At S the Hessian is negative definite on the transverse hyperplane. The unstable eigenvector for gradient ascent has nonzero x component: its Hessian quadratic form is positive, whereas that hyperplane is strictly negative. This argument also holds for a smooth positive-definite Riemannian metric. One half-branch enters the cap at height >s; the other enters outside. The first cannot exit because its height increases and every boundary value is <=s. Its omega-limit is a critical point in the compact cap. Gradient monotonicity and finiteness of the critical set make this limit a single point; it cannot be S, so it is M. The other branch cannot enter C or tend to its interior point M. Exactly one branch ends at M.

For general kappa>0 apply the normalized theorem to f/(6*kappa). Its pins have gap r^3/6 and arbitrary birth b/(6*kappa). Its derivative norms and transverse curvature are divided by 6*kappa. Condition (4) becomes exactly (1). The geometric ridge is unchanged. The reduced third derivative is >3*kappa/2, the longitudinal drop/rise is at least (27/16)*kappa*r^3, and the excess over the specified gap is (11/16)*kappa*r^3. The radial drop exceeds (26400/121)*kappa*r^3.

An arbitrary orthonormal frame can be used in this pathwise argument as long as its product chart is embedded. Scaling or rotating the function here is not a proof that SIDE24's conditional covariance, pin energy or normalizer are invariant under that operation.

## 6. What is imported for the 2D probability calculation

Use the fixed-axis Q,Q^W law from Section 1 and the same random-variable coupling as #57 Sections 5 and 7. The following are retained premises, not newly rerun numerical programs:

* Z>=(17/10)r^2, six-pin regression energy<8, and ||det H_M det H_S||_2/Z<23/5.
* q=f_yy(M) has density <=1/3 and ||q||_8<5. On typed support q<0.
* There is T>=1, JOINTLY independent of q, with ||T||_8<320 and M_3<=T+8|q|; also ||M_4||_8<340.
* In the SAME coupling, at q=-ru, the full typed weight satisfies the pointwise bound

      W<=r^4[T^2*u^2+3*T^3*u],       u>=0.                (15)

  Internal correlations among the residual field, axial/mixed variables and transverse innovation are retained. Independence is only from q. The coefficient 3 includes the saddle's mixed-square contribution.
* The pinned-genericity candidate #49 supplies the full-probability Morse/distinct-value locus for each fixed r. No simultaneous null-set statement over all parameters is imported.

These facts are read from exact source bytes and their explicit chain. Hash verification does not establish their truth; H3 and LPW interval programs were not rerun by this delivery. The stronger geometry does not depend on #56's endpoint spectral candidate.

## 7. The smaller exceptional layer and its retained far branch

Set G_new={-q>8r M_3^2, rM_4<=1/20}. Section 2-5 give global pairing on G_new. Unlike an inference from a local-event probability, this uses a direct bound on G_new's complement.

Put lambda=-q>0 on typed support. Failure of the depth condition implies

    lambda<=8r(T+8lambda)^2
          <=16rT^2+1024r lambda^2.

Thus the failure lies in

    {0<lambda<=32rT^2} union {lambda>1/(2048r)}.            (16)

The large-lambda branch is essential to the quadratic inequality and has NOT been discarded.

For the near event, disintegrate on q independent of T. Use its density cap 1/3, dq=r du, and (15):

    E[W 1_near] <= (r^5/3) E integral_0^(32T^2)
                                   [T^2 u^2+3T^3 u]du
      = (r^5/3)[(32^3/3)E T^8+(3*32^2/2)E T^7].         (17)

Division by the full Z and the moment bounds gives C3_new*r^3. Weighted Cauchy-Schwarz, INCLUDING its square root, and eighth-moment Markov give

    Q^W{lambda>1/(2048r)} < (23/5)(10240r)^4,
    Q^W{M_4>1/(20r)}      < (23/5)(6800r)^4.              (18)

Together (17),(18) prove (2),(3) at the imported premises. The same single complement controls nonadjacency and the event that both unstable branches end at M; no independence or equality of these events off G_new is asserted.

The earlier LPW lower premise, when separately consumed, is unaffected: c_LPW=13/468574870385996070912000 and r<=1/356352 still compose with this upper bound for two-sided cubic order in the one fixed-axis law. This delivery does not re-establish that lower premise or give a limiting coefficient.

## 8. Exact comparison and remaining work

The predecessor constants from #57/#58 are

    C3_old=(10/51)[(256^3/3)320^8+(3*256^2/2)320^7],
    C4_old=(23/5)[81920^4+6800^4].

Exact arithmetic checks C3_old>500*C3_new, C4_old>500*C4_new and

    C3_new+C4_new/20 <2.4*10^23.

Their absorbed-constant ratio is about 511.803 (display only; the comparison >500 is rational). The tighter good event is strictly weaker than the old sufficient event; neither claims to characterize all successful pairs.

For the three-dimensional probability program the deterministic matrix/ridge obstacle now has a concrete candidate solution, but the probability of a small transverse eigenvalue and intersections of matrix boundary strata remains to be proved under the correct eight-pin law. Full-annulus expected counts, 24-jet modulus certificates, all-angle/all-mark estimates and lifetime integration remain distinct targets. Small failure probability alone does not bound a count on the failure event.

Source identities and external reconnaissance are in SOURCE_OBSERVATIONS.json and RECONNAISSANCE.md. Tests check exact algebra, finite examples and deliberate mutations, not arbitrary continuous fields or the analytic imported chain. Review should focus on the vector average in (7), all partial-block norms, the h'' cancellation, entire radial side, marked scaling, and the independence/density/normalizer powers in (17).
