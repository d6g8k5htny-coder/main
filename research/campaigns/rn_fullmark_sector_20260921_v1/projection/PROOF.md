# Full-mark projection and adjugate energy on a bounded RN sector

This is an author-side successor under the unchanged normalized SIDE24 law,
r=1/20, b=6/5, six pins at(-1/40,0),(1/40,0), and full height window
[57599/48000,6/5]. It proves rectangle inequalities, not a spatial partition,
an integral budget, a full annulus, all radii, all pin orientations, RN/q0
closure, event identification, canonical promotion or independent acceptance.

## Source and domain

The adjacent bernstein.py and successor_wedge_v2.py are byte-exact copies of the
already frozen successor. Their hashes and byte counts are checked before use
and again after each rectangle. The caller must freshly authenticate the N6
adapter and its runtime source dependencies; copied source-binding labels alone
are not freshness evidence. No original repository or archive bytes are edited.

The enlarged allowed rectangle is x in[99/1000,3/25], y in[-33/2800,33/2800].
Every positive-area exact subrectangle has a center with each absolute coordinate
below17 and halfwidths below1. Its center-to-pin displacements have absolute
coordinates below18. These satisfy the original N6 center, normalized image-tail
and L2 remainder domains; no derivative-tail extrapolation is introduced.
The field is stationary, so its proved derivative variance caps and Taylor L2
remainders hold along the entire segment from a center to its target. At order6,
f,fx,fy remainders require derivatives of total order at most8; the original
authored adapter supports every needed per-axis covariance derivative through18.

The unchanged N6 routine provides actual common-law conditional Taylor mean
and covariance polynomials, plus unconditional L2 remainder caps eps_i.
Conditioning contracts centered L2 norm. Conditional mean error is at most
sqrt(E6)*eps_i, with the original checked six-pin energy E6. The covariance
polynomial and actual covariance always refer to the same six-pin law.

The inherited exact rational matrix T only selects coordinates. Its measured
covariance P is retained; P is not set equal to identity. For transformed
remainders eta_i=sum_j |T_ij|eps_j, covariance entry error is
c_ij=sigma_i eta_j+sigma_j eta_i+eta_i eta_j, where sigma_i bounds the polynomial
standard deviation. The inherited determinant perturbation retains all
linear/quadratic/cubic products. A strictly positive actual covariance determinant,
together with source-law PSD, supplies actual positive definiteness G>0.
Independent-entry interval-hull positive definiteness is not required or claimed.

## A. Fixed rational three-jet projection

For any nonzero exact rational vector a and target w(v)=(v,0,0), covariance
Cauchy-Schwarz gives

q(v,y)=(w-mu)^T Sigma^-1(w-mu) >= [a^T(w-mu)]^2/(a^T Sigma a).

The default direction is chosen from a rational midpoint approximation to
Sigma(center)^-1(w(v_mid)-mu(center)), normalized exactly by its largest absolute
coordinate. This is a selection heuristic only: actual polynomial variance is
evaluated afterward. An explicitly supplied nonzero rational direction is equally
valid; no special optimality or approximate inverse identity enters the proof.

Form P_delta=a_f*v-a^T P_mu and P_var=a^T P_cov a, retaining all covariance
cross terms. The full mark interval occurs in the constant coefficient of P_delta.
Interval coefficient multiplication contains every fixed mark's actual polynomial;
it may overestimate because it forgets correlations but cannot exclude a mark.
The projected L2 error is eps_a=sum_i |a_i|eps_i. Mean error is
e_mu=sqrt(E6)*eps_a and variance error is e_var=2 sigma_a eps_a+eps_a^2.

For exact Q>=0, form the common spatial polynomial R=P_delta^2-Q P_var.
If its certified lower bound is L and M bounds |P_delta|, then

[a^T(w-mu)]^2-Q(a^T Sigma a) >= L-2M e_mu-Q e_var.

The mean-error square is nonnegative and is the only discarded quadratic term.
The linear mean-error cross term and full variance error are retained. A
nonnegative margin establishes q>=Q at every point and every height. Failure
retains only the independently valid separate-range quotient; it never grants Q.

## B. Full adjugate energy when a constant direction is too weak

Work entirely in the transformed three-jet coordinates. Let P be the actual
Taylor covariance, G=P+E the actual covariance, and |E_ij|<=c_ij. Let
d=T(w-P_mu), delta=T(w-mu)=d+e, with |e_i|<=a_i=sqrt(E6)*eta_i.
The exact preconditioner changes neither q nor its energy threshold.

Write A=adj(P), H=adj(G)-A. Every A entry is its signed two-by-two minor,
formed as a common polynomial before range evaluation. Specifically adj(P)_ij
removes row j and column i. If M_ij bounds |P_ij|, each product perturbation
has bound M_ab c_cd+c_ab M_cd+c_ab c_cd. Summing the two minor-product
bounds gives C_ij>=|H_ij|, including the quadratic error products.

Let L_i bound |d_i|, F_i bound |(A d)_i| as a common polynomial, and e_D bound
|det(G)-det(P)| from the inherited determinant computation. Then

delta^T adj(G) delta-Q det(G)
 >= d^T A d-Q det(P)
    -[sum_ij C_ij L_i L_j
      +2 sum_i a_i F_i
      +2 sum_ij a_i C_ij L_j
      +Q e_D].

To verify this, expand the difference into d^T H d+2e^T A d+2e^T H d
+e^T adj(G)e-Q(det(G)-det(P)). Actual G>0 implies adj(G)>0, so
e^T adj(G)e>=0 and may be discarded from a lower bound. All remaining terms
are bounded as displayed. This does not assume P or its interval hull is SPD.
If the right side is nonnegative, division by det(G)>0 proves q>=Q.
All determinants and all error terms in this energy calculation use the same
transformed coordinates. Only the separate density determinant later divides
by det(T)^2 to restore original coordinates.

## C. Keeping height dependence through the final enclosure

Write v=v_mid+s, |s|<=h=1/96000. Then d=d0+s*t with t=T[:,0]. Height enters
every transformed coordinate, not merely coordinate0. Form spatial polynomials

R0=d0^T A d0-Q det(P), R1=2 t^T A d0, R2=t^T A t.

Their spatial degrees are at most36,30,24 respectively. Elevate their exact
interval Bernstein control nets to common spatial degrees. At each spatial
control position with entries(b0,b1,b2), the three degree-two height controls are

b0-h*b1+h^2*b2, b0-h^2*b2, b0+h*b1+h^2*b2.

The spatial and height Bernstein basis functions are nonnegative and sum to1;
the minimum lower endpoint of these controls therefore bounds R0+sR1+s^2R2
throughout the entire spatial×height domain. Every operation rounds outward.
This preserves the single shared height variable through the cancellation.

The quantities d and A d are affine in s. At each spatial point their absolute
values attain their maximum over s at an endpoint; exact spatial enclosure at
both height endpoints supplies L_i and F_i. No mark sampling is used.

## Evidence and unresolved coverage

The first eight in-memory region probes are retained as diagnostic history in
PROBES.json, explicitly noncertifying. The enlarged whole box failed determinant
positivity. Four positive-y quarterrectangles had positive determinant but their
constant center height-residual projections failed Q=20,40,60,80,103. This is why
the full adjugate argument is included; a failed projection is not a field
counterexample and does not justify a spatial conclusion.

The parent's separate finite-cover runner supplies exact geometry, fresh runtime
source checks before/after, retained failures, full successful cell budgets,
normal/optimized replays, source-bound density majorant and one-time area
accounting. This module alone grants none of these. Synthetic controls test
adjugate orientation, quadratic covariance error, mean-linear error, the full
height endpoints, common Bernstein degrees, and original domain/source refusal.
Same-provider review earns zero organizational-independence credit.
