# Higher homology: attaching obstructions and isolated-fold unit incidence

Incorporated on 8 October 2026 by OpenAI/Codex root, task
01a11cbf-068d-7102-861e-75814e715c98, under GitHub307 claim6070584604
and amendments6071035594/6071181708. Mathematical author:
OpenAI/Codex universality_review. This source preserves Sections1-12
of the frozen 44,258-byte/884-line preparation at SHA256
`de1cad971e113f097a8e29ba82fb295d93b23f57aeda5b265e6f3c88cef47f2e`.
That actual preparation has two separate fresh FULL nonauthor mathematical
PASS verdicts. This changed introduction, source-binding section and README
are a new construction, pending their own incorporated-source review.
The initial worktree base is native spectral draft330 head
`0c8ad2e7acbbf288b6ef2eeb4f96b6824ba8abaa`; no main landing is implied.
All performers are source/route-exposed OpenAI/Codex AI agents;
organizational independence0. No compiler, numerical, blind or scientific
promotion is supplied by these conventional analytic results.

Adjacent handle indices and a small height gap alone permit global
attaching alternatives. An isolated regular cubic fold, with the
same-field continuation and complete critical-list control stated below,
supplies a unit incidence. That forces an actual short ordinary interval
in the appropriate degree over every fixed coefficient field. Mixed
transverse inertia changes the degree under these hypotheses.

## 1. Conventions and the exact deterministic claim

Let X be a closed oriented smooth d-manifold, d>=2, with a fixed
background Riemannian metric. The probabilistic specialization below
is the flat torus R^d/(L Z^d). Fix any coefficient field K. All
homology and ordinary persistence use K. Essential intervals are
excluded from the finite-lifetime population.

For a field F, its decreasing superlevel filtration {F>=h} is the
increasing sublevel filtration of g=-F at time tau=-h. An ordinary
finite Hp interval born at field height b and dying at s<b has lifetime
ell=b-s>0. The birth and death critical points are ordered in that
filtration, regardless of whether either is a local extremum of F.

Consider actual C4 fields F_r, r down to zero, converging to F_0
globally in C4. In fixed physical coordinates (s,t) near x, where
t is in R^(d-1), require:

* x is the only degenerate critical point of F_0;
* F_0(x)=b, grad F_0(x)=0, F_0,ss(x)=0,
  F_0,st(x)=0 and F_0,sss(x)=12k, with k>0;
* A_0=F_0,tt(x) is nonsingular, with j negative eigenvalues;
* every other critical point of F_0 is Morse, and their critical
  values are distinct and different from b;
* F_r has the literal two pinned critical points
  M_r=(-r/2,0), S_r=(r/2,0), with

      F_r(M_r)=b, F_r(S_r)=b-k r^3.                     (1)

These are assumptions on a same physical field continuation, not
on unrelated arbitrary jets. In the Gaussian application they are
proved on each fixed contact-law fiber almost surely by SG below.
They imply the other critical set is finite: the regular contact
is isolated, and every other root is isolated on a compact complement.

**Deterministic unit-fold theorem.** Under these assumptions, for
all sufficiently small r, F_r is globally Morse with distinct
critical values and its actual ordinary finite persistence contains
the interval

      Hp: birth M_r, death S_r, lifetime k r^3,
      p=d-j-1.                                         (2)

Both endpoints have their unique birth/death roles in this interval.
No other critical value lies between them. No H0 elder enclosure,
negative-definite transverse block, preselected global attaching map,
or simultaneous event for all contact targets is assumed.

The threshold depends on this realization and these fixed marks.
There is no uniform physical chart radius or deterministic convergence
rate. This assertion concerns the actual filtration of F_r; an
adapted gradientlike field is only a tool to calculate its handle
incidence. It does not replace F_r by a different persistence law.

## 2. Transverse inertia and the persistence degree

At the two pins the axial Schur complement tends, after division
by r, to -6k at M_r and +6k at S_r. The transverse blocks tend to
A_0. Thus, for small r,

      index_F(M_r)=j+1, index_F(S_r)=j.                 (3)

Here index is the number of negative Hessian eigenvalues. Since
Hess g=-Hess F, their sublevel handle degrees for g are

      degree(M_r)=d-j-1=p,
      degree(S_r)=d-j=p+1.                              (4)

The earlier cell is M_r, at tau_a=-b; the later cell is S_r, at
tau_b=-b+k r^3. In particular a negative-definite transverse A_0
(j=d-1) gives p=0, but an index-j block with j<d-1 gives a higher
homology interval. The saddle type must be the full inertia in (3).
A determinant-sign predicate from d=2 is not a valid general-d
replacement.

One can obtain the signs in (3) directly from the reduced ridge
constructed in Section 4; that proof also establishes the exact
number of local roots. No conclusion about an actual interval follows
from (3)-(4) alone. The unit attaching incidence is the decisive step.

## 3. Finite-chain lemma, and what zero incidence really permits

First consider a finite filtered chain complex over K. Its old
subcomplex C^- is followed by one generator a of degree p and then
one generator b' of degree p+1, with no intervening generator. Suppose

      partial b'=u a+z, u!=0, z in C^-_p.              (5)

The chain identity gives partial a=-u^(-1) partial z.
Hence a+u^(-1)z is a cycle. It cannot have been a boundary before
the later cell enters: any earlier boundary belongs to C^-, while
this cycle has coefficient one on a. It gives a new p-dimensional
homology direction. The next boundary kills that direction.

Equivalently, reduce the boundary columns in filtration order.
All earlier columns have zero a-row. The b'-column retains its
nonzero a coefficient after reduction against them; a is its latest
nonzero row. Thus its persistence pivot is a. This creates the exact
finite interval [tau_a,tau_b) in degree p. The earlier cell cannot
instead kill an old H_(p-1) class under (5). The conclusion is
independent of old attaching maps, old cycles and characteristic:
a signed incidence +/-1 is a unit in every field.

For later use, there is a version requiring no global choice of
a cellular boundary matrix. Suppose X^- is the sublevel set before
the first handle, X^a after it and X^+ after the second, and

      H_*(X^a,X^-)=K in degree p,
      H_*(X^+,X^a)=K in degree p+1.                    (6)

If the triple connecting homomorphism

      delta:H_(p+1)(X^+,X^a)->H_p(X^a,X^-)

is a unit, the triple exact sequence gives H_*(X^+,X^-)=0.
The inclusion X^- -> X^+ is consequently a homology isomorphism.
The pair sequence for X^a shows that H_p(X^-) injects into
H_p(X^a) with quotient K: the next connecting map is zero because
its composition with the surjective delta is zero. The map to
H_p(X^+) kills one direction complementary to the old image and
is an isomorphism on that old image. Therefore the newly born
interval dies at the next handle. This also proves (2) from a
relative unit incidence, without rearranging critical values by
handle dimension or assuming a global Morse-Smale complex.

There are real attaching alternatives when (5) is absent:

* For p>=1, take an old p-cycle c, a new p-cycle a at time t_1,
  a (p+1)-cell b' at t_2 just above t_1 with partial b'=c,
  and a later (p+1)-cell e with partial e=a. The two Hp bars
  are [t_0,t_2) and [t_1,t_3), not [t_1,t_2). This is realized
  by a filtered CW complex: begin with a point and S^p, wedge
  another S^p, and attach the two filling disks in the stated order.
* For p=0, begin with old vertices c_0,c_1, add the new vertex a,
  then an edge joining c_0,c_1, then a later edge joining a,c_0.
  The middle edge's boundary has no a coefficient. The old
  c_1 component dies first; the new a component has a long bar.
  The oldest c_0 component is essential. This is an actual filtered
  graph and respects the ordinary H0 augmentation constraint.
* An earlier p-cell can instead kill an old H_(p-1) class and a
  later (p+1)-cell create H_(p+1): set partial a=c_old and
  partial b'=0. For p>=2 this is realized by filling an old
  S^(p-1) and then wedging S^(p+1). For p=1 use an edge joining
  two old components, then wedge S^2. Again the a coefficient
  in partial b' is zero.

These are rigorous filtered-chain/CW diagnostics. They are NOT
realizations of the isolated generic fold in Section 1. Calling
their arbitrary nearby cell times a geometric birth-death fold
would erase exactly the unit-incidence obligation.

Even a nonzero a coefficient is insufficient for *that exact pair*
if a younger p-cell c enters between a and b': with
partial b'=a+c, the reduction pivot is c. An isolated limiting
contact whose other critical heights avoid b excludes this extra
cell. Both unit incidence and no intervening critical event matter.

## 4. Exact transverse splitting at finite regularity

We prove the geometric input without assuming a smooth unfolding
parameter lambda=r^2. The C4 continuation alone suffices.

Shrink one fixed physical cylinder about x. Invertibility of A_0
and the implicit function theorem give unique transverse critical
ridges t=h_r(s), with h_r converging to h_0 in C3. At the pins
h_r(+-r/2)=0. Put phi_r(s)=F_r(s,h_r(s)). Since

      phi_r'(s)=F_r,s(s,h_r(s)),

phi_r is C4, although the ridge need only be C3. At contact,
h_0'(0)=-A_0^(-1)F_0,ts(0)=0. Differentiating the displayed
identity gives phi_0'(0)=phi_0''(0)=0 and phi_0'''(0)=12k.
The term containing h_0'' is multiplied by F_0,st=0.
On a smaller fixed cylinder phi_r''' stays positive for small r.

The function phi_r' is therefore strictly convex. Its pinned roots
are s=-r/2 and s=r/2; they are its only roots and phi_r'<0
between them. Its derivatives at the left/right root are respectively
negative/positive. Every local full critical point lies on the ridge,
so these are exactly the two local critical points. For F_0 the
expansion phi_0'(s)=6k s^2+o(s^2) proves contact is isolated.

Now write, exactly rather than as a Taylor truncation,

      F_r(s,h_r(s)+t)=phi_r(s)+t^T B_r(s,t)t,
      B_r(s,t)=integral_0^1 (1-v) F_r,tt(s,h_r(s)+v t)dv.  (7)

The symmetric B_r is C2 and converges in C2 on the fixed cylinder.
At (0,0) its limit is A_0/2. Diagonalize A_0 once. On a sufficiently
small matrix neighborhood, signed LDL factorization followed by
positive square roots of the absolute pivots gives a smooth matrix
map S(B), with

      B=S(B)^T J S(B),
      J=diag(-I_j,+I_(d-1-j)).                          (8)

The pivots are bounded away from zero on this neighborhood. Set
w=S(B_r(s,t))t. Its derivative in t at zero is invertible. The
inverse function theorem, after shrinking once more, supplies
uniformly invertible C2 coordinates (s,w), converging in C2 with r,
in which

      F_r=phi_r(s)-|w_-|^2+|w_+|^2.                   (9)

This is an exact transverse splitting. No concavity is asserted
when w_+ is nonempty. Uniform invertibility here is for this fixed
realization's nonzero A_0; it is not an inverse-curvature moment
or a common radius over the Gaussian law.

Outside this cylinder, use implicit continuations of the finitely
many old Morse critical points. On their compact complement and
outside a smaller contact neighborhood, grad F_0 has a positive
floor. Global C4 convergence therefore excludes any additional
root. The old critical values converge to their distinct heights,
all separated from b. Together with (1) this proves F_r is globally
Morse/distinct eventually and that no critical value lies between
F_r(S_r) and F_r(M_r).

## 5. A unique local connection and a uniform no-escape argument

In the splitting coordinates define the ascent vector field for F_r

      dot s=phi_r'(s), dot w_-=-2w_-, dot w_+=2w_+.      (10)

It is descending for g_r=-F_r. Along the axial segment between
the two roots, s decreases from S_r to M_r. The scalar autonomous
equation has exactly one such trajectory up to time translation:
separation of variables determines it, and the simple endpoint
zeros put the limits at infinite times.

Any trajectory confined to this chart and converging to S_r as
time -> -infinity and M_r as time -> +infinity must have w=0.
A nonzero w_- explodes at the first limit; a nonzero w_+ explodes
at the second. Thus the connecting trajectory is unique locally.
Along it the unstable tangent of S_r is spanned by the axial
direction and the w_+ directions; the stable tangent of M_r is
spanned by the axial direction and the w_- directions. They sum
to the whole tangent space and intersect in the orbit direction.
The connection is transverse. On a regular level between the
critical heights, the corresponding attaching/belt spheres meet
transversely in one point.

Pull (10) back to physical coordinates. The uniformly invertible
coordinate changes give constants c,C>0, independent of small r,
such that the resulting X_r satisfies

      dF_r(X_r)>=c |grad F_r|^2,
      |X_r|<=C |grad F_r|.                              (11)

Choose fixed concentric physical balls B_R and B_(2R) inside a
common splitting neighborhood, with both pins in B_(R/2). Make
X_r equal to this product field on B_(2R), and blend with a
background ascent gradient outside, using a nonnegative partition.
The estimates (11) survive the blend; the resulting field is C1,
is zero precisely at the critical points and is strictly ascending
elsewhere. On the compact annulus B_(2R) minus B_R, F_0 has no
critical point. For all small r there is a common gamma>0 with
|grad F_r|>=gamma there.

For any portion of a trajectory traversing that annulus, parameterize
by background spatial arclength l. Equations (11) yield

      dF_r/dl=dF_r(X_r)/|X_r| >= (c/C)gamma.

Traversing from radius R to 2R costs at least

      Delta_*=(c/C)gamma R>0                            (12)

in F height. A trajectory connecting S_r to M_r has total height
increase k r^3. Once k r^3<Delta_*, it cannot exit B_(2R).
It is therefore entirely in the product region and is the unique
transverse trajectory in (10). This excludes additional global
connecting paths that could otherwise cancel the local signed
incidence. It is a gradient-trajectory energy argument, not an
assertion that all superlevel paths are trapped in a concave cylinder.

The adapted field can depend on the realization and r. Handle
incidence and persistence are invariants of the original function,
so proving uniqueness for this one admissible field suffices.
No uniqueness claim for every physical metric is needed.

## 6. The classical unit-incidence import and the C4 bridge

The classical smooth handle fact used here is specific: if two
consecutive critical points have sublevel degrees p and p+1,
the coefficient of the first handle in the later relative boundary
is the signed intersection number of the later attaching sphere
with the earlier belt sphere on a separating regular level.
One transverse point gives +/-1. In Milnor's primary account this
is the relative boundary formula of Corollary 7.3. The unique-orbit
geometric cancellation criterion is separately Theorem 5.4; it has
no middle-dimension or simply-connected restriction. Its stronger
algebraic cancellation variants must not be substituted for the
unique-point criterion.

We do not silently apply smooth handle theory to a C4 field. Here
is the finite-regularity bridge. For a fixed sufficiently small r,
F_r is Morse/distinct on the compact X. A sufficiently close smooth
approximation in C4 has exactly the same critical branches and
indices; the segment joining it to F_r stays in a C2 neighborhood
with no new critical point or critical-value crossing. The finitely
many values vary C1 along this segment. A smooth-in-height increasing
reparametrization aligns them to the original values: use disjoint
height bumps, equal to one near the moving values, whose amplitudes
are their small value shifts. Taking the approximation close enough
keeps the derivative positive.

Let g_v be the resulting aligned path of negative fields, v in [0,1].
It is C4 in space, C1 in v; g_0=-F_r, and g_1 is smooth in space.
Its critical values and order are constant. Select one regular
level tau_i in each gap, including levels below/above all critical
values. On small disjoint collars of these finitely many levels the
gradient has a uniform positive floor along the path. Set

      Y_v=-sum_i chi_i(g_v) (partial_v g_v)
                                grad g_v/|grad g_v|^2,  (13)

where each chi_i is one near tau_i and supported in its collar.
The ordinary time-dependent flow is an ambient diffeomorphism
and carries every {g_0<=tau_i} onto {g_v<=tau_i}, simultaneously.
Indeed the derivative of g_v along the flow is zero on the relevant
boundary; that hypersurface is invariant in the extended (v,x)
space, so no trajectory crosses it. The same ambient maps respect
all the inclusions between these finitely many sublevel sets.
Thus their finite persistence modules are isomorphic. They determine
the whole barcode because each between-critical band is carried by
the ordinary critical-free level flow. Aligned values retain the
actual endpoints, not only their ordering.

The final smooth approximation may be made arbitrarily close in
C4 to F_r after alignment. This requires choosing its accuracy
relative to this r's shrinking critical-value gap; no uniform
smoothing tolerance is asserted. Choose the final C4 error also
tending to zero as r tends to zero. Its ridge splitting, positive
third derivative, complete critical list and annulus gradient floor
then satisfy Sections 4-5. The two roots can move slightly from
the literal pins, but remain the same continued pair, and the
aligned gap is still k r^3. The product field is now smooth.
It supplies exactly one transverse attaching/belt intersection.
Apply the smooth relative unit-incidence formula, then the relative
lemma (6), and transfer the interval through (13) to F_r.
This proves the deterministic theorem (2).

For each fixed r the product flow is a conventional pseudogradient:
it strictly decreases g and Xg has a nondegenerate local maximum
at each nondegenerate root. If a chosen smooth handle convention
requires the usual linear field in Morse coordinates at a root,
one may use the one-dimensional Morse coordinate for phi there
and adjust its positive axial speed locally. The transverse coordinate
planes and the unique connection are unchanged; the annulus field
is unchanged. This removes a convention issue without assuming a
global Morse-Smale perturbation or changing a critical value.

The bridge uses ordinary smooth approximation, implicit function
continuation and a C1 time-dependent flow on critical-free collars.
It supplies the required finite filtered homology equivalence;
it does not assert an all-orders smooth family in r or lambda.
Antony's primary unique-trajectory theorem assumes a smooth
one-parameter birth-death unfolding and smooth metric family.
Its main theorem corroborates the genuine smooth-fold picture,
but its mixed unfolding derivative is not an extra premise silently
imposed on this C4 regression continuation.

## 7. Borel selectors and unique finite-bar counting in degree p

On the open C4 Morse/distinct locus, each field has finitely many
critical points. At a nondegenerate critical point, crossing its
g-level attaches one handle, of degree equal to index_g. Its
relative homology is K in that degree. The long exact sequence
shows that the event either births a class in that degree or kills
a class one degree below, with rank one; no other degree changes.
These statements for C4 follow from Section 6's smooth approximation
and regular-level equivalence. In particular every finite Hp interval
has exactly one degree-p birth critical point and one degree-(p+1)
death critical point. Each critical point has at most one endpoint
role. Essential classes have a birth and no finite death.

For clarity, finite persistence does not need an unspecified global
chain basis: sample the sublevel sets at one regular value in each
critical gap. The finite-dimensional homology vector spaces and
their inclusion maps form a finite persistence module over K.
Successive basis elimination gives its interval decomposition.
Because the values are distinct and each handle changes one rank,
an interval's endpoints identify unique critical points. The same
argument is equivalently ordinary filtered boundary reduction when
a compatible handle-cell model is used.

Define sigma_p(F,M,S)=1 if F is Morse/distinct and that ordered
pair is precisely one of its finite Hp intervals; set it to zero
otherwise. It includes the appropriate inertia condition. This is
a Borel mark of the full C4 field and ordered pair, as follows.

In a small C2 neighborhood of one good field, all critical branches
and their order persist by the implicit function theorem and a
compact gradient floor. Apply the aligned-level isotopy (13) along
any sufficiently short segment in this neighborhood. Its sampled
persistence module is constant. Hence the pairing of the labeled
critical branches is locally constant. On a countable chart cover
of the separable C4 good locus, each finite branch graph is continuous
and its selected endpoint relation is fixed. Taking their countable
union gives the Borel graph of all selected pairs. The complement
of the open good locus is Borel and receives zero. This proves the
selector, not merely an unproved assertion that a root count is
measurable. It allows globally different attaching maps in different
Morse chambers; a critical-value crossing may change their pairing.

The unique finite counting assertion is consequently

      N_(p,t)(F)=sum_(M!=S, grad F(M)=grad F(S)=0)
          sigma_p(F,M,S) 1{0<F(M)-F(S)<=t}.             (14)

It is jointly Borel in field and t on good charts, extended by zero
on bad fields. It excludes all essential Hp intervals. On the
connected torus finite intervals exist only in p=0,...,d-1; no
degree-(d+1) handle is available to kill a top-dimensional birth.
We will make no claim for extended, relative, zigzag or torsion-valued
persistence, whose endpoint rules are different.

## 8. Probabilistic interfaces: what transfers and what is newly proved

For the rest let the actual law be SG's fixed centered stationary
Gaussian field on X=R^d/(L Z^d), d>=2. Its symmetric spectral
weights q_n>=0 sum to one. A simple sufficient verifier is full
positivity on |n|infinity<=5 and

      sum_n sqrt(q_n)(1+|2pi n/L|)^4<infinity.          (15)

More generally SG's explicitly proved finite carrier ranks plus
its actual C4 version with E||F||C4^(2d)<infinity suffice. They are
the joint value/gradient rank at every distinct pair, the named
contact rows plus all independent entries of A, and the contact
rows plus one/two distinct off-contact first jets. SG proves the
residual ranks on the conditioned contact kernel, including the
Bayes-weighted tail. No arbitrary-jet interpolation, four-site K7
rank, empirical covariance or covariance-only non-Gaussian premise
is substituted. All constants are for this fixed law, d and L.

For p in {0,...,d-1}, set j=d-p-1 and use the typed determinant
weight

      W_(p,r)=|det H_M det H_S|
           1{index H_M=j+1, index H_S=j},
      V_(p,r)=W_(p,r)/r^2.                              (16)

The selector sigma_p and this weight supply the following five
interfaces. Their names describe mathematical premises, not a new
registry or implementation.

| Interface | Exact source of its proof here |
| --- | --- |
| COUNT_p | Section 7's Borel unique finite interval pairs, combined with SG's all-Borel two-site finite-incidence formula and two-height disintegration. |
| CONTACT_p | SG's literal observation rows, canonical full-field regression and determinant identity, with the inertia cone in (17) below. |
| SELECT_p | The new deterministic unit-fold theorem on the actual contact-good coupling; Sections 9-10 prove the weighted convergence. |
| ENVELOPE_p | W_(p,r) is bounded by the untyped determinant product, so SG's degree-2d norm and Gaussian target envelope apply. |
| FAR_p | SG's original two-site observation covariance floor and conditional determinant moments, with sigma_p<=1. |

COUNT does not inherit an H0 elder selector. SELECT does not inherit
H0 concavity when A is mixed. Their missing higher-degree mechanisms
are proved respectively by finite persistence/regular-level isotopy
and by unit incidence. The other three analytic interfaces retain
the exact same law and rows.

## 9. Contact cone, same-field genericity and weighted selection

SG's literal contact observations, in row order, are

      U_r=((F(M)+F(S))/2,(F(S)-F(M))/r,
          (F_u(S)-F_u(M))/r,
          (6/r^2)[F_u(M)+F_u(S)-2(F(S)-F(M))/r],
          ((F_vi(M)+F_vi(S))/2,(F_vi(S)-F_vi(M))/r)_i),
      v_r=(b-k r^3/2,-k r^2,0,12k,(0,0)_i), k>0.

They have 2d+2 components. Their contact limit pins F=b,
grad F=0, the axial Hessian column V=0 and F_uuu=12k.
The Gaussian contact density pi_r(v_r) converges to pi_0(v_0),
and its covariance has a common floor in all frames at small r.
The original two-site height/gradient density is exactly
12r^(-(d+3)) pi_r(v_r).

On the canonical full-field conditional coupling Q_r, SG proves
F_r -> F_0 globally C4 and a bound
||F_r||C4<=C||F||C4+C(|b|+k). Under Q_0 all other critical points
are Morse, have distinct heights, and avoid b almost surely,
for each fixed contact target/frame. The contact transverse block
A_0 has a full Gaussian density on symmetric matrices. It is
nonsingular almost surely. These are actual conditional statements,
not a disintegration of unconditional genericity onto every null
target or an uncountable simultaneous event.

For the endpoint Hessians the exact block identity is

      H_i=[r alpha_i, r beta_i^T; r beta_i,A_i],
      det H_i/r=alpha_i det A_i-r beta_i^T adj(A_i)beta_i.

Its validity for singular A_i follows because both sides are
polynomials. Here alpha_M -> -6k, alpha_S -> 6k, beta_i is bounded
and A_i -> A_0. The congruence diag(r^(-1/2),I) has limit
diag(-6k,A_0) at M and diag(6k,A_0) at S. Thus if A_0 is nonsingular
its inertia gives exactly (3) on its index-j cone. A different
inertia eventually fails (16); at a singular limit the determinant
product tends to zero irrespective of indicator oscillations.
Consequently

      V_(p,r) -> V_(p,0)
         =36k^2(det A_0)^2 1{index A_0=j},
      V_(p,r)<=C(1+||F_r||C3)^(2d).                    (17)

Each index cone is open and nonempty. Full conditional Gaussian
matrix support makes

      z_(p,0)(b,k,u)=E_Q0 V_(p,0)

strictly positive and finite at every fixed b,k>0,u. Domination
gives E_Qr V_(p,r) -> z_(p,0). On V_(p,0)>0, the contact-good
realization meets Section 1, so its actual pinned pair satisfies
sigma_p(F_r,M_r,S_r)=1 for all sufficiently small r. On
V_(p,0)=0 the product V_(p,r)(1-sigma_p) tends to zero without
requiring any finite-r genericity on that weight-zero support.
The same norm moment yields

      E_Qr[V_(p,r)(1-sigma_p)]->0,
      E_Qr[V_(p,r) sigma_p]->z_(p,0).                  (18)

No normalized weighted law is needed outside a compact small-r
band where its normalizer is positive. The unnormalized formulation
(18) has no global inverse-normalizer or inverse-curvature premise.

SG's target coercivity and Gaussian regression bounds give one
all-mark envelope, also for (16):

      B_(p,r)=12 pi_r(v_r) E_Qr[V_(p,r) sigma_p]
        <=H(b,k)=C(1+|b|+k)^(2d) exp[-c(b^2+k^2)],
      integral_Rx(0,infinity) k^(-2/3)H db dk<infinity.  (19)

The constants can be enlarged over the finite set of p. They
are not common over arbitrary spectral laws or dimensions.

## 10. Actual Hp lifetime density and its physical coefficient

Apply SG's all-Borel area identity with mark sigma_p and the
inertia predicate in (16). A single absolute determinant product
enters. Two-height disintegration gives the finite Borel density
representative

      nu_p(ell)=L^(-d) integral_(M!=S) integral_R
        p_O(b,0,b-ell,0)
        E_Q[|det H_M det H_S| sigma_p] db dM dS.         (20)

The equality is an expected finite-interval counting identity,
not a claim that all critical pairs are bars. The Gaussian kernel
is the actual full-field regression version at the stated target;
the selector is zero on its bad-field subset. There is no inherited
continuity assertion for this discontinuous global mark.

A common spatial translation carries all sublevel sets and their
inclusions homeomorphically, with the same critical labels and heights.
Thus sigma_p is translation invariant. Stationarity then removes the
midpoint x from the conditional kernel; its integral is exactly L^d.
This justifies cancellation of the per-volume factor in the near
formulas below. Orthogonal transverse frame changes preserve the
Hessian inertia, determinant square and observation-density product;
local Borel frame charts suffice.

For distances at least r_0, SG's original two-site covariance
floor and conditional determinant moments give

      0<=nu_p,far(ell)<=C, 0<ell<=1.                    (21)

It also gives a finite bound on any compact positive-lifetime band.
In the unique short-distance band, use midpoint x and displacement
r u, with u oriented from birth point M to death point S. The
exact flat Jacobian factors are

      observations: 12 r^(-(d+3)),
      spatial polar: r^(d-1) dr dx d sigma(u),
      two heights: r^3 db dk,
      determinants: r^2 E_Qr[V_(p,r) sigma_p].

Their product is r B_(p,r) dr db dk dx d sigma(u).
This radial order, rather than the cubic gap alone, fixes the
exponent. Set ell=k r^3 in the specific density (20):

      ell^(1/3) nu_p,near(ell)
        =integral 1{k>=ell/r_0^3}
          B_(p,(ell/k)^(1/3))(b,k,u)/(3k^(2/3))
                                             db dk d sigma(u).

Equations (18)-(19) give the full all-mark dominated limit.
Equation (21) removes the far term, so

      nu_p(ell) ~ c_(q,p) ell^(-1/3),
      E N_(p,t)/L^d ~ (3/2)c_(q,p) t^(2/3),
      c_(q,p)=4 integral k^(-2/3) pi_0(v_0)
                             z_(p,0) db dk d sigma(u).  (22)

The coefficient is finite and strictly positive for every
p=0,...,d-1 in this admitted Gaussian spectral class. The density
statement is pointwise for the explicitly specified finite Borel
representative (20). Arbitrary null-set changes to a Radon-Nikodym
version need not retain a pointwise asymptotic. Density continuity
is not asserted. The cumulative statement follows by integration,
not by differentiating a cumulative asymptotic. The missed near
candidate population is only o(ell^(-1/3)); no O(1) near remainder
is proved.

Here is the exact law-specific Gaussian coefficient. Let
G=grad F, V_u=(F_uu,F_uv_1,...,F_uv_(d-1)),
A_u=Hess F restricted to u-perp, and
tau_u^2=Var(F_uuu|G=0)>0. Define

      D_(u,j)=E[(det A_u)^2 1{index A_u=j}|V_u=0].       (23)

For the fixed spectral law, all these are its actual covariances
and conditional Gaussian matrix law. Stationary parity makes
(G,F_uuu) independent of (F,V_u,A_u). Integrating the birth b
in (22) removes conditioning on F=b but retains V_u=0 and the
correlations of V_u with A_u. It gives

      pi_0(v_0) z_(p,0), integrated in b
         =36k^2 p_G(0) p_Vu(0) phi_tau_u(12k) D_(u,j).

The remaining k factor in (22) is

      144 integral_0^infinity k^(4/3) phi_tau(12k)dk
        =12^(-1/3) integral_0^infinity z^(4/3) phi_tau(z)dz
        =Gamma(7/6) tau^(4/3)/(24^(1/3)sqrt(pi)).

Thus the physical coefficient is

      c_(q,p)=Gamma(7/6)/(24^(1/3)sqrt(pi))
          *integral_S^(d-1) p_G(0) p_Vu(0)
                    tau_u^(4/3) D_(u,d-p-1) d sigma(u).  (24)

There is no factor one half for ordered birth-to-death directions.
The numeral 24 is a change-of-variables constant, not the torus
side L. The coefficient depends on q,d,L, and its transverse
cone; covariance-independent amplitudes or infinite-cutoff uniformity
are not inferred. Its leading value is independent of the chosen
coefficient field K because the local incidence is +/-1. Global
bar pairings and lower-order populations can depend on K.

Conditioned only on V_u=0, A_u is centered. Its symmetry under
A_u -> -A_u gives D_(u,j)=D_(u,d-1-j); hence

      c_(q,p)=c_(q,d-1-p).                              (25)

This is a Gaussian cone identity, not a claim that the actual
entire barcode in those two degrees is identical. Summing over
p replaces the cone in (23) by the full second determinant moment,
since singular matrices have probability zero. In d=2 the two
leading coefficients c_(q,0),c_(q,1) are equal; no finite H2
lifetime population is created by these ordinary filtrations.

## 11. A degree-marked iid-copy corollary, with its actual cluster input

The preceding result admits the same narrow process regime as SG;
this section explains the additional global step rather than inferring
Poisson from the first moment. Define

      N_t^all=sum_(p=0)^(d-1) N_(p,t)

on the good locus and zero otherwise. The Borel construction in
Section 7 applies simultaneously to this finite set of degrees.
A single field has a finite positive lifetime list, so its process
on bounded scaled-lifetime windows tends to empty almost surely;
the mean also tends to zero at order t^(2/3).

At a fixed contact-good realization with V_(p,0)>0, the unit-fold
theorem supplies the actual Hp bar and consumes both new critical
points' unique roles. Every other finite bar, in every degree,
uses two of the finitely many old critical branches. Their limiting
heights are distinct, so the minimum of all their nonzero possible
endpoint gaps is positive. This is true whether or not their old
pairing changes with r. There are no extra critical branches near
b. At r=(ta/k)^(1/3), with fixed 0<a<=T, exactly the pinned bar
has lifetime at most Tt eventually. Hence N_(Tt)^all=1 there.
When V_(p,0)=0, the normalized determinant weight tends to zero
regardless of the number or pairing of finite-r bars.

Insert the bounded Borel mark 1{N_(Tt)^all>=2} into (20), and
sum over p. Pointwise the weighted contact integrand tends to
zero by the preceding endpoint exhaustion. Its bound is a finite
multiple of

      a^(-1/3) k^(-2/3) H(b,k), 0<a<=T,

integrable over the complete marks, including a down to zero.
Far anchored counts are O(t) by (21). Therefore

      E[N_(Tt)^all 1{N_(Tt)^all>=2}]=o(t^(2/3)).         (26)

This does not bound the second factorial moment. It deliberately
inserts a bounded global mark into a once-counted endpoint sum,
not an additional unbounded witness count or a four-site rank claim.

Give each finite bar marks (p,a,b,k,M,u), with a=ell/t,
k=ell/dist(M,S)^3 and ordered u from its birth point M to its
death point S. In the near band these are unique physical
displacements. A fixed Borel shortest-displacement convention
outside that band is harmless because its contribution is O(t).
The limiting one-bar mean measure is

      d eta_(q,p)=4 a^(-1/3) k^(-2/3) pi_0(v_0)
               z_(p,0)(b,k,u) da db dk dx d sigma(u).    (27)

Its mass on a<=T is (3/2)L^d c_(q,p) T^(2/3); the lifetime
marginal is L^d c_(q,p) a^(-1/3) da. Midpoint x and the actual
birth position M=x-r u/2 have the same limiting mark. In higher
degrees M is the birth critical point, not generally a maximum.

For n independent whole copies of this one fixed actual law,
choose t_n down to zero with n t_n^(2/3)->lambda in (0,infinity).
For any compact-supported nonnegative process test f, the error
between its single-copy expected sum of 1-exp(-f) and its
single-copy expected 1-exp(-sum f) is bounded by the left side
of (26). Independence and the elementary Laplace product give
the Poisson random measure with intensity

      lambda sum_(p=0)^(d-1) delta_p tensor eta_(q,p).    (28)

The same all-mark bound proves convergence on finite windows a<=T.
The resulting limiting degree-mark populations are independent
Poisson restrictions of this marked measure; the finite-field
degree populations are not asserted independent.
This is an independent-whole-copy rare-event limit. It is not a
spatial expanding-domain limit, a within-one-field Poisson law,
a coalescing-contact factorial theorem or the separately owned
regional additional-witness estimate. Those regimes do not follow
from (26). No K7 factorial side result is inherited from K5 positivity.

## 12. Failure boundaries and what the result does not universalize

The reusable higher-degree mechanism is now explicit:

1. Nonzero cubic axial derivative and nonsingular transverse block
   give the adjacent degrees and exact local product connection.
2. Same-field C4 convergence and off-contact genericity give the
   complete critical list, old height gaps and annulus gradient floor.
3. The small height gap plus that floor excludes other connecting
   trajectories; the relative coefficient is a unit.
4. No intervening critical value lets that unit create and kill
   the new persistence direction at those exact two endpoints.
5. The Borel once-count mark plus the actual probabilistic contact,
   all-mark envelope and far interfaces give the lifetime density.

Removing a premise has an identifiable obstruction, rather than
an automatic replacement law. Arbitrary adjacent cells with zero
incidence can exchange long bars as Section 3 shows. A second
critical level entering the shrinking band can change the exact
pivot even with a nonzero incidence. A singular transverse block
is outside the regular-fold theorem. Uncontrolled global continuation
can create extra endpoints or destroy the fixed-annulus floor.
Other singularity orders/contact densities can change the radial
weight and exponent. A non-Gaussian law needs its own actual
contact density, conditional selection and envelope; its covariance
does not supply them.

There is therefore no H0 elder-rule generalization by terminology:
in p>0 the algebraic pivot and unit relative boundary replace elder
component enclosure. No arbitrary higher-homology fold, arbitrary
manifold, non-Gaussian class, numerical sample, formal theorem or
coefficient interval is admitted by the word 'generic'. The stated
Gaussian spectral class is admitted by its named ranks, moments,
conditional genericity and the new topological argument above.

The qualitative unit-fold theorem works on the deterministic
oriented closed manifold in Section 1, but the density/process
specialization here is only the exact flat-torus Gaussian law.
A different geometry still needs its actual all-Borel count,
contact observations, volume Jacobian, envelope and far contracts.
No new scientific promotion or model-status change is made.

## 13. Exact operational sources, review extents and classical imports

The following incorporated operational bindings were freshly hashed by
root. SG's mathematical Sections1-11 are byte-identical to its original
825-line preparation; the original source consumed by the Hk author and
the new incorporated SG wrapper remain separately identified.

| Symbol | Incorporated proof | Bytes / lines | SHA256 |
| --- | --- | --- | --- |
| SG | [spectral Gaussian admission](../spectral_gaussian_admission/PROOF.md) | 42478 / 826 | 6c0f34cb318d337b3b8c2c306f3b965f5e5a9d2fe38d5e001b8044de74b7f225 |
| ST | [regular-fold structural theorem](../fold_structural_universality/PROOF.md) | 43927 / 804 | 3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531 |
| IG | [periodized Gaussian H0](../periodized_gaussian_h0/PROOF.md) | 34629 / 637 | c8bd489ca3fbb3760f7b2d53e7385131ac5336c51ba983e00e4d540fe2f00aaa |
| PP | [iid short-bar process](../iid_short_bar_process/PROOF.md) | 30935 / 618 | 320bc988c468bea4938bb777a2d05952c6ab8e0a384f33725b50de847aeec1eb |
| ET | [finite elder transfer](../finite_elder_transfer/PROOF.md) | 23774 / 458 | c9e4fb09900888cd2b1bd96b49f85072b51c822b2503fe9f3d8891049bf66201 |

The Hk mathematical author consumed SG's original42038B/825-line source,
SHA25643e8180947b559e53f7afcf95ba61136f3fffcbe05a19e7a5953018fda27e294:
a full prior saved-file read and fresh lines1-645/740-825 during the Hk
construction. Optional diagnostics646-739 are not a new Hk premise.
The current SG source has the same36656B Sections1-11 core, SHA256
4cad43d3f80a8e47da886762d402c4981fe6834e278579b36673055539d6cf32;
root checked actual complete original/current core and incorporation diff.
No new whole current826-line SG read is attributed to the Hk author.
ST/IG/PP received that author's full actual reads in the immediately
preceding spectral derivation, including the separately filled truncated
ST middle; their identities were freshly rechecked during Hk preparation.
ET's incorporated bytes were rehashed; the author's full original ET read
was prior, with no fresh full incorporated ET wrapper read claimed.

The proof consumes SG Sections2-5 for actual C4 law/moments, named finite
ranks and all-Borel incidence; Section6 for every fixed contact posterior's
genericity and complete old critical list; Section7 for the radial and
coefficient ledger and far bound; Section8 for the independently explained
bounded-cluster/Laplace mechanism. SG's H0 selector is supplied here by
Sections3-7's new relative unit-incidence/persistence argument. ST supplies
the observation/determinant ledger, IG the preceding physical-law route,
and PP the process-regime distinction. No K7/factorial theorem is consumed.

Fresh FULL original-source mathematical reviews are separate:
final_contract_review read all884 actual lines in four untruncated ranges,
verified the exactde1cad identity, and found no must-fix; benchmark_formal_audit
independently read all884 actual lines and found no must-fix. Both freshly
rehashed the five original declared inputs. Final's prior separate full
actual input audits include SG original/current; this pass does not claim
five extra dependency reads. Benchmark's prior full input audits and ET
authorship remain separately exposed; this pass does not claim five new
whole dependency rereads. Root read original actual1-310/311-620/621-884
untruncated. Changed incorporated bytes/readme await their own full review.

Final's new Hk primary exposure was bounded parsed Milnor Section5 setup
and Theorem5.4, Section7 setup/Corollary7.3, bounded Laudenbach convention
and theorem clauses, and bounded Antony introduction/main theorem/normal
form. Benchmark's new Hk exposure was bounded parsed Milnor Section5 and
Section7 setup/Lemma7.2/Corollary7.3 and Laudenbach convention/extra-orbit
statement; no new Antony reading is credited to that reviewer. Root's new
bounded Milnor find results supplied actual Corollary7.3 and Theorem5.4
passages, without full-book or raw-PDF credit. Neither root nor reviewers
inherit the author's complete six-page Laudenbach read described below.

Original author supplied C/window/S/ST/IG/PP/SG. Benchmark authored CG/ET
and catalogue code. Final authored the separate fold-chart route and
provided earlier carrier witnesses. Root supplied bounded prefile unit
incidence, IGtheta and PP endpoint/Borel challenges. These are substantial
same-provider source and route exposures, with organizational independence0.
The [review record](REVIEW.md) retains exact original/current verdicts.

The following author-side primary-exposure account and import ledger is
preserved from the original preparation. Its final request for exact-file
review is historical; the two new original-source verdicts are recorded
above and separately from pending incorporated-source review.

Fresh primary readings for the new topological step are explicit:

* [Milnor, Lectures on the h-Cobordism Theorem](https://webhomes.maths.ed.ac.uk/~v1ranick/surgery/hcobord.pdf),
  1965 primary book scan: bounded parsed-PDF readings of the
  Section 5 setup/first cancellation theorem (printed pages 45-48,
  PDF pages 49-52), and Section 7's relative boundary formula and
  handle homology construction (printed pages 85-90, PDF pages
  89-94). Corollary 7.3 is the incidence import actually used.
  Other cancellation hypotheses elsewhere in that book are not
  imported. A bounded earlier opening also included its equal-N
  Morse perturbation lemma, but the new topology proof does not
  depend on a new high-dimensional Sard import.
* [Laudenbach, A proof of Morse's theorem about cancellation](https://arxiv.org/pdf/1307.2545),
  arXiv:1307.2545v1: the complete parsed six-page/291-line primary
  text was read. Its unique transverse orbit criterion and
  pseudogradient convention corroborate the geometric mechanism.
  Its additional global orbit condition is not assumed implicitly
  as a replacement for the direct relative incidence calculation.
* [Antony, Gradient Flow Line Near Birth-Death Critical Points](https://arxiv.org/pdf/1706.07746),
  arXiv:1706.07746v3: bounded actual parsed reading of the main
  theorem, normal-form discussion and introductory outline only,
  not the full 39-page proof. Its smooth unfolding and metric
  assumptions are stated in Section 6 and are not claimed satisfied
  merely by the C4 contact coupling.

These were primary web text reads, not downloaded PDF byte-custody
verifications. No raw-PDF SHA or full-book/full-Antony read is
invented. No failed primary web operation was retried. The new construction
itself proves the adapted-flow no-escape step and the C4-to-smooth
filtration equivalence. Classical imports remaining are smooth
Morse handle attachment and its relative incidence formula,
ordinary smooth approximation/implicit continuation/ODE flow,
finite persistence interval decomposition, and SG's explicitly
declared locally Lipschitz area formula/Gaussian regression.
All integrability and limiting uses are supported by the actual
SG moments, target coercivity and domains in Sections 8-11.

The strongest proved preparation claim is the unit-fold mechanism,
and, under the actual SG spectral assumptions, the resulting
ordinary Hp density and independent-copy degree-marked process.
An exact-file nonauthor attack must still assess these new
topological imports/bridges and composition. No review, compiler,
organizational independence or scientific credit is manufactured
by the author-side derivation or saved-byte readback.
