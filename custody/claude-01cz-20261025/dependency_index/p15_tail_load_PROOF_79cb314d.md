# Even-capacity reference tails and instance-adaptive overlap budgets

Object: OA-P15-TAIL-LOAD-20260928-v1.
Author: OpenAI / ChatGPT, foreground continuation, 28 September 2026.
Disposition: AUTHOR-SIDE MATHEMATICAL CANDIDATE; source-bound nonauthor review required.
Scientific effect: NONE. No existing proof, governing status, graph or prize changes.

## 1. Results and exact scope

This additive note sharpens two steps in the P15 overlap packet at Math- commit
`23b7558943f69a7f479251566b19ba58e75cfc26`. Its original proof documents are not
modified. The independent-provider review of that packet is not presumed complete.
The arguments used here are displayed below rather than imported as accepted
premises. SOURCE_MAP.json identifies the related source objects and their roles.

Let X be a finite nonempty set covered by nonempty blocks X_i. Every coordinate
belongs to at most two blocks. Each capacity a_i is a positive EVEN integer,
including the capacities of inactive blocks. Define

    D = {U subset X: |U intersect X_i| <= a_i for every i}.

Let chi_D(U) be the minimum number of members of D partitioning U, with
chi_D(empty)=0. Assume |X_i|=a_i d_i+1, with positive integer d_i, and put
K=max_i d_i >=4. Let O_K(D)={U:chi_D(U)>K}. For independent original Bernoulli
probabilities p_v in [0,1], put H_D=-log mu_p(D), possibly infinite. Admissible
prices satisfy 0<=c_v<=min(1,-log(1-p_v)), taking the upper bound to be one at
p_v=1. A generator family covers O_K(D) when every obstruction contains a member;
a generator g costs product_{v in g}c_v. The empty generator costs one.

**Theorem 1 (same palette, tighter even-capacity factor).** In this class,

    covercost_c(O_K(D)) <= min(1, rho_9 H_D),                       (T1)
    rho_9 = 2/[9-log(36e^2-63e+28)] < 1/2.

A rational outward enclosure, not a floating-point fit, is

    0.47734798895766998533 < rho_9 < 0.47734798895766998534.          (T2)

The former packet supplied 2/log(512/25)<2/3. The new bound has the SAME
structural hypotheses and K, with no restriction on the Bernoulli vector or
transformed prices. It is not a claim that rho_9 is the optimal COVER factor.
The reference-tail optimization used to obtain it is exact, as stated below.

**Theorem 2 (actual-coordinate load).** More generally, suppose a nonempty family of
nonempty blocks indexed by J really covers an obstruction family, and D is
contained in each local event A_i={|U intersect X_i|<=a_i}. Put p_*=1-e^(-1) and

    H_i=-log P(Bin(|X_i|,p_*)<=a_i)>0,
    beta_J=max_{v in X} sum_{i in J: v in X_i} 1/H_i.                (T3)

For independent original coordinates and the same prices,

    covercost_c(O) <= min(1, beta_J H_D).                          (T4)

The statement presupposes a genuine setwise cover and positive reference
hazards; it does not assert the palette formula for an arbitrary downset.
It needs neither an even-capacity nor a bipartite hypothesis ONCE a cover has
been established. Section 2 establishes such a cover for Theorem 1. The earlier
bipartite class also has a displayed palette proof in the referenced packet.

Choose J to contain one representative of each inclusion-minimal active block
(d_i=K). Repeated blocks and redundant strict supersets can be removed because
their containment events are covered by smaller retained generators. In the
even/read-two class beta_J<=2/(-log q_9)=rho_9. If the retained blocks are
disjoint, beta_J<=1/(-log q_9). Unequal reference hazards are retained separately;
a shared coordinate is charged the SUM of its incident weights, never their max.

## 2. Why the even/read-two full-block family is a cover

For every U subset X,

    chi_D(U)=max_i ceil(|U intersect X_i|/a_i).                     (T5)

The lower bound is immediate. For the upper bound write k for the right side;
the empty set is trivial. Represent each ORIGINAL coordinate of U as one edge
of the block-incidence multigraph: a shared coordinate joins its two block
vertices, and a private coordinate joins its block to a private degree-one
vertex. Coordinates are never duplicated as independent random variables.

Split the edges incident to block i among a_i/2 vertices as evenly as possible.
Every split degree is at most ceil(2|U intersect X_i|/a_i)<=2k. Pair odd-degree
vertices and add auxiliary edges. Each odd degree was at most 2k-1, so the
augmented degrees are even and at most 2k. Orient an Euler tour in each component.
Each vertex now has at most k incoming and k outgoing edges. Replace each vertex
by its outgoing and incoming copies. An oriented edge joins its tail's outgoing
copy to its head's incoming copy, producing a bipartite multigraph of maximum
degree at most k.

A bipartite multigraph of maximum degree k can be regularized to a k-regular
one by adding isolated vertices to balance its two parts, then adding edges
between degree deficits. For a left vertex set S, its k|S| edges end in a set
of total degree at most k|N(S)|, proving Hall's condition. Remove a perfect
matching repeatedly to obtain k edge colors. Remove all added edges. In the
original split graph a color has at most one incoming and one outgoing edge
at each vertex, hence at most two. Combining the a_i/2 vertices permits at most
a_i original coordinates per color in X_i. This proves (T5).

For an inactive i, d_i<K gives |X_i|=a_i d_i+1<=a_i K. For an active i,
|U intersect X_i|>a_i K holds exactly when X_i subset U. Consequently

    O_K(D)=union_{i:d_i=K}{U:X_i subset U}.                        (T6)

Thus J described above is a valid cover at the SAME K. This proves the needed
cover premise independently of a status label on the older packet. It does not
reintroduce that packet's excluded arbitrary macro-clutter or odd capacities.

## 3. Local price-to-hazard transfer

For a proper decreasing event A containing the empty configuration, on its
finite coordinate set S define p_v(t)=1-e^(-t_v), 0<=t_v<=1, and
F(t)=-log mu_{p(t)}(A). Since p_v(t)<1 and A contains the empty set, its probability
is positive. Holding other coordinates fixed gives mu(A)=A_0+B_0 e^(-t_v) with
A_0,B_0>=0, by decreasingness. Direct differentiation yields

    partial_v^2 F=-A_0 B_0 e^(-t_v)/(A_0+B_0 e^(-t_v))^2 <=0.

F is nonnegative. Its one-coordinate chord inequality implies
F(t)>=t_v F(t with t_v=1). Iterating gives

    F(t)>=F(1,...,1) product_{v in S}t_v.                          (T7)

For arbitrary probabilities set t_v=min(1,-log(1-p_v)); then p_v(t)<=p_v.
Decreasingness implies -log mu_p(A)>=F(t). Therefore

    product_{v in S}c_v <= [-log mu_p(A)]/F(1,...,1).              (T8)

This holds as an extended inequality at zero event probability. A zero price is
not a reason to delete a generator from the setwise cover. Applying (T8) to A_i
uses reference hazard H_i, not the hazard of the full-block containment event.

## 4. Weighted entropy with actual coordinate loads

We prove the needed finite product inequality rather than presume independence
of overlapping events. If H_D is finite, set nu=mu_p(.|D). Let KL denote relative
entropy. Then KL(nu||mu_p)=H_D. Since the S_i=X_i marginal of nu is supported on
A_i,

    KL(nu_{S_i}||mu_{S_i})
       =KL(nu_{S_i}||mu_{S_i}(.|A_i)) - log mu_p(A_i)
       >= h_i,       h_i=-log mu_p(A_i).                         (T9)

All A_i have positive probability because D does. Order the original coordinates
and write

    b_v=E_nu KL(nu(v | all earlier coordinates)||mu_v)>=0.

Product form of mu and the chain rule give sum_v b_v=H_D. For a subset S, the
law conditioned on earlier coordinates IN S is a mixture of laws conditioned
on all earlier coordinates. Convexity of KL in its first entry gives

    KL(nu_S||mu_S)<=sum_{v in S} b_v.                            (T10)

For arbitrary nonnegative weights w_i, multiply (T9)-(T10), sum, and exchange
finite sums:

    sum_i w_i h_i
       <=sum_v b_v sum_{i:v in S_i}w_i
       <=[max_v sum_{i:v in S_i}w_i] H_D.                        (T11)

This is a weighted read-incidence / Finner-type inequality, not a new claim about
arbitrary dependent measures. Deterministic original coordinates have zero
conditional divergence and can be removed from the product support.

Take w_i=1/H_i on the genuine cover J. Equation (T8) followed by (T11) proves
price_c(J)<=beta_J H_D. Comparing with the unit-price empty-generator cover proves
(T4). If H_D=infinity use that empty generator directly; if H_D=0 the finite
argument gives a zero-price covering family. No infinity divided by infinity
or logarithm of a zero REFERENCE probability is involved.

A sum, not a maximum, at each shared coordinate is necessary for (T11). Two
identical events depending on one Bernoulli(1/2) coordinate, both of probability
1/2 and both of weight one, have local hazard sum2log2 but global hazardlog2.
Their load is two. This is a counterexample to a max-instead-of-sum entropy step,
not a claim that keeping duplicate generators is optimal.

## 5. Exact worst reference tail among positive even capacities

Let

    q_9=P(Bin(9,1-e^(-1))<=2)
        =e^(-9)[1+9(e-1)+36(e-1)^2]
        =e^(-9)(36e^2-63e+28).                                  (T12)

**Lemma (sharp reference-tail maximum).** For every even a>=2 and integer K>=4,

    P(Bin(aK+1,1-e^(-1))<=a) <= q_9,                            (T13)

and equality is attained at a=2,K=4.

For a=2, monotonicity in the number of trials gives (T13). If a>=4 is even, put
s=E[5^(-Bernoulli(1-e^(-1)))]=(e+4)/(5e). The elementary inequality e>8/3 gives
s<1/2. Markov's inequality applied to 5^(-Bin(n,p_*)), with n=aK+1>=4a+1, yields

    P(Bin(n,p_*)<=a) <=5^a s^n
        <(1/2)(5/16)^a <=625/131072.                             (T14)

There is no omitted a=3 case: the theorem's capacity is EVEN. No assertion for
an odd-capacity nonbipartite graph is licensed by this estimate.

We certify 625/131072<q_9<1/64 using rational arithmetic. Let l=1957/720 and
u=87/32. The exponential series through degree6 plus its geometric tail gives

    l<e<31967/11760<u.

Let P(z)=36z^2-63z+28. P'(z)=72z-63>0 on [l,u]. Thus

    P(l)/u^9 < q_9 < P(u)/l^9.                                 (T15)

The two strict margins are the exact positive rationals

    P(l)/u^9 - 625/131072
      =87187623163627321174969/8421039761612032386662400,
    1/64 - P(u)/l^9
      =12311490319341451572976881157/26946192308072632700222520394048.

This proves the reference maximum and -log q_9>log64. Also e<11/4 implies
e^2<121/16<8, hence log2>2/3. Therefore

    rho_9=2/(-log q_9)<2/log64=1/(3log2)<1/2.                   (T16)

Every active even block has H_i>=-log q_9. Every original coordinate belongs
to at most two retained blocks. Equations (T3)-(T4) now prove (T1).

This is exact sharpness of the uniform REFERENCE tail over the displayed integer
parameter family. The entropy bound, local transfer and empty-generator choice
may lose slack, so optimality of the COVER factor does not follow.

## 6. Certified numerical interval and operational use

The executable certificate brackets e by its positive series and geometric tail.
It brackets log x using x=2^j y with 1<=y<2, and

    log y=2 sum_{n=0}^{m-1}t^(2n+1)/(2n+1)+R,
    t=(y-1)/(y+1),
    0<=R<=2t^(2m+1)/[(2m+1)(1-t^2)].

The same formula gives log2. Monotonic interval operations on
2/[9-log P(e)] give (T2) without using floats to choose or certify endpoints.

For an instance, first establish an exact palette theorem and identify its
active full-block cover. Remove duplicates and strict supersets. Evaluate each
positive reference H_i, using outward enclosures when numerical certification is
needed. Use a certified UPPER bound on 1/H_i in the load at every original
coordinate. The maximum load is a safe instance-specific factor. One may then
use the full-block cover or the empty-generator cap. An uncertified numerical
hazard is not substituted for a proved lower bound.

## 7. Verification and acceptance limits

The small standard-library suite checks the nine-trial polynomial identity,
strict rational margins, interval directions and tails, weight addition at shared
coordinates, inactive-block exclusion, duplicate/superset removal, and the integer-
weighted probability form of (T11) on a heterogeneous finite Bernoulli grid.
It includes the max-instead-of-sum counterexample. Finite tests do not prove
(T7)-(T11) for every probability vector; their analytic proofs are above.

This is an additive author-side candidate, not a governing acceptance. It does
not modify either #118 source proof, revive #116's unrelated blocked executable,
claim an optimal cover factor, retain arbitrary macro-clutter constraints, or
permit odd capacities in the nonbipartite case. Read-only targeted CI checks
finite certificates and existing repository gates, not the truth of a new full
analytic theorem or organizational reviewer independence.
