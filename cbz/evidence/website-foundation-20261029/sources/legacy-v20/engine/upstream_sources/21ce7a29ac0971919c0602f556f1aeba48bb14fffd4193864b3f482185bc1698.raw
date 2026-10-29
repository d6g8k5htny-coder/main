# P15: overlapping original-coordinate blocks with an exact palette and a sub-3/4 budget

Object: OA-P15-BIPARTITE-OVERLAP-20260928-v1.
Author: OpenAI / ChatGPT, foreground continuation, 28 September 2026.
Disposition: AUTHOR-SIDE MATHEMATICAL CANDIDATE; NONAUTHOR REVIEW REQUIRED.
Scientific effect: NONE. No existing theorem body, verdict, graph or prize is modified.

## 1. What changes, and what does not

The existing `frontiers/full_price_20260924/PROOF.md` treats disjoint capacity
blocks, with an additional macro-clutter. Its full transformed-price theorem
uses independence of local restrictions on disjoint coordinate sets. That
independence is generally false once original coordinates are shared.

This note proves a different structural extension: capacity blocks MAY OVERLAP,
provided their intersection graph is bipartite. We drop the additional
macro-clutter restriction. The new conclusion is therefore not a blanket
extension of every earlier realized family, nor an arbitrary-downset result.
Both its coloring assertion and its probability budget are proved below on the
actual original coordinates; it does not simply invoke P15-B with unverified
local covers. Auxiliary graph vertices do not duplicate random coordinates.

Let X be a finite ground set, covered by nonempty blocks X_i (i in I). Fix positive
integer capacities a_i. Assume a partition I=L disjoint-union R such that
blocks on the SAME side are pairwise disjoint. Equivalently the graph joining
intersecting blocks is bipartite. A coordinate belongs to at most one block on
each side and thus to at most two blocks overall. Repeated equal blocks on
opposite sides cause no difficulty.

Define the decreasing family

    D = {U subset X : |U intersect X_i| <= a_i for every i}.             (O1)

Let chi_D(U) be the least number of sets in D whose disjoint union is U, with
chi_D(empty)=0. Let O_k(D)={U:chi_D(U)>k}. A generator cover G means that every
U in O_k(D) contains some g in G; generators need not themselves be minimal.
Its price is sum_{g in G} product_{v in g} c_v. The empty generator costs one;
the empty FAMILY costs zero. Repeated generators are counted only once.

## 2. Exact original-coordinate coloring

**Theorem O1.** For every U subset X,

    chi_D(U) = max_i ceil(|U intersect X_i|/a_i).                       (O2)

In particular the lower bounds from individual capacities are simultaneously
attainable despite overlaps.

*Proof.* Write k for the right side. The lower bound follows because a union of
k members of D contains at most k a_i coordinates in each block. The empty U
case is immediate. Otherwise k>=1.

Construct a bipartite multigraph whose original edges correspond ONE-TO-ONE to
coordinates of U. A coordinate in both X_l and X_r is an edge between l and r.
A coordinate in only one block is joined to its own auxiliary degree-one
vertex on the other side. Because X covers the ground set, no coordinate lacks
an endpoint. Parallel edges are retained with their distinct coordinate labels.

At each real vertex i, split its incident edges among a_i auxiliary vertices,
with loads differing by at most one. Every such vertex has degree at most
ceil(|U intersect X_i|/a_i)<=k. An original edge retains both its assigned
endpoints and its SINGLE original label. Private auxiliary vertices have
degree one.

For completeness, a bipartite multigraph of maximum degree at most k has a
proper edge coloring with at most k colors. Add isolated vertices to make its
two sides have the same size N. Their total degree deficits from k are equal,
namely kN minus the current edge count. Add auxiliary edges between vertices
with positive deficits until every vertex has degree k. Parallel auxiliary
edges are allowed. For S on the left, its k|S| incident edges end in N(S), whose
total degree is k|N(S)|, so |N(S)|>=|S|. Hall's condition gives a perfect
matching. Remove one perfect matching and repeat in the remaining regular
bipartite multigraph, using one new color each time. Discard all auxiliary
edges. This proves the asserted edge-coloring fact.

Return each original coordinate to the color of its edge. At most one edge of
a given color uses any split vertex, so at most a_i coordinates of that color
lie in X_i. Each color class belongs to D. This proves (O2). The construction
uses independent Bernoulli variables only later: graph splitting has not
altered the ground set or introduced probabilistic independence. QED.

## 3. The same palette and the exact obstruction

Now assume

    |X_i| = a_i d_i+1,    d_i positive integers,
    K = max_i d_i >= 4.                                                (O3)

Only the maximal demand K must be at least four. Lower-demand blocks, including
demand one, are allowed. Set I_*={i:d_i=K} and let G_* be the distinct full
blocks X_i for i in I_*.

**Corollary O2.** At the SAME palette K,

    O_K(D) = union_{i in I_*} {U subset X : X_i subset U}.               (O4)

Thus G_* is a genuine cover; its inclusion-minimal distinct members are exactly
the minimal K-obstructions. The whole ground set has chi_D(X)=K+1, so the
obstruction is nonempty. If k>K then O_k(D) is empty.

*Proof.* By (O2), a K-obstruction must have |U intersect X_i|>Ka_i for some i.
If d_i<K, then |X_i|=a_i d_i+1<=a_i K. If d_i=K, the only way to exceed Ka_i is
to contain all of X_i. These implications are reversible. Also
ceil((a_i d_i+1)/a_i)=d_i+1, which proves the whole-ground assertion. QED.

The smaller-demand full blocks are NOT charged to the cover. This avoids
incorrectly requiring a demand-one block to satisfy a standalone same-demand
budget when it is not a generator at the global palette.

## 4. Local hazard transfer for all transformed prices

For independent original probabilities p_v in [0,1], write mu_p for the
product measure. Put phi(p)=min(1,-log(1-p)), with phi(1)=1, and assume
0<=c_v<=phi(p_v).

We first reproduce the elementary separate-concavity lemma in the earlier
full-price source, rather than import a stronger independence claim.
Let A be a proper decreasing family on a nonempty finite coordinate set which
contains the empty set. For t in [0,1]^n, set p_j(t)=1-exp(-t_j) and

    F(t)=-log mu_{p(t)}(A),    H_A=F(1,...,1)>0.

The probability stays positive on this cube. With other coordinates fixed,
mu(A)=A_0+B_0 exp(-t_j), where A_0>=0 and B_0>=0 by decreasingness. Consequently

    partial_j^2 F = -A_0 B_0 exp(-t_j)/(A_0+B_0 exp(-t_j))^2 <= 0.

Only separate, not joint, concavity is needed. Since F>=0, its chord bound
implies F(t)>=t_j F(t with t_j=1). Iterate over all coordinates:

    F(t) >= H_A product_j t_j.

For arbitrary p_j, take t_j=phi(p_j) and p'_j=1-exp(-t_j)<=p_j. Monotonicity of
a decreasing event gives mu_p(A)<=mu_{p'}(A). Thus, as an extended inequality,

    product_j c_j <= [-log mu_p(A)]/H_A.                              (O5)

No division by a zero coordinate probability is made. If mu_p(A)=0, its
hazard is infinite; if some c_j=0, the generator remains in the setwise cover.

For A_i={|U intersect X_i|<=a_i}, let h_i=-log mu_p(A_i) and

    H_i=-log P(Bin(|X_i|,p_*)<=a_i),   p_*=1-exp(-1).

Equation (O5) gives price(X_i)<=h_i/H_i. We need this only for i in I_*.

## 5. A uniform reference hazard for the active blocks

Define

    q_5=P(Bin(5,p_*)<=1)=(5e-4)e^(-5),
    h_5=-log q_5=5-log(5e-4),    rho_bip=2/h_5.                      (O6)

**Lemma O3.** If a>=1 and n>=4a+1, then

    P(Bin(n,p_*)<=a) <= q_5,  hence H_(n,a)>=h_5.                  (O7)

*Proof.* For a=1 this is monotonicity in n and the exact n=5 formula. For a>=2,
put s=E[5^(-Bernoulli(p_*))]=(e+4)/(5e). Since e>8/3, s<1/2. Markov's
inequality, applied in its valid direction to 5^(-Bin(n,p_*)), gives

    P(Bin(n,p_*)<=a) <= 5^a s^n
       <= 5^a s^(4a+1) < (1/2)(5/16)^a <= 25/512.

This is a bound uniform over ALL a>=2, not a finite-grid inference. The
inequalities 8/3<e<11/4 give

    q_5 > (28/3)(4/11)^5 > 25/512,

where the difference in the final comparison is the positive rational
2601239/247374336. This proves (O7). QED.

Every active block has n=a_i K+1>=4a_i+1, so H_i>=h_5. No estimate on the
reference hazard of inactive blocks is required.

## 6. Replace false independence by two-sided Cauchy-Schwarz

Let A_L=intersection_{i in I_* intersect L} A_i, and define A_R analogously.
Blocks within either side are disjoint original coordinate sets. Therefore

    mu_p(A_L)=product_{i in I_* intersect L} mu_p(A_i),
    mu_p(A_R)=product_{i in I_* intersect R} mu_p(A_i).

There is in general NO independence between A_L and A_R. Instead,

    mu_p(D) <= mu_p(A_L intersect A_R)
       <= [mu_p(A_L) mu_p(A_R)]^(1/2),                               (O8)

by Cauchy-Schwarz for the two indicators. Taking logarithms when probabilities
are positive, and treating zero cases by extended hazards, gives

    sum_{i in I_*} h_i <= 2[-log mu_p(D)].                           (O9)

Empty side intersections have probability one. Combining (O5), (O7), (O9),
and the possible removal of duplicate full-block generators proves

    price_c(G_*) <= sum_{i in I_*} h_i/h_5
                  <= rho_bip[-log mu_p(D)].                         (O10)

**Theorem O4 (full-price overlapping-block budget).** Under (O1)-(O3), for every
independent probability vector and every transformed price vector,

    covercost_c(O_K(D)) <= min(1, rho_bip[-log mu_p(D)]),
    rho_bip = 2/[5-log(5e-4)] < 3/4.                                (O11)

If the uncapped bound is at least one, use the empty generator; otherwise use
G_*. If mu_p(D)=0 the empty generator deals with the infinite hazard. All-zero
prices or probability-one good events create no undefined operation. The
palette is K=max_i d_i, not a sum of palettes or a twofold inflation.

This is an upper bound, NOT a claim that rho_bip is optimal. Setwise coloring
(O2) is exact; budget-factor sharpness is a different question.

## 7. Short exact certificate and numerical enclosure

The positive exponential series through degree six, with a geometric tail,
gives

    8/3 < 1957/720 < e < 31967/11760 < 87/32 < 11/4.

Thus 5e-4<307/32. The positive Taylor polynomial for exp(7/3) through degree six
satisfies

    sum_{j=0}^6 (7/3)^j/j! - 307/32 = 129055/209952 > 0.

It follows that log(5e-4)<7/3, h_5>8/3, and rho_bip<3/4. This exact rational
certificate is sufficient for (O11); no decimal estimate is needed.

A separate rational-series calculation gives the outward enclosure

    0.73015826409534942932 < rho_bip < 0.73015826409534942934.          (O12)

The code encloses e by a positive series and its geometric tail. It encloses
logarithms by binary reduction and the positive atanh series with remainder
2t^(2m+1)/((2m+1)(1-t^2)), t=(x-1)/(x+1). Monotonic interval operations yield
(O12). Floating-point arithmetic is not used to choose its endpoints.

The number is not directly comparable as an improvement on the old sharp
0.84548 factor: the old result allowed maximal demand two and disjoint blocks,
whereas this one uses maximal demand at least four and a different overlap
structure. The substantive advance is handling shared ORIGINAL coordinates.

## 8. Two false extensions excluded by exact counterexamples

**Read-two alone does not give the same coloring theorem.** Take nine original
coordinates and blocks

    X_0={0,1,4,5,6}, X_1={0,1,2,3,7}, X_2={2,3,4,5,8},
    a_i=1, d_i=4, K=4.

Every coordinate belongs to at most two blocks. For U={0,1,2,3,4,5}, no full X_i
is contained in U, but every two distinct coordinates of U share a block.
Equivalently U is the six edges of a triangle with two parallel edges per
side; every matching has size at most one. Thus chi_D(U)=6>4, although the
individual capacity lower bound is four. The full-block family fails to
cover this K-obstruction. The intersection graph is the excluded odd triangle.
This example rules out omitting bipartiteness, not all possible covers on
non-bipartite systems.

**Overlap invalidates multiplying all local good probabilities.** Take blocks
{0,1} and {1,2}, capacities one, and independent probabilities 1/2. Their good
probabilities are 3/4 each, while the joint good probability is 5/8>9/16.
The invalid disjoint-block inequality fails. The squared Cauchy-Schwarz
inequality (5/8)^2<= (3/4)^2 remains true.

These are exact finite counterexamples, not asymptotic heuristics. Demand-one
counterexamples to older standalone budgets are also unchanged: allowing an
inactive demand-one block here is not accepting its standalone K=1 budget.

## 9. Scope, attribution, and verification

The new proof is self-contained apart from Hall's elementary matching theorem;
the regular bipartite edge-coloring reduction is displayed, not claimed as an
invention. Separate concavity and Cauchy-Schwarz are classical. This note makes
no novelty-priority claim. It closes an author-side overlapping-family route,
not a governing P15 prize, an arbitrary macro-clutter, arbitrary read-two
systems, or an unrestricted downset statement.

`test_overlap.py` checks exact coloring against independent partition dynamic
programming, shared-edge consistency, parallel edges, heterogeneous capacities,
maximal-demand generator selection, all subsets of a nine-coordinate overlap
instance, rational probability grids, exact counterexamples, and rational
series bounds. The runner rejects semantic changes in both ordinary and
optimized Python. Such tests do not prove an analytic inequality for a
continuum of probabilities; Sections 4-7 supply that argument. No independent
review, Lean formalization, or theorem-status promotion is implied.

Source identities and limited primary-source reconnaissance are in
SOURCE_MAP.json and RECONNAISSANCE.md. The prior interrupted 31/11 collision
test claim is NOT evidence for this distinct suite. All execution counts
reported here are recomputed for this packet.
