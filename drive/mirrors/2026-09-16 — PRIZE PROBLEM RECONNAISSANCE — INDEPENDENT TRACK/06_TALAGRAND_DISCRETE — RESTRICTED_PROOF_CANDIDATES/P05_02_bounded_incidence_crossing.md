# PR-TAL-004 — Arbitrary crossings with bounded coordinate incidence

2026-09-16. Complete author-side proof; external review and novelty UNESTABLISHED. This extends the mathematical CLASS beyond laminar scopes and beyond any fixed bound on the chromatic number of their crossing graph. It does not establish universal constants for arbitrary downsets.

## 1. Precise statements

Let D be any finite capacity downset, |S intersect B|<=r_B. Let d>=1 be an integer such that every coordinate belongs to at most d scopes. One may first remove vacuous, duplicate, implied, and zero constraints as in PR-TAL-003; it suffices that the residual positive-scope representation has incidence at most d.

For uniform p:

(A) With K=d+1, mu_p(D)>=1-1/(2K) implies that D^(K) is(p/4)-small. One explicit cost bound is

    3d/[5(2d+1)] <3/10<1/2.

(B) With K=4d+1, mu_p(D)>=1-1/K implies that D^(K) is p-small. The explicit cost is at most

    max(1,3d/5)/(4d) <=1/4.

These constants are independent of the number of elements, number and sizes of scopes, and capacities. Their K DEPENDS ON d. Fixed d is a genuine restriction.

The results hold for arbitrary independent p_i<=1/2 coordinatewise. For arbitrary p_i in[0,1], A holds at p_i/8 and B at p_i/2 by coordinate capping. The uniform high-p case is handled directly, not capped.

## 2. Deterministic greedy decomposition — where incidence is used

For an integer t>=2, suppose S avoids the zero scopes and |S intersect B|<=t*r_B for every positive scope. Process its elements in a fixed order and color with

    K=d(t-1)+1

colors, never putting more than r_B elements of any color in B. When processing x in B, at most

    floor((|S intersect B|-1)/r_B)<=t-1

colors are saturated in B. At most d such scopes contain x, so at most d(t-1) colors are forbidden. At least one color remains. This proves that S is a union of K D-members.

Consequently all singletons in zero scopes, together with all(t*r_B+1)-subsets of each residual B, cover D^(K). This is an INCLUSION of the obstruction in the covered family, not equality. The extra '+1' color and '-1' previous-element count are both load-bearing.

The triangle with pair scopes of capacity1 and S={1,2,3} satisfies all doubled capacities but cannot be split into2 admissible parts. It is the smallest counterexample to dropping the '+1' in this argument when d=2.

## 3. Read-d inequality — no false scope independence

This is a standard product-space inequality (Finner/Shearer/read-k), not a new invention. A self-contained version is as follows. For nonnegative functions f_B depending only on independent coordinates in B, and weights theta_B>=0 with sum_(B contains x)theta_B<=1 for every coordinate,

    E product_B f_B^(theta_B) <= product_B (E f_B)^(theta_B).  (1)

Proof in our finite Bernoulli setting: integrate one coordinate at a time. Holder in that coordinate, with exponents1/theta_B for its incident nonzero-weight factors, gives an upper bound by the product of their coordinate integrals raised to theta_B. If the weights sum to less than1, include a constant-one factor for the unused exponent. The resulting functions depend only on the remaining coordinates; repeat. Each original f_B is integrated once in each of its own variables. The final expression is the right side. Zero-weight factors are omitted; zero functions follow by approximation or directly. This proves(1) without an independence assumption on overlapping events.

For indicators of successful positive-scope constraints and theta_B=1/d, the product on the left is their intersection indicator. If u=mu_p(D_R) and q_B is each residual failure probability, then

    u^d <= product_B(1-q_B),
    sum_B q_B <= sum_B[-log(1-q_B)] <=d[-log u].      (2)

Scope events are generally dependent. It is precisely the exponent d that pays for overlap.

## 4. Probability and cover-cost assembly

Extract the rank-zero union Z; write mu_p(D)=z*u and l0=-log z,l1=-log u. A bad S meeting Z is covered by singleton generators. Their cost at p/L is at most l0/L. On X minus Z, the symmetric-polynomial cover cost is the sum over scopes; repeated generators can be removed, and counting them several times is a safe upper bound.

For A use t=2,L=4 and PR-TAL-003(3). Every q_B<=1/(2K)<=1/4, so residual cover cost<=(3/5)sum_B q_B<=(3d/5)l1. Total cost<=l0/4+(3d/5)l1<=(3d/5)(l0+l1). The assumption bounds the last sum by1/(2d+1), proving A.

For B use t=5,L=1 and PR-TAL-003(1),(2). Every q_B<=1/K<=1/5, so residual cost<=(3d/5)l1. Add l0 and bound by max(1,3d/5)(l0+l1). The assumption bounds l0+l1 by1/(K-1)=1/(4d), proving B.

For uniform p>=1/2, the good-event assumptions exceed1/2, so the two-partition argument makes the entire obstruction empty. Arbitrary coordinates are treated by p'_i=min(p_i,1/2) and monotonicity exactly as in PR-TAL-003.

## 5. A genuinely new covered class relative to Phase04

Take the ground set to be edges of any finite hypergraph of rank at most d (parallel distinct edges allowed). For each vertex v impose a degree capacity r_v on selected incident edges. Every ground coordinate(edge) participates in at most d constraints. Hence A/B apply to these capacitated hypergraph-matching families, even with arbitrary intersection pattern among scopes.

For ordinary graph edges, d=2. The result gives3 copies with(p/4)-smallness under mu_p(D)>=5/6, and9 copies with p-smallness under mu_p(D)>=8/9.

For the complete simple graph on m>=4 vertices, its m vertex-star scopes are pairwise crossing: each pair intersects in their one common edge, and neither contains the other. Their crossing graph is K_m. Its chromatic number is unbounded although coordinate incidence is always2. Thus no FIXED number of laminar layers in the existing star representation captures this whole family, whereas this theorem does. This comparison is about representations/classes, not a claim that no alternate presentation exists for any particular downset.

## 6. Exact scope

The decomposition and read-d inequality are standard ingredients assembled here with explicit bounds. We do not claim a previously unknown greedy-coloring or Holder theorem. Novelty of the combined discrete-convexity subclass result requires specialist review.

This is not the unrestricted prize: arbitrary downsets can require unbounded incidence in their minimal-forbidden-set capacity presentation. Nor does it prove that every small-probability increasing family is p-small. The constructive cover and good-event hypothesis both matter.
