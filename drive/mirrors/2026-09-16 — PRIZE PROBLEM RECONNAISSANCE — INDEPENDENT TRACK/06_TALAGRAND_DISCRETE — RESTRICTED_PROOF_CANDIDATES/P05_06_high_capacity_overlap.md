# PR-TAL-008 — Twenty copies under an exponential overlap condition

2026-09-16. Complete author-side argument; no external independent review; historical novelty UNESTABLISHED. Standard ingredients are the asymmetric Lovasz Local Lemma and fractional Holder/Finner. They are not new inventions. Unlike PR-TAL-004, this theorem allows UNBOUNDED coordinate incidence, but imposes the explicit capacity-weighted condition below. It is still not the unrestricted discrete Convexity Conjecture.

## 1. Statement and scope

Extract rank-zero coordinates Z and prune vacuous/implied constraints as in PR-TAL-003. Suppose every remaining positive capacity obeys r_B>=100 and

    eta := max_(x not in Z) sum_(B contains x) (3/4)^(r_B) <= 1/200.  (H)

There is no bound on the ground-set size, number of constraints, crossing pattern, scope sizes, or coordinate incidence. The capacities and overlaps must satisfy (H).

**Theorem.** For uniform p in[0,1],

    mu_p(D)>=19/20  =>  D^(20) is p-small,

with a generator cover of total p-cost at most1/19<1/2. For unequal independent probabilities all<=1/2, the same conclusion holds. For arbitrary independent coordinate probabilities,

    mu_p(D)>=21/22  =>  D^(22) is p-small,

with cover cost at most1/21, by PR-TAL-007.

These constants do not grow with coordinate incidence within class(H). For any finite incidence d, (H) is guaranteed if every r_B>=R>=100 and d*(3/4)^R<=1/200; however the theorem uses the actual weighted sum and does not require a uniform capacity. This is an all-dimensional restricted-class theorem, not a proof of(H) for all downsets.

## 2. Deterministic decomposition by a random-coloring existence proof

Assume S avoids Z and |S intersect B|<=5r_B for every positive scope. Color its elements independently with20 colors. Let E_B be the event that some color has more than r_B members inside B. A union bound over colors and(r_B+1)-subsets gives

    P(E_B) <=20*binom(5r_B,r_B+1)*20^(-(r_B+1))
           <=20*(e/4)^(r_B+1)
           <=20*(17/25)^(r_B).                         (1)

Here binom(n,j)<=(en/j)^j follows from j!>=(j/e)^j; the latter follows by integrating log x. The elementary bound e<68/25 was established rationally in PR-TAL-003. Replacing5r_B by an upper bound on the actual integer scope count is safe.

The variable-dependency graph joins B,C if their intersections with S overlap. Set x_B=(3/4)^(r_B)<1/2. By(H),

    sum_(C neighbor B) x_C <= |S intersect B| eta <=r_B/40.

Since log(1-x)>=-2x for0<=x<=1/2,

    product_(C neighbor B)(1-x_C) >=exp(-r_B/20).

Using exp(1/20)<=1/(1-1/20)=20/19, equations(1) yield

    P(E_B)/[x_B product_neighbors(1-x_C)]
       <=20*[(68/75)*exp(1/20)]^(r_B)
       <=20*(272/285)^(r_B)<1.                         (2)

The last inequality follows from the exact integer comparison20*272^100<285^100 and the ratio272/285<1. Thus it holds at EVERY integer rank>=100, not only sampled ranks.

The asymmetric Local Lemma implies a coloring with no E_B. Every color class then belongs to D. This proves

    {S avoiding Z: all |S intersect B|<=5r_B} subset D_(20).

Consequently the zero singletons and all(5r_B+1)-subsets of every residual B cover D^(20). This is an inclusion, not an exact characterization of D_(20).

## 3. Self-contained finite Local Lemma used in Section2

For events E_i with a dependency graph in which E_i is independent of the joint sigma-algebra of its nonneighbors, suppose0<=x_i<1 and

    P(E_i)<=x_i product_(j neighbor i)(1-x_j).

Induct on the size of S to prove P(E_i | all E_j^c,j in S)<=x_i whenever the conditioning probability is positive. Split S into neighbors S1 and nonneighbors S2. The numerator is at most P(E_i | all E_j^c,j in S2)=P(E_i). The denominator P(all E_j^c,j in S1 | all E_j^c,j in S2) is, by a sequential chain and the smaller-set induction, at least product_(j in S1)(1-x_j). Division gives the desired bound. The induction also establishes positivity of each finite conditioning intersection. Applying the chain to all complements gives probability at least product_i(1-x_i)>0. This proves the existence conclusion.

Independent color variables supply the required joint nonneighbor independence. Pairwise event independence alone would not suffice. The numerical companion produces actual finite colorings for sample instances but does not replace this all-instance proof or claim a generally efficient deterministic coloring algorithm.

## 4. Probability budget for the witness cover

For p_i<=1/2 and mu_p(D)>=19/20, every residual scope failure probability q_B is at most1/20<1/5. The positive-rank bound of PR-TAL-003, because r_B>=100>=5, gives

    e_(5r_B+1)(p_B) <= (3/5)^(r_B) q_B.

The weighted Finner budget of PR-TAL-006 applies with a_B=(3/5)^(r_B). Its maximum coordinate load obeys

    Lambda=max_x sum_(B contains x)a_B <=eta<=1/200.

Let l0=-log P(no Z), l1=-log mu_p(D_R), so l0+l1=-log mu_p(D). The full cover cost is at most

    l0 + Lambda*l1 <=l0+l1 <=(1/20)/(19/20)=1/19.

If there are no positive scopes, Lambda=0 and the zero-singleton cover alone proves the statement. No division by a zero load or zero failure probability occurs.

For uniform p>=1/2 the general two-partition lemma makes D_(2)=2^X, so the obstruction is empty. For arbitrary probabilities, split coordinates at1/2 as in PR-TAL-007 and reserve two pieces for the heavy part. Restricting scopes to the light part does not increase eta; newly vacuous constraints are deleted, and the remaining capacities stay>=100. With the stronger assumption mu_p(D)>=21/22, the light cover costs at most1/21. This gives22 pieces without probability dilution.

## 5. A nonvacuous family with incidence tending to infinity

For each integer m>=100, let C have30m coordinates and introduce additional coordinates y_1,...,y_m. Use scopes B_i=C union{y_i}, each of capacity m. Every core coordinate is in m scopes, and every pair of scopes crosses. Nevertheless eta=m(3/4)^m<=100(3/4)^100<1/200: the sequence decreases for m>=100.

The20-piece obstruction is nonempty because S=C has30m coordinates and each feasible piece contains at most m core coordinates. It cannot be covered by20 feasible pieces.

At uniform p=1/(100m), a scope occupancy Y has mean lambda=(30m+1)/(100m)<=31/100. Markov applied to Y(Y-1) yields

    P(Y>=m+1)<=lambda^2/[m(m+1)].

The union bound over m scopes gives failure at most(31/100)^2/(m+1)<1/20. Thus the theorem's probability hypothesis holds, its bad family is nonempty, and incidence grows without bound. No limiting or infinite-ground-set measure is being used: this is one finite system for each m.

## 6. Why this is a useful but incomplete extension

PR-TAL-004 replaces laminarity with bounded incidence but lets the number of pieces grow with that incidence. The present theorem keeps20 pieces while allowing incidence to grow when capacities suppress the weighted load. Laminarity and bounded crossing-graph chromatic number are not assumed.

Both necessary interfaces are controlled by the SAME structural quantity: the Local Lemma uses eta to prove decomposition, and Finner uses the smaller Lambda to bound cover cost. Without(H), neither step is justified by this argument. In particular rank-one common-cause systems and arbitrary minimal-forbidden-set presentations need not satisfy(H). Duplicating coordinates, inflating capacities, or changing the probability parameter would require a separate measure-preserving reduction; none is asserted.

This is a restricted theorem derived from classical tools, not a claim of historical novelty or a solution of Talagrand's original unrestricted problem.
