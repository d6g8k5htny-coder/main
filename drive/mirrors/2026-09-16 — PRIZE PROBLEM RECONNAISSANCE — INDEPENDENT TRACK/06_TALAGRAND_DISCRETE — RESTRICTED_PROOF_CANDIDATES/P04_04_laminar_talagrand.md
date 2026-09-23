# PR-TAL-001 — Explicit dimension-free covers for laminar capacity families

2026-09-16. Complete author-side argument below with exact finite companions. External review and historical novelty are UNESTABLISHED. This is a structural subclass theorem, NOT the full discrete Talagrand conjecture. Definitions are matched to Ascoli--He--Park--Talagrand, arXiv:2608.11183v1, Definition1.1 and Conjecture1.2; their general theorem is not assumed as an input.

The constraint class is the standard laminar-matroid independent-set class (Fife--Oxley, arXiv:1606.08354); laminar presentation and redundancy removal are prior mathematical structures, not inventions of this package.

## 1. Definitions and statements

Let X be any finite ground set. A collection L of subsets is laminar when any two are disjoint or one contains the other. Give B in L a nonnegative integer capacity r_B and set

    D={S subset X : |S intersect B|<=r_B for every B in L}.

There is no bound on |X|, |L|, the capacities, or nesting depth. This is a decreasing family. Put D_(k)={union of k members of D} and D^(k)=2^X\D_(k).

For a vector t=(t_i), a generator G has product cost t^G=product_(i in G)t_i. A family I is t-small when it is covered by supersets of generators with total cost at most1/2. For a constant vector t this is sum_G t^|G|, the standard p-small notion. Bernoulli coordinates are always independent.

**Theorem A (two copies, constant dilution).** For uniform p in [0,1], if mu_p(D)>=3/4, then D^(2) has a cover of cost at most29/60<1/2 at parameter p/4.

**Theorem B (eight copies, no dilution).** For uniform p in [0,1], if mu_p(D)>=7/8, then D^(8) has a cover of cost at most15/49<1/2 at parameter p.

These verify the respective modern and original formulations of the conjecture on this laminar subclass. They make NO assertion for arbitrary decreasing families.

More generally, if all p_i<=1/2, the same conclusions hold coordinatewise at p_i/4 and p_i. For arbitrary independent p_i in [0,1], the conclusions hold at p_i/8 and p_i/2, respectively. Uniform high p is handled separately and loses no extra factor.

## 2. Normalize redundant constraints

Drop vacuous constraints r_B>=|B|. For equal scopes keep the least capacity. If B is properly contained in C and r_C<=r_B, the C constraint implies the B constraint; discard B. This preserves D and every enlarged-capacity family with capacities k*r_B.

After these operations, scopes with the same capacity are pairwise disjoint. Indeed laminarity allows only containment or disjointness, and a contained equal-capacity scope would have been discarded.

If a removed constraint is implied by another removed constraint, follow the strictly increasing inclusion chain until a retained constraint is reached. The procedure is finite, includes rank-zero constraints, and does not assume all different ranks are independent.

## 3. A simultaneous balanced-coloring lemma

For any S subset X and any integer k>=1, there is a k-coloring of S such that every laminar scope contains color-class sizes differing by at most1.

Proof. View the laminar family as a forest; append X as an unconstrained root if necessary. Induct upwards. Each child has a balanced color vector. Its baseline count is the same in every color, with a surplus1 on some colors. Permute the child's colors so those surpluses go to the currently least populated aggregate colors. A balanced aggregate remains balanced after this operation. Each selected element not in a child is treated as a one-element child. A permutation of all colors within one subtree preserves every balance property already proved in it. This proves the lemma at every node and the root. QED.

If |S intersect B|<=kr_B, balance gives at most ceil(|S intersect B|/k)<=r_B elements of each color in B. Conversely a union of k admissible sets has at most kr_B elements in B. Therefore

    D_(k)={S : |S intersect B|<=kr_B for every B}.     (1)

A cover of D^(k) is obtained by taking ALL (kr_B+1)-subsets of each B. Duplicate generators need not be counted twice; counting them multiple times only gives a safe upper estimate.

For nonuniform p, the cost contribution of B at p/L is

    L^(-(kr_B+1)) e_(kr_B+1)(p_B),                    (2)

where e_j denotes the j-th elementary symmetric polynomial in the coordinates belonging to B. If j>|B|, this is zero.

## 4. A probability budget for each capacity layer

First suppose all p_i<=1/2 and mu_p(D)>=1-epsilon. Let

    q_B=P(Z_B>r_B),  Z_B=|X_p intersect B|,
    lambda_B=E Z_B=sum_(i in B)p_i.

Every q_B<=epsilon. For a fixed capacity r the scopes are disjoint, so their failure events are independent. Since D is contained in the event that all these particular constraints hold,

    product_(r_B=r)(1-q_B)>=1-epsilon.

Taking reciprocals and using product(1+q_B/(1-q_B))>=1+sum q_B gives

    sum_(r_B=r)q_B <= epsilon/(1-epsilon).             (3)

This is the entire role of laminarity in the probability step. It is not an assumption of independence among nested constraints.

## 5. Single-block tail-to-cover inequality

Fix one block with capacity r and mean lambda. For independent Bernoulli coordinates <=1/2,

    q=P(Z>=r+1)>=P(Z=r+1)>=e_(r+1)(p)*exp(-2lambda).  (4)

Indeed the exact atom is product(1-p_i)e_(r+1)(p_i/(1-p_i)); the odds only increase each symmetric coefficient, and log(1-p_i)>=-2p_i.

For nonnegative variables,

    e_a e_b >= binom(a+b,a)e_(a+b),
    e_b <= lambda^b/b!.

The first follows by retaining disjoint index selections in the product; each (a+b)-subset is obtained binom(a+b,a) times. Thus for k>=2,

    e_(kr+1)(p)
      <= q exp(2lambda)lambda^((k-1)r)
                     *(r+1)!/(kr+1)!.               (5)

We also need control of lambda from q. If lambda>r, Cantelli's inequality and Var(Z)<=lambda give

    q>= (lambda-r)^2/[lambda+(lambda-r)^2].            (6)

For completeness, the one-sided variance inequality follows by applying Markov to (Z-lambda-t)^2 on Z-lambda<=-a and minimizing over t>=0; this yields P(Z-lambda<=-a)<=Var/(Var+a^2). With a=lambda-r, (6) follows. When lambda<=r the upper-mean bounds used below hold automatically.

If q<=1/4, (6) implies 3(lambda-r)^2<=lambda whenever lambda>r. For r=0 it gives q>=lambda/(1+lambda), so

    lambda<=q/(1-q).                                  (7)

All formulas remain valid at q=lambda=0. We never divide by a zero q.

## 6. Proof of Theorem A for p_i<=1/2

Take epsilon<=1/4 and k=2. For r=0, the cost at p/4 is lambda/4<=q/3.

For r=1,...,6, the following u_r satisfy 3(u_r-r)^2>=u_r and u_r>=r:

    r:   1     2     3      4      5      6
    u:  9/5    3    21/5   16/3   13/2   38/5.

The positive root of the quadratic is at most u_r, hence lambda<=u_r. Define the EXACT rational coefficients

    a_r=(68/25)^ceil(2u_r) * u_r^r
                         * (r+1)!/[(2r+1)!4^(2r+1)].  (8)

We use e<68/25, obtained from sum_(j=0)^6 1/j! plus the tail bound 1/4410. Equation(5) then bounds the cost by a_r q. The six exact rational comparisons give

    a_1<514/1000, a_2<178/1000, a_3<176/1000,
    a_4< 62/1000, a_5< 23/1000, a_6< 21/1000.

These six small, fully specified rational inequalities are verified by the included Fraction code; no decimal transcendental evaluation is consumed.

For r>=7, (6) implies lambda<=5r/4. At lambda=5r/4 the quadratic condition would require r<=20/3, so larger lambda is impossible. Also

    (2r+1)!/(r+1)! >= (sqrt(2)r)^r.

To verify this, pair the r factors r+2,...,2r+1 from opposite ends. Every paired product exceeds2r^2; any unpaired middle factor exceeds sqrt(2)r.

Since e^(5/2)<13 and 65/(4sqrt(2))<12, equation(5) implies

    e_(2r+1)(p)/4^(2r+1) <= q*(1/4)*(3/4)^r.          (9)

The two numerical inequalities can be checked algebraically from (68/25)^5<169 and 65^2<2*48^2.

Sum the capacity layers using (3). Let

    B=1/3+sum_(r=1)^6 a_r+(3/4)^7.

The last term is exactly sum_(r>=7)(1/4)(3/4)^r. The preceding rational caps give

    B <1/3+974/1000+(3/4)^7 <29/20.

Therefore the total cover cost is

    cost <= B*epsilon/(1-epsilon) < (29/20)/3=29/60.

This proves Theorem A on the entire arbitrary-rank, arbitrary-depth family for p_i<=1/2. The infinite geometric estimate is a proof for all capacities; tests at finitely many r do not replace it.

## 7. Proof of Theorem B for p_i<=1/2

Now epsilon<=1/8 and k=8, with no dilution. For r=0, equation(7) yields lambda<=8q/7.

For r>=1, equation(6) and q<=1/8 imply lambda<2r. Equation(5) gives

    e_(8r+1)(p) <= q exp(4r)(2r)^(7r)(r+1)!/(8r+1)!.

Using the increasing function log x,

    log[(8r+1)!/(r+1)!]
      >=integral_(r+1)^(8r+1)log x dx
      >=integral_r^(8r)log x dx
      =7r log r+8r log8-7r.

Consequently

    e_(8r+1)(p) <= q [e^11/2^17]^r <q*2^(-r),        (10)

where (68/25)^11<2^16 supplies the exact elementary bound. Summing rank layers with (3),

    cost <= [8/7+sum_(r>=1)2^(-r)]*epsilon/(1-epsilon)
         = (15/7)*epsilon/(1-epsilon)
         <=15/49<1/2.

This proves Theorem B for small individual probabilities.

## 8. Uniform high p and arbitrary product measures

For UNIFORM p>=1/2, a general decreasing family with mu_p(D)>1/2 satisfies D_(2)=2^X. Randomly partition X into two colors; each color has law mu_(1/2), hence probability greater than1/2 of belonging to D. The union bound makes the probability that BOTH belong to D positive. Such a partition covers X by two members of D. Downward closure gives every subset the same property. Both hypotheses of Theorems A/B imply this; D^(2) and D^(8) are then empty.

This argument uses p>=1/2, NOT p>=1/k. At the boundary mu_p(D)=1-1/k, a random k-partition union bound can be zero and cannot justify an empty obstruction. Example: X={x}, D={empty}, p=1/8 has mu_p(D)=7/8 and nonempty D^(8); its legitimate cover cost is1/8. This exact counterexample is a regression test.

For arbitrary nonuniform p_i, put p'_i=min(p_i,1/2). Monotonicity gives mu_(p')(D)>=mu_p(D). Since p_i/8<=p'_i/4 and p_i/2<=p'_i, the low-coordinate theorems supply the claimed factors8 and2. No independence is asserted for overlapping scope events; coordinate independence alone is retained.

## 9. Exact certificate algorithm

A cover can be stored compactly as the pairs (B,kr_B+1), meaning all subsets of that size in B. Its exact rational cost is computed by symmetric-polynomial dynamic programming without enumerating the cover. A laminar-tree generating-function recursion computes mu_p(D): multiply child count polynomials and uncovered Bernoulli factors, and truncate at EACH node's capacity. Coefficients remain unnormalized, so probabilities are not accidentally conditioned midway.

For arbitrary rational input the checker verifies containment/type conditions and exact cost. Small tests independently enumerate all sets, actual k-fold unions, and generator subsets. Those numerical tests do not prove the arbitrary-dimension theorem; Sections2--8 do.

## 10. Scope and priority

Laminar capacity systems include partition capacities and nested quotas, but not arbitrary downsets. Three crossing pair constraints of capacity1 on a3-element ground set have each doubled capacity satisfied by X, yet D contains only empty/singleton sets and two members cannot cover X. Thus (1) fails without its structural hypothesis.

The proof uses standard coloring, Bernoulli, factorial, and capacity-layer ideas. No historical originality has been established. Primary Talagrand literature was checked to freeze the target; absence of the word 'laminar' in two recent papers is not a novelty certificate. The continuous conjecture, the discrete conjecture, and the fractional expectation-threshold conjecture are distinct. This result does not close any unrestricted prize statement.
