# PR-TAL-002 — Fixed numbers of laminar layers, and the obstruction to arbitrary overlap

2026-09-16. Complete author-side derivation from PR-TAL-001; external review/novelty unresolved. This is NOT a universal-constant theorem for all downsets.

## 1. Refinement lemma

Let d>=1 and D=D_1 intersect ... intersect D_d, with each D_i decreasing. If S is a union of k members of EVERY D_i, then S is a union of at most k^d members of D.

Proof. For each i, turn its cover into a partition of S by assigning an element to one covering member; shrinking members preserves membership in D_i. Intersect all choices of partition cells across i. The at most k^d resulting sets partition S; every such cell is a subset of one member of every D_i, hence belongs to D. Empty pieces can be added to make exactly k^d members. QED.

Consequently

    D^(k^d) subset union_(i=1)^d D_i^(k).            (1)

This is an INCLUSION, not an equality. Even when D^(k^d) is empty, component obstruction covers can be nonempty.

## 2. Intersection of d laminar capacity families

Assume each D_i has a laminar capacity description. No bound is placed on its depth, number of blocks, capacities, or ground set.

**Theorem A.** Put K=2^d. Under uniform Bernoulli p, if

    mu_p(D)>=1-1/(2K),

then D^(K) is (p/4)-small. The explicit cover cost is at most

    d*B/(2^(d+1)-1) <29/60<1/2,

where B<29/20 is PR-TAL-001's exact rank sum.

Proof. For p<1/2, each D_i has measure at least1-epsilon with epsilon=1/(2^(d+1)). PR-TAL-001's stronger intermediate estimate gives a cover of D_i^(2) of cost at most B*epsilon/(1-epsilon). Take the union of these covers and use (1). Since 3d<=2^(d+1)-1, the advertised bound follows. For p>=1/2, the general two-partition argument makes D_(2)=2^X directly, so the obstruction is empty. QED.

**Theorem B.** Put K=8^d. Under uniform p, if

    mu_p(D)>=1-1/K,

then D^(K) is p-small, with explicit cost at most

    15d/[7(8^d-1)] <=15/49<1/2.

Proof. Apply PR-TAL-001's eight-copy estimate to each family with epsilon=8^(-d), followed by (1). The inequality 8^d-1>=7d is Bernoulli's inequality. The high-p case is again empty. QED.

For arbitrary independent coordinate probabilities, Theorems A/B remain valid with factors8 and2, respectively. They use the coordinate cap p'_i=min(p_i,1/2) exactly as in PR-TAL-001.

An equivalent structural condition is that the crossing graph on the constraint scopes has chromatic number at most d: two scopes are adjacent when they overlap but neither contains the other. Each color class is laminar. The proofs depend on that fixed d, not on the number of scopes within a color class.

## 3. Why the number of copies cannot simply be retained

Three pair scopes {1,2},{1,3},{2,3}, each capacity1, form three laminar layers separately. The whole three-element set satisfies every doubled capacity. But the intersection D consists of sets of size at most1, so two D members do not cover the whole set. Separate two-colorings need not be one common two-coloring.

Refining to a vector of colors is valid. Pretending all layers use the SAME two colors is not.

## 4. Why the unrestricted conjecture is not closed

The integer K grows with d. General decreasing families may require unbounded d in a capacity description; replacing a dimension-dependent complexity parameter by an exponential number of copies violates the conjecture's demand for one universal fixed number.

Two ingredients of the present proof break under arbitrary overlap:

- the exact balanced-coloring/decomposition criterion;
- independence of scopes at the same capacity after pruning.

The natural next research problem is a structural replacement for BOTH, such as a dimension-independent decomposition/uncrossing lemma plus an overlap-controlled probability budget. Merely performing the same union bound over all crossing scopes would introduce an uncontrolled multiplicity factor.

This identifies an interface for further work; it is not evidence that the required general uncrossing exists.

## 5. Concrete tests

`verify_intersections.py` checks exact product-measure cost and partition refinements over112 small cases, and checks every subset for the coverage implication. It separately rejects the false same-color and false equality inferences. The theorem for arbitrary dimension and d is the written proof above, not those tests.
