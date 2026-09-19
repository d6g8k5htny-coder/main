# P11-C — Closure under graph/rank-three substitution, with one palette

Grade: author-side complete derivation, external review and novelty pending. Exact mathematical dependencies: P10-A/B (the supplied frozen common-palette and threshold-scalar proofs), P11-A/B. No q0 theorem, empirical fit, or external computational approval is consumed.

## 1. Graph-enriched hierarchy: same64 colors

Consider any finite read-once monotone formula: each original coordinate labels one leaf only. Internal nodes may be:
- any unweighted threshold gate, of arbitrary arity;
- any zero-preserving monotone gate of arity<=64;
- any zero-preserving gate with minimal obstructions of size<=2, of ARBITRARY arity.

For independent original coordinates p_v, let D be the root good family. Then

    D^(64) has a cover of p-cost <=min(1,-log mu_p(D)).

Proof. At every node maintain c_v<=phi(q_v), where q_v is ORIGINAL failure probability and c_v is a cover cost for the64-piece obstruction. Leaf c=p. Disjoint child supports give the product law of active children. P10-A uses one64-color palette throughout and composes generator costs as products on disjoint blocks. Threshold nodes consume P10-B; small gates consume P10-C's singleton-port reduction; graph gates consume P11-A at w_i=c_i. No depth-dependent factor occurs. QED.

This closes P10's explicit open interface for arbitrary-arity graph primitive gates. In particular the path-on65-ports gate formerly used to distinguish the old representation class is now admitted. This does not imply every monotone function is admitted.

## 2. Rank-three-enriched hierarchy

Replace64 by K=262144, and additionally allow arbitrary-arity minimal-obstruction-rank<=3 gates. Arbitrary gates of arity<=K may be used. The same induction, now using P11-B, gives

    D^(262144) has p-cost <=min(1,-log mu_p(D)).

P10-B works for every K>=64. P11-A also works for larger palettes since D^(K) is a subset of D^(64). P11-B supplies exactly the fixed palette chosen. Threshold fan-in, depth, total variables, number of nodes and maximal ORIGINAL witness rank may all be unbounded. No shared-coordinate independence is assumed: leaves must remain distinct.

Consequently either hierarchy satisfies the following at its stated K:

    mu_p(D)>=3/4 => cost<3/10<1/2;
    mu_p(D)>=2/3 => cost<5/12<1/2.

The first uses exp(3/10)>4/3; the second exp(5/12)>3/2. These are finite-Taylor elementary bounds, not sampled logarithms.

## 3. The graph gate is genuinely more general than a threshold

A graph gate outputs bad when two active child indices form an edge of a supplied graph. Arbitrary graphs and singleton-forbidden indices are allowed. A path on65 vertices is neither a64-input gate nor a general threshold but is now explicitly covered. The graph result also permits nonempty64-piece obstructions: a complete graph on65 ports has an obstruction on the full set, while sufficiently small p makes its independent-set probability large.

A rank-three gate may have any antichain of pairs/triples/singletons; unbounded codegree is allowed. Root witness rank may grow through disjoint composition, so the hierarchy theorem is not merely the same fixed original-rank theorem relabeled.

## 4. Cheap discrepancies and weighted intersections

If D subset D0 and D0\D is covered by an increasing generator family R of p-cost beta, then

    D^(K) subset D0^(K) union <R>.

For a member of either hierarchy and mu_p(D)>=3/4, beta<=1/5 gives cost<=log(4/3)+1/5<1/2. Arbitrary crossing exceptional witnesses are allowed ONLY with the explicit cover certificate.

More generally, if D=intersection_(i=1)^m D_i, each D_i has a cover of its K_i-obstruction costing beta_i at the SAME coordinate vector, then

    D^(product_i K_i) subset union_i D_i^(K_i),
    total cover cost <=sum_i beta_i.

The intersection palette is a PRODUCT; the disjoint-substitution palette is COMMON. The distinction is essential and has a finite counterexample in P11-D.

## 5. Scope left open

No decomposition of every downset into these hierarchies plus a cheap discrepancy is proved. Primitive gates of arbitrary minimal rank and arity remain outside the fixed-constant conclusions unless they separately satisfy the potential inequality. Neither independent review, novelty, external release nor the unrestricted prize status is promoted.
