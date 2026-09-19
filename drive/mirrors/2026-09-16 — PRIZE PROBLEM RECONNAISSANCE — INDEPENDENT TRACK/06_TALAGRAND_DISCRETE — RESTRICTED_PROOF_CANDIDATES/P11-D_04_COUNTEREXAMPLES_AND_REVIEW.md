# P11-D — Closure boundaries, counterexamples, and author-side review

This is a mathematical scope check by the exposed author, not a blind or independent review. The proofs P11-A/B/C remain subject to external review. Neither fixed-rank existence nor generic second-moment/coloring techniques are claimed novel.

## 1. What has actually been closed at author-side proof level

P10 NEXT_WORK asked for graph-like and bounded-rank primitive gates satisfying a potential inequality at all child probabilities, not merely a constant smallness statement at one good-probability threshold. P11-A supplies that interface for minimal rank<=2 at K=64; P11-B supplies it for rank<=3 at K=262144. P11-C substitutes those interfaces into P10-A without increasing palettes with depth.

The result is about cover costs w_i <= phi(p_i), not just coordinate probabilities p_i. This is why earlier graph/rank-three smallness theorems were insufficient on their own. The inequalities use original p for random events and w for generator cost throughout.

## 2. Common palette versus overlapping intersection: exact counterexample

On vertices {0,1,2}, let D1 be the independent sets of path edges01,12, and D2 the independent sets of edge02. Both whole ground sets are unions of two allowed sets, so D1^(2) and D2^(2) are empty. But D1 intersect D2 is the independent-set family of the triangle and its full three-vertex set cannot be covered by two allowed parts.

Thus neither

    (D1 intersect D2)^(2) subset D1^(2) union D2^(2)

nor a general same-palette intersection rule is valid. Four product colors suffice; the general product-of-palettes inclusion remains valid. P10-A's common-palette identity is specific to DISJOINT SUBSTITUTION, not arbitrary intersections on the same coordinates.

## 3. Independent colors on cores and residuals cannot share labels

For one edge joining a core vertex to a residual vertex, assigning both local color0 creates a monochromatic crossing edge. A cover consisting only of internal monochromatic edges would be empty and miss it. The palettes8/56,256/768,and128/128 in P11-A/B are explicitly disjoint. Product colors combine the high-pair and low-triple decompositions only after each of those decompositions is valid.

## 4. The two-sprinkle law is exact and asymmetric

At p=1/2, two samples of parameterp/2 have union probability7/16, not1/2. The used values q=p/2 and t=p/(2-p) give q+t-qt=p exactly. The high pair is chosen from the first sample only; the second sample remains independent of this choice. Extension vertices must be distinct and outside the selected pair.

Raw pair-codegree is not weighted extension mass. Eleven extensions of individual probability1/1000 exceed a raw count threshold10 but have total mass11/1000; their completion probability is far below99/100. P11-B uses the weighted sum, not the number of extensions.

## 5. Zero-cost generators are still generators

If coordinatev has p_v=0 and is a forbidden singleton, the generator{v} has costzero but is still needed to cover sets containingv. The empty cover fails on such sets. Conversely an EMPTY GENERATOR covers every set and costsone, even on an empty ground set. Both endpoint cases are retained in the code.

## 6. All-probability versus restricted numerical examples

For graph gates we tested the actual K=64 obstruction of a65-clique and a130-leaf hierarchy of65 disjoint OR children. Those obstructions are nonempty. For the K=262144 rank-three theorem, the small enumerations primarily test exact logic, probability, and the proof's local interfaces, not a nonempty full-palette obstruction. Reduced-palette experiments only test coverage mechanics; they are not falsely claimed to prove the large-palette cost theorem.

The written all-rank-three theorem is nonvacuous: a complete3-uniform hypergraph on2K+1 vertices needsK+1 colors; sufficiently small p makes the good-probability premise hold. That enormous instance is a mathematical example, not an enumerated computational test.

## 7. What remains unclosed

No universal all-arity/all-rank gate potential theorem has been proved. No decomposition of every monotone family into the enlarged read-once hierarchy plus a cheap exceptional cover has been proved. Repeated original coordinates remain outside the product-independence proof. The broader all-rank Phase09 result and this hierarchy result have different hypotheses and neither subsumes the other without an exact comparison.

All current proofs are author-side; source priority and external independent mathematical acceptance remain unestablished. No q0 or original-prize theorem status changes follow from these derivations.
