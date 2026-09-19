# P09-E — Exact advance, established overlap, and remaining problem

## 1. No new bounded-rank existence claim

Park--Pham, *A Proof of the Kahn--Kalai Conjecture*, arXiv:2203.17207, establishes p_c(I) <= C q(I) log ell(I). The role of ell as the maximal minimal-witness size, at least two, is also restated in Ascoli--He--Park--Talagrand, arXiv:2608.11183v1, Remark 1.6.

For uniform p with mu_p(D)>=3/4, I=D^c has mu_p(I)<=1/4. Hence p<p_c(I) for a nontrivial family. If ell(I)<=max(2,r), this gives a cover of I itself at parameter p/[C log max(2,r)]. Therefore bounded-rank smallness in the qualitative sense was available before this campaign. The ordinary obstruction I=D^(1) is larger than D^(k), so it covers those smaller obstruction families too.

We must not characterize Phases07--09 as the first resolution of the bounded-rank subclass. They are local direct derivations with specified quantitative tradeoffs. The comparison does not refute their proofs, but it changes their novelty and priority interpretation.

## 2. The actual tradeoff studied here

Prior general result: one piece is enough when dilution is allowed to grow logarithmically with witness rank, up to its universal theorem constant.

P09-D: dilution is bounded by 6 independent of rank while pieces grow as

    1600 * 4^(r-3) * (r!/6)^5.

P09-C: at rank four, 6,400 pieces and dilution 8.

We have not identified a literature record or optimality for these constants. A focused search is not a priority certificate. The nonuniform product extension requires a separate literature comparison rather than silently importing a uniform-p theorem into that setting.

## 3. The unrestricted quantifier gap

What we prove:

    for every finite r, for all rank-at-most-r families, K_r pieces suffice.

What the unrestricted discrete convexity conjecture requests:

    there exist fixed K and L that work for all finite families, with no rank-dependent choice.

These are not equivalent. Every finite family has finite rank, but that observation does not interchange the order of the quantifiers.

Our summable sprinkling schedule removes cumulative dilution as an obstacle in this particular construction. The remaining cost is repeated product-color refinement. Bounding or replacing that accumulated color complexity is a precise next target.

## 4. Concrete failed shortcuts

A. Large expected extension count does not imply large extension-hit probability. A link star with a rare center can have many expected edges while all hit events share that rare center. The high-core admission condition must use actual completion probability or a proved lower bound for it.

B. A link-mass bound computed at t=p/r^2 cannot be used unchanged at z=p/L_r. For an r-edge its weight changes by (r^2/L_r)^r.

C. Two proper k-colorings for two constraint families do not generally combine into one proper k-coloring of their union. Ordered-pair refinement is safe; replacing its product count by a maximum is false. On four vertices, two bipartite graphs can have union K4.

D. A cover of cost 0.49 is not a proof that the covered bad event has probability zero. A small cover is allowed to overcover and leave a nonzero obstruction.

E. A test program whose only guards are Python assert statements does not remain fail-closed under python -O. Phase08's isolated corrupted copy passes in optimized mode. This is a test-integrity failure, not a refutation of the rank-three statement.

## 5. Bounded next work

First obtain an external mathematical review of the probability-core induction and all-rank quantifiers. Our test code checks finite interfaces; it cannot supply that external review.

Then examine whether the coloring used at one rank can be chosen compatibly with the next rank, rather than multiplying palettes. A valid argument must preserve the core-to-edge implication for every original edge and must not merely assume that two separately proper colorings align.

An alternative is to compare this induction with existing fractional-smallness tools and identify a fractional certificate that has a uniformly bounded support/cost complexity. Frankston--Kahn--Park's pair-supported theorem can only be consumed once its actual fractional premise is established.

Do not spend the next campaign refining the decimal cost cap. The advance needed is structural, not arithmetic.
