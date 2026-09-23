# P13-W3 — Arbitrary-depth disjoint weighted hierarchies

Status: author-side corollary, conditional only on the explicitly cited author-side substitution proof P10-A and the newly written P13-W1/W2. It is not an external acceptance decision. The elementary substitution identity is restated here, so no inaccessible premise is silently consumed.

## Exact substitution identity

For pairwise disjoint blocks X_i, decreasing child families D_i containing the empty set, and a decreasing outer family A containing the empty set, put

\[
 J(U)=\{i:U\cap X_i\notin D_i\},\qquad D=\{U:J(U)\in A\}.
\]

For a positive integer K let J_K(U)={i:U\cap X_i notin (D_i)_(K)}. Then

\[
 U\in D_{(K)}\iff J_K(U)\in A_{(K)}. \tag{1}
\]

Forward: in any K-part good partition every hard block is active in at least one color; the union of the K good outer active-index sets contains J_K(U). Reverse: partition J_K(U) into K outer-good sets; put each hard block entirely into its assigned color; partition each soft block internally into K child-good pieces. Disjoint supports allow these assignments to be combined. The same K labels are used, not K times K.

If child obstruction covers have costs c_i, a generator S of an outer obstruction cover is lifted to all Cartesian unions of one child generator for every i in S. The total description cost is product_(i in S)c_i. Disjointness is essential. Duplicate descriptions can only inflate the upper bound, so deduplication is safe. Empty generators and zero-cost families retain their distinct semantics.

## The hierarchy theorem

Fix d_*>=1 and K=408d_*. Take any finite rooted monotone formula with each original coordinate occurring exactly once. Each internal node may be the OR of at most d_* positive weighted threshold failures:

\[
 F_v=1\iff\exists j\le d_v:\ \sum_i b_{vji}F_i\ge1,
 \qquad b_{vji}\ge0,\quad 1\le d_v\le d_*.
\]

Arity, depth, number of nodes, coefficient sizes and ratios, and original minimal-witness rank are arbitrary. Normalize any positive gate capacities to one; identically-good gates may be retained, while constant-bad gates must be simplified before applying the empty-set formulation.

Let D be the root-good family. Under arbitrary independent original coordinate probabilities p,

\[
 \boxed{\operatorname{covercost}_{\mathbf p}(D^{(408d_*)})
 \le\min\{1,-\log\mu_{\mathbf p}(D)\}.} \tag{2}
\]

At each node use the invariant c_v<=phi(q_v), where q_v is its original failure probability. At a leaf use its singleton generator, cost p_v<=phi(p_v). Disjoint original supports make child failures independent. P13-W2 supplies the outer certificate at prices c_i<=phi(q_i); (1) and Cartesian lifting give the node's K-obstruction certificate. If d_v<d_*, a certificate for its smaller palette covers the larger-palette obstruction. Induction on the finite tree proves (2).

Consequently mu_p(D)>=3/4 implies a no-dilution cover of cost<3/10, uniformly in depth, all fan-ins, all coefficient ratios, and maximal original witness rank. This closes P12's terminal-only weighted-gate interface at author-side proof level.

Every ordinary unweighted threshold is a positive weighted threshold, so this includes the previous threshold-only hierarchy class. It may also include other primitives already proved phi-compatible for the same K (for example the attached graph gate when K>=128). No substitution of generic inflation hypotheses by hazard-only statements is permitted.

## Multiple simultaneous rows versus repeated variables

The d rows of a SINGLE gate may overlap arbitrarily on their common child indices; P13-W2 handles that dependence by the same original product vector and a deterministic max-coefficient surrogate. In contrast, different child subtrees must still be disjoint. Repeating an original coordinate in two subtrees is not made safe by this result.

For a regression, using the same x twice in an AND gives actual failure p, not p^2. Cloning cannot make an overlapping formula read-once without changing the probability law and the decomposition identity.

## Nonvacuity

For d_*=1, K=408. A weighted root with 817 inputs whose positive weights all lie in [2/5,7/10] has every good set of size at most two. Thus the full input set cannot be covered by408 good pieces. Taking sufficiently small independent probabilities gives arbitrarily high good-event probability. Unequal weights can make some pairs good and other pairs bad, so the root need not be an unweighted cardinality threshold. Arbitrarily large witness ranks and arbitrary depths are separately allowed by the theorem; no claim that every example simultaneously exercises every parameter is made.

## Still open

This does not prove a palette independent of the number of resource rows, arbitrary shared-variable composition, or the unrestricted discrete-convexity conjecture. It does not automatically discharge q0 analytic uniformity or review gates. Historical novelty and external correctness review remain open.
