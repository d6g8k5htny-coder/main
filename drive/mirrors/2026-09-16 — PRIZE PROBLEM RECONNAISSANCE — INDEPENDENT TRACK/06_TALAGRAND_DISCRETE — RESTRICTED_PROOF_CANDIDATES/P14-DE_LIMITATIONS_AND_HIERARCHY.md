# P14-D/E — A necessary scope boundary and exact common-palette composition

Author-side proofs; no independent review or historical novelty assessment.

## D1. Fractional row width is not universally bounded

Take the independent-set family of K_(m,m), with m>=2. Describe it using one strict row (x_u+x_v)/2<1 for each edge uv. Thus a_i=1/2 and each normalized row is the indicator of its two endpoints.

A perfect matching, weight1 on each of its m edges, gives a fractional row cover of mass m. The dual assignment y_i=1/2 on all2m vertices has row load1 and objective m. Therefore tau_*=m exactly.

Yet the entire graph is2-colorable, so D^(2) is empty and has zero cover cost. The graph is connected. This is a limitation of the fractional scalar-envelope method, NOT a difficult or false case of the desired Talagrand conclusion.

## D2. Changing scalar weights cannot uniformly repair this example

Suppose ANY scalar sandwich a satisfies the hypotheses of P14-A for this same D. Every graph edge is bad; by contraposition of(S1), a_u+a_v>=1. Sum over a perfect matching: a(X)>=m. Each whole bipartition class is good. At least one of them has a-weight>=m/2, so(S2) requires kappa>=(m/2), and in fact strict kappa>max(side weights)>=m/2.

Thus even the optimized scalar-sandwich width is unbounded on a connected family with a trivial2-coloring. No universal palette theorem follows solely by insisting that every downset admit this one-dimensional sandwich with bounded width.

## D3. Other unsafe shortcuts

- Algebraic matrix rank does not certify a fractional cover. The certificate must satisfy every coordinate inequality exactly.
- A floating-point dual objective is not an upper bound for tau; primal feasibility gives an upper bound, dual feasibility gives a lower bound.
- Deleting low-probability or zero-probability coordinates can destroy SETWISE cover completeness. Zero-price generators may still be required.
- The family {a(U)<1} is only a safe sufficient packing surrogate; it need not equal D.
- Approximate coefficient sampling is not the hypothesis(FC).
- Counting a few independent profiles, or finding a feasible solution on a subset of rows/columns, does not certify the original instance unless the exact residual inequalities are also checked.

## E. Disjoint hierarchy consequence

For completeness, let X_i be disjoint; D_i and A decreasing, all containing the empty set. Define J(U)={i:U intersect X_i notin D_i} and D={U:J(U) in A}. Let J_K(U)={i:U intersect X_i notin (D_i)_(K)}.

Then

    U in D_(K)  iff  J_K(U) in A_(K).                 (E1)

Forward: a K-part good decomposition makes every hard child active in at least one color. Its hard-child set is covered by K allowed outer active-index sets; decreasingness proves the implication. Reverse: partition the hard-child indices into K outer-good sets. Put every hard child's entire selected block in its assigned color; split each soft child into K child-good parts. Disjoint supports allow the assignments to be combined without conflicts, and each color's active-index set is contained in its assigned outer-good set.

An outer generator S lifts to Cartesian unions of one child generator for each i in S. Their total description cost equals the product of the child costs because the supports are disjoint. Duplicate descriptions can only enlarge that upper count. Empty family and empty generator have different costs and must not be conflated.

Consequently, for any finite read-once hierarchy whose nonsingleton primitive families admit scalar sandwiches of widths<=kappa_*, K=ceil(408 kappa_*) is a common palette and

    covercost_p(D_root^(K))<=min(1,-log(mu_p(D_root))).  (E2)

The same applies when the sandwich is certified by fractional row covers of mass<=tau_* at each node. Arbitrary depth, row count, coefficient ratios, fan-in, and original witness rank are allowed. The width bound and original support-disjointness are essential. Repeating a variable in two AND inputs yields probability p, not p^2; copying variables does not establish disjointness.

A single primitive may contain arbitrarily overlapping resource rows, as P14-B permits. That is different from overlapping supports across child subtrees.

## What actually remains

This resolves the raw-row-count limitation for systems with a uniformly certified fractional/sandwich width. It does not remove every structural parameter or settle the unrestricted conjecture. The connected bipartite example proves that a different structured-coloring mechanism is needed when this width is large; earlier arbitrary-graph arguments supply such a mechanism for that particular class.
