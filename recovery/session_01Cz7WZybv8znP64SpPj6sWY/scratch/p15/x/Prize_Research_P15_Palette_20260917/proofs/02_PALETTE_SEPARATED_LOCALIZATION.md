# P15-B — Palette-separated localization of obstruction covers

Status: complete author-side argument; historical novelty and independent external review unresolved. Local scalar/hazard estimates are consumed from the exact P14 sources copied under inputs/. The coloring amalgamation is proved here. We do not claim to have invented coloring or hypergraph multicoloring.

## B1. Exact finite objects

Let D be a decreasing family on finite X containing the empty set. Let H be its inclusion-minimal forbidden sets. Partition X into nonempty disjoint blocks X_1,...,X_b. Define the ACTUAL local restrictions

    D_i = D intersect 2^(X_i).

For each block choose a nonempty palette P_i subset [K] and a positive local demand k_i<=|P_i|. Require, for every minimal forbidden e meeting more than one block,

    intersection_{i:e intersects X_i} P_i = empty.         (PS)

This is checked on the block support of EACH crossing witness, not on a sampled set of witnesses. The palettes themselves may overlap. Pairwise disjointness is sufficient but not necessary.

Suppose G_i is a generator cover of D_i^(k_i), with all its generators contained in X_i. Then

    G = union_i G_i covers D^(K).                         (B1)

Proof: let U avoid G. Each U intersect X_i can be partitioned into at most k_i locally good pieces. Map their labels injectively into P_i. Merge pieces with the same global label. If a merged piece were globally bad, it would contain a minimal forbidden e. If e lies in one block, local goodness gives a contradiction. Otherwise its color would lie in every palette indexed by the blocks e meets, contradicting (PS). Thus all K merged parts are globally good.

This proof is setwise and remains valid at zero-probability outcomes. It needs only local obstruction covers, not fixed-coloring certificates. The local partitions may be chosen separately for each U.

## B2. Hazard aggregation despite crossing constraints

Select ORIGINAL coordinates independently with arbitrary probabilities p_v. Let q_i=1-mu_p(D_i). Independence on the disjoint blocks gives

    P(all local restrictions good)=product_i(1-q_i).

Since global goodness implies every local restriction is good,

    mu_p(D) <= product_i(1-q_i).                          (B2)

This is generally an INEQUALITY. Crossing forbidden sets make it strict; they do not invalidate it.

Let c_v<=phi(p_v), phi(t)=min(1,-log(1-t)), phi(1)=1. Assume the local G_i have costs at most phi(q_i), at these SAME original-coordinate prices. Then

    cost_c(G) <= sum_i phi(q_i)
              <= sum_i[-log(1-q_i)]
               = -log(product_i(1-q_i))
              <= -log mu_p(D).                          (B3)

If this sum exceeds one, replace the union by the family containing the empty generator. Therefore

    covercost_c(D^(K)) <= min(1,-log mu_p(D)).             (B4)

If some local good probability is zero, global good probability is zero and the trivial cap handles the endpoint. Zero-cost generators are retained when needed for completeness. D=empty is outside the empty-admissible formulation and trivially covered at cost one.

No independence of crossing-witness events was assumed. No copied coordinates were introduced. This is NOT the read-once substitution identity from P10: D need not be determined by the vector of local bad/good flags. Rather, crossing constraints are explicitly neutralized by (PS).

## B3. When (PS) is exact

For the fixed block palettes, (PS) is equivalent to the following uniform amalgamation property:

> Every coloring on every U that uses each block's allowed palette and is locally proper yields globally good color classes.

Sufficiency is the argument in B1. Conversely, if a crossing minimal e has a common palette color gamma, take U=e and color every vertex gamma. Each proper block fragment e intersect X_i is good by minimality, but the complete gamma-class is bad. Thus automatic safety fails.

This necessity is ONLY for safety under every allowed local coloring. A failure of (PS) is NOT a counterexample to the existence of a good global coloring or a different cheap cover.

## B4. Readily consumed local certificates

Each block may use any correctly bound compatible primitive:

- a P14 scalar sandwich of width kappa_i, with k_i=ceil(408 kappa_i);
- its exact fractional row-cover certificate;
- an already established graph/threshold compatible bound, with its exact source and hypotheses;
- an explicit proper coloring of its entire ground set using k_i colors. This last case has empty obstruction and uses the empty cover of cost zero.

A useful special case is a properly t-colored crossing SUPPORT hypergraph with all scalar widths<=kappa. Give each macro-color a disjoint group of k=ceil(408 kappa) labels. The theorem supplies K=t*k, independent of the number of blocks. Different scalar widths may use distinct demands and an exactly checked system of palettes. The parameter is the number of actually used GLOBAL labels, not the sum of all local demands.

## B5. Small exact example: overlapping palettes are genuinely useful

Take three disjoint two-vertex blocks. Forbid each internal pair and all eight transversal triples (one vertex from each block). Each local family is properly2-colorable. Give the blocks palettes {0,1}, {1,2}, {0,2}. The triple intersection is empty, although every pairwise palette intersection is nonempty. The theorem gives a proper3-coloring with empty cover. Two colors are impossible: each block would have one vertex of each color, so an all-red transversal would be forbidden. Thus the chromatic number is exactly3. Requiring all three palettes to be pairwise disjoint would unnecessarily use6 colors.

## B6. Scope and algorithmic interpretation

The source of local families is their exact restrictions; replacing one by a convenient stronger condition loses (B2) unless a separate budget is proved. Actual generator costs need not be probabilities, but local guarantees must use the same prices. Fractional assignments of color labels do not constitute integer palettes. The generic certificate checker may inspect every explicit witness; succinct families require their own proved structural formulas. No polynomial-time discovery algorithm or universally bounded palette certificate is asserted.
