# PR-TAL-005 — Two-copy covers for capacitated bipartite edge families

2026-09-16. Complete author-side argument, no external review or novelty claim. The splitting and edge-coloring ingredient is classical bipartite multigraph coloring; the exact probability/cost assembly is supplied here. No general discrete Convexity Conjecture is solved.

## 1. Family

Let G=(U,V,E) be ANY finite bipartite multigraph, with distinct edge identities. For each vertex v specify an integer capacity r_v>=0. Put

    D={S subset E:deg_S(v)<=r_v for every v}.

No bound on vertex count, degree, parallel edges, or capacities is imposed.

The theorem, for uniform p in[0,1], is:

    mu_p(D)>=3/4 => D^(2) is(p/4)-small, cost<=2/5;
    mu_p(D)>=4/5 => D^(5) is p-small, cost<=3/10.

The same holds coordinatewise when all edge probabilities<=1/2. Arbitrary edge probabilities instead allow factors8 and2 as before.

## 2. Exact decomposition for every k

    S belongs to D_(k) iff deg_S(v)<=k*r_v for all v.       (1)

Necessity follows by counting. For sufficiency remove edges touching capacity-zero vertices, which cannot occur in S. At every vertex, distribute its incident S-edges among at most r_v clones with clone degrees<=k. This is possible because deg_S(v)<=kr_v. Each edge receives one left clone and one right clone, producing a bipartite multigraph of maximum degree<=k.

Pad the two sides to the same number of vertices and add dummy edges until the graph is k-regular: total degree deficiency is the same on the two sides, so pair any positive left and right deficiencies, allowing parallel dummy edges. Hall's condition holds in a k-regular bipartite graph since k|A| edges out of any left set A enter its neighbors, whose total capacity is at most k|N(A)|. Hall's theorem itself follows from the usual augmenting-path proof; iteratively augment until no unmatched left vertex remains (an unsuccessful augment exposes a Hall violation). Thus a perfect matching exists. Remove it and repeat to obtain k matchings. Delete dummy edges and identify the clones. Each original vertex has at most r_v edges of each color. This proves(1).

The implementation uses this actual clone/regularize/augment procedure rather than assuming that one coloring works for both endpoint constraints. A second small-instance implementation tests decomposability by exhaustive subset dynamic programming.

## 3. Cover costs

Each edge belongs to at most2 vertex scopes. Consequently the read-d calculation of PR-TAL-004 with d=2 controls the sum of positive-scope failure probabilities by2*l1. Rank-zero edge union Z is treated exactly once, with log cost l0 and mu_p(D)=exp(-l0-l1).

For the two-copy theorem, cover each positive scope by its(2r_v+1)-subsets. The coefficient bound3/5 gives residual cost<=(6/5)l1 at p/4. Zero edges cost<=l0/4. Since mu_p(D)>=3/4,

    cost <=(6/5)(l0+l1)<=(6/5)*(1/3)=2/5.

For the five-copy theorem use the(5r_v+1)-subsets with no dilution. Their coefficient is also at most3/5 under q<=1/5. The total cost is at most l0+(6/5)l1<=(6/5)(1/4)=3/10.

For uniform p>=1/2 both obstructions are empty by the two-partition argument. For arbitrary edge probabilities use coordinate capping. These are independent-coordinate product measures; no independence between adjacent vertex events is asserted.

## 4. Comparison and limits

An arbitrary pair of overlapping laminar systems had only a four-piece guarantee from Phase04's product-refinement method. For row/column partition capacities, equivalently bipartite degree-capacity systems, (1) gives an exact k-piece criterion and improves the two-piece conclusion. The graph coloring lemma is not claimed new.

For a nonbipartite triangle with capacities1, S=all three edges satisfies degree<=2 but needs3 matchings. Hence (1) with k=2 cannot be generalized to every graph. That counterexample and the crossing-scope triangle are both explicit mutation controls.
