# P15-D — Graph-coupled scalar modules without a bounded macro chromatic number

Status: complete author-side derivation, consuming P14-A and the exact ATTACHED P11 graph proof restated below. No external review or historical novelty conclusion. Graph substitution is established; the claimed interface is a product-measure obstruction-cover inequality, not a new definition of substitution.

## D1. Exact class and theorem

Partition finite X into disjoint nonempty blocks X_i. Let A_i be decreasing families on X_i. Let G be ANY finite simple graph on the block indices. Define

    D={U: U intersect X_i in A_i for all i,
          and {i:U intersects X_i} is independent in G}.     (D1)

Equivalently, retain the local minimal witnesses and, for every macro-edge ij, forbid EVERY pair uv with u in X_i and v in X_j. Completeness of this cross-block lifting is essential; one cannot replace it by a few cross edges while retaining the proof's probability comparison.

After separating genuine singleton-forbidden original coordinates, suppose each actual local restriction admits a P14 scalar sandwich of width<=kappa, kappa>=1. Put

    L=ceil(408 kappa),   K=256 L.

Then for all independent original probabilities p_v and all prices z_v<=phi(p_v),

    covercost_z(D^(K)) <= min(1,-log mu_p(D)).              (D2)

No bound on number of blocks, macro-graph degree/chromatic number, number of resource rows, or original witness rank is imposed. The local width bound remains essential. If there are no singleton forbidders and Q=1-mu_p(D)<=2/3, the constructed uncapped cost is at most

    (181/192) [-log mu_p(D)].                              (D3)

At kappa=1 the fixed palette is104448. It is conservative, not claimed optimal. The result does not require the local-failure tests and block-occupancy tests to be independent; they are not independent in general.

## D2. The local covers

Assume first that all original singletons are allowed and Q<=2/3. Let Q_i=P(U_i notin A_i). Since every local failure is a global failure, Q_i<=Q. P14-A's low-failure result at prices z<=2p gives a cover of A_i^(L) of cost<=Q_i/2.

The local good events ARE independent because their coordinate blocks are disjoint. Hence, writing H=-log mu_p(D),

    sum_i Q_i <= -log(product_i(1-Q_i)) <= H.              (D4)

The second inequality uses D subset intersection_i local-good. It does not assert equality with D.

## D3. Occupancy probabilities and exact generator prices

Let q_i=P(U_i nonempty)=1-product_{v in X_i}(1-p_v). These occupancy indicators are independent across blocks under the ORIGINAL product law.

The nonempty event in block i has the cover of all its singleton coordinates, price cost sum_{v in X_i}z_v. If that cost exceeds one, use the empty generator instead. Thus a finite occupancy cover has cost

    c_i=min(1,sum_{v in X_i}z_v)
       <=min(1,sum_{v in X_i}[-log(1-p_v)])=phi(q_i).       (D5)

This cost is not generally q_i. No conditioning on local goodness is performed. An empty generator has cost one and is distinct from an empty family.

Let Q_G be the probability the occupied macro-indices contain a G-edge. By the complete-transversal definition(D1), Q_G<=Q. This inference would fail for incomplete cross-block lifting.

## D4. A refined graph coloring certificate

The attached P11-A source (SHA25621c3d944ccc5dc1de3146b10ee7ddd1b5a869a0f618359e906415786b11eec24) proves that at Q_G<=2/3 and prices c_i<=2q_i, a128-coloring has monochromatic-edge cost<=(85/96)Q_G. Its generators are the actual monochromatic graph edges. For clarity, the complete estimate is:

- Peel vertices of current weighted neighborhood mass>2. Disjoint first-selected core events give 1-product_C(1-q_i)<=7Q_G/6. A convex logarithm chord on[0,2/3] gives sum_C q_i<5Q_G/2 for Q_G>0.
- At prices c<=2q, the core price sum is at most5Q_G. Give it32 colors: cost<=25Q_G^2/64<=25Q_G/96.
- In the residual weighted degree<=2. Its edge-count second moment gives edge probability mass<=5Q_G/(1-Q_G)<=15Q_G. Inflated edge prices multiply by at most4;96 disjoint residual colors cost<=5Q_G/8.
- Crossing core/residual edges cannot be monochromatic. Total is85Q_G/96. At Q_G=0 use all original zero-price edges as a one-color zero-cost certificate before invoking strict estimates.

Independently split each of these128 color classes into two colors and retain only edges whose endpoints get the same refining color. Every edge monochromatic in the new256-coloring is retained. Expected retained cost is half, so some deterministic refinement has cost<=(85/192)Q_G. Refinement is valid here because every generator is an edge, not a singleton.

By(D5), this applies to the true occupancy vector q and costs c. Lift every macro generator ij to all unions of one occupancy generator from each block. Disjoint supports make total description cost exactly c_i c_j; duplicate lifted generators may be removed. The lifted cover of the macro256-piece obstruction has cost<=(85/192)Q_G.

## D5. Product-color assembly and budget

If U avoids every local L-obstruction generator and every lifted macro256-obstruction generator, its occupied macro-indices are256-colorable and each U_i is L-decomposable. Assign a macro color to each occupied block and combine it with the local part color. There are at most256L global colors. Each color's occupied indices are graph-independent and each within-block fragment is locally good. Every part is therefore in D.

The union of these covers costs at most

    (1/2) sum_i Q_i + (85/192)Q_G
       <= (1/2+85/192)H = (181/192)H.                     (D6)

Here both sums are bounded separately by H. We NEVER multiply probabilities of the local-good and macro-good events: they share coordinates and are generally dependent.

Cap at one when needed. If Q>2/3, phi(Q)=1 and the trivial empty-generator cover suffices. If Q=0, retain zero-price minimal forbidden generators, giving cost zero.

For singleton forbidders J, first set them absent and work on the remaining blocks, deleting any empty block. The remaining class retains(D1). Good probability factors as product_{v in J}(1-p_v) times the remaining good probability. Singleton hazard costs add exactly, proving(D2) without a larger palette. A singleton at probability one triggers the trivial Q=1 case.

## D6. Why this is a controlled shared-use result

Within(D1), the same original block coordinates determine both its local failure and its occupancy. They have not been cloned into independent copies. Each probability comparison uses the true law. The theorem handles this particular repeated use through a two-part cost allocation and setwise product-color assembly.

It does not license arbitrary overlapping formulas. In particular, if only one pair uv crosses two blocks while other block vertices are almost surely present, macro occupancy can fail with probability one although original edge failure is tiny.

## D7. Composition as a certified primitive

Because(D2) holds at transformed prices z<=phi(p), a gate of exactly this class is a compatible primitive. The P14-E disjoint-substitution identity may therefore place such gates at any finite depth, provided each original coordinate occurs once across the primitive inputs and a uniform local width kappa is certified. All levels use the same K. This is distinct from repeatedly applying the block product-color construction itself and multiplying its palette again without a primitive-interface proof.

The result combines an arbitrary-chromatic graph mechanism with scalar modules, rather than requiring either global small scalar width or a bounded proper macro coloring. Historical novelty of this particular cover transfer remains unestablished.
