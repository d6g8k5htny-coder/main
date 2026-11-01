# P15: actual-coordinate realizations with nonempty covers and exact palette demand

**Object:** P15-REALIZED-COVERS-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side mathematical derivation; no unrestricted prize closure or independent review.

## 1. The source interface, not a substitute hypergraph

The original source is P15-B, `P15-B_PALETTE_SEPARATED_LOCALIZATION.md`, Drive `19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j`, 6286 bytes, SHA256 `9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3`. It was read in full. P15-B requires ACTUAL local restrictions, compatible local covers at the same original-coordinate prices, and palette separation on EVERY crossing minimal original forbidden set. Main issues59/61 and the later consecutive-palette construction left that application check open in general.

This note supplies an explicit, nontrivial family for which all those inputs are proved, including NONEMPTY local generator covers. It does not claim that every existing P15 downset belongs to this family or that the previous abstract examples already specified these original coordinates. The price range is deliberately stated as 0<=c_v<=p_v, a subset of P15-B's possible transformed-price range c_v<=phi(p_v). No assertion is made here for the additional prices p_v<c_v<=phi(p_v).

For a decreasing family D, write I_k(D) for sets partitionable into at most k members of D, and O_k(D)=2^X\I_k(D) for its obstruction. A generator cover G of O_k(D) means every obstructed set contains some g in G. A generator's price is product_(v in g)c_v, and family price is the sum of generator prices. The empty generator has price1; the empty family has price0. This notation makes explicit the obstruction sense of the original D^(k) cover statement.

## 2. A complete original-coordinate construction

Let H be a clutter on b macro-indices: distinct nonempty edges, no edge properly containing another. Require every edge to have size at least2. Empty H is allowed. Choose positive integers a_i,d_i and disjoint original-coordinate blocks X_i with

    n_i=|X_i|=a_i d_i+1.

Define the downset

    D={U subset X: |U intersect X_i|<=a_i for every i,
                   supp(U) contains no edge of H},
    supp(U)={i:U intersect X_i is nonempty}.              (P1)

**Proposition P1.** The inclusion-minimal forbidden ORIGINAL sets are exactly

(a) every (a_i+1)-subset of a single X_i;
(b) for each e in H, EVERY transversal {x_i:i in e}, with x_i in X_i.

Every set in (a) is minimal, because its proper subsets fit the local capacity and occupy one block. A set in (b) fits every local capacity since a_i>=1; removing a vertex leaves a proper subset of e, which contains no H-edge by the clutter property. Thus it is minimal. Conversely, any bad set either exceeds a local capacity and contains a set in (a), or has support containing an H-edge and contains a transversal from (b). This proves completeness, not just existence of some witnesses.

In particular the crossing-support hypergraph is EXACTLY H, and the actual local restriction is

    D_i=D intersect 2^(X_i)={S subset X_i: |S|<=a_i}.     (P2)

The restriction does not inherit an artificial stronger condition. Neither a sampled list of transversals nor an incomplete lifting can replace (b).

## 3. Real, positive-price local obstruction covers

The k-fold decomposability of a capacity family is exact:

    I_k(D_i)={S:|S|<=a_i k}.

Necessity follows by summing the k capacities; sufficiency partitions S into successive groups of at most a_i elements. For k=d_i and n_i=a_i d_i+1, the only obstructed subset of X_i is X_i itself. Hence

    G_i={X_i}                                           (P3)

is an exact local obstruction cover. It is not the empty family arising from proper colorability of the entire block with d_i colors.

Select the ORIGINAL coordinates independently with arbitrary probabilities p_v in [0,1], and let 0<=c_v<=p_v. With q_i=P(U_i notin D_i),

    price(G_i)=product_(v in X_i)c_v
       <=product_(v in X_i)p_v
       =P(U_i=X_i)<=q_i<=phi(q_i),                      (P4)

where phi(t)=min(1,-log(1-t)), phi(1)=1. When all c_v>0 the local price is positive. When a zero price occurs the generator is still retained for setwise coverage.

The independent-block calculation uses (P2), so

    mu_p(D)<=product_i(1-q_i),
    sum_i price(G_i)<=sum_i q_i
        <=-log(product_i(1-q_i))<=-log mu_p(D).           (P5)

If some q_i=1 the trivial price-one cover handles the endpoint. When the summed price exceeds1 use {empty set}; otherwise retain union_i G_i. This produces the capped hazard bound whenever the palette condition below holds. Crossing events are not declared independent, and no coordinates are cloned.

## 4. The exact number of colors needed for this cover

Let K_H(d) be the minimum size K of an integer palette universe admitting subsets P_i with |P_i|>=d_i and

    intersection_(i in e) P_i = empty, for every e in H. (P6)

Equivalently each label's incident macro-indices form an independent set of H. By discarding excess labels within blocks, one may take |P_i|=d_i. This is the actual macro-demand coloring problem; fractional palettes are not used.

**Theorem P2.** For D in (P1), the fixed family G={X_1,...,X_b} covers O_K(D) if and only if

    K>=K_H(d).                                          (P7)

For sufficiency take a set U avoiding all generators. Then |U_i|<=a_i d_i. Split U_i into at most d_i groups of size<=a_i and assign them distinct labels from P_i. A global label's block support is H-independent by (P6), and it uses at most a_i vertices in each block. Each resulting color class is in D. Thus U is K-decomposable.

For necessity choose U with |U_i|=a_i d_i in EVERY block, omitting one original vertex per block. It avoids every generator X_i. Any D-coloring of U needs at least d_i distinct colors in block i because of its capacity a_i. If a label occurs in every block of an H-edge, completeness of the transversal forbidders makes that label class bad. Thus its used-color sets form palettes satisfying (P6); at least K_H(d) colors are necessary. For K<K_H(d), this U is an uncovered obstruction.

Combining (P4)-(P7), for K>=K_H(d),

    covercost_c(O_K(D)) <= min(1,-log mu_p(D)).            (P8)

Optimality in (P7) is for the SPECIFIED union-of-block-generators cover. It does not prove that no different cover with the same budget works at fewer colors. It is not unrestricted prize optimality.

For comparison, the chromatic number of the ENTIRE ground set X is K_H(d+1), because ceil(n_i/a_i)=d_i+1. The same palette argument gives both inequalities. Therefore cover demand and whole-ground colorability are distinct quantities.

## 5. Realized structural subclasses

For any H whose independent sets form a loopless matroid, the established matroid palette formula in main issue59 can be applied to this actual family. It is not necessary to establish a new matroid theorem here. For a consecutive-capacity presentation, the corresponding consecutive-palette theorem can similarly be consumed only when its exact complete-support hypotheses hold.

A fully elementary case is H consisting of all (s+1)-subsets of b indices, 1<=s<=b. Each label may occur in at most s blocks, and

    K_H(d)=max(max_i d_i, ceil(sum_i d_i/s)).             (P9)

The two lower bounds follow from one block and from total label incidences. For the matching construction, concatenate segments of lengths d_i along the cyclic word 0,1,...,K-1. Each block has distinct labels because d_i<=K. Each label occurs at most ceil(sum d_i/K)<=s times. This proves (P9) without enumerating individual labels when demands are large.

The macro triangle at unit demands needs3 colors, although a formula considering only size-two edge loads would give2. The theorem uses K_H itself or a proved structural method, not an unrestricted edge-load formula.

## 6. A nonempty-cover realization of the 816-label benchmark

Take six blocks, a_i=1 and d_i=408. Thus EACH original block has409 vertices and the original ground set has2454 vertices. H consists of all15 four-block subsets. Every D-color class contains at most one vertex per block and occupies at most three blocks.

The original minimal forbidders are exactly

    6*binom(409,2)=500616 internal pairs,
    15*409^4=419743994415 crossing transversal quadruples. (P10)

These are counts obtained from a proved complete structural formula, not a claim that the program enumerated more than four hundred billion sets.

Use408 labels on macro-indices {0,2,4} and408 different labels on {1,3,5}, giving

    K_H(408,...,408)=816.

The actual cover has six generators, the six FULL409-vertex blocks. At816 colors it covers the obstruction; at815 it does not, because omitting one vertex from each block leaves2448 vertices, each color holds at most3, and815*3<2448. Pairwise-disjoint local palettes would have used2448 labels. The whole ground set, unlike the cover-avoiding sets, needs818 colors, since2454/3=818.

This is a NEW specified original-coordinate family instantiating the previous abstract benchmark; it is not a statement that every earlier six-block application had409 vertices per block or these complete transversals.

A concrete low-failure, positive-price instance is p_v=c_v=1/40900. The cover price is exactly

    6/(40900^409)>0.

A union bound on internal pairs and four-block occupancy gives

    Q=1-mu_p(D)
      <=500616/(40900^2)+15/(100^4)
       =2449227/8180000000 < 3/10000.                    (P11)

For the second term each block's occupancy probability is at most409p=1/100, and occupancies of distinct blocks are independent. This is only a union bound, not an equality of the global bad probability. Thus the hazard budget is not being satisfied solely in the trivial high-failure regime; the actual-good probability exceeds0.9997. The cost comparison (P5) supplies (P8), rather than comparing this positive price to an unproved numerical hazard lower bound.

## 7. Exact verification and boundaries

The executable finite checker constructs actual block coordinates for small cases. It independently enumerates minimal bad subsets, tests equality with (a)-(b), computes actual restrictions, compares original-coordinate colorability with residual-demand dynamic programming, and checks every cover-avoiding subset in declared bounded examples. A removed transversal is a negative control: its now-good original set still violates the macro support condition, so incomplete lifting is detected. The large816 example uses the proved formulas (P9)-(P11), not an unbounded enumeration.

The original P15-D source (Drive `1z-vrL-f5tDBbqybco-ZiS7X9vJq70ifA`) was also read. It treats complete GRAPH-transversal coupling, arbitrary macro chromatic number and a different scalar/hazard budget. This note does not improve its256 factor or extend it to arbitrary hypergraphs at the same fixed palette. It supplies a separate actual-realization theorem whose palette remains K_H(d).

The unrestricted P15 prize, full transformed-price range, arbitrary input support identification, and nonauthor review remain open. Classical coloring, hypergraph demand coloring, union bounds and the source P15-B amalgamation are not claimed as inventions. The contribution supplied here is the exact compatible original-coordinate family, nonempty covers, and source-hypothesis verification that were absent from a merely abstract palette output.
