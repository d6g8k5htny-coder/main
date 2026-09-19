# Phase07 next work v2

PR-TAL-015 closes arbitrary capacity-one overlap at author-side proof
level with 11 pieces and no dilution under
(`\mu`{=tex}\_{`\mathbf `{=tex}p}(`\mathcal `{=tex}D)`\ge3`{=tex}/4). Do
not spend the next campaign trying to solve graph overlap again unless
independent review breaks PR-TAL-015.

Primary frontier:

1.  **Rank \>=3 high-dependency witness hypergraphs.** PR-TAL-016
    already handles bounded directed dependency load.
2.  Develop a greedy common-cause extraction theorem for hyperedges
    analogous to PR-TAL-015. Candidate cores may be vertices, pairs, or
    smaller intersections of many witnesses.
3.  Search for counterexamples showing that vertex-only extraction is
    insufficient in rank 3.
4.  Test sunflower/common-core decompositions: if many high-dependency
    witnesses share a small intersection, pay once for that intersection
    even when it is not itself a violation, because p-small covers may
    overcover.
5.  Compare every pair-supported reduction against Frankston--Kahn--Park
    before claiming novelty.
6.  Attempt a rank-3 theorem before arbitrary rank.
7.  Preserve exact separation between author-side completion and
    external review.

Secondary: adversarially review PR-TAL-015, especially the disjoint
first-core-selected events and the conversion from product probability
to singleton generator cost.
