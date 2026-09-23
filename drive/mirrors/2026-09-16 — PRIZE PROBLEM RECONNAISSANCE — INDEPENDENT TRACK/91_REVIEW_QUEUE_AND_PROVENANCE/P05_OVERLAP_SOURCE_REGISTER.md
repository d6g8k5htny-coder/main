# Phase05 source and priority register

Accessed 2026-09-16. Research was based on the exact prior phase files and the primary sources below. All Phase05 theorems are author-side arguments; no external reviewer or formal proof assistant has endorsed them. A bounded keyword search is not evidence of historical novelty.

S1. Ruben Ascoli, Xiaoyu He, Jinyoung Park, Michel Talagrand, *A reformulation of the discrete Convexity Conjecture via k-thresholds*, arXiv:2608.11183v1. https://arxiv.org/html/2608.11183v1
Used for Definition1.1 (k unions, obstruction, p-smallness), Conjecture1.2 and its original no-dilution formulation. It distinguishes the discrete problem from the continuous one and gives other graph-property results. Its general theorems are not used to establish the present restricted capacity results. The statement here is a direct restriction of its downset formulation, not a claim that matching or graph-containment results are being superseded.

S2. Dmytro Gavinsky, Shachar Lovett, Michael Saks, Srikanth Srinivasan, *A Tail Bound for Read-k Families of Functions*, arXiv:1205.1478. https://arxiv.org/abs/1205.1478
Prior context for bounded coordinate dependence, fractional Holder/Finner/Shearer inequalities. The finite product inequality needed here is proved coordinate by coordinate in PR-TAL-004. Read-k is standard terminology and not new research architecture terminology invented by this project.

S3. Robin A. Moser, Gabor Tardos, *A constructive proof of the general Lovasz Local Lemma*, arXiv:0903.0544v3. https://arxiv.org/abs/0903.0544
Standard local-lemma and variable-resampling framework. PR-TAL-008 writes out the elementary finite asymmetric existence proof. The finite companion's successful resampling runs are examples, not a reconstruction of the full Moser--Tardos runtime theorem. No new Local Lemma or efficient general derandomization is claimed.

S4. Jinyoung Park, *A dimension-free comparison between expectation thresholds and fractional expectation thresholds*, arXiv:2609.14681v1. https://arxiv.org/html/2609.14681v1
Current context: dimension-free comparison with a max(1,loglog(1/q)) factor. Not consumed as a theorem premise. The present structural-subclass constant bounds do not remove that factor for arbitrary families.

S5. Tara Fife and James Oxley, *Laminar matroids*, arXiv:1606.08354. https://arxiv.org/abs/1606.08354
Classical identification of laminar capacity downsets. Laminar balancing, bipartite edge coloring, Hall augmenting paths, Cantelli, symmetric-polynomial estimates, and fractional Holder are established methods. Our exact constant assembly and subclass boundaries require separate priority review.

Prior project sources copied byte-exactly in prior_sources/: PR-TAL-001 and PR-TAL-002 from Phase04. Their sha256 identities are included in MANIFEST.sha256 and the prior-source register. The all-rank coefficients of PR-TAL-001 are re-evaluated as exact Fractions; computational checks do not replace its written all-rank argument.

## Priority boundary

Targeted searches included Talagrand convexity with bounded degree, read-k, laminar families, and matchings; these did not establish a prior result with exactly these constants. That is not a novelty certificate. The safe description is: restricted results derived within this campaign from standard combinatorial/probability tools, available for independent verification and priority comparison.

The earlier reconnaissance document is preserved historically. Its initial scouting claims, prize valuations, and model-news assertions are not automatically carried forward as verified current facts. No prize-success percentage or monetary valuation is inferred here.
