# Source and novelty register — 2026-09-16

These are source identities and scoped uses, not automatic acceptance of everything in a source. No outside proof was silently merged into a current claim. Primary papers were checked for the exact discrete-convexity statement and current frontier. The local proofs are self-contained except for explicitly reused elementary/classical results.

## External sources

**S1. Ruben Ascoli, Xiaoyu He, Jinyoung Park, Michel Talagrand, _A reformulation of the discrete Convexity Conjecture via k-thresholds_, arXiv:2608.11183v1.**
https://arxiv.org/html/2608.11183v1
https://arxiv.org/abs/2608.11183
Used: Definition1.1, p-small cover definition, Conjecture1.2 and its footnote distinguishing the original no-dilution formulation. The source says the discrete problem remains open while the continuous problem has a recent solution. It supplies the TARGET, not a proof of our laminar theorem. It also contains substantial graph-family results; our restricted result is not claimed to dominate those.

**S2. Jinyoung Park, _A dimension-free comparison between expectation thresholds and fractional expectation thresholds_, arXiv:2609.14681v1, 13 September2026.**
https://arxiv.org/html/2609.14681v1
The reported comparison is q_f(F)<=Kq(F)max(1,loglog(1/q(F))). This is a dated primary preprint, not a statement of prize adjudication. Its loglog factor is not a universal constant. We inspected its definitions, main theorem, and stated relevance to discrete convexity; this package does not reproduce or extend its general proof.

**S3. _On p-Spread Measures_, arXiv:2609.08967.**
https://arxiv.org/abs/2609.08967
Related primary lead, cited by S2 for fractional discrete convexity. Only abstract/metadata were consulted in this pass. No new theorem consumes its proof.

**S4. Tara Fife and James Oxley, _Laminar Matroids_, arXiv:1606.08354; European Journal of Combinatorics62 (2017),206--216.**
https://arxiv.org/abs/1606.08354
https://repository.lsu.edu/mathematics_pubs/1189/
The capacity-defined downsets considered here are a standard class: independent sets of laminar matroids. Laminar representations and removal of redundant capacity constraints are not new inventions. Our balanced-coloring argument is supplied directly; no claim is made of originating matroid union/coloring.

**S5. Michael Elkin, _An Improved Construction of Progression-Free Sets_, arXiv:0801.4310, and the classical Behrend construction as written in Phase03.**
https://arxiv.org/abs/0801.4310
Only the elementary sphere construction already proved in the baseline is consumed. No claim that we invented the Behrend density lower bound, or that our constants improve the classical construction.

**S6. Thomas Bloom and Olof Sisask, _Breaking the logarithmic barrier in Roth's theorem on arithmetic progressions_, arXiv:2007.03528.**
https://arxiv.org/abs/2007.03528
Used by Phase03 for the harmonic k=3 endpoint. The new all-scale counting proof instead assumes sigma>1 and does not require a new Roth estimate. Do not confuse the already-known k=3 case with the remaining all-length prize question.

## Baseline

The complete original Phase03 ZIP is preserved under baseline with its SHA256 in the intake receipt. The read inputs were PR-AP-009,010,012, NEXT_WORK, and the review packet. Hash verification confirms identities, not truth. The new theorem PR-AP-013 strengthens the sigma>1 limsup statement to a full limit; it does not retroactively mark Phase03 independently reviewed.

## Bounded priority search, not a novelty certificate

Searches included combinations of `Talagrand discrete convexity laminar matroid`, `Talagrand partition matroid`, `convexity matroid union`, `arithmetic progression reciprocal maximizer`, and `progression-free weighted greedy`. They identified S4 as required structural prior work but did not establish whether the exact constants or proof assembly in PR-TAL-001/002 have appeared before. Text searches of S1/S2 for matroid/laminar were not matches; absence of those words is weak evidence and cannot establish priority.

The weighted extremizer and greedy-limit statements may be standard consequences in weighted extremal combinatorics. The local-marginal construction is an elementary consistency obstruction, not a claimed novel hierarchy lower bound. All new historical novelty fields remain UNESTABLISHED.

No researcher was contacted, no correction sent, no formal artifact from another project rebuilt, no prize submitted, and no full original prize solution claimed.
