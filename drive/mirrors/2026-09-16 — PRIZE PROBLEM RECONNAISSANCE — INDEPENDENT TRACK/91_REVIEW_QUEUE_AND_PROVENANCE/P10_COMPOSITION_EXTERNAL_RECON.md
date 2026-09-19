# P10-COMPOSITION — External reconnaissance

Task: extend Phase09-G block compression to disjoint-variable monotone substitutions, especially arbitrary-depth threshold trees, without multiplying palettes. Input: Phase09 ZIP SHA-256 03683fb71b0ca550dbcff87329fb9702dfddfe0852804a538e3622d8877f7295. Session: P10-20260917T0204Z. Local campaign date: 2026-09-16 America/Chicago.

Policy: OP-RECON-20260916-v1.0. Mode: FRESH. These are author-side searches; no blind-review or organizational independence is claimed.

Queries actually executed through web search (2026-09-17 UTC):
1. “Talagrand discrete convexity conjecture hypergraph substitution read once monotone Boolean formulas”
2. “hypergraph blow up chromatic number smallness expectation threshold block substitution”
3. “Talagrand convexity conjecture bounded rank product measures rank independent k thresholds”
Follow-up exact-phrase queries are recorded in QUERY_LOG.json. Only public problem terminology was disclosed.

Sources inspected:
- Ascoli–He–Park–Talagrand, arXiv:2608.11183v1, https://arxiv.org/html/2608.11183v1, Definitions 1.1/1.3, Conjecture 1.2, Remark 1.6, Theorem 1.8: FULL_TEXT. Exact definition and quantifier reference; graph-containment reduction is PARTIAL, not a theorem for arbitrary read-once threshold trees.
- Park–Pham, https://arxiv.org/html/2203.17207, introduction/Theorem 1.1: FULL_TEXT. Bounded-rank existence is already known via rank-dependent dilution; do not claim new existence on that basis.
- Pham, https://arxiv.org/html/2412.03540v1, abstract/introduction: FULL_TEXT. Bounded-support fractional rounding is related, not automatically applicable without a fractional certificate.
- “Functions that are read-once on a subset of their inputs,” DOI 10.1016/0166-218X(93)90105-W: discovery abstract only; full DOI open failed. Read-once representation has extensive prior literature. AND/OR read-once functions differ from the threshold-gate class to be defined.

Assessment: no source located within these searches and access limits establishing the exact proposed common-palette/hazard-cover theorem. This is not a novelty certificate. Develop the exact substitution identity, then a scalar Bernoulli-tail inequality, before claiming arbitrary-depth closure. Explicitly test shared-variable counterexamples; no independence may be assumed for overlapping children.

Disposition: PROCEED_WITH_INSIGHTS after this memo's raw upload/readback. Restricted candidate proofs remain author-side. The unrestricted conjecture and historical novelty remain open. No q0 source, status, or Claude work is consumed or modified.

Custody: separate SHA-256/size sidecar and deployment receipt; no circular self-hash.
