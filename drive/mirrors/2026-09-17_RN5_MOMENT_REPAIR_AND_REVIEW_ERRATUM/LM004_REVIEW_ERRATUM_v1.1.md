# LM004 review erratum v1.1 — false duplicate-term finding withdrawn

2026-09-17 · ROUND5-20260917-b9c2 · OpenAI / Codex.

The earlier reconciliation report incorrectly stated that the derivative supremum appears twice in the frozen C1 norm. The actual source contains it exactly once. **Withdraw that finding and the suggested source cleanup. The frozen mathematical document needs no edit for this issue.**

## Exact identities

Affected report: `REV-AUTO-RV-LM004-20260917165201`, Drive `1CHErajW7G2y8y7Mr0bKYMy2lJbA2HIAt`, 5,093 bytes, SHA-256 `c0571b7affc23c0d44e3479cb7ebafafbcfd49001a43097d43f00553f1d75f81`.

Mathematical target: LCR-DER-016-v1.1 / P02-LM-004, Drive `1tPt0KKJEKE6c-7xNwJW82dooRkDyANnToRErc3oWX84`. The unique body between its line-anchored frozen markers, normalized to LF with one final LF, remains **10,416 bytes**, SHA-256 `d43179f3abfd7e93f988725f3b89a5826cc71e0fa04840f65c7f93f3359d0f87`.

Fresh complete export: 13,321 bytes, SHA-256 `acdc92856ff4670a4cf1cbda0bc328e99b648d55153c2e39a3dfe43861e200ee`.

## Cause and corrected A1 entry

An earlier display read overlapped line 180 in ranges 27–180 and 180–360. That duplicated a line in the displayed output, not in the document. A second overlap at line 360 had the same presentation mechanism. The current check extracts the entire frozen body and counts the norm-section terms directly.

Corrected A1: **PASS.** The norm is the maximum of one field supremum and one operator-Jacobian supremum. Euclidean vector norm, operator norm, gradient symmetry, open neighborhood, and exact pins are explicit. There is no duplicate-term source defect.

The reproduced norm section has exactly one `sup_{x in D} ||E(x)||_2` and one `sup_{x in D} ||DE(x)||_op`.

## Technical disposition and validation limits

The prior exact-body technical reconciliation remains the recorded disposition. This erratum adds no new full theorem review and changes no accepted independence predicate. The outstanding LCR-REQ-039 obligation remains one qualifying organizationally distinct exact-object verdict on the body above.

The replacement verifier reproduces the full source identity and checks the affected display claim directly. It also checks the cone losses with exact rational squared comparisons:

\[
8192^{-2}(1+16^{-2})<8000^{-2},\qquad
8192^{-2}(1+1/16)^2(1+16^{-2})<7000^{-2}.
\]

It verifies 63/128-3/64-1/8192>44/100 and -3+1/4=-11/4. These are exact arithmetic checks. They do not automate the imported geometry, unstable-branch argument, or theorem composition. The previous report's blanket verifier wording should be read with that limitation; no unavailable old script is relied upon here.

## Lineage and restoration

This is an **author-line correction of the earlier OpenAI review**, with full exposure to the target and prior conclusions. It earns zero organizational-independence credit. The broader RN5 work also continues the OpenAI author lineage.

Retain the old report's bytes as historical evidence, mark its A1 duplicate/cleanup claim superseded, and route the current queue to this numbered erratum. The old report may be consulted for history together with this correction. Restoring its original filename or parent does not restore the false finding's authority. No frozen mathematical source is changed.

Future source-defect findings should bind to one reconstructed source buffer or use disjoint line ranges with original line coordinates. A repeated line in a tool display alone is insufficient evidence of duplicate content.

Reproducible extraction and checks: `round5/verify_round5.py` in the RN5 delivery bundle; result `round5/output/VERIFICATION.json`.
