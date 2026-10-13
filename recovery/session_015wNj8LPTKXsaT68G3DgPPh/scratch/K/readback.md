## DELIVERED — shard K (`certificates/**`) Wave 0 shadow record; review requested; claim continues until review

Dylan Roy — delegated AI evidence maintenance. Actual performer: Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). This completes the inventory for assignment 6018787719 / PICKUP 6018824915. Scientific effect NONE.

- **Record hash:** everything below the horizontal rule is the record. Its file bytes are **10908 B, SHA-256 `8dbee606deeb75a976f695fd27e5fd5ca46f2327839fb16a9df618154c4df185`**, without the trailing footer.
- **Result:** 14/14 objects accounted for. 8 are already compliant, 6 need additive supplements (K-S1 to K-S7), and none are blocked or out of scope. Two blocked axes (B-1 mathematical review, B-2 custody trust anchor) remain visible and are not counted as finished.
- **Execution:** one new, labeled targeted run (K-S4). Nothing in the repository was written.
- **Review request:** one named nonauthor review of this exact record, by a **non-Claude** lane chosen by the coordinator. The reviewer should check each table row against the native objects, plus the K-S3, K-S4 and K-S6 facts. Wave 1 conversion to `reviews/retrofit_20261006/K_certificates/` waits for the landed contract (main#276) and a separate integrator.
- **Routing:** B-2 and the optional Hessian `cmp` step go to the workflow's owner, through you. This shard edits no workflow.
- **Status:** the claim stays held for answering review findings until 18:00Z. If no review is assigned by then, I release it with the record frozen as is. No 15:30Z checkpoint is needed, since delivery came first.

---

### Shard K record — `certificates/**` evidence inventory (Wave 0 shadow, proposed, not committed)

**Scope.** Math- `08f86862f859ac4804b4fd6c6477a7ed0421e37f`. `certificates/` tree `ebed1369bd8599d96087cf2df0538647a5b7a6a9`, 14 blobs. Assignment 6018787719, PICKUP 6018824915.
- **Performer:** Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Exposure: none to these paths before this pickup. Credit 0, scientific effect NONE.
- **Requirements applied.** Classes are judged against the requirements in force (AGENTS.md, OP-WORKFLOW-20260930, FORMAL_REQUIRED_CHECKS) and against the **proposed** v1.1 evidence axes. Main#276 and Math-#381 are unmerged, so no class here applies a landed closure contract.
- **No change to the objects.** Nothing in the 14 objects or in any shared file was edited.

#### 1. Membership (14/14, all authenticated at base)

| # | Path (under `certificates/`) | Git blob | Bytes | SHA-256 (prefix) | Class | Supplement IDs |
|---|---|---|---:|---|---|---|
| 1 | `bf_six_pin_hessian_20260926/CERTIFICATE.json` | `085bd198078a91a3bd3d8a879a526740fd7ae6a4` | 1252 | `db781538` | needs additive supplement | K-S4, K-S7 |
| 2 | `bf_six_pin_hessian_20260926/PROOF.md` | `201a29fa0fd724feb3f3a06693700dc051b306bb` | 7552 | `83aa1749` | needs additive supplement (review axis blocked) | K-S1, K-S2, K-S5, K-S6 |
| 3 | `bf_six_pin_hessian_20260926/RECONNAISSANCE.md` | `91c3b930232e5e30ddbd9ac9603e87ae7e87a65c` | 1864 | `7fdc141b` | already compliant | (K-S6 records its citations) |
| 4 | `bf_six_pin_hessian_20260926/SOURCE_MANIFEST.json` | `169de6966f53b1f7a4afe60416b32475bbd2f1c3` | 1497 | `a0e67573` | already compliant | none |
| 5 | `bf_six_pin_hessian_20260926/hessian_certificate.py` | `933d7afa4355d7a63cfcc67f39e868c850355760` | 6079 | `ce238132` | already compliant | (K-S4, K-S5 cover execution) |
| 6 | `bf_six_pin_hessian_20260926/test_hessian_certificate.py` | `bb620109be099a168f9a0517e6389d748697efed` | 6266 | `9392a1f2` | already compliant | none |
| 7 | `bf_six_pin_scalar_20260926/CERTIFICATE.json` | `1fae1333027d212fd05645f765c398b9d847ead8` | 688 | `266880aa` | needs additive supplement | K-S4, K-S7 |
| 8 | `bf_six_pin_scalar_20260926/PROOF.md` | `09944573562981401059bb31c68ab2909a28fd0c` | 4654 | `9e5edf50` | needs additive supplement (review axis blocked) | K-S1, K-S2, K-S5, K-S6 |
| 9 | `bf_six_pin_scalar_20260926/RECONNAISSANCE.md` | `b188fefc7edaac47188b593ceed440ad3c1f2de9` | 1618 | `53e92980` | already compliant | (K-S6 records its citations) |
| 10 | `bf_six_pin_scalar_20260926/SOURCE_MANIFEST.json` | `d1838a08748b9f77feb4e4afec6425b2db4e51f1` | 1479 | `25288c80` | already compliant | none |
| 11 | `bf_six_pin_scalar_20260926/certificate.py` | `ca16ed3ee490257f647b187d588023f154a69180` | 5631 | `5b3e5dd4` | already compliant | (K-S4, K-S5 cover execution) |
| 12 | `bf_six_pin_scalar_20260926/test_certificate.py` | `ff5b7732413c0ceea928a3f393ee674b3bfb599e` | 5436 | `13a43033` | already compliant | none |
| 13 | `bf_source_integrity.py` | `e0a225994d053ab9fcfef15c732c6ed6d4055155` | 3206 | `7161e490` | needs additive supplement | K-S1, K-S2, K-S3 |
| 14 | `test_bf_source_integrity.py` | `48c671550f751d5b0e885f1c5cbf288d9337c76f` | 4683 | `d9ce1591` | needs additive supplement | K-S3 |

**Counts:** 14 eligible = 8 already compliant + 6 needing additive supplements + 0 blocked + 0 out of scope. Two evidence **axes** are blocked; they are listed in §3, are not counted as objects, and are not counted as finished.

**Identity checks:**
- The certificates tree is `ebed1369` at all four points checked:
  - #89's CI-tested head `eb41d3e4bc3a601d7414c7f52d209e68bc282bbb`;
  - merge commit `0be9c764eb5bb0158ebb5fb51b2d09d3d604d2f0`;
  - head `458d4da3` of hosted run 37176387961;
  - this base.
- The only commit touching these paths up to base is `0be9c764` (#89).
- Both SOURCE_MANIFEST.json files bind their 5 package files at `source_commit` `1ec186bb529d36b90d498d34063b0bd4c022639a` (that commit exists in the history, parent `81de9a6a`). All 10 bound blobs equal the base blobs.
- `bf_source_integrity.py` line 14 pins both manifest SHA-256 values, and they equal the base manifests.

#### 2. Proposed additive supplements (Wave 1 only, after a landed contract; facts are recorded as found)

- **K-S1, provenance.**
  - Both PROOF.md headers say `Author: OpenAI / ChatGPT`, a self-declared header. The branch is `chatgpt/bf-scalar-certificate-20260926`.
  - Commits `81de9a6a`, `1ec186bb` and `e2e427e9` carry Dylan Roy as git author with no session trailer, so the **session/model is UNKNOWN**.
  - Verifier and test successor `85979c15` is authored by `copilot-swe-agent[bot]`, requested in issue comment 5856595796 and reported in 5856625765.
  - Merge commit `eb41d3e4` was authored by Cursor Agent. `cursor[bot]` merged the PR at 2026-09-27T17:38:51Z. **The integrating provider is UNKNOWN** beyond that bot identity.
- **K-S2, review record.**
  - PR #89 has exactly three reviews, all `copilot-pull-request-reviewer[bot]`, all COMMENTED, none APPROVE:
    - 5328555779 @`1ec186bb`, which requested manifests;
    - 5328575240 @`e2e427e9`, which flagged the verifier boundary and CI enforcement;
    - 5330629317 @`e2e427e9`, which said the verifier and workflow are PR-controlled and need a protected trust anchor.
  - Both review threads are marked resolved; the **resolver is UNKNOWN**:
    - RC 4113833623 (symlinked `--root`) was addressed by `85979c15`;
    - RC 4115656719 (trust anchor), posted on the final head `eb41d3e4`, has **no later code change**.
  - **No nonauthor mathematical review record exists.** This agrees with navigation row 37 ("no nonauthor review record found") and the PR body ("The reviewer did NOT approve the mathematical argument"). Disposition stays AUTHOR_DERIVATION / REVIEW_REQUIRED. Automated custody reviews are not mathematical review and earn no independence credit.
- **K-S3, stale identity in PR #89's body.**
  - The body states `bf_source_integrity.py` as 3110 B, SHA-256 `5a8bf4be…`. That is the `e2e427e9` version, blob `355cb3e2`.
  - The base file is the Copilot successor `e0a22599` (3206 B, SHA-256 `7161e490…`). The test changed from `d06ef6f1` (4485 B) to `48c67155` (4683 B).
  - The supplement records the successor identity next to the PR body's. It neither edits the body nor declares it wrong for its own commit.
- **K-S4, recorded versus observed execution.**

  | Kind | Record | Commit | Workflow blob | Result |
  |---|---|---|---|---|
  | recorded (author comment 5851780538) | runs 36286057342 (BF), 36286057295 (downstream) | `81de9a6a`, first commit only | (at that head) | success. The "16/16 local, both modes" in that comment is author-side |
  | recorded hosted | run 36337635067 `scalar-arithmetic`; run 36337635091 `math-downstream-gates` | final head `eb41d3e4` | `9fef6d25` | success, 2026-09-27T17:37–38Z |
  | recorded hosted | run 37176387961, all steps (four other 2026-10-03/04 runs succeeded and were not audited individually) | `458d4da3` (identical certificates tree) | `ccba07ef` (after Dependabot `6b254fde`, 2026-10-04) | success |
  | **new observation (C8, 2026-10-06, local CPython 3.11.15, clean base worktree)** | `hessian_certificate.py` and `certificate.py` with no arguments, normal and `-O` | base | not applicable | stdout byte-identical to the stored CERTIFICATE.json in both modes (`db781538…`, `266880aa…`) |

  - The workflow has **no push trigger**, so no run exists on the merge commit or on main.
  - `bf-six-pin-scalar` is **not a required check**, and no required workflow runs these tests.
  - The new observation closes one specific gap: CI regenerates and compares the scalar certificate, but for the Hessian it runs only `--check` plus mode equality. It is not historical runtime evidence and not a review.
- **K-S5, where the commands come from.**
  - Prose: scalar PROOF.md lines 84–87 and Hessian PROOF.md lines 167–170 (`python3 -B -S …`).
  - Machine: workflow `.github/workflows/bf-six-pin-scalar.yml` `ccba07ef` (`python` 3.11). The workflow adds a scalar regeneration with `cmp`, and has no Hessian byte comparison.
  - RESULTS-style argv: not recorded.
- **K-S6, consumed versus cited.**
  - Consumed repository sources: **none**. Each analytic dependency is the packet's own PROOF.md.
  - The Hessian proof re-derives b(r) itself (PROOF.md lines 57–66) and cites the scalar package at `81de9a6a` (PROOF SHA-256 `9e5edf50…`, equal to base). That is the same author lineage, **not independent evidence**.
  - Cited only:
    - Math-#87 @`0fab9330` (merged; noted as having a denominator-order defect; re-derived, not imported);
    - Rivera–Vanneuville arXiv:1711.05012 and the Python-FLINT Arb docs (scalar);
    - NIST DLMF 4.2.19 and Beliaev–Cammarota–Wigman arXiv:1911.03455v3, abstract only (Hessian).
  - Both reconnaissance notes disclaim novelty and priority.
- **K-S7, aliases and normalization.**
  - Citers: `PROOF_INDEX.md:81` and `docs/integration/2026-09-29-navigation-refresh.md:37`. These are citations, not extra evidence.
  - No GRAPH node, no other Math- consumer and no reference in main @`cddbb7f6`.
  - Object IDs: `OA-BF-SIX-PIN-HESSIAN-20260926-v1`, with an `object_id` field and `review_status`; and `OA-BF-SIX-PIN-SCALAR-20260926-v1`, stated only in PROOF.md, whose certificate has no `object_id` and uses the key `review`. A supplement records the mapping without editing either file.

#### 3. Blocked axes (outside evidence maintenance; visible, not finished)

- **B-1, nonauthor mathematical review of both PROOF.md files.** Not recorded since 2026-09-27. Only an actual eligible reader can supply it. This record supplies nothing toward it and does not change REVIEW_REQUIRED.
- **B-2, protected trust anchor for custody (Copilot 5330629317 / RC 4115656719).** The verifier and workflow remain PR-controlled, and the thread was resolved without a change. Any fix is a workflow or required-check design decision.
- **Routed, not edited:** an optional Hessian regeneration-and-`cmp` step, and the trust-anchor question. Both go to the coordinator for the workflow's owner. This shard writes no workflow.

#### 4. Validation and limits

- **Checked:**
  - 14/14 coverage;
  - full blob, tree and SHA-256 identities;
  - the manifest-to-base match;
  - verifier pins;
  - tree equality across the four commits;
  - every citation of these objects in both pinned baselines;
  - no promotion of the CERTIFICATE `REVIEW_REQUIRED` fields.
- **UNKNOWN stays UNKNOWN:** author session and model, review-thread resolver, integrator provider.
- **Not done:** no unit-test replay. Recorded hosted runs over identical bytes exist, and per the assignment I reused them. No theorem, proof or review claim. Byte identity establishes no mathematics.
- **Delta queue:** empty. Main is still `08f86862`, with no post-cut change to these paths.


---
_Generated by [Claude Code](https://claude.ai/code)_
