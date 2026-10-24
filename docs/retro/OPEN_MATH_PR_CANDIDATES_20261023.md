# Open Math pull-request artifacts: retroactive source overlay (23 October 2026)

**Candidate inventory only — NOT A MATHEMATICAL RESULT.** Scientific effect: NONE. This file is the human-readable companion to [OPEN_MATH_PR_CANDIDATES_20261023.json](OPEN_MATH_PR_CANDIDATES_20261023.json).

## Boundary and source identity

- Frozen baseline: `d6g8k5htny-coder/Math-` default main at **`9f3fb6918b28e6378c034ebc19e76d432f5391cb`** (native complete recursive Git tree: 3,216 entries; `truncated:false`).
- Candidate population: **33 open Math pull requests**, including Draft and stacked branches; each exact-head PR's changed-file list was paginated. **#491 had 184 changed files (100+84)** and **#477 had 249 (100+100+49)**. A first page of 100 was never assumed complete.
- Published complement, not a replacement for the Chief of Staff's [313-unit *landed* Math/main inventory](https://github.com/d6g8k5htny-coder/main/pull/527): [original source-bound 33-PR overlay report](https://github.com/d6g8k5htny-coder/main/pull/527#issuecomment-6370536039), actual [pickup/release](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6370267167). The original 313-unit census, 291 proposed orphan units, graph statuses, GB-8a–x assignments and all mathematics remain with their existing owners.
- Scope: one candidate unit is the second-level directory path under `frontiers/`, `reviews/`, `coefficients/`, `companions/` or other research roots that is not present at the *frozen default cut* but appears in an **open PR delta**. A unit may contain many files, a sequence of revisions or tests. **It is not one theorem or one verified implementation.**

## Results at this immutable cut

| Area | Distinct pending units | Representative open PRs |
| --- | ---: | --- |
| Formal C293/C294/C296/C299/source-lifetime packages | 5 | Math-#482, #487, #488, #491 with stacked #496/#497, #477 |
| Compact-source-cutoff revision lineage | 7 | #414, #415, #424, #425, #426, #427, #430 |
| Fold-affine and LM006 coefficient proof/code/test packets | 3 | #389, #372, #380/#385 (one shared packet) |
| Engineering review-protocol/test directories | 2 | #379, #370 |
| **Total** | **17** | **Distinct directories, not 17 mathematical theorems** |

All 17 candidate directories are absent from the frozen Math-main tree and absent from the 313-unit landed ledger at its source cut. Other open PRs modify *existing* directories (for example elder-cusp parity in #492, and downstream hard-gate objects in #490/#397); they are not ignored, but do not increase this denominator.

Every proposed unit, open PR number and full head SHA, representative path, verification boundary and status are in the machine-readable JSON. Its baseline status is **`PENDING_OPEN_PR`**, never `accepted` or `landed`.

## Authenticated source-to-implementation samples

### C293 to C294 to C299 (formal source provenance, not theorem implication)

At [Math-#482](https://github.com/d6g8k5htny-coder/Math-/pull/482), the C293 `FINAL_SOURCE_INVENTORY.json` declares five Lean files. All **5/5** whole file bodies were read at its immutable PR head, matching expected UTF-8 byte lengths and SHA-256.

At [Math-#487](https://github.com/d6g8k5htny-coder/Math-/pull/487), C294 declares seven Lean files. All **7/7** whole file bodies likewise match. Three modules have the same Git blobs in both packages:

| Shared source module | Identical Git blob |
| --- | --- |
| `Lifetime.lean` | `f2508f7f2e3fd42f22f5aa52bc5ead5ff9e73b9a` |
| `TorusTopology.lean` | `ed678ea322fe5315b3aaff44e4e2f06b018399d8` |
| `MorseFiniteCritical.lean` | `989fabc6e8f04b0c4a36dff9b21d33b1299f7b92` |

[Math-#491](https://github.com/d6g8k5htny-coder/Math-/pull/491) C299's `HOSTED_SOURCE_MANIFEST.json`, Git blob `18d2570643e45e2de06e89ab1a96e9332b0f6064`, expressly declares those three modules' **origin at C294 head `7af365e8c8db1ae0275d4efe08a70a678f153ca2`**. This is a tangible historical *source byte/reuse* chain. It does **not** show C299's complete proof is accepted, that a build succeeded in this stage or that the three theorems are logically interchangeable. #496/#497 are stacked, different heads of that same prospective C299 packet.

### Distinct coefficient and fold-affine packets

At [Math-#385](https://github.com/d6g8k5htny-coder/Math-/pull/385), **4/4** LM006 `SOURCES.json` rows (`NOTE.md`, `oned.py`, `test_oned.py`, `RESULTS.json`) were whole-file byte/SHA-256/Git-blob verified at the proposed head. #380 touches the same unit and must not be counted again.

At [Math-#389](https://github.com/d6g8k5htny-coder/Math-/pull/389), **6/6** declared `SOURCE_FILES.json` rows, including `Contract.lean`, `FoldAffineBridge.lean`, `replay.py`, `test_contract.py`, README and its workflow, match their full declared identities. Across the C293, C294, LM006 and fold-affine samples, **22/22 manifest-listed instances** were authenticated. Neither their mathematical proofs nor their executable test suites were rerun in this audit. A hash helper that failed its own known-answer test was discarded before using the corrected, self-tested implementation.

## Fail-closed migration rule for future work

1. **While an associated PR is open**, keep exact commit/path/blob/declared source links as `PENDING_OPEN_PR`. Report missing bytes as `NOT_VERIFIED`, not as nonexistence or acceptance.
2. **For stacked PRs**, retain their ordered dependency/base relationships. Do not treat modified files from an unmerged parent as newly accepted source.
3. **Only after a protected, reviewed merge**, recheck the new default Math SHA and the exact landed source bytes, and update the existing CoS inventory through its current owner. A merge does not itself satisfy the separate mathematical predicates.
4. **Do not manufacture downstream-graph nodes for all 17 units.** The existing `frontiers/downstream_gate_20260925/GRAPH.json` is a specific proof-obligation graph. The complete artifact inventory and that graph have different semantics. Preserve original failures, theorem/status classifications and open premises.
5. Never infer that a `.py` file implements a proof from naming alone. These representative paths are candidates supported by their existing manifest or changed-file association; formal verification, actual execution, reviewer acceptance and dependency consumption require their own authenticated evidence.

This packet is documentation and source accounting only. It changes no workflow, hard gate, proof, Artifact Index identity, theorem classification, graph edge, permission or required check. Publication is a Draft proposal with separate review and guarded integration still pending.
