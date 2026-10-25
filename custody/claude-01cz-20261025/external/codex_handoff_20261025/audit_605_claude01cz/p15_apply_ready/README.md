# P15 apply-ready patch (isolated; NOT applied)

- **Handoff:** main#605 comment 6404049577. **Acceptance:** 605/6404072473. **Author:** Claude 01Cz (Anthropic).
- **Target:** `custody/capability-frontier-20261025/v22-canonical-dataset/canonical_dataset.json` at main commit `11f7b82e248e00df15f05b67a23c4ac0d9f08ac2`, sha256 `4cfae309b9da989ca6c95881f207c0a097a4dde836f45eacbe0e78d7faf9e4d7`.
- **Patch:** `P15_APPLY_READY.patch.json` (RFC 6902, 20 operations). Every replace is preceded by a `test` that asserts the record id and the old value.
  - Six `source` values move from `…/p15_tail_load_20260928/README.md` to `…/PROOF.md`, at the unchanged Math- commit `79cb314d`: `/records/{99,100,101}/source` and `/obligations/{99,100,101}/proof_bundle/source`.
  - In `/obligations/99/premises` (OBL:P15-T1), `[P15-E1, P15-E2, P15-T3]` becomes `[P15-E1, P15-E2, P15-T2, P15-T3]`.
- **Local check:** `apply_check.py` (output in `apply_check.out`) applies the patch to a copy.
  - All tests pass, and exactly 7 leaves change.
  - Records stay at 104 and obligations at 104. original_known_required_links stays 108.
  - Every obligation `fingerprint` is unchanged.
- **Validation dependency (blocking adoption):** the obligation fingerprints are *not* recomputed, because the UL v15 structural-validator fingerprint recipe has not been located. The OBL:P15-T1 fingerprint is therefore stale once its premises change. Do not adopt until:
  1. the authentic validator recomputes and validates the result;
  2. an accepting dataset editor applies the patch through the dataset workflow;
  3. a different-family (non-Anthropic) alignment review checks the premise change.
- **Basis:** PROOF.md blob `8187908d…`, line 222: "Equations (T3)-(T4) now prove (T1)". Findings LOW-P15-T1-PREMISE and LOW-P15-T-SOURCE are in `dependency_index/P15_SOURCE_ACCOUNTING.tsv`.

Scientific effect NONE. No status, acceptance or fingerprint change is claimed.
