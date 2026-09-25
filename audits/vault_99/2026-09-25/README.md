# 99_DO_NOT_OPEN forensic census — 2026-09-25

## CURRENT — continuation v2: INTAKE FROZEN; AUDIT INCOMPLETE

Read [INDEX_v2.json](INDEX_v2.json) first. It records all **80 observed Drive IDs**, the original 39-item baseline, 41 post-hold intake items, and exact hashes/IDs for the full JSON registry, Excel workbook and replay package. Older preliminary labels below are historical, not blanket review clearance. The current tracker clears four archive redundancy dispositions and leaves 76 rows open.

**Post-hold intake was real, not the earlier search omission.** The live external manifest reports 29 START_HERE copies moved at approximately 11:47 CT and 12 lane-index items at approximately 11:50 CT, after issue #91's 16:12:28Z (11:12 CT) freeze. The GitHub-only rule did not stop the Drive hygiene workflow. The owner-directed hold is now prepended, with revision-guarded writes, to the live external manifest and the live Active Research lane index. Original logs are retained. No vault source was moved, renamed, deleted, restored or edited by this continuation. No model was delegated vault access. This is a workflow hold, not a per-model ACL lock.

**Actual checks:** four vaulted ZIPs byte-match three freshly downloaded live KEEP files; all outer ZIP CRC checks pass. Fifteen raw historical sources are fingerprinted. All 6,748 data rows across eight CSVs were structurally parsed, with no inconsistent column counts. These are identity/structure checks, not mathematical proof or full semantic review.

**Research-branch correction:** default-main search was an inadequate absence test. Nine selected interfaces are crosswalked against research-hardening tip `077464ef5e2859ce98cbb9307799d5867a820eaf`. RN/JETMOD nodes, frozen-versus-register layers, preferred RN engine bindings and the scoped RN5 moment hold already exist. Do not replace these with older vault statuses.

**Executable component:** [code/reverse_impact.py](code/reverse_impact.py) computes reverse reachability over the union of old and new edges, including sub-obligations and removed dependencies. Its 22 distinct unittest controls passed in normal and optimized Python. The full tests, actual ledger fixture, logs, workbook and manifest are in the hash-bound replay package. No live research-CI gate deployment or mathematical acceptance is claimed. Issue #90 also clarifies that required REFUTED premises block, and supersession needs a reviewed edge replacement/removal.

Audit outputs are OUTSIDE the vault:
- Folder: https://drive.google.com/drive/folders/1RypKdFWeuADp7JFsGFSjeXJ035fa1bcL
- Excel: https://drive.google.com/file/d/19_ifS1viysWtGdN3kryb8vhSbPS1JNpi/view
- Full JSON registry: https://drive.google.com/file/d/1CzpACKNUKh6yj7kP-ehLVhyOeRooQpmt/view
- Replay package: https://drive.google.com/file/d/1JBbE6IiMERNC3vb21Sjku8J4LpVHdQFv/view
- Progress/scope report: https://drive.google.com/file/d/1Xk7b-22CjRj9NNFf-aQoQZoyiVaz7ju6/view

The vault does not reopen from a count, workbook, test pass or merge. Content/successor checks of the remaining records and actual graph-gate integration remain open.

---

## Historical first-pass record (retained verbatim below)

Owner-directed one-time audit. **Do not use this directory as mathematical authority.**

## Freeze
Drive vault: `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`.

During this pass no item is to be added, moved, renamed, deleted, restored, promoted, or used as authority. OpenAI/ChatGPT in the owner-directed session is sole auditor; assistance requires an explicit named delegation recorded in GitHub issue #91.

## Baseline
Provider direct-child enumeration on 2026-09-25 returned **39 direct children** when the folder itself was fetched. An earlier generic Drive search exposed only 26 and was incomplete; it is not used as the census boundary. `baseline.csv` now records all 39 direct children and preliminary classifications.

## Important early finding
The vault is not pure disposable clutter. Multiple `CENSUS_DUMP_NOT_HOLD` files contain historical mathematical maps, claim inventories, RN/JETMOD state, code/ZIP inventories, and source bindings. They remain non-authoritative, but unique facts must be diffed against current Git/Drive sources before final disposition.

## Manifest inconsistency
The initial generic Drive search was incomplete: it omitted 8 census CSVs and 5 logged binary/code vault items. Exact-ID checks and direct folder fetch verified those omitted objects are still parented in the vault. The live manifest's 23 census-dump claim matches the direct folder enumeration. This is a provider-enumeration pitfall, not evidence of deletion.

## Classification vocabulary
- SAFE_SUPERSEDED
- DUPLICATE_WITH_VERIFIED_SUCCESSOR
- IMPORTANT_EXTRACT_REQUIRED
- UNIQUE_RECOVERY_REQUIRED
- PROVENANCE_KEEP
- NEEDS_REVIEW

No classification promotes scientific status.

## Next checks
1. Reconcile live manifest rows against actual parent membership by exact Drive ID.
2. Diff `FULL_MATH_CLAIM_INVENTORY`, `LANE_MATH_MAP`, `LANE_RN_UNIF`, `FULL_DOCS_MATH_READ`, and code/ZIP census against current Git.
3. Bind any unique surviving fact to a modern Git source record without reviving stale statuses.
4. Only after every baseline item is terminally classified, release the vault freeze and install register-before-move intake.

## Hash-verified duplicate pass — 2026-09-25

The sole-auditor pass independently downloaded the relevant vaulted objects and live successors where available:

- PRE_REVIEW vault twin `19Th…` and EXACT_R2 vault twin `1djA…` both SHA256 `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`, exactly matching live PKG-05 `101yMW5gQO16crGZyGesOw9KEBYuB7dLV`.
- V3.4 audit-replay twin `1TSH…` SHA256 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`, exactly matching live closure `13QS9QQHxSiuLPIPSh9o5plS5HkClwSmz`.
- G9/R2 recovery twin `18Pd…` SHA256 `5393ec94ae6a05c31cc2fd22d45ff7667f9286c9f6b100c4425409786c7d1f58`; the existing external receipt independently records the same hash for live KEEP `1Uz1xeynsY17xHDTfisVC6ImztJOXZbFp`.
- The 7201-byte `rnu_ds3_scalar_SUPERSEDED.py` is **not a byte duplicate** of the 9704-byte preferred `rnu_ds3.py`: old SHA `c3d5adc8…116da`, successor SHA `bd3074fd…d9421`. Its classification is therefore `SAFE_SUPERSEDED`, not `DUPLICATE_WITH_VERIFIED_SUCCESSOR`.
- The orphan PRE_REVIEW SHA sidecar contains the same digest line as live PKG-05 sidecar `1OL1Ht3_As41rWfy-BsLHCaD1De-pyuuj`.

No Drive object was moved, renamed, deleted, restored, or promoted.

## Important-content triage

Sampled census bodies confirm the early warning: these are not safe to discard as a class. In particular, `LANE_MATH_MAP`, `LANE_RN_UNIF`, `FULL_DOCS_MATH_READ`, `FULL_MATH_CLAIM_INVENTORY`, `FULL_ZIP_AND_CODE_CENSUS`, and `FULL_CODE_CENSUS` carry historical source IDs, hashes, dependency statements, and open-status evidence. Exact GitHub code searches for several distinctive census names/hashes returned no match on current `main`. They remain `IMPORTANT_EXTRACT_REQUIRED` until the unique facts are source-bound into modern Git records or matched to exact live successors.
